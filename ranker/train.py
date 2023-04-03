import time
import numpy as np
import pandas as pd
import torch
from torch_geometric.graphgym import optim

from ranker.comparison import davidScore, serialRank_matrix
from ranker.metrics import calculate_upsets, generation_accuracy
from ranker.model import DIGRAC_Ranking
from ranker.param_parser import parameter_parser, add_min_args
from ranker.preprocess import load_data
from ranker.utils import scipy_sparse_to_torch_sparse, get_powers_sparse
from scipy.stats import kendalltau, rankdata

device = 'cpu'
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
GNN_variant_names = ['dist', 'innerproduct', 'proximal_dist', 'proximal_innerproduct', 'proximal_baseline']
NUM_GNN_VARIANTS = len(GNN_variant_names)  # number of GNN variants for each architecture

upset_choices = ['upset_simple', 'upset_ratio', 'upset_naive']
NUM_UPSET_CHOICES = len(upset_choices)
args = parameter_parser()
args = add_min_args(args)
torch.manual_seed(args.seed)
save_name_base = "generation"
method_name = "DIGRAC"
random_seed = 10
model, split, new_path = None, None, None


def adj_to_edges(A):
    src = []
    dst = []
    weight = []
    for a in range(0, A.shape[0]):
        for b in range(0, A.shape[0]):
            if A[a][b] != 0:
                src.append(a)
                dst.append(b)
                # the output of the line below is a numpy array
                tensor_array = A[a][b].cpu().detach().numpy()

                weight.append(tensor_array.min())
    df = pd.DataFrame({"Src": src, "Dst": dst, "Weight": weight})
    return df


def evalutaion(logstr, score, A_torch, label_np, val_index, test_index, SavePred, save_path, split, identifier_str):
    kendalltau_full = np.zeros((3, 2))
    kendalltau_full[:] = np.nan
    if score.min().detach().item() < 0:
        if score.min().detach().item() > -1:
            score = (score + 1) / 2
        else:
            score = torch.sigmoid(score)
    upset1 = calculate_upsets(A_torch, score, device=args.device)
    upset2 = calculate_upsets(torch.transpose(A_torch, 0, 1), score, device=args.device)
    if upset1.detach().item() < upset2.detach().item():
        upset_ratio = upset1
    else:
        upset_ratio = upset2
        score = -score
    pred_label = rankdata(-score.detach().cpu().numpy(), 'min')
    upset_simple = calculate_upsets(A_torch,
                                    torch.FloatTensor(-pred_label.reshape(pred_label.shape[0], 1)).to(args.device),
                                    device=args.device,
                                    style='simple').detach().item()
    upset_naive = calculate_upsets(A_torch,
                                   torch.FloatTensor(-pred_label.reshape(pred_label.shape[0], 1)).to(args.device),
                                   style='naive', device=args.device).detach().item()
    upset_full = [upset_simple, upset_ratio.detach().item(), upset_naive]
    if SavePred:
        np.save(save_path + identifier_str + '_pred' + str(split), pred_label)
        np.save(save_path + identifier_str + '_scores' + str(split), score.detach().cpu().numpy())

    logstr += '\n From ' + identifier_str + ':,'
    if label_np is not None:
        # test

        tau, p_value = kendalltau(pred_label[test_index], label_np[test_index])
        generation_accuracy(pred_label, adj_to_edges(A_torch), save_path, identifier_str, generation=args.generation,
                            tau=tau)
        outstrtest = 'Test kendall tau: ,{:.3f}, kendall p value: ,{:.3f},'.format(tau, p_value)
        kendalltau_full[0] = [tau, p_value]

        # val
        tau, p_value = kendalltau(pred_label[val_index], label_np[val_index])
        outstrval = 'Validation kendall tau: ,{:.3f}, kendall p value: ,{:.3f},'.format(tau, p_value)
        kendalltau_full[1] = [tau, p_value]

        # all
        tau, p_value = kendalltau(pred_label, label_np)
        outstrall = 'All kendall tau: ,{:.3f}, kendall p value: ,{:.3f},'.format(tau, p_value)
        kendalltau_full[2] = [tau, p_value]

        logstr += outstrtest + outstrval + outstrall
    logstr += 'upset simple:,{:.6f},upset ratio:,{:.6f},upset naive:,{:.6f},'.format(upset_simple,
                                                                                     upset_ratio.detach().item(),
                                                                                     upset_naive)
    return logstr, upset_full, kendalltau_full


class Trainer(object):
    """
    Object to train and score different models.
    """

    def __init__(self, args, random_seed, save_name_base, test=False):
        """
        Constructing the trainer instance.
        :param args: Arguments object.
        """
        self.args = args
        self.device = args.device
        self.random_seed = random_seed

        self.label, self.train_mask, self.val_mask, self.test_mask, self.features, self.A = load_data(args, test)
        self.features = torch.FloatTensor(self.features).to(args.device)
        self.args.N = self.A.shape[0]
        self.A_torch = torch.FloatTensor(self.A.toarray()).to(device)

        self.nfeat = self.features.shape[1]
        if self.label is not None:
            self.label = torch.LongTensor(self.label).to(args.device)
            self.label_np = self.label.to('cpu')
            self.args.K = int(self.label_np.max() - self.label_np.min() + 1)
        else:
            self.label_np = None
        self.num_clusters = self.args.K

        save_name = save_name_base + 'Seed' + str(random_seed) + 'Dimensions' + str(self.features.shape[1])

        self.log_path = os.path.join(os.path.dirname(os.path.realpath(
            __file__)), args.log_root, args.dataset, save_name, str(args.generation))

        if os.path.isdir(self.log_path) == False:
            try:
                os.makedirs(self.log_path)
            except FileExistsError:
                print('Folder exists!')

        self.splits = self.args.num_trials
        if self.test_mask is not None and self.test_mask.ndim == 1:
            self.train_mask = np.repeat(
                self.train_mask[:, np.newaxis], self.splits, 1)
            self.val_mask = np.repeat(
                self.val_mask[:, np.newaxis], self.splits, 1)
            self.test_mask = np.repeat(
                self.test_mask[:, np.newaxis], self.splits, 1)

    def train(self, model_name):
        self.model_name = model_name
        #################################
        # training and evaluation
        #################################
        if model_name in ['DIGRAC', 'ib']:
            if self.args.upset_ratio_coeff + self.args.upset_margin_coeff == 0:
                raise ValueError('Incorrect loss combination!')
            # (the last two dimensions) rows: test, val, all; cols: kendall tau, kendall p value
            kendalltau_full = np.zeros([NUM_GNN_VARIANTS, self.splits, 3, 2])
            kendalltau_full[:] = np.nan
            kendalltau_full_latest = kendalltau_full.copy()

            upset_full = np.zeros([NUM_GNN_VARIANTS, self.splits, NUM_UPSET_CHOICES])
            upset_full[:] = np.nan
            upset_full_latest = upset_full.copy()

            args = self.args
            A = scipy_sparse_to_torch_sparse(self.A).to(self.args.device)
            if model_name == 'DIGRAC':
                norm_A = get_powers_sparse(self.A, hop=1, tau=self.args.tau)[
                    1].to(self.args.device)
                norm_At = get_powers_sparse(self.A.transpose(), hop=1, tau=self.args.tau)[
                    1].to(self.args.device)
            for split in range(self.splits):
                if self.args.baseline == 'davidScore':
                    score = davidScore(self.A)
                else:
                    raise NameError('Please input the correct baseline model name from:\
                        SpringRank, syncRank, serialRank, btl, davidScore, eigenvectorCentrality,\
                        PageRank, rankCentrality, SVD_RS, SVD_NRS instead of {}!'.format(self.args.baseline))
                score_torch = torch.FloatTensor(score.reshape(score.shape[0], 1)).to(self.args.device)
                if score.min() < 0:
                    score_torch = torch.sigmoid(score_torch)
                upset1 = calculate_upsets(self.A_torch, score_torch, device=args.device)
                upset2 = calculate_upsets(torch.transpose(self.A_torch, 0, 1), score_torch, device=args.device)

                if upset1.detach().item() > upset2.detach().item():
                    score = -score
                score_torch = torch.FloatTensor(score.reshape(score.shape[0], 1)).to(self.args.device)
                model = self.build_model(model_name, score_torch)

                if self.args.optimizer == 'Adam':
                    opt = optim.Adam(model.parameters(), lr=self.args.lr,
                                     weight_decay=self.args.weight_decay)
                elif self.args.optimizer == 'SGD':
                    opt = optim.SGD(model.parameters(), lr=self.args.lr,
                                    weight_decay=self.args.weight_decay)
                else:
                    raise NameError('Please input the correct optimizer name, Adam or SGD!')
                M = self.A_torch

                if self.test_mask is not None:
                    train_index = self.train_mask[:, split]
                    val_index = self.val_mask[:, split]
                    test_index = self.test_mask[:, split]
                    if args.AllTrain:
                        # to use all nodes
                        train_index[:] = True
                        val_index[:] = True
                        test_index[:] = True
                #################################
                # Train/Validation/Test
                #################################
                best_val_loss = 1000.0
                early_stopping = 0
                log_str_full = ''
                train_with = self.args.train_with
                if self.args.pretrain_with == 'serial_similarity' and args.train_with[:8] == 'proximal':
                    serialRank_mat = serialRank_matrix(self.A[train_index][:, train_index])
                    serialRank_mat = serialRank_mat / max(0.1, serialRank_mat.max())
                    serial_matrix_train = torch.FloatTensor(serialRank_mat.toarray()).to(self.args.device)

                for epoch in range(args.epochs):
                    if self.args.optimizer == 'Adam' and epoch == self.args.pretrain_epochs and self.args.train_with[
                                                                                                :8] == 'proximal':
                        opt = optim.SGD(model.parameters(), lr=10 * self.args.lr,
                                        weight_decay=self.args.weight_decay)
                    start_time = time.time()
                    ####################
                    # Train
                    ####################

                    model.train()
                    if model_name == 'DIGRAC':
                        _ = model(norm_A, norm_At, self.features)
                    if train_with == 'dist' or (
                            epoch < self.args.pretrain_epochs and self.args.pretrain_with == 'dist'):
                        score = model.obtain_score_from_dist()
                    elif train_with == 'innerproduct' or (
                            epoch < self.args.pretrain_epochs and self.args.pretrain_with == 'innerproduct'):
                        score = model.obtain_score_from_innerproduct()
                    else:
                        score = model.obtain_score_from_proximal(train_with[9:])

                    if self.args.upset_ratio_coeff > 0:
                        train_loss_upset_ratio = calculate_upsets(M[train_index][:, train_index], score[train_index],
                                                                  device=args.device).to(device)
                    else:
                        train_loss_upset_ratio = torch.ones(1, requires_grad=True).to(device)
                    if self.args.upset_margin_coeff > 0:
                        train_loss_upset_margin = calculate_upsets(M[train_index][:, train_index], score[train_index],
                                                                   style='margin', margin=self.args.upset_margin,
                                                                   device=args.device).to(device)
                    else:
                        train_loss_upset_margin = torch.ones(1, requires_grad=True).to(device)

                    train_loss = self.args.upset_ratio_coeff * train_loss_upset_ratio + self.args.upset_margin_coeff * train_loss_upset_margin
                    if self.args.pretrain_with == 'serial_similarity' and epoch < self.args.pretrain_epochs and args.train_with[
                                                                                                                :8] == 'proximal':
                        pretrain_outside_loss = torch.mean(
                            (model.obtain_similarity_matrix()[train_index][:, train_index] - serial_matrix_train) ** 2)
                        train_loss += pretrain_outside_loss
                        outstrtrain = 'Train loss:, {:.6f}, upset ratio loss: {:6f}, upset margin loss: {:6f}, pretrian outside loss: {:6f},'.format(
                            train_loss.detach().item(),
                            train_loss_upset_ratio.detach().item(), train_loss_upset_margin.detach().item(),
                            pretrain_outside_loss.detach().item())
                    else:
                        outstrtrain = 'Train loss:, {:.6f}, upset ratio loss: {:6f}, upset margin loss: {:6f},'.format(
                            train_loss.detach().item(),
                            train_loss_upset_ratio.detach().item(), train_loss_upset_margin.detach().item())
                    opt.zero_grad()
                    try:
                        train_loss.backward()
                    except RuntimeError:
                        log_str = '{} trial {} RuntimeError!'.format(model_name, split)
                        log_str_full += log_str + '\n'
                        print(log_str)
                        if not os.path.isfile(self.log_path + '/' + model_name + '_model' + str(split) + '.t7'):
                            torch.save(model.state_dict(), self.log_path +
                                       '/' + model_name + '_model' + str(split) + '.t7')
                        torch.save(model.state_dict(), self.log_path +
                                   '/' + model_name + '_model_latest' + str(split) + '.t7')
                        break
                    opt.step()
                    ####################
                    # Validation
                    ####################
                    model.eval()

                    if model_name == 'DIGRAC':
                        _ = model(norm_A, norm_At, self.features)
                    if train_with == 'dist' or (
                            epoch < self.args.pretrain_epochs and self.args.pretrain_with == 'dist'):
                        score = model.obtain_score_from_dist().to(self.args.device)
                    elif train_with == 'innerproduct' or (
                            epoch < self.args.pretrain_epochs and self.args.pretrain_with == 'innerproduct'):
                        score = model.obtain_score_from_innerproduct().to(self.args.device)
                    else:
                        score = model.obtain_score_from_proximal(train_with[9:]).to(self.args.device)

                    if self.args.upset_ratio_coeff > 0:
                        val_loss_upset_ratio = calculate_upsets(M[val_index][:, val_index], score[val_index],
                                                                device=args.device).to(device)
                    else:
                        val_loss_upset_ratio = torch.ones(1, requires_grad=True).to(device)
                    if self.args.upset_margin_coeff > 0:
                        val_loss_upset_margin = calculate_upsets(M[val_index][:, val_index], score[val_index],
                                                                 style='margin', margin=self.args.upset_margin,
                                                                 device=args.device).to(device)
                    else:
                        val_loss_upset_margin = torch.ones(1, requires_grad=True).to(device)

                    val_loss = self.args.upset_ratio_coeff * val_loss_upset_ratio + self.args.upset_margin_coeff * val_loss_upset_margin

                    outstrval = 'val loss:, {:.6f}, upset ratio loss: {:6f}, upset margin loss: {:6f},'.format(
                        val_loss.detach().item(),
                        val_loss_upset_ratio.detach().item(), val_loss_upset_margin.detach().item())

                    duration = "---, {:.4f}, seconds ---".format(
                        time.time() - start_time)
                    log_str = ("{}, / {} epoch,".format(epoch, args.epochs)) + \
                              outstrtrain + outstrval + duration
                    log_str_full += log_str + '\n'
                    print(log_str)

                    ####################
                    # Save weights
                    ####################
                    save_perform = val_loss.detach().item()
                    if save_perform <= best_val_loss:
                        early_stopping = 0
                        best_val_loss = save_perform
                        torch.save(model.state_dict(), self.log_path +
                                   '/' + model_name + '_model' + str(split) + '.t7')
                    else:
                        early_stopping += 1
                    if early_stopping > args.early_stopping or epoch == (args.epochs - 1):
                        torch.save(model.state_dict(), self.log_path +
                                   '/' + model_name + '_model_latest' + str(split) + '.t7')
                        break

                status = 'w'
                if os.path.isfile(self.log_path + '/' + model_name + '_log' + str(split) + '.csv'):
                    status = 'a'
                with open(self.log_path + '/' + model_name + '_log' + str(split) + '.csv', status) as file:
                    file.write(log_str_full)
                    file.write('\n')
                    status = 'a'

        return model, split, self.log_path

    def testing(self, split, model_name, args, log_path, model):
        train_index = np.full((self.A.shape[0],), True)
        val_index = np.full((self.A.shape[0],), True)
        test_index = np.full((self.A.shape[0],), True)
        if args.AllTrain:
            # to use all nodes
            train_index[:] = True
            val_index[:] = True
            test_index[:] = True
        test_A = self.A[test_index][:, test_index]
        self.A_torch = scipy_sparse_to_torch_sparse(self.A[test_index][:, test_index]).to(self.args.device).to_dense()
        M = self.A_torch
        self.label_np = torch.tensor(self.label_np.numpy()[test_index])
        base_save_path = log_path + '/' + model_name
        logstr = ''
        norm_A = get_powers_sparse(test_A, hop=1, tau=args.tau)[
            1].to(self.args.device)
        norm_At = get_powers_sparse(test_A.transpose(), hop=1, tau=self.args.tau)[
            1].to(self.args.device)
        test_feature = torch.tensor(self.features.cpu().numpy()[test_index]).to(self.args.device)
        # latest
        model = self.load_model(log_path, model_name, split, model)
        model.eval()
        model = self.set_model(norm_A, norm_At, test_feature, model_name, model)
        score_model = self.set_initializer(self.args.train_with, model, self.args)

        all_loss = self.calculate_loss(M, score_model)
        logstr += 'all loss: ,{:.3f},'.format(
            all_loss.detach().item())
        val_index = np.full(len(self.label_np), True)
        test_index = np.full(len(self.label_np), True)
        score = model.obtain_score_from_dist()
        logstr, upset_full, kendalltau_full = evalutaion(logstr, score,
                                                         self.A_torch,
                                                         self.label_np,
                                                         val_index,
                                                         test_index,
                                                         self.args.SavePred,
                                                         base_save_path,
                                                         split, 'dist_latest')
        score = model.obtain_score_from_innerproduct()
        logstr, upset_full, kendalltau_full = evalutaion(logstr, score,
                                                         self.A_torch,
                                                         self.label_np,
                                                         val_index,
                                                         test_index,
                                                         self.args.SavePred,
                                                         base_save_path,
                                                         split,
                                                         'innerproduct_latest')

        print(logstr)

        torch.cuda.empty_cache()

    def predict(self, model_name, save_path, test_A, split, model):
        A_torch = scipy_sparse_to_torch_sparse(test_A).to(self.args.device).to_dense()
        norm_A = get_powers_sparse(test_A, hop=1, tau=self.args.tau)[
            1].to(self.args.device)
        norm_At = get_powers_sparse(test_A.transpose(), hop=1, tau=self.args.tau)[
            1].to(self.args.device)
        test_feature = torch.tensor(self.features.cpu().numpy()).to(self.args.device)

        model = self.load_model(save_path, model_name, split, model)
        model.eval()
        model = self.set_model(norm_A, norm_At, test_feature, model_name, model)
        score_model = self.set_initializer(self.args.train_with, model, self.args)
        upset_ratio, pred_label, df = self.get_score(score_model, A_torch, self.args)
        return df

    def calculate_loss(self, M, score_model):
        if self.args.upset_ratio_coeff > 0:
            all_loss_upset_ratio = calculate_upsets(M, score_model, device=args.device).to(device)
        else:

            all_loss_upset_ratio = torch.ones(1, requires_grad=True).to(device)
        if self.args.upset_margin_coeff > 0:
            all_loss_upset_margin = calculate_upsets(M, score_model, style='margin',
                                                     margin=self.args.upset_margin, device=args.device).to(device)
        else:
            all_loss_upset_margin = torch.ones(1, requires_grad=True).to(device)

        all_loss = self.args.upset_ratio_coeff * all_loss_upset_ratio + self.args.upset_margin_coeff * all_loss_upset_margin
        return all_loss

    def get_score(self, score, A_torch, args):
        if score.min().detach().item() < 0:
            if score.min().detach().item() > -1:
                score = (score + 1) / 2
            else:
                score = torch.sigmoid(score)
        upset1 = calculate_upsets(A_torch, score, device=args.device)
        upset2 = calculate_upsets(torch.transpose(A_torch, 0, 1), score, device=args.device)
        if upset1.detach().item() < upset2.detach().item():
            upset_ratio = upset1
        else:
            upset_ratio = upset2
            score = -score
        pred_label = rankdata(-score.detach().cpu().numpy(), 'min')
        df = adj_to_edges(A_torch)
        df['pred_weight'] = df.apply(lambda x: 1 if pred_label[int(x['Src'])] > pred_label[int(x['Dst'])] else -1,
                                     axis=1)
        return upset_ratio, pred_label, df

    def set_model(self, norm_A, norm_AT, test_feature, model_name, model):
        if model_name == 'DIGRAC':
            _ = model(norm_A, norm_AT, test_feature)
        return model

    @staticmethod
    def set_initializer(initializer_name, model, args):
        if initializer_name == 'dist':
            score_model = model.obtain_score_from_dist().to(args.device)
        elif initializer_name == 'innerproduct':
            score_model = model.obtain_score_from_innerproduct().to(args.device)
        else:
            score_model = model.obtain_score_from_proximal(initializer_name[9:]).to(args.device)
        return score_model

    def build_model(self, model_name, score_torch):
        if model_name == 'DIGRAC':
            model = DIGRAC_Ranking(num_features=self.nfeat, dropout=self.args.dropout, hop=self.args.hop,
                                   fill_value=self.args.tau,
                                   embedding_dim=self.args.hidden * 2,
                                   Fiedler_layer_num=self.args.Fiedler_layer_num, alpha=self.args.alpha,
                                   trainable_alpha=self.args.trainable_alpha, initial_score=score_torch,
                                   prob_dim=self.num_clusters, sigma=self.args.sigma).to(self.args.device)
        else:
            raise NameError('Please input the correct model name from:\
                SpringRank, syncRank, serialRank, btl, davidScore, eigenvectorCentrality,\
                PageRank, rankCentrality, mvr, DIGRAC, ib, instead of {}!'.format(model_name))
        return model

    def save_model(self):
        pass

    def save_logs(self):
        pass

    def load_model(self, log_path, model_name, split, model):
        base_save_path = log_path + '/' + model_name
        model.load_state_dict(torch.load(
            base_save_path + '_model_latest' + str(split) + '.t7'))
        return model


def run_train():
    print(args.device)
    for i in range(1, 30):
        if i > 2:
            trainer = Trainer(args, random_seed, save_name_base, True)
            test = trainer.testing(split, method_name, args, new_path)
        trainer = Trainer(args, random_seed, save_name_base)
        model, split, new_path = trainer.train(method_name)
        args.generation += 1
