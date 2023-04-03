import argparse
import os
import torch


def parameter_parser():
    """
    A method to parse up command line parameters.
    """
    parser = argparse.ArgumentParser(description="Ranking.")

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--debug', '-D', action='store_true', default=False,
                        help='Debugging mode, minimal setting.')
    parser.add_argument('--seed', type=int, default=31, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01,  # default = 0.01
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=32,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--sigma', type=float, default=1.0,
                        help='(Initial) Sigma in the Gaussian kernel, actual sigma is this times sqrt(num_nodes), default 1.')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='(Initial) learning rate for proximal gradient step.')
    parser.add_argument('--trainable_alpha', action='store_true', default=False,
                        help='Whether to set the proximal gradient step learning rate to be trainable.')
    parser.add_argument('--Fiedler_layer_num', type=int, default=5,
                        help='The number of proximal gradient steps in calculating the Fiedler vector.')
    parser.add_argument('--train_with', type=str, default='dist',
                        help='To train GNNs with dist.')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='Optimizer to use. Adam or SGD in our case.')
    parser.add_argument('--pretrain_with', type=str, default='dist',
                        help='Variant to pretrain with, dist')
    parser.add_argument('--pretrain_epochs', type=int, default=50,
                        help='Number of epochs to pretrain.')
    parser.add_argument('--baseline', type=str, default='SpringRank',
                        help='The baseline model used for obtaining rankings as initial guess.')
    parser.add_argument("--seeds",
                        nargs="+",
                        type=int,
                        help="seeds to generate random graphs.")
    parser.set_defaults(seeds=[10, 20, 30, 40, 50])

    # synthetic model hyperparameters below
    parser.add_argument('--p', type=float, default=0.05,
                        help='Probability of the existence of a link within communities, with probability (1-p), we have 0.')
    parser.add_argument('--N', type=int, default=350,
                        help='Number of nodes in the directed stochastic block model.')
    parser.add_argument('--K', type=int, default=5,
                        help='Number of clusters.')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Training ratio during data split.')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                        help='Test ratio during data split.')
    parser.add_argument('--hop', type=int, default=2,
                        help='Number of hops to consider for the random walk.')
    parser.add_argument('--tau', type=float, default=0.5,
                        help='The regularization parameter when adding self-loops to an adjacency matrix, i.e. A -> A + tau * I, where I is the identity matrix.')
    parser.add_argument('--num_trials', type=int, default=1,
                        help='Number of trials to generate results.')
    parser.add_argument('--ERO_style', type=str, default='uniform',
                        help='ERO rating style, uniform or gamma.')
    parser.add_argument('--eta', type=float, default=0.1,
                        help='Direction noise level in the meta-graph adjacency matrix, less than 0.5.')
    parser.add_argument('--upset_ratio_coeff', type=float, default=1.0,
                        help='Coefficient of upset ratio loss.')
    parser.add_argument('--upset_margin_coeff', type=float, default=0.0,
                        help='Coefficient of upset margin loss.')
    parser.add_argument('--upset_margin', type=float, default=0.01,
                        help='Margin of upset margin loss.')
    parser.add_argument('--archive_size', type=int, default=10000, help="Size of archive which is used for train")
    parser.add_argument('--generation', default=1,
                        type=int,
                        help="generation to use for test")
    parser.add_argument('--early_stopping', type=int, default=200,
                        help='Number of iterations to consider for early stopping.')
    parser.add_argument('--fill_val', type=float, default=0.5,
                        help='The value to be filled when we originally have 0, from meta-graph adj to meta-graph to generate data.')
    parser.add_argument('--regenerate_data', action='store_true', help='Whether to force creation of data splits.')
    parser.add_argument('--load_only', action='store_true', help='Whether not to store generated data.')
    parser.add_argument('-AllTrain', '-All', action='store_true',
                        help='Whether to use all data to do gradient descent.')
    parser.add_argument('-SavePred', '-SP', action='store_true', help='Whether to save predicted labels.')
    parser.add_argument('--log_root', type=str,
                        default=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'logs/'),
                        help='The path saving model.t7 and the training process')
    parser.add_argument('--data_path', type=str,
                        default=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/'),
                        help='Data set folder.')
    parser.add_argument('--dataset', type=str, default='ERO/', help='Data set selection.')

    args = parser.parse_args()

    if args.train_with in ['dist', 'innerproduct']:
        args.optimizer = 'Adam'

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if (not args.no_cuda and torch.cuda.is_available()) else "cpu")
    if args.debug:
        args.num_trials = 2
        args.seeds = [10]
        args.epochs = 2
        args.pretrain_epochs = 1
        args.log_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../debug_logs/')
    return args


def add_min_args(args):
    args.dataset = 'mine/'
    args.pretrain_with = 'serial_similarity'
    args.tau = 0.013
    args.upset_margin_coeff = 1
    args.upset_ratio_coeff = 0
    args.SavePred = True
    args.baseline = 'davidScore'
    args.train_with = 'innerproduct'
    args.seeds = [10, 20]
    args.dimension = 10

    return args
