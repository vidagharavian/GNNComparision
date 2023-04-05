import itertools

import dgl
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score

from ranker.DGL_presentation import create_archive
from ranker.MyData import MyDataDataset, GraphSAGE, DotPredictor, MLPPredictor


def load_edges(generation,archive=None):
    if archive is not None:
        edge = pd.read_csv(f"generations/{generation}.csv")
    else:
        edge =archive
    positive = edge[edge['Weight']==1]
    src = edge['Src']
    dst = edge['Dst']
    edge_list = np.unique(pd.concat([src,dst]))
    g = MyDataDataset(positive,edge_list)[0]
    u, v = g.edges()
    eids = np.arange(g.number_of_edges())
    eids = np.random.permutation(eids)
    test_size = int(len(eids) * 0.1)
    train_size = g.number_of_edges() - test_size
    test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
    train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]

    # Find all negative edges and split them for training and testing
    adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
    adj_neg = 1 - adj.todense() - np.eye(g.number_of_nodes())
    neg_u, neg_v = np.where(adj_neg != 0)

    neg_eids = np.random.choice(len(neg_u), g.number_of_edges())
    test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
    train_neg_u, train_neg_v = neg_u[neg_eids[test_size:]], neg_v[neg_eids[test_size:]]
    train_g = dgl.remove_edges(g, eids[:test_size])
    return test_neg_u, test_neg_v,train_neg_u, train_neg_v, test_pos_u, test_pos_v,train_pos_u, train_pos_v,g,train_g

archive = create_archive(4,15000)
test_neg_u, test_neg_v, train_neg_u, train_neg_v, test_pos_u, test_pos_v, train_pos_u, train_pos_v,g,train_g =load_edges(4,archive)

def apply_edges(train_pos_u,train_pos_v,train_neg_u,train_neg_v,g):
    train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.number_of_nodes())
    train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.number_of_nodes())
    return train_neg_g,train_pos_g

train_neg_g,train_pos_g = apply_edges(train_pos_u,train_pos_v,train_neg_u,train_neg_v,g)
test_neg_g, test_pos_g = apply_edges(test_pos_u,test_pos_v,test_neg_u,test_neg_v,g)

model = GraphSAGE(train_g.ndata['feat'].shape[1], 64,32,0.2)
# You can replace DotPredictor with MLPPredictor.
pred = MLPPredictor(32)


def compute_loss(pos_score, neg_score):

    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
    return F.binary_cross_entropy_with_logits(scores, labels)

def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return roc_auc_score(labels, scores)

optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=0.01)

# ----------- 4. training -------------------------------- #
all_logits = []
for e in range(500):
    # forward
    h = model(train_g, train_g.ndata['feat'])
    pos_score = pred(train_pos_g, h)
    neg_score = pred(train_neg_g, h)
    loss = compute_loss(pos_score, neg_score)

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if e % 5 == 0:
        print('In epoch {}, loss: {}'.format(e, loss))

# ----------- 5. check results ------------------------ #
from sklearn.metrics import roc_auc_score
with torch.no_grad():
    pos_score = pred(test_pos_g, h)
    neg_score = pred(test_neg_g, h)
    print('AUC', compute_auc(pos_score, neg_score))





