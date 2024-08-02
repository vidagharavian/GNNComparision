import dgl
import numpy as np
import pandas as pd
import torch
from dgl.data import DGLDataset
from dgl.nn.pytorch import SAGEConv
from dgllife.model import HadamardLinkPredictor
from numpy import double
from torch import nn
import dgl.function as fn
import torch.nn.functional as F



device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class MyDataDataset(DGLDataset):
    def __init__(self, edges_data, node_list):
        self.edges_data = edges_data
        self.node_list = node_list
        super().__init__(name='generation_data')

    def process(self):
        nodes_data = pd.read_csv('../ranker/features.csv')
        nodes_data = nodes_data[nodes_data.index.isin(self.node_list)]
        nodes_data.reset_index(inplace=True)
        self.edges_data['Src'] = [nodes_data[nodes_data['index'] == x].index.values[0] for x in self.edges_data['Src']]
        self.edges_data['Dst'] = [nodes_data[nodes_data['index'] == x].index.values[0] for x in self.edges_data['Dst']]
        nodes_data.drop(columns=['index'], inplace=True)
        nodes_data = np.round(nodes_data.to_numpy(), 2).astype(np.float64)
        node_features = torch.from_numpy(nodes_data).type(torch.FloatTensor)
        edge_features = torch.from_numpy(self.edges_data['Weight'].to_numpy(dtype=double))
        edges_src = torch.from_numpy(self.edges_data['Src'].to_numpy(dtype=int))
        edges_dst = torch.from_numpy(self.edges_data['Dst'].to_numpy(dtype=int))

        self.graph = dgl.graph((edges_src, edges_dst), num_nodes=nodes_data.shape[0])
        self.graph.ndata['feat'] = node_features
        self.graph.edata['weight'] = edge_features

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1


# ----------- 2. create model -------------- #
# build a two-layer GraphSAGE model
class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats, r_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, 'mean').to(device)
        self.conv3 = SAGEConv(h_feats, r_feats, 'mean').to(device)

    def forward(self, x, g):
        h = self.conv1(g, x).to(device)
        h = F.relu(h)
        h = self.conv3(g, h).to(device)
        return h

    def reset_params(self):
        for layer in self.children():
            if hasattr(layer,"reset_parameters"):
                layer.reset_parameters()


class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'.
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            return g.edata['score'][:, 0]


class MLPPredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats).to(device)
        self.W2 = nn.Linear(h_feats,1).to(device)

    def apply_edges(self, edges):
        """
        Computes a scalar score for each edge of the given graph.

        Parameters
        ----------
        edges :
            Has three members ``src``, ``dst`` and ``data``, each of
            which is a dictionary representing the features of the
            source nodes, the destination nodes, and the edges
            themselves.

        Returns
        -------
        dict
            A dictionary of new edge features.
        """
        h = torch.cat([edges.src['h'], edges.dst['h']], 1).to(device)
        x=F.sigmoid(self.W2(F.relu(self.W1(h)).to(device))).to(device).squeeze(1)
        return {'score': x}

    def reset_params(self):
        for layer in self.children():
            layer.reset_parameters()

    def forward(self, g, h):
        g=g.to(device)
        with g.local_scope():
            g.ndata['h'] = h.to(device)
            g.apply_edges(self.apply_edges)
            return g.edata['score']

class ScorePredictor(nn.Module):
    def __init__(self, num_classes, in_features):
        super().__init__()
        self.W1 = nn.Linear(in_features, int(in_features/2)).to(device)
        self.W2 = nn.Linear(int(in_features/2), num_classes).to(device)
        self.activation = nn.LogSoftmax(dim=1)

    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        score = self.W2(F.relu(self.W1(h_u-h_v).to(device))).to(device)
        return {'score': self.activation(score)}

    def forward(self, edge_subgraph, x):
        with edge_subgraph.local_scope():
            edge_subgraph.ndata['h'] = x
            edge_subgraph.apply_edges(self.apply_edges)
            return edge_subgraph.edata['score']


class MyHadamardLinkPredictor(HadamardLinkPredictor):

    def apply_edges(self,edges):
        x= F.sigmoid(super().forward(edges.src['h'], edges.dst['h'])).to(device).squeeze(1)
        return {'score': x}

    def forward(self, g, h):
        g = g.to(device)
        with g.local_scope():
            g.ndata['h'] = h.to(device)
            g.apply_edges(self.apply_edges)
            return g.edata['score']

    def reset_params(self):
        self.reset_parameters()

class Model(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_classes):
        super().__init__()
        self.gcn = GraphSAGE(in_features, hidden_features,out_features)
        self.predictor = ScorePredictor(num_classes,out_features)

    def forward(self,x,edge_subgraph):
        x = self.gcn(x,edge_subgraph)
        return self.predictor(edge_subgraph, x)