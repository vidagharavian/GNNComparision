import numpy as np
import pandas as pd
import dgl
import networkx as nx
import torch

from sklearn.preprocessing import StandardScaler
from torch import nn

from torch_geometric.data import HeteroData, Data
import torch_geometric.transforms as T

def read_data(edge_path="edges.csv",feature_path="features.csv",edge_vector=None):
    if edge_vector is None:
        edges_data = pd.read_csv(edge_path)
    else:
        edges_data = edge_vector
    edges_data = edges_data.groupby(['Src', 'Dst']).agg({'Weight': 'sum'})
    edges_data['Weight'] = [1 if i > 0 else 0 for i in edges_data['Weight']]
    edges_data.reset_index(inplace=True)
    feature_data = pd.read_csv(feature_path)
    negetive_edges = edges_data[edges_data['Weight']==0]
    posetive_edges = edges_data[edges_data['Weight'] == 1]
    src = posetive_edges['Src'].to_numpy()
    dst = posetive_edges['Dst'].to_numpy()

    edge_attr = torch.tensor(posetive_edges['Weight'].to_numpy(), dtype=torch.float)
    feature_data=feature_data.values
    edge_index = torch.tensor([src,
                               dst], dtype=torch.long)
    x=torch.tensor(feature_data, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index,edge_attr=edge_attr)
    return data ,posetive_edges,negetive_edges

def plot_network(nx_g):
    pos = nx.kamada_kawai_layout(nx_g)
    nx.draw(nx_g, pos, with_labels=True)
    print("plotted")

def covert_feature(feature,df):
    ratings_feature_id = torch.from_numpy(df[feature].values)
    return ratings_feature_id


def create_archive(generation,archive_size=500):
    edges =[]
    data_set =pd.DataFrame()
    for i in range(generation,0,-1):
        edge =pd.read_csv(f"../ranker/generations/{i}.csv")
        edges.append(edge)
        data_set = pd.concat(edges)
        if len(data_set)> archive_size:
            data_set = data_set[:archive_size]
            break
    return data_set

