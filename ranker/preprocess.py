import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp
from torch_geometric.data import Data
from torch_geometric.transforms import RandomNodeSplit

from config import path
from ranker.extract_network import extract_network


def get_train_test_dataset(data):
    msk = RandomNodeSplit(split="train_rest", num_splits=1, num_val=0.2, num_test=0.1)
    g = msk(data)
    train_mask = g.train_mask.data.numpy().astype('bool_')
    val_mask = g.val_mask.data.numpy().astype('bool_')
    test_mask = g.test_mask.data.numpy().astype('bool_')
    return test_mask, val_mask, train_mask


def read_data(edge_path="edges.csv", feature_path="features.csv", edge_vector=None):
    if edge_vector is None:
        edges_data = pd.read_csv(edge_path)
    else:
        edges_data = edge_vector
    feature_data = pd.read_csv(feature_path)
    edges_data = edges_data.groupby(['Src', 'Dst']).agg({'Weight': 'sum'})
    edges_data['Weight'] = [1 if i > 0 else -1 for i in edges_data['Weight']]
    edges_data.reset_index(inplace=True)
    src = edges_data['Src'].to_numpy()
    dst = edges_data['Dst'].to_numpy()
    features =np.unique(np.concatenate((src,dst))).tolist()
    feature_data=feature_data.loc[features]
    feature_data.reset_index(inplace=True)
    src = [feature_data.index[feature_data['index']==i].values[0] for i in src]
    dst = [feature_data.index[feature_data['index'] == i].values[0]  for i in dst]
    feature_data.drop(columns=['index'],inplace=True)
    edge_atter = edges_data['Weight'].to_numpy()
    feature_data = feature_data.values
    edge_index = torch.tensor([src,
                               dst], dtype=torch.long)
    x = torch.tensor(feature_data, dtype=torch.float)
    A = sp.csr_matrix((edge_atter, (src, dst)), shape=(feature_data.shape[0], feature_data.shape[0]))
    label = edges_data.groupby('Src').agg({'Weight': 'sum'}).sort_values(by='Weight')
    label['Label'] = [i for i in range(1, len(label) + 1)]
    label.reset_index(inplace=True)
    label = label.sort_values(by='Src')
    label = label['Label'].to_numpy()
    # A, label = extract_network(A, label)
    data = Data(x=x, edge_index=edge_index, y=label, A=A, edge_attr=torch.LongTensor(edge_atter))
    # train_mask = g.train_mask.data.numpy().astype('bool_')
    # val_mask = g.val_mask.data.numpy().astype('bool_')
    # test_mask = g.test_mask.data.numpy().astype('bool_')
    return label, x, A,data


def load_data(args,test=False):
    label = None
    train_mask = None
    val_mask = None
    test_mask = None
    x=None
    A=None
    if args.dataset[:4] == 'mine':
        if test:
            label, x, A, data = read_data(f"generations/{args.dimension}/{args.generation}.csv",f"generations/{args.dimension}/features.csv")
        else:
            archive = create_archive(f"{path}/",args.generation, archive_size=args.archive_size)
            print(f"archive_size{len(archive)}")
            label, x, A,data = read_data(f"{path}/{args.generation}.csv",f"{path}/features.csv", edge_vector=archive)
            train_mask, test_mask, val_mask = get_train_test_dataset(data)
    return label, train_mask, test_mask, val_mask, x, A


def create_archive(save_path,generation, archive_size=16000):
    edges = []
    data_set = pd.DataFrame()
    for i in range(generation, 0, -1):
        edge = pd.read_csv(f"{save_path}{i}.csv")
        edges.append(edge)
        data_set = pd.concat(edges)
        if len(data_set) > archive_size:
            data_set = data_set[:archive_size]
            break
    return data_set
