import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling

from ranker.DGL_presentation import read_data, create_archive


class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels,out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)


    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(
            dim=-1
        )  # product of a pair of nodes on each edge

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()


def train_link_predictor(
        model, train_data, val_data, optimizer, criterion,negetive_edges,n_epochs=500
):
    accuracy=[]
    lost =[]
    neg_src = negetive_edges['Src'][:train_data.edge_label_index.size(1)].to_numpy()
    neg_dst = negetive_edges['Dst'][:train_data.edge_label_index.size(1)].to_numpy()
    neg_edge_index = torch.tensor([neg_src,
                                   neg_dst], dtype=torch.long)
    edge_label,edge_label_index=negetive_labeling(train_data,neg_edge_index)
    neg_val_src = negetive_edges['Src'][train_data.edge_label_index.size(1):train_data.edge_label_index.size(1)+val_data.edge_label_index.size(1)].to_numpy()
    neg_val_dst = negetive_edges['Dst'][train_data.edge_label_index.size(1):train_data.edge_label_index.size(
        1) + val_data.edge_label_index.size(1)].to_numpy()
    val_neg_edge_index= torch.tensor([neg_val_src,
                                   neg_val_dst], dtype=torch.long)
    val_edge_label , val_edge_label_index = negetive_labeling(val_data,val_neg_edge_index)
    for epoch in range(1, n_epochs + 1):

        model.train()
        optimizer.zero_grad()
        z = model.encode(train_data.x, train_data.edge_index)

        # sampling training negatives for every training epoch
        # neg_edge_index = negative_sampling(
        #     edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
        #     num_neg_samples=train_data.edge_label_index.size(1), method='sparse')
        # neg_edge_index=negetive_edges.

        out = model.decode(z, edge_label_index).view(-1)
        loss = criterion(out, edge_label)
        loss.backward()
        optimizer.step()

        val_auc = eval_link_predictor(model, val_data,val_edge_label,val_edge_label_index)
        if val_auc < 0.6:
            n_epochs+=100

        if epoch % 10 == 0:
            if epoch > 10:
                lost.append(loss.detach().numpy().min())
                accuracy.append(val_auc)
            print(f"Epoch: {epoch:03d}, Train Loss: {loss:.3f}, Val AUC: {val_auc:.3f}")
    # plot_accuracy(lost,accuracy)
    return model


def plot_accuracy(accuracy,val_acc):

    plt.plot()
    plt.plot(accuracy)
    # plt.plot(val_acc)
    plt.title('train loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    # plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    plt.plot()
    plt.plot(val_acc)
    # plt.plot(val_acc)
    plt.title('Model Accuray')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    # plt.legend(['train', 'val'], loc='upper left')
    plt.show()




@torch.no_grad()
def eval_link_predictor(model, data,val_edge_label=None,val_edge_label_index=None):
    model.eval()
    z = model.encode(data.x, data.edge_index)
    out = model.decode(z, data.edge_label_index if val_edge_label_index is None else val_edge_label_index).view(-1).sigmoid()

    return roc_auc_score(data.edge_label.cpu().numpy() if val_edge_label is None else val_edge_label , out.cpu().numpy())

def negetive_labeling(train_data,neg_edge_index):
    edge_label_index = torch.cat(
        [train_data.edge_label_index, neg_edge_index],
        dim=-1,
    )
    edge_label = torch.cat([
        train_data.edge_label,
        train_data.edge_label.new_zeros(neg_edge_index.size(1))
    ], dim=0)
    return edge_label,edge_label_index


import torch_geometric.transforms as T

split = T.RandomLinkSplit(
    num_val=0.05,
    num_test=0.1,
    is_undirected=True,
    add_negative_train_samples=False,
    neg_sampling_ratio=1.0,

)
def run_for_all():
    graph,posetive,negetive = read_data()
    train_data, val_data, test_data = split(graph)
    neg_src = negetive['Src'][train_data.edge_label_index.size(1)+val_data.edge_label_index.size(1):].to_numpy()
    neg_dst = negetive['Dst'][train_data.edge_label_index.size(1)+val_data.edge_label_index.size(1):].to_numpy()
    neg_edge_index = torch.tensor([neg_src,
                                       neg_dst], dtype=torch.long)
    edge_label,edge_label_index=negetive_labeling(train_data,neg_edge_index)
    print("train_data:" ,train_data)
    model = Net(len(graph.x[0]), 128, 64)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()
    model = train_link_predictor(model, train_data, val_data, optimizer, criterion,negetive)

    test_auc = eval_link_predictor(model, test_data)
    print(f"Test: {test_auc:.3f}")

def run_for_generation(model,optimizer,criterion,i):
        archive = create_archive(i,archive_size=10000)
        print(f"{i} generation")
        graph, posetive, negetive_m = read_data(f"generations/{i}.csv",edge_vector=archive)
        train_data, val_data, test_data = split(graph)
        graph, posetive, negetive = read_data(f"generations/{i}.csv")
        _, _, test_data = split(graph)
        neg_src = negetive['Src'][train_data.edge_label_index.size(1) + val_data.edge_label_index.size(1):].to_numpy()
        neg_dst = negetive['Dst'][train_data.edge_label_index.size(1) + val_data.edge_label_index.size(1):].to_numpy()
        neg_edge_index = torch.tensor([neg_src,
                                       neg_dst], dtype=torch.long)
        edge_label, edge_label_index = negetive_labeling(test_data, neg_edge_index)
        print("train_data:", train_data)
        test_auc_before=0
        if i> 5:
            test_auc_before = eval_link_predictor(model, test_data, edge_label, edge_label_index)
            print(f"before train Test: {test_auc_before:.3f}")
        model = train_link_predictor(model, train_data, val_data, optimizer, criterion, negetive_m)

        test_auc_after = eval_link_predictor(model, test_data,edge_label, edge_label_index )
        print(f"Test: {test_auc_after:.3f}")
        return model , test_auc_before,test_auc_after

# model = Net(2, 128, 64)
# optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
# criterion = torch.nn.BCEWithLogitsLoss()
measure =[]
objective = 10
for i in range(1,200):
    model = Net(objective, 128, 64)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()
    model,test_auc_before,test_auc_after =run_for_generation(model,optimizer,criterion,i)
    measure.append({"generation":i,"archive_train_set":test_auc_before , "generation_train_set":test_auc_after})
df=pd.DataFrame.from_records(measure)
df.to_csv(f"measure_{objective}.csv")

