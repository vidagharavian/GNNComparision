import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from stellargraph import StellarGraph
from stellargraph.layer import GCN, LinkEmbedding
from stellargraph.mapper import FullBatchLinkGenerator
from tensorflow import keras
pop_size=100
benchmark='RosenBrock'
dimension = 10

def positive_and_negative_links(pos_edges,neg_edges):
    pos = pos_edges[["Src", "Dst"]].values.tolist()
    if not isinstance(neg_edges,list):
        neg_tuple =neg_edges[["Src", "Dst"]].values.tolist()
    else:
        neg_tuple =neg_edges
    neg = random.choices(neg_tuple,k=len(pos))
    return pos, neg

def build_graph(generation,square_edges=None):
    if square_edges is None:
        square_edges =pd.read_csv(f"generations/{generation}.csv")

    negetive_edges = square_edges[square_edges['Weight'] == 0]
    positive_edges = square_edges[square_edges['Weight'] == 1]
    all_positive = positive_edges[["Src", "Dst"]].values.tolist()
    # keep older edges in graph, and predict more recent edges
    edges_train, edges_test = train_test_split(positive_edges, test_size=0.25)

    features = pd.read_csv(f"features.csv")
    pos, neg = positive_and_negative_links(edges_train,negetive_edges)
    negetive_edges=negetive_edges[["Src", "Dst"]].values.tolist()
    negetive_edges = [i for i in negetive_edges if i not in neg]
    pos_test, neg_test = positive_and_negative_links(edges_test,negetive_edges)
    G_test, edge_ids_test, edge_labels_test,all_positive = get_labels(pos_test,neg_test,features,all_positive)

    print(G_test.info())
    # edge_splitter_train = EdgeSplitter(G_test)
    G_train, edge_ids_train, edge_labels_train ,all_positive= get_labels(pos,neg,features,all_positive)
    print(G_train.info())
    return G_train, edge_ids_train, edge_labels_train,G_test, edge_ids_test, edge_labels_test

def get_labels(pos_test,neg_test,node_features,all_positive):
    edge_ids_test =np.array(pos_test+neg_test)
    remain_list = [i for i in all_positive if i not in pos_test]
    edge_labels_test = np.repeat([1, 0], [len(pos_test), len(neg_test)])
    G_test = StellarGraph(node_features, pd.DataFrame(list(remain_list),columns=["source","target"]), source_column="source", target_column="target", node_type_default="superior", edge_type_default="dominant")
    return G_test, edge_ids_test, edge_labels_test,remain_list





def build_model(epochs,G_train,edge_ids_train,edge_labels_train,G_test,edge_ids_test, edge_labels_test):
    train_gen = FullBatchLinkGenerator(G_train, method="gcn")
    train_flow = train_gen.flow(edge_ids_train, edge_labels_train)
    test_gen = FullBatchLinkGenerator(G_test, method="gcn")
    test_flow = test_gen.flow(edge_ids_test, edge_labels_test)
    gcn = GCN(
        layer_sizes=[64, 64], activations=["relu", "relu"], generator=train_gen, dropout=0.3
    )
    x_inp, x_out = gcn.in_out_tensors()
    prediction = LinkEmbedding(activation="relu", method="ip")(x_out)
    prediction = keras.layers.Reshape((-1,))(prediction)
    model = keras.Model(inputs=x_inp, outputs=prediction)

    model.compile(
        optimizer=keras.optimizers.Adam(lr=0.01),
        loss=keras.metrics.binary_crossentropy,
        metrics=["acc"],
    )

    init_train_metrics = model.evaluate(train_flow)
    init_test_metrics = model.evaluate(test_flow)

    print("\nTrain Set Metrics of the initial (untrained) model:")
    for name, val in zip(model.metrics_names, init_train_metrics):
        print("\t{}: {:0.4f}".format(name, val))

    print("\nTest Set Metrics of the initial (untrained) model:")
    for name, val in zip(model.metrics_names, init_test_metrics):
        print("\t{}: {:0.4f}".format(name, val))
    history = model.fit(
        train_flow, epochs=epochs, validation_data=test_flow, verbose=2, shuffle=False
    )
