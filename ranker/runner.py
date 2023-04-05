import pandas as pd
from stellargraph.layer import GraphSAGE
from stellargraph.mapper import GraphSAGELinkGenerator

from ranker.GraphSAGE import GraphSage
from ranker.linkClassifier import build_graph


def create_archive(save_path,generation, archive_size=16000):
    edges = []
    data_set = pd.DataFrame()
    for i in range(generation, 0, -1):
        edge = pd.read_csv(f"{save_path}/{i}.csv")
        edges.append(edge)
        data_set = pd.concat(edges)
        if len(data_set) > archive_size:
            data_set = data_set[:archive_size]
            break
    return data_set


def main():
    pop_size = 100
    benchmark = 'RosenBrock'
    dimension = 10
    generations =200
    graph_sage = GraphSage(40, 20, [5,5], [10, 10])
    for i in range(1,generations):
        if i<3:
            data_set =create_archive("generations",i)
            G_train, edge_ids_train, edge_labels_train,G_test, edge_ids_test, edge_labels_test=build_graph(i,data_set)
            graph_sage.prepare_data(G_train, edge_ids_train, edge_labels_train, G_test, edge_ids_test, edge_labels_test)
            model=graph_sage.build_model()
            graph_sage.model_fit(model,graph_sage.train_flow,test_flow=graph_sage.test_flow)
            print("after")
            graph_sage.evaluate(model, graph_sage.test_flow)

        else:
            G_train, edge_ids_train, edge_labels_train, G_test, edge_ids_test, edge_labels_test = build_graph(i)
            graph_sage.prepare_data(G_train, edge_ids_train, edge_labels_train, G_test, edge_ids_test, edge_labels_test)
            graph_sage.evaluate(model,graph_sage.test_flow)
            model = graph_sage.build_model()
            graph_sage.model_fit(model, graph_sage.train_flow, test_flow=graph_sage.test_flow)
            print("after")
            graph_sage.evaluate(model, graph_sage.test_flow)
main()




