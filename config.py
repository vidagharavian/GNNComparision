import itertools

import pandas as pd
import torch
from ranker.MyData import GraphSAGE, MLPPredictor



class Config():
    benchmark = 'RosenBrock'
    dimension = 30
    pop_size = 100
    generations = 300
    archive_size = 1000
    current_gen = 1
    counter = 0
    last_model_test_accuracy = 0

    hash_dict = {}
    data_set = []

    # You can replace DotPredictor with MLPPredictor.

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # pred = MyHadamardLinkPredictor(in_feats=64,
    #                                       hidden_feats=32,
    #                                       num_layers=32,
    #                                       n_tasks=1,
    #                                       dropout=0.1).to(device)
    pred = None
    last_model = None
    optimizer = None

    def __init__(self):
        self.pred= MLPPredictor(64).to(self.device)
        self.last_model = GraphSAGE(self.dimension, 128, 64).to(self.device)
        self.optimizer = torch.optim.Adam(itertools.chain(self.last_model.parameters(), self.pred.parameters()), lr=0.0001)
        torch.save(self.pred.state_dict(), "pred.t7")
        torch.save(self.optimizer.state_dict(), "opt.t7")
        torch.save(self.last_model.state_dict(), "model.t7")

    def surrogate_use_permission(self,generation):
        if self.last_model is None or generation < 3:
            return False
        if self.last_model_test_accuracy > 0.70:
            return True
    @classmethod
    def get_test_split(cls,generation):
        if generation < 4:
            return 1
        else:
            return 0.3

    def create_edge_vector_generation(self,df):
        source, label, target = df['source'], df['label'], df['target']
        data_set = []
        for num, j in enumerate(label):
            # array_source = change_to_array(source[num])
            # array_target = change_to_array(target[num])
            hashed_source = self.hashFloatArray(source[num])
            hashed_target = self.hashFloatArray(target[num])
            data_set.append({"Src": self.hash_dict[hashed_source], "Dst": self.hash_dict[hashed_target], "Weight": j})
        data = pd.DataFrame.from_records(data_set)
        return data
    @staticmethod
    def hashFloatArray(arr):
        h = ''
        for i in arr:
            n = hash(i)
            h += str(n)
        return h

    def create_feature_vector(self,df, save=True):
        source, target = df['source'], df['target']
        for i, j in zip(source, target):
            # i = change_to_array(i)
            hashed_i = self.hashFloatArray(i)
            if hashed_i not in self.hash_dict.keys():
                self.hash_dict[hashed_i] = len(self.data_set)
                self.data_set.append(i)
            # j = change_to_array(j)
            hashed_i = self.hashFloatArray(j)
            if hashed_i not in self.hash_dict.keys():
                self.hash_dict[hashed_i] = len(self.data_set)
                self.data_set.append(j)
        data = pd.DataFrame.from_records(self.data_set)
        if save:
            data.to_csv("./features.csv", index=False)
        else:
            return data

    def reset_params(self):
        self.hash_dict={
        }
        self.data_set=[]
        self.current_gen=1
        self.last_model.load_state_dict(torch.load("model.t7"))
        self.optimizer.load_state_dict(torch.load("opt.t7"))
        self.pred.load_state_dict(torch.load("pred.t7"))




