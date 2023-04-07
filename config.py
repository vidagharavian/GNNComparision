import itertools

import torch

from ranker.MyData import GraphSAGE, MLPPredictor

benchmark ='RosenBrock'
dimension = 20
pop_size = 100
generations = 200
archive_size = 15000
last_model_test_accuracy = 0
last_model = GraphSAGE(dimension, 64,32,0.2)
hash_dict = {}
data_set = []

# You can replace DotPredictor with MLPPredictor.
pred = MLPPredictor(32)

optimizer = torch.optim.Adam(itertools.chain(last_model.parameters(), pred.parameters()), lr=0.01)


def surrogate_use_permission(generation):
    if last_model is None or generation < 3:
        return False
    if last_model_test_accuracy > 0.70:
        return True

def get_test_split(generation):
    if generation<4:
        return 1
    elif generation< 10:
        return 0.2
    else:
        return 0.1



