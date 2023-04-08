import itertools

import torch
from ranker.MyData import GraphSAGE, MLPPredictor, MyHadamardLinkPredictor

benchmark ='Ackley'
dimension = 10
pop_size = 100
generations = 30
archive_size = 15000
last_model_test_accuracy = 0
last_model = GraphSAGE(dimension, 64,32)
hash_dict = {}
data_set = []

# You can replace DotPredictor with MLPPredictor.
pred = MLPPredictor(32)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# pred = MyHadamardLinkPredictor(in_feats=32,
#                                       hidden_feats=16,
#                                       num_layers=2,
#                                       n_tasks=1,
#                                       dropout=0.1).to(device)
optimizer = torch.optim.Adam(itertools.chain(last_model.parameters(), pred.parameters()), lr=0.001)
def surrogate_use_permission(generation):
    if last_model is None or generation < 3:
        return False
    if last_model_test_accuracy > 0.70:
        return True

def get_test_split(generation):
    if generation<4:
        return 1
    else:
        return 0.2



