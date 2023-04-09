

import dgl
import numpy as np
import pandas as pd


from ranker.DGL_presentation import create_archive
from ranker.Node2Vec import load_edges, apply_edges, train_model, load_edges_test, test, load_edge_pred, prediction
import warnings
warnings.filterwarnings("ignore")

dimension = 20
benchmark ='Ackley'
pop_size = 100



# You can replace DotPredictor with MLPPredictor.



def train_in_generation(generation,model,pred,optimizer):
    val_neg_u, val_neg_v, train_neg_u, train_neg_v, val_pos_u, val_pos_v, train_pos_u, train_pos_v, g, train_g = load_edges(
        generation, create_archive(generation, 15000))
    train_neg_g, train_pos_g = apply_edges(train_pos_u, train_pos_v, train_neg_u, train_neg_v, g)
    val_neg_g, val_pos_g = apply_edges(val_pos_u, val_pos_v, val_neg_u, val_neg_v, g)
    return train_model(model, train_g, train_pos_g, pred, train_neg_g, optimizer, val_pos_g, val_neg_g)



def test_in_generation(generation,model,pred):
    test_neg_u, test_neg_v, test_pos_u, test_pos_v, g = load_edges_test(generation)
    test_neg_g, test_pos_g = apply_edges(test_pos_u, test_pos_v, test_neg_u, test_neg_v, g)
    acc=test(pred, test_pos_g, test_neg_g, g, model)
    print('AUC', acc)
    return acc


def pred_in_generation(edges,model,df_list,pred):
    pred_u,pred_v,g = load_edge_pred(edges,df_list)
    train_neg_g = dgl.graph((pred_u, pred_v), num_nodes=g.number_of_nodes())
    return prediction(pred,train_neg_g,g, model)


def main(generations:int):
    global model
    df = pd.DataFrame({"generation":np.arange(1,generations,1)})
    test_roc =[]
    for generation in range(1,generations):
        print(f"generation: {generation}")
        if generation < 4:
            model=train_in_generation(generation,model)
            test_roc.append(np.nan)
        else:
            generation_roc =test_in_generation(generation,model)
            test_roc.append(generation_roc)
            model=train_in_generation(generation,model)
    df['test_roc']=test_roc
    df.to_csv(f"output/{benchmark}_d{dimension}_pop{pop_size}_g{generations}.csv")


# main(200)


