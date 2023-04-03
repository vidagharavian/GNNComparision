import pandas as pd
from numpy import load
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, accuracy_score

data=load('logs/mine/K5dropout50ratio_coe0margin_coe100margin1withinnerproductFiedler5sigma100alpha100hid32lr10usedavidScoretrials1train_r80test_r10AllFalseSeed10/03-29-14:05:38/DIGRACinnerproduct_pred0.npy', mmap_mode='r')
edges = pd.read_csv("edges.csv")
edges['pred_weight'] = edges.apply(lambda x: 1 if data[x['Src']] > data[x['Dst']] else 0,axis=1)
n = confusion_matrix(edges['Weight'],edges['pred_weight'])
s=accuracy_score(edges['Weight'],edges['pred_weight'])

print(s)