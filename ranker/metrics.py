import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from texttable import Texttable
import latextable


default_compare_names_all = ['DIGRAC']
default_metric_names = ['test kendall tau', 'test kendall p', 'val kendall tau', 'val kendall p', 'all kendall tau', 'all kendall p']
def print_performance_mean_std(dataset:str, results:np.array, compare_names_all:list=default_compare_names_all,
                               metric_names:list=default_metric_names, print_latex:bool=True, print_std:bool=True):
    r"""Prints performance table (and possibly with latex) with mean and standard deviations.
        The best two performing methods are highlighted in \red and \blue respectively.

    Args:
        dataset: (string) Name of the data set considered.
        results: (np.array) Results with shape (num_trials, num_methods, num_metrics).
        compare_names_all: (list of strings, optional) Methods names to compare.
        metric_names: (list of strings, optional) Metrics to use (deemed better with larger values).
        print_latex: (bool, optional) Whether to print latex table also. Default True.
        print_std: (bool, optinoal) Whether to print standard deviations or just mean. Default False.
    """
    t = Texttable(max_width=120)
    t.set_deco(Texttable.HEADER)
    final_res_show = np.chararray(
        [len(metric_names)+1, len(compare_names_all)+1], itemsize=100)
    final_res_show[0, 0] = dataset+'Metric/Method'
    final_res_show[0, 1:] = compare_names_all
    final_res_show[1:, 0] = metric_names
    std = np.chararray(
        [len(metric_names), len(compare_names_all)], itemsize=20)
    results_std = np.transpose(np.round(np.nanstd(results,0),2))
    results_mean = np.transpose(np.round(np.nanmean(results,0),2))
    for i in range(results_mean.shape[0]):
        for j in range(results_mean.shape[1]):
            final_res_show[1+i, 1+j] = '{:.2f}'.format(results_mean[i, j])
            std[i, j] = '{:.2f}'.format(1.0*results_std[i, j])
    if print_std:
        plus_minus = np.chararray(
            [len(metric_names), len(compare_names_all)], itemsize=20)
        plus_minus[:] = '$\pm$'
        final_res_show[1:, 1:] = final_res_show[1:, 1:] + plus_minus + std
    if len(compare_names_all)>1:
        red_start = np.chararray([1], itemsize=20)
        blue_start = np.chararray([1], itemsize=20)
        both_end = np.chararray([1], itemsize=20)
        red_start[:] = '\\red{'
        blue_start[:] = '\\blue{'
        both_end[:] = '}'
        for i in range(results_mean.shape[0]):
            if metric_names[i] in ['test kendall tau', 'val kendall tau', 'all kendall tau']:
                best_values = -np.sort(-results_mean[i])[:2] # the bigger, the better
            else:
                best_values = np.sort(results_mean[i])[:2] # the smaller, the better
            final_res_show[i+1, 1:][results_mean[i]==best_values[0]] = red_start + final_res_show[i+1, 1:][results_mean[i]==best_values[0]] + both_end
            if best_values[0] != best_values[1]:
                final_res_show[i+1, 1:][results_mean[i]==best_values[1]] = blue_start + final_res_show[i+1, 1:][results_mean[i]==best_values[1]] + both_end

    t.add_rows(final_res_show)
    print(t.draw())
    if print_latex:
        print(latextable.draw_latex(t, caption=dataset +
                                    " performance.", label="table:"+dataset) + "\n")

def calculate_upsets(A: torch.FloatTensor,
                     score: torch.FloatTensor, style: str='ratio', margin: float=0.01,device="cuda")-> torch.FloatTensor:
    r"""Calculate upsets from rankings (with ties). 
    Convention: r_i (the score for the i-th node) larger, better skill, smaller ranks, larger out-degree.

    Args:
        A: (torch.FloatTensor) Adjacency matrix.
        score: (torch.FloatTensor) Ranking scores, with shape (num_nodes, 1).
        style: (str, optional) Styles of loss to choose, default ratio.
        margain: (float, optional) Margin for which we need to hold for the margin version, default 0.01.
        
    :rtype: 
        upset: (torch.FloatTensor) Portion of upsets, take average so that the value is bounded and normalized.
        """
    # for numerical stability
    epsilon = torch.FloatTensor([1e-8]).to(score)
    epsilon=epsilon.to(device)
    A=A.to(device)
    M = A - torch.transpose(A, 0, 1)
    M= M.to(device)
    indices = (M != 0)
    T1 = score - score.T
    if style == 'simple':
        upset = torch.mean(torch.pow(torch.sign(T1[indices]) - torch.sign(M[indices]), 2))
    elif style == 'margin':
        upset = torch.mean((M + torch.abs(M)).multiply(torch.nn.ReLU()(-T1 + margin))[indices]) # margin loss
    elif style == 'naive':
        upset = torch.sum(torch.sign(T1[indices]) != torch.sign(M[indices]))/torch.sum(indices)
    else: # 'ratio'
        T2 = score + score.T + epsilon
        T = torch.div(T1, T2)
        M2 = A + (A.T).to(device) + epsilon
        M3 = torch.div(M, M2)
        powers = torch.pow((M3-T)[indices], 2)
        upset = torch.mean(powers)# /M.numel()
        # the value is from 0 to 2, as the difference is from 0 to 2 (1 and -1)
    return upset

def generation_accuracy(pred_rank,edges,save_path,identifier_str,generation,tau):
    data = pred_rank
    edges['pred_weight'] = edges.apply(lambda x: 1 if data[int(x['Src'])] > data[int(x['Dst'])] else -1, axis=1)
    s = accuracy_score(edges['Weight'], edges['pred_weight'])

    print(f"model accuracy generation {generation}, {s}")
    df=pd.DataFrame({"base_line":[identifier_str],"generation":[generation],"accuracy":[s],"test_size":[len(pred_rank)],"tau":[tau]}).to_csv(f"{save_path}_{identifier_str}_accuracy.csv")



