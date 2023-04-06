import pandas as pd
from matplotlib import pyplot as plt


def plot_accuracy(function,pop,generation):
    for i in [10,20]:
        m = f"output/{function}_d{i}_pop{pop}_g{generation}.csv"
        accuracy =pd.read_csv(m)["test_roc"]
        plt.plot(accuracy,label=f"d{i}")
        # plt.plot(val_acc)
    plt.title('ROC-AUC Curve score')
    plt.ylabel('Score')
    plt.xlabel('Generation')
    plt.legend()
    # plt.legend(['train', 'val'], loc='upper left')
    plt.show()

def svr(accuracy,generation):
    pass


plot_accuracy("RosenBrock",100,200)
