import pandas as pd
from matplotlib import pyplot as plt


def plot_accuracy(accuracy):

    plt.plot()
    plt.plot(accuracy)
    # plt.plot(val_acc)
    plt.title('ROC-AUC Curve score')
    plt.ylabel('Score')
    plt.xlabel('Generation')
    # plt.legend(['train', 'val'], loc='upper left')
    plt.show()

def svr(accuracy,generation):
    pass


plot_accuracy(accuracy=pd.read_csv("output/RosenBrock_d20_pop100_g50.csv")["test_roc"])
