from matplotlib import pyplot as plt


def plot_accuracy(accuracy,val_acc):

    plt.plot()
    plt.plot(accuracy)
    # plt.plot(val_acc)
    plt.title('Train Loss')
    plt.ylabel('Loss')
    plt.xlabel('Generation')
    # plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    plt.plot()
    plt.plot(val_acc)
    # plt.plot(val_acc)
    plt.title('Model ROC')
    plt.ylabel('ROC')
    plt.xlabel('Generation')
    # plt.legend(['train', 'val'], loc='upper left')
    plt.show()
