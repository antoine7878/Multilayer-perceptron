import numpy as np
import matplotlib.pyplot as plt

from .metrics import accuracy, confusion, to_pred, roc
from .models import Sequential
from .utils import dehot


def eval(args):
    model = Sequential.load(args.file)

    x = np.load("data/x_eval.npy")
    y = np.load("data/y_eval.npy")

    y_pred = model(x)
    val_loss = model.loss(y, y_pred).mean()
    val_acc = accuracy(y, y_pred)

    roc(y, y_pred)

    y_pred = to_pred(y_pred)
    y_pred = dehot(y_pred)
    y = dehot(y)


    tp, tn, fp, fn = confusion(y, y_pred)

    print("Validation:")
    print(f"Examples count: {x.shape[0]}")
    print(f"Loss: {val_loss:#.4g}")
    print(f"Accuracy: {val_acc:.2%}")
    print(f"Recall: {tp / (tp + fn):#.2g}")
    print(f"FPR: {fp / (fp + tn):#.2g}")
    print(f"Precision: {tp / (tp + fp):#.2g}")
    print(f"f1 score: {2 * tp / (2 * tp + fp + fn):.2%}")
    print("Confusion matrix:")
    print(np.array([[tp, fn], [fp, tn]]))
    plt.show()
