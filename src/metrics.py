import numpy as np
import matplotlib.pyplot as plt


def accuracy(y: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.argmax(y, axis=1) == np.argmax(y_pred, axis=1))


def to_pred(x: np.ndarray) -> np.ndarray:
    result = np.zeros_like(x)
    max_indices = np.argmax(x, axis=1)
    result[np.arange(x.shape[0]), max_indices] = 1
    return result


def true_positive(y: np.ndarray, y_pred: np.ndarray):
    return int(np.logical_and(y == 1, y_pred == 1).sum())


def false_positive(y: np.ndarray, y_pred: np.ndarray):
    return int(np.logical_and(y == 0, y_pred == 1).sum())


def true_negative(y: np.ndarray, y_pred: np.ndarray):
    return int(np.logical_and(y == 0, y_pred == 0).sum())


def flase_negative(y: np.ndarray, y_pred: np.ndarray):
    return int(np.logical_and(y == 1, y_pred == 0).sum())


def confusion(y: np.ndarray, y_pred: np.ndarray) -> tuple:
    tp = true_positive(y, y_pred)
    tn = true_negative(y, y_pred)
    fp = false_positive(y, y_pred)
    fn = flase_negative(y, y_pred)
    return tp, tn, fp, fn


def recal(y, y_pred):
    try:
        tp = true_positive(y, y_pred)
        fn = flase_negative(y, y_pred)
        return tp / (tp + fn)
    except Exception as _:
        return 0


def fpr(y, y_pred):
    try:
        fp = false_positive(y, y_pred)
        tn = true_negative(y, y_pred)
        return fp / (fp + tn)
    except Exception as _:
        return 1


def roc(y, y_pred):
    y = y[:, 0]
    y_pred = y_pred[:, 0]

    def compute_roc(y, y_score):
        thresholds = np.sort(np.unique(y_score))[::-1]
        tpr_list = []
        fpr_list = []

        for thresh in thresholds:
            y_pred = (y_score >= thresh).astype(int)

            tp, tn, fp, fn = confusion(y, y_pred)

            tpr = tp / (tp + fn) if (tp + fn) else 0
            fpr = fp / (fp + tn) if (fp + tn) else 0

            tpr_list.append(tpr)
            fpr_list.append(fpr)

        return fpr_list, tpr_list

    fpr, tpr = compute_roc(y, y_pred)

    plt.plot(fpr, tpr, label="ROC Curve")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
