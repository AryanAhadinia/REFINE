import numpy as np

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score


def get_best_threshold(gt, pr):
    precision, recall, thresholds = precision_recall_curve(gt, pr)
    f1 = 2 * (precision * recall) / (precision + recall)
    f1[np.isnan(f1)] = 0
    best_index = np.argmax(f1)
    best_threshold = thresholds[best_index]
    best_precision = precision[best_index]
    best_recall = recall[best_index]
    best_f1 = f1[best_index]
    return best_threshold, best_precision, best_recall, best_f1


def accuracy(gt, pr_classes):
    return accuracy_score(gt, pr_classes)


def sre(gt, pr_classes):
    return np.linalg.norm(gt) / np.linalg.norm(gt - pr_classes)


def mcc(gt, pr_classes):
    return matthews_corrcoef(gt, pr_classes)


def evaluate(ground_truth, observed, predicted):
    # all of these are numpy arrays with shape (n, n)
    # where n is the number of nodes
    # ground_truth is the ground truth adjacency matrix
    # observed is the observed adjacency matrix
    # predicted is the predicted adjacency matrix

    ob = observed.flatten()
    gt = ground_truth.flatten()
    pr = predicted.flatten()

    gt = gt[ob == 0].flatten()
    pr = pr[ob == 0].flatten()

    best_threshold, precision, recall, f1 = get_best_threshold(gt, pr)
    auc = roc_auc_score(gt, pr)

    pr_classes = (pr > best_threshold).astype(int)

    accuracy_score = accuracy(gt, pr_classes)
    sre_score = sre(gt, pr_classes)
    mcc_score = mcc(gt, pr_classes)

    return {
        "predicted": predicted,
        "best_threshold": best_threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy_score,
        "auc": auc,
        "sre": sre_score,
        "mcc": mcc_score,
    }
