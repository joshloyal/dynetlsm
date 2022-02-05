import numpy as np

from sklearn.metrics import roc_auc_score, mutual_info_score
from sklearn.metrics.cluster import entropy

from .array_utils import triu_indices_from_3d
from .array_utils import nondiag_indices_from_3d


def network_auc(Y_true, Y_pred, is_directed=False, nan_mask=None):
    if is_directed:
        indices = nondiag_indices_from_3d(Y_true)
    else:
        indices = triu_indices_from_3d(Y_true, 1)

    y_fit = Y_pred[indices]
    y_true = Y_true[indices]

    if nan_mask is not None:
        y_fit = y_fit[~nan_mask]
        y_true = y_true[~nan_mask]

    return roc_auc_score(y_true, y_fit)


def out_of_sample_auc(y_true, y_pred, test_indices):
    indices = triu_indices_from_3d(y_true, k=1)
    return roc_auc_score(y_true[indices][test_indices], y_pred)


def _network_auc_directed():
    y_true, y_fit = [], []

    indices = np.triu_indices_from(Y_true[0], 1)
    for t in range(Y_true.shape[0]):
        y_fit.append(Y_pred[t][indices])
        y_true.append(Y_true[t][indices])

    return roc_auc_score(np.hstack(y_true), np.hstack(y_fit))


def _network_auc_undirected(Y_true, Y_pred):
    y_true, y_fit = [], []

    indices = triu_indices_from_3d(Y_true, 1)
    y_fit = Y_pred[indices]

    return roc_auc_score(np.hstack(y_true), np.hstack(y_fit))


def variation_of_information(labels_true, labels_pred):
    entropy_true = entropy(labels_true)
    entropy_pred = entropy(labels_pred)
    mutual_info = mutual_info_score(labels_true, labels_pred)

    return entropy_true + entropy_pred - 2 * mutual_info
