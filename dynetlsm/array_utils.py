import numpy as np


def triu_indices_from_3d(Y, k=0):
    return np.nonzero(~np.stack(
        [np.tri(Y.shape[1], Y.shape[2], k=k-1, dtype=np.bool) for
            t in range(Y.shape[0])]
    ))


def diag_indices_from_3d(Y):
    return np.nonzero(np.stack(
        [np.eye(Y.shape[1], Y.shape[2], dtype=np.bool) for
            t in range(Y.shape[0])]
    ))


def nondiag_indices_from_3d(Y):
    return np.nonzero(~np.stack(
        [np.eye(Y.shape[1], Y.shape[2], dtype=np.bool) for
            t in range(Y.shape[0])]
    ))


def nondiag_indices_from(Y):
    return np.nonzero(~np.eye(Y.shape[0], Y.shape[1], dtype=np.bool))
