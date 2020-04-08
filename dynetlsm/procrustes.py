import numpy as np
import numpy.linalg as linalg


def flatten_array(X):
    return X.reshape(np.prod(X.shape[:-1]), -1)


def compute_procrustes_rotation(X, Y):
    """X is the reference and Y is matching X"""
    X_center = X - np.mean(X, axis=0)
    Y_center = Y - np.mean(Y, axis=0)

    u, s, v = linalg.svd(np.dot(X.T, Y))

    return np.dot(v, u.T)


def static_procrustes_rotation(X, Y):
    """Rotate Y to match X"""
    A = compute_procrustes_rotation(X, Y)
    return np.dot(Y - np.mean(Y, axis=0), A)


def longitudinal_procrustes_rotation(X_ref, X):
    """A single procrustes transformation applied across time."""
    n_time_steps, n_nodes = X.shape[:-1]

    X_ref = flatten_array(X_ref)
    X = flatten_array(X)
    X = static_procrustes_rotation(X_ref, X)
    return X.reshape(n_time_steps, n_nodes, -1)


def longitudinal_procrustes_transform(X, means, copy=True):
    if copy:
        # copy data over
        X = X.copy()

        if means is not None:
            means = means.copy()

    # apply procrustes transformation to samples past the tuning phase
    n_samples = X.shape[0]
    X_ref = X[0]
    for i in range(1, n_samples):
        X_new = X[i]

        P = compute_procrustes_rotation(X_ref, X_new)
        X[i] = np.dot(X_new, P)

        if means is not None:
            mu_new = means[i]
            means[i] = np.dot(mu_new, P)

    return X, means
