import numpy as np

from math import ceil

from sklearn.utils import check_random_state

from ..array_utils import triu_indices_from_3d


MAX_INT = np.iinfo(np.int32).max

__all__ = ['train_test_split']


def train_test_split(Y, test_size=0.1, random_state=None):
    """Split dyads into training and testing subsets.

    Parameters
    ----------
    Y : array-like, shape  (n_time_steps, n_nodes, n_nodes)
    """
    n_time_steps, n_nodes, _ = Y.shape

    random_state = check_random_state(random_state)

    # number of dyads in an undirected graph with n_nodes nodes
    n_dyads = int(0.5 * n_nodes * (n_nodes - 1))
    test_size_type = np.asarray(test_size).dtype.kind
    if test_size_type == 'f':
        n_test = ceil(test_size * n_dyads)
    else:
        n_test = int(test_size)

    Y_new = np.zeros_like(Y)
    for t in range(n_time_steps):
        tril_indices = np.tril_indices_from(Y[t], k=-1)

        perm = random_state.choice(
            np.arange(n_dyads), size=n_test, replace=False)
        test_indices = perm

        Y_vec = Y[t][tril_indices]
        Y_vec[perm] = -1.0
        Y_new[t][tril_indices] = Y_vec
        Y_new[t] += Y_new[t].T


    triu_indices =  triu_indices_from_3d(Y_new, k=1)
    test_indices = Y_new[triu_indices] == -1
    return Y_new, test_indices
