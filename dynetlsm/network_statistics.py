import numpy as np

from scipy.sparse import csgraph
from sklearn.preprocessing import LabelEncoder

from .array_utils import nondiag_indices_from


def is_dynamic(Y):
    return Y.ndim == 3


def num_edges(Y, is_directed=False):
    return np.sum(Y) if is_directed else 0.5 * np.sum(Y)


def density(Y, is_directed=False):
    n_edges = num_edges(Y, is_directed=is_directed)
    n_nodes = Y.shape[1] if is_dynamic(Y) else Y.shape[0]

    n_possible = n_nodes * (n_nodes - 1)
    if is_dynamic(Y):
        n_possible *= Y.shape[0]

    if not is_directed:
        n_possible *= 0.5

    return n_edges / n_possible


def modularity(Y, z, is_directed=False):
    if is_dynamic(Y):
        n_time_steps = Y.shape[0]
        mod_ave = 0
        for t in range(n_time_steps):
            mod_ave += static_modularity(Y[t], z[t],
                                         is_directed=is_directed)
        return mod_ave / n_time_steps

    return static_modularity(Y, z, is_directed=is_directed)


def static_modularity(Y, z, is_directed=False):
    """modularity for a static network."""
    if is_directed:
        n_edges = Y.sum()
        degree = 0.5 * (Y.sum(axis=0) + Y.sum(axis=1))
    else:
        n_edges = Y.sum() / 2
        degree = Y.sum(axis=0)
    degree = degree.reshape(-1, 1)

    encoder = LabelEncoder().fit(z)
    groups = encoder.transform(z)
    n_groups = encoder.classes_.shape[0]

    A = 0.5 * (Y + Y.T) if is_directed else Y
    B = A - np.dot(degree, degree.T) / (2 * n_edges)
    S = np.eye(n_groups)[groups.astype(np.int)]

    return np.trace(S.T @  B @ S) / (2 * n_edges)


def connected_nodes(Y, is_directed=False, size_cutoff=1):
    # NOTE: weak connections essentially treats the graph as undirected
    n_components, labels = csgraph.connected_components(Y,
                                                        directed=is_directed,
                                                        connection='weak')

    if n_components == 1:
        return np.arange(Y.shape[1])

    component_sizes = np.bincount(labels)
    non_singletons = np.where(component_sizes > size_cutoff)[0]

    return np.in1d(labels, non_singletons)
