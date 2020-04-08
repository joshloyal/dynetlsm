import networkx as nx
import numpy as np
import scipy.linalg as linalg

from scipy.sparse import csgraph
from scipy.optimize import minimize

from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances, euclidean_distances

from .procrustes import static_procrustes_rotation


__all__ = ['calculate_distances', 'generalized_mds', 'longitudinal_kmeans',
           'initialize_radii']


def calculate_distances(X, metric='euclidean', squared=False):
    """Calulates the pairwise distances between latent positions X."""
    if X.ndim == 2:
        return pairwise_distances(X, metric=metric)

    n_time_steps, n_nodes, _ = X.shape

    dist = np.empty((n_time_steps, n_nodes, n_nodes))
    for t in range(n_time_steps):
        if metric == 'euclidean':
            dist[t] = euclidean_distances(X[t], squared=squared)
        else:
            dist[t] = pairwise_distances(X[t], metric=metric)

    return dist


def shortest_path_dissimilarity(Y, unweighted=True):
    """Calculate the shortest-path dissimilarty of a static graph."""
    dist = csgraph.shortest_path(Y, directed=False, unweighted=unweighted)

    # impute unconnected components with the largest distance plus 1
    inf_mask = np.isinf(dist)
    dist[inf_mask] = np.max(dist[~inf_mask]) + 1

    return dist


def generalized_mds(Y, n_features=2, is_directed=False, unweighted=True,
                    lmbda=10, random_state=None):
    """Generalized Multi-Dimension Scaling (Sarkar and Moore, 2005)."""
    is_dynamic_graph = Y.ndim == 3
    if not is_dynamic_graph:
        Y = np.expand_dims(Y, axis=0)

    n_time_steps, n_nodes, _ = Y.shape

    # calculate shortest-path dissimilarity for each time step
    D = np.empty((n_time_steps, n_nodes, n_nodes))
    for t in range(Y.shape[0]):
        D[t] = shortest_path_dissimilarity(Y[t], unweighted=unweighted)

    # compute latent positions based on MDS
    X = np.empty((n_time_steps, n_nodes, n_features))

    # classical multi-dimensional scaling for t = 1
    X[0] = MDS(dissimilarity='precomputed',
               n_components=n_features,
               random_state=random_state).fit_transform(D[0])

    # minimize the objective function found in Sarkar and Moore
    H = np.eye(n_nodes) - (1. / n_nodes) * np.ones((n_nodes, n_nodes))
    for t in range(1, n_time_steps):
        alpha = 1 / (1 + lmbda)
        beta = lmbda / (1 + lmbda)
        XXt = alpha * np.dot(H, np.dot(-0.5 * D[t] ** 2, H))
        XXt = XXt + beta * (np.dot(X[t-1], X[t-1].T))

        # the optimum is the eigen-decomposition of XXt
        evals, evecs = linalg.eigh(XXt)

        # flip so in descending order
        evecs = evecs[:, ::-1]
        evals = evals[::-1]

        # extract features (top n_features eigenvectors scaled by eigenvalue)
        X[t] = evecs[:, :n_features] * np.sqrt(evals[:n_features])

        # procrustes transformation to fix rotation invariance
        X[t] = static_procrustes_rotation(X[t-1], X[t])

    # the directed model scales the space so that it is roughly [-1, 1],
    # i.e. same scale as the radii
    if is_directed:
        X /= n_nodes

    return X if is_dynamic_graph else np.squeeze(X)


def longitudinal_kmeans(X, n_clusters=5, var_reg=1e-3,
                        fixed_clusters=True, random_state=None):
    """Longitudinal K-Means Algorithm (Genolini and Falissard, 2010)"""
    n_time_steps, n_nodes, n_features = X.shape

    # vectorize latent positions across time
    X_vec = np.moveaxis(X, 0, -1).reshape(n_nodes, n_time_steps * n_features)

    # perform normal k-means on the vectorized features
    kmeans = KMeans(n_clusters=n_clusters,
                    random_state=random_state).fit(X_vec)

    # this method assigns a single cluster to each point across time.
    labels = kmeans.labels_.reshape(-1, 1)
    labels = np.hstack([labels] * n_time_steps).T

    # un-vectorize centers, shape (n_time_steps, n_centers, n_features)
    centers_vec = kmeans.cluster_centers_
    if fixed_clusters:
        centers = np.empty((n_clusters, n_features))
        for k in range(n_clusters):
            muk = centers_vec[k].reshape(-1, n_time_steps).T
            centers[k] = muk.mean(axis=0)  # average position overtime
    else:
        centers = np.empty((n_time_steps, n_clusters, n_features))
        for k in range(n_clusters):
            centers[:, k] = centers_vec[k].reshape(-1, n_time_steps).T

    # calculate cluster variances (assumed spherical and constant over-time)
    variances = np.zeros(n_clusters, dtype=np.float64)
    for k in range(n_clusters):
        for t in range(n_time_steps):
            variances[k] += np.var(X[t][labels[t] == k], axis=0).mean()
        variances[k] /= n_time_steps

    # clusters with a single data point will have zero-variance.
    # assign a fudge factor in this case
    variances[variances == 0.] = var_reg

    return centers, variances, labels


def initialize_radii(Y, reg=1e-5):
    """Initialize radii to normalized average of out-degree and in-degree
    over time.
    """
    radii = 0.5 * (Y.sum(axis=(0, 1)) + Y.sum(axis=(0, 2)))
    radii /= Y.sum()

    # radii can be zero if no edges are present. Add a small amount
    # of social reach to each radii in this case.
    if np.any(radii == 0.):
        radii += reg
        radii /= np.sum(radii)

    return radii
