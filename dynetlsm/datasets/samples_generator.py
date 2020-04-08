"""
Generate samples of synthetic data sets.
"""
import six

import numpy as np

from math import ceil

from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances
from sklearn.utils import check_random_state

from ..network_likelihoods import directed_network_probas
from ..latent_space import calculate_distances


__all__ = ['make_gaussian_mixture',
           'make_dynamic_gaussian_mixture',
           'make_blobs_network',
           'make_latent_space_network',
           'make_dynamic_latent_space_network',
           'make_splitting_blobs',
           'network_from_dynamic_latent_space',
           'synthetic_network',
           'synthetic_directed_network',
           'synthetic_undirected_network',
           'merging_synthetic_network',
           'simple_splitting_synthetic_undirected_network',
           'synthetic_dynamic_network',
           'make_splitting_network']


def make_gaussian_mixture(n_samples=100, n_features=2, alpha=2,
                          cluster_std=0.5, std_shape=2, std_scale=4,
                          center_box=(-5.0, 5.0),
                          shuffle=True, random_state=None):
    """
    Generate a non-parametric gaussian mixture using a chinese restaurant
    process.
    """
    rng = check_random_state(random_state)

    # sample cluster labels
    _, nk = sample_chinese_restaurant(n_samples=n_samples,
                                      alpha=alpha,
                                      random_state=rng)

    # sample cluster centers
    n_centers = nk.shape[0]
    centers = rng.uniform(center_box[0], center_box[1],
                          size=(n_centers, n_features))

    if isinstance(cluster_std, six.string_types):
        # sample variances from an inverse gamma
        cluster_std = 1. / rng.gamma(shape=std_shape,
                                     scale=1. / std_scale,
                                     size=n_centers)
        cluster_std = np.sqrt(cluster_std)
    else:
        cluster_std = n_centers * [cluster_std]

    # sample data points
    X = []
    z = []
    for i in range(n_centers):
        X.append(rng.normal(loc=centers[i], scale=cluster_std[i],
                            size=(nk[i], n_features)))
        z += [i] * nk[i]

    X = np.concatenate(X)
    z = np.asarray(z, dtype=np.int)

    if shuffle:
        indices = np.arange(n_samples)
        rng.shuffle(indices)
        X = X[indices]
        z = z[indices]

    return X, z


def make_dynamic_gaussian_mixture(n_samples=100, n_features=2, n_time_steps=3,
                                  alpha=2, sticky_proba=0.75,
                                  mixing_lambda=0.5,
                                  center_step_size=1.0,
                                  cluster_std=0.5, std_shape=2, std_scale=4,
                                  center_box=(-5.0, 5.0),
                                  shuffle=True, random_state=None):
    """
    Sample from a Dynamic RCRP Gaussian Mixture.
    """
    rng = check_random_state(random_state)

    # sample cluster labels from a RCRP
    z = sample_recurrent_chinese_restaurant(n_samples=n_samples,
                                            n_time_steps=n_time_steps,
                                            alpha=alpha,
                                            sticky_proba=sticky_proba,
                                            random_state=rng)

    X = np.empty((n_time_steps, n_samples, n_features), dtype=np.float64)

    # sample clusters at the first time step
    n_clusters = np.unique(z).shape[0]
    nt = np.bincount(z[0])
    centers = rng.uniform(center_box[0], center_box[1],
                          size=(n_clusters, n_features))
    if isinstance(cluster_std, six.string_types):
        # sample variances from an inverse gamma
        cluster_std = 1. / rng.gamma(shape=std_shape,
                                     scale=1. / std_scale,
                                     size=n_clusters)
        cluster_std = np.sqrt(cluster_std)
    else:
        cluster_std = n_clusters * [cluster_std]

    for k in range(n_clusters):
        if k < nt.shape[0] and nt[k] > 0:
            X[0, z[0] == k, :] = rng.normal(loc=centers[k],
                                            scale=cluster_std[k],
                                            size=(nt[k], n_features))

    # sample remaining time steps
    nt_prev = nt
    for t in range(1, n_time_steps):
        nt = np.bincount(z[t])
        for k in range(n_clusters):
            if k < nt.shape[0] and nt[k] > 0:
                if k < nt_prev.shape[0] and nt_prev[k] > 0:
                    centers[k] = rng.normal(loc=centers[k],
                                            scale=center_step_size,
                                            size=(1, n_features))

                mu = (mixing_lambda * centers[k] +
                        (1 - mixing_lambda) * X[t-1, z[t] == k, :])
                X[t, z[t] == k, :] = rng.normal(loc=mu,
                                                scale=cluster_std[k],
                                                size=(nt[k], n_features))
        nt_prev = nt

    return X, z


def make_blobs_network(n_nodes=100, n_features=2, intercept=0, coef=1,
                       make_directed=False, n_centers=4,
                       cluster_std=0.4, center_box=(-1.0, 1.0),
                       metric='euclidean', shuffle=True, random_state=None):
    """
    Generate a network from a latent space containing a parametric
    gaussian mixture model.
    """
    rng = check_random_state(random_state)

    X, z = make_blobs(n_samples=n_nodes, n_features=n_features,
                      centers=n_centers, cluster_std=cluster_std,
                      center_box=center_box, shuffle=shuffle,
                      random_state=rng)

    # sample the adjacency matrix
    dij = pairwise_distances(X, metric=metric)
    eta = intercept - coef * dij
    pij = np.exp(eta) / (1 + np.exp(eta))
    Y = rng.binomial(1, pij).astype(np.int)
    if not make_directed:
        Y = np.triu(Y, 1)
        Y += Y.T

    return Y, X, z


def make_latent_space_network(n_nodes=100, n_features=2, intercept=0,
                              make_directed=False,
                              alpha=2, cluster_std=0.5, std_shape=2,
                              std_scale=10, center_box=(-5.0, 5.0),
                              metric='euclidean',
                              shuffle=True, random_state=None):
    """
    Generate a network from latent space containing a CRP Gaussian Mixture.
    """
    rng = check_random_state(random_state)

    X, z = make_gaussian_mixture(n_samples=n_nodes, n_features=n_features,
                                 alpha=alpha, cluster_std=cluster_std,
                                 std_shape=std_shape, std_scale=std_scale,
                                 center_box=center_box, shuffle=shuffle,
                                 random_state=rng)

    # sample the adjacency matrix
    dij = pairwise_distances(X, metric=metric)
    eta = intercept - dij
    pij = np.exp(eta) / (1 + np.exp(eta))
    Y = rng.binomial(1, pij).astype(np.int)
    if not make_directed:
        Y = np.triu(Y, 1)
        Y += Y.T

    return Y, X, z


def network_from_latent_space(X, intercept=1, coef=1,
                              make_directed=False,
                              metric='euclidean', random_state=None):
    rng = check_random_state(random_state)

    n_nodes = X.shape[0]
    Y = np.zeros((n_nodes, n_nodes), dtype=np.float64)

    # sample the adjacency matrix
    dij = pairwise_distances(X, metric=metric)
    eta = intercept - coef * dij
    pij = np.exp(eta) / (1 + np.exp(eta))
    Y = rng.binomial(1, pij).astype(np.int)
    if not make_directed:
        Y = np.triu(Y, 1)
        Y += Y.T

    return Y

def network_from_dynamic_latent_space(X, intercept=1, coef=1,
                                      radii=None,
                                      metric='euclidean', random_state=None):
    rng = check_random_state(random_state)

    n_time_steps, n_nodes, _ = X.shape
    Y = np.zeros((n_time_steps, n_nodes, n_nodes), dtype=np.float64)
    dij = calculate_distances(X)
    probas = np.zeros((n_time_steps, n_nodes, n_nodes), dtype=np.float64)
    if radii is not None:
        probas = directed_network_probas(
            dij, radii, intercept[0], intercept[1])

    for t in range(n_time_steps):
        # sample the adjacency matrix
        if radii is None:
            eta = intercept - coef * dij[t]
            pij = np.exp(eta) / (1 + np.exp(eta))
            probas[t] = pij
        else:
            pij = probas[t]
        Y[t] = rng.binomial(1, pij).astype(np.int)
        if radii is None:
            Y[t] = np.triu(Y[t], 1)
            Y[t] += Y[t].T

    return Y, probas

def make_dynamic_latent_space_network(n_nodes=100, n_time_steps=3, n_features=2,
                                      intercept=1, coef=5,
                                      make_directed=False,
                                      metric='euclidean',
                                      alpha=2, sticky_proba=0.9,
                                      mixing_lambda=0.9,
                                      center_step_size=0.25,
                                      cluster_std=0.2, std_shape=2, std_scale=4,
                                      center_box=(-1.0, 1.0),
                                      shuffle=True, random_state=None):

    rng = check_random_state(random_state)

    X, z = make_dynamic_gaussian_mixture(n_samples=n_nodes,
                                         n_features=n_features,
                                         n_time_steps=n_time_steps,
                                         alpha=alpha,
                                         sticky_proba=sticky_proba,
                                         mixing_lambda=mixing_lambda,
                                         center_step_size=center_step_size,
                                         cluster_std=cluster_std,
                                         std_shape=std_shape,
                                         std_scale=std_scale,
                                         center_box=center_box,
                                         shuffle=shuffle, random_state=rng)

    Y = np.zeros((n_time_steps, n_nodes, n_nodes), dtype=np.float64)
    for t in range(n_time_steps):
        # sample the adjacency matrix
        dij = pairwise_distances(X[t], metric=metric)
        eta = intercept - coef * dij
        pij = np.exp(eta) / (1 + np.exp(eta))
        Y[t] = rng.binomial(1, pij).astype(np.int)
        if not make_directed:
            Y[t] = np.triu(Y[t], 1)
            Y[t] += Y[t].T

    return Y, X, z


def make_splitting_blobs(n_nodes=100, random_state=123):
    rng = check_random_state(random_state)
    lmbda = 0.8
    mu = np.array([[1.5, 0.0],
                   [-1.5, 0.0],
                   [-1.5, 1.5],
                   [-1.5, -1.5],
                   [-1.5, -1.5]])
    sigma = (np.array([0.15, 0.15, 0.05, 0.05, 0.1]) * 4).reshape(-1, 1)

    X, z = [], []

    # t = 0
    X1 = sigma[0] * rng.randn(n_nodes).reshape(n_nodes // 2, 2) + mu[0]
    X2 = sigma[1] * rng.randn(n_nodes).reshape(n_nodes // 2, 2) + mu[1]
    X.append(np.vstack((X1, X2)))

    z0 = np.zeros(n_nodes, dtype=np.int)
    z0[(n_nodes // 2):] = 1
    z.append(z0)

    # t = 1 (move around according to lmbda * mu + (1 - lmbda) * xit

    # re-sample z0 based on distances
    z0 = rng.binomial(1, p=np.array([0.1, 0.9])[z])
    X1 = sigma[z0] * rng.randn(n_nodes, 2) + lmbda * mu[z0] + (1 - lmbda) * X[0]
    X.append(X1)
    z.append(z0)


    # t = 2 (split into two clusters)
    z1 = np.zeros(n_nodes, dtype=np.int)
    z1[int(n_nodes * 0.5):int(n_nodes * 0.75)] = 2
    z1[int(n_nodes * 0.75):] = 3
    z.append(z1)

    X1 = sigma[z1] * rng.randn(n_nodes, 2) + lmbda * mu[z1] + (1 - lmbda) * X[1]
    X.append(X1)

    # t = 3 resample
    z.append(z1)

    X1 = sigma[z1] * rng.randn(n_nodes, 2) + lmbda * mu[z1] + (1 - lmbda) * X[2]
    X.append(X1)

    # t = 4 (merge)
    z2 = np.zeros(n_nodes, dtype=np.int)
    z2[int(n_nodes * 0.75):] = 4
    z.append(z2)
    X1 = sigma[z2] * rng.randn(n_nodes, 2) + lmbda * mu[z2] + (1 - lmbda) * X[3]
    X.append(X1)

    X = np.stack(X, axis=0)
    z = np.stack(z, axis=0)
    Y, _ = network_from_dynamic_latent_space(X,
                                             intercept=1,
                                             coef=1, random_state=123)

    return Y, X, z


def synthetic_network(n_nodes=100, lmbda=0.9, intercept=1.0, random_state=123):
    rng = check_random_state(random_state)

    centers = np.array([[-2.0, 0.0],
                        [-1.0, 0.0],
                        [0.0, 3.0],
                        [2.0, 0.0],
                        [1.0, 0.0],
                        [0.0, -2.0]])
    #stds  = 1. / rng.gamma(shape=2, scale=100, size=6)
    stds  = 1. / rng.gamma(shape=1, scale=1/0.103, size=6).reshape(-1, 1)

    X, z = [], []

    # t = 0
    X1 = np.sqrt(stds[0]) * rng.randn(n_nodes).reshape(n_nodes // 2, 2) + centers[0]
    X2 = np.sqrt(stds[3]) * rng.randn(n_nodes).reshape(n_nodes // 2, 2) + centers[3]
    X.append(np.vstack((X1, X2)))

    z0 = np.zeros(n_nodes, dtype=np.int)
    z0[(n_nodes // 2):] = 3
    z.append(z0)

    # t = 1 (move around according to lmbda * mu + (1 - lmbda) * xit

    # re-sample z0 based on distances
    z1 = rng.binomial(1, p=np.array([0.2, 0.0, 0.0, 0.8])[z0])
    z1[z1 == 1] = 3
    #z1 = z0.copy()
    X1 = np.sqrt(stds[z1]) * rng.randn(n_nodes, 2) + lmbda * centers[z1] + (1 - lmbda) * X[0]
    X.append(X1)
    z.append(z1)

    z2 = rng.binomial(1, p=np.array([0.2, 0.0, 0.0, 0.8])[z1])
    z2[z2 == 1] = 3
    #z1 = z0.copy()
    X2 = np.sqrt(stds[z2]) * rng.randn(n_nodes, 2) + lmbda * centers[z2] + (1 - lmbda) * X[1]
    X.append(X2)
    z.append(z2)

    # t = 2 (split into two clusters)
    z3 = z2.copy()
    zero_mask = z3 == 0
    zero_mask[:int(n_nodes/2)] = 0
    z3[zero_mask] = 2
    one_mask = z3 == 0
    one_mask[int(n_nodes/8):] = 0
    z3[one_mask] = 2

    #z3[int(n_nodes * 0.75):] = 3
    z.append(z3)

    X3 = np.sqrt(stds[z3]) * rng.randn(n_nodes, 2) + lmbda * centers[z3] + (1 - lmbda) * X[2]
    X.append(X3)

    ## resample
    z4 = z3.copy()
    z.append(z4)

    X4 = np.sqrt(stds[z4]) * rng.randn(n_nodes, 2) + lmbda * centers[z4] + (1 - lmbda) * X[3]
    X.append(X4)

    X = np.stack(X, axis=0)
    z = np.stack(z, axis=0)
    Y, _ = network_from_dynamic_latent_space(X,
                                             intercept=intercept,
                                             coef=1, random_state=123)

    return Y, X, z, centers, stds


def make_splitting_network(n_nodes=100, lmbda=0.9, intercept=1.0,
                           random_state=None):
    rng = check_random_state(random_state)

    # cluster parameters
    lmbda
    mu = np.array([[-2.0, 0.0],
                   [ 2.0, 0.0],
                   [ 0.0, 3.0]])
    sigma  = 1. / rng.gamma(shape=1, scale=1/0.103, size=3).reshape(-1, 1)
    std = np.sqrt(sigma)

    # init_w
    init_w = 0.5

    X, z = [], []

    # t = 0
    z1 = rng.binomial(1, p=0.5, size=n_nodes)
    X1 = std[z1] * rng.randn(n_nodes, 2) + mu[z1]
    X.append(X1)
    z.append(z1)

    # t = 1 (resample)
    trans_w = 0.8
    z2 = np.zeros(n_nodes, dtype=np.int)

    zero_mask = z1 == 0
    z2[zero_mask] = rng.binomial(1, p=1 - trans_w, size=np.sum(zero_mask))

    one_mask = z1 == 1
    z2[one_mask] = rng.binomial(1, p=trans_w, size=np.sum(one_mask))

    X2 = std[z2] * rng.randn(n_nodes, 2) + lmbda * mu[z2] + (1 - lmbda) * X[0]
    X.append(X2)
    z.append(z2)

    # t = 2 (resample)
    trans_w = 0.8

    z3 = np.zeros(n_nodes, dtype=np.int)

    zero_mask = z2 == 0
    z3[zero_mask] = rng.binomial(1, p=1 - trans_w, size=np.sum(zero_mask))

    one_mask = z2 == 1
    z3[one_mask] = rng.binomial(1, p=trans_w, size=np.sum(one_mask))

    X3 = std[z3] * rng.randn(n_nodes, 2) + lmbda * mu[z3] + (1 - lmbda) * X[1]
    X.append(X3)
    z.append(z3)

    # t = 3 (split)
    z4 = np.zeros(n_nodes, dtype=np.int)

    zero_mask = z3 == 0
    z4[zero_mask] = rng.choice([0, 1, 2], p=[0.5, 0.25, 0.25],
                               size=np.sum(zero_mask))

    one_mask = z3 == 1
    z4[one_mask] = rng.choice([0, 1, 2], p=[0.25, 0.5, 0.25],
                               size=np.sum(one_mask))

    X4 = std[z4] * rng.randn(n_nodes, 2) + lmbda * mu[z4] + (1 - lmbda) * X[2]
    X.append(X4)
    z.append(z4)

    # t = 4 (resample)
    z5 = np.zeros(n_nodes, dtype=np.int)

    zero_mask = z4 == 0
    z5[zero_mask] = rng.choice([0, 1, 2], p=[0.8, 0.1, 0.1],
                               size=np.sum(zero_mask))

    one_mask = z4 == 1
    z5[one_mask] = rng.choice([0, 1, 2], p=[0.1, 0.8, 0.1],
                               size=np.sum(one_mask))

    two_mask = z4 == 2
    z5[two_mask] = rng.choice([0, 1, 2], p=[0.1, 0.1, 0.8],
                               size=np.sum(two_mask))

    X5 = std[z5] * rng.randn(n_nodes, 2) + lmbda * mu[z5] + (1 - lmbda) * X[3]
    X.append(X5)
    z.append(z5)

    X = np.stack(X, axis=0)
    z = np.stack(z, axis=0)
    Y, _ = network_from_dynamic_latent_space(X,
                                             intercept=intercept,
                                             coef=1, random_state=123)

    return Y, X, z, mu, sigma


def synthetic_directed_network(n_nodes=25, n_time_steps=5, n_features=2,
                               sigma_sq='auto', sigma_scale=5, X_scale=1.0,
                               lmbda=0.9, squared=False,
                               beta_in=1.0, beta_out=2.0,
                               random_state=42):
    rng = check_random_state(random_state)

    if sigma_sq == 'auto':
        sigma_sq = 1 / (sigma_scale * n_nodes) ** 2

    # sample latent positions
    X = []

    # initial positions (drawn from a 5-component mixture)
    n_components = 5
    means = (2 / n_nodes) * rng.randn(n_components, n_features)
    z = rng.choice(np.arange(n_components), size=n_nodes)

    X0 = np.sqrt(sigma_sq) * rng.randn(n_nodes, n_features) + means[z]
    X.append(X0)

    for t in range(1, n_time_steps):
        Xt = (lmbda * means[z] + (1 - lmbda) * X[t-1] +
                    np.sqrt(sigma_sq) * rng.randn(n_nodes, n_features))
        X.append(Xt)
    X = np.stack(X, axis=0)
    X *= X_scale

    # radi
    X_norm = np.linalg.norm(X[0], axis=1)

    alpha = n_nodes / (X_norm * np.max(X_norm))
    radii = rng.dirichlet(alpha)

    if squared:
        radii **= 2

    # sample network
    dist = calculate_distances(X, squared=squared)
    probas = directed_network_probas(dist, radii, beta_in, beta_out)
    Y = rng.binomial(1, probas).astype(np.float64)

    # braodcast z across time
    z = np.repeat(z.reshape(1, -1), n_time_steps, axis=0)

    return Y, X, z, radii, np.array([beta_in, beta_out])


def synthetic_undirected_network(n_nodes=100, n_time_steps=5, n_groups=6,
                                 intercept=1.0, lmbda=0.8, sticky_const=20.,
                                 sigma_shape=6, sigma_scale=20,
                                 random_state=42):
    rng = check_random_state(random_state)

    # group locations
    mus = np.array([[-3, 0],
                    [3, 0],
                    [-1.5, 0],
                    [1.5, 0],
                    [0, 2.0],
                    [0, -2.0]])

    if n_groups > 6:
        raise ValueError("Only a maximum of six groups allowed for now.")

    # group spread
    sigmas = np.sqrt(1. / rng.gamma(shape=sigma_shape, scale=sigma_scale,
                                    size=n_groups))

    # sample initial distribution
    w0 = rng.dirichlet(np.repeat(10, n_groups))  # E[p] = 1 / n_groups

    # set-up transition distribution
    with np.errstate(divide='ignore'):
        wt = 1. / pairwise_distances(mus)

    # only took necessary groups
    wt = wt[:n_groups][:, :n_groups]
    diag_indices = np.diag_indices_from(wt)
    wt[diag_indices] = 0
    wt[diag_indices] = sticky_const * np.max(wt, axis=1)
    wt /= wt.sum(axis=1).reshape(-1, 1)

    # run data generating process
    X, z = [], []

    # t = 0
    z0 = rng.choice(np.arange(n_groups), p=w0, size=n_nodes)
    X0 = np.zeros((n_nodes, 2), dtype=np.float64)
    for group_id in range(n_groups):
        group_count = np.sum(z0 == group_id)
        X0[z0 == group_id, :] = (sigmas[group_id] * rng.randn(group_count, 2) +
                                    mus[group_id])
    X.append(X0)
    z.append(z0)

    for t in range(1, n_time_steps):
        zt = np.zeros(n_nodes, dtype=np.int)
        for group_id in range(n_groups):
            group_mask = z[t - 1] == group_id
            zt[group_mask] = rng.choice(np.arange(n_groups), p=wt[group_id, :],
                                        size=np.sum(group_mask))

        Xt = np.zeros((n_nodes, 2), dtype=np.float64)
        for group_id in range(n_groups):
            group_mask = zt == group_id
            group_count = np.sum(group_mask)
            Xt[group_mask, :] = (
                sigmas[group_id] * rng.randn(group_count, 2) + (
                    lmbda * mus[group_id] + (1 - lmbda) * X[t-1][group_mask, :])
            )

        X.append(Xt)
        z.append(zt)

    X = np.stack(X, axis=0)
    z = np.vstack(z)

    Y, _ = network_from_dynamic_latent_space(X, intercept=intercept,
                                             random_state=rng)

    return Y, X, z, intercept


def merging_synthetic_network(n_nodes=120, n_time_steps=6, n_groups=6,
                              intercept=1.0, lmbda=0.8, sticky_const=20.,
                              sigma_shape=6, sigma_scale=20, is_directed=False,
                              random_state=42):
    rng = check_random_state(random_state)

    # group locations
    mus = np.array([[-3, 0],
                    [3, 0],
                    [-1.5, 0],
                    [1.5, 0],
                    [0, 2.0],
                    [0, -2.0]])

    # negative / positive groups merge
    new_mus = np.array([[-1.5, -2/3.],
                        [1.5, 2/3.]])

    if is_directed:
        mus /= 100.
        new_mus /= 100.

    n_groups_new = new_mus.shape[0]
    n_groups_total = n_groups + n_groups_new

    if n_groups > 6:
        raise ValueError("Only a maximum of six groups allowed for now.")

    # group spread
    if is_directed:
        sigma_scale = 1e5
        sigma_shape = 13
    sigmas = np.sqrt(1. / rng.gamma(shape=sigma_shape, scale=sigma_scale,
                                    size=n_groups_total))
    # sigmas post merge are twice os large on average
    #sigmas[n_groups:n_groups_total] = np.sqrt(
    #    1. / rng.gamma(shape=sigma_shape, scale=sigma_scale / 2.,
    #                   size=n_groups_new)
    #)

    # sample initial distribution
    w0 = rng.dirichlet(np.repeat(10, n_groups))  # E[p] = 1 / n_groups

    # set-up transition distribution
    with np.errstate(divide='ignore'):
        wt = 1. / pairwise_distances(mus)

    # only took necessary groups
    wt = wt[:n_groups][:, :n_groups]
    diag_indices = np.diag_indices_from(wt)
    wt[diag_indices] = 0
    wt[diag_indices] = sticky_const * np.max(wt, axis=1)
    wt /= wt.sum(axis=1).reshape(-1, 1)

    # run data generating process
    X, z = [], []

    # t = 0
    z0 = rng.choice(np.arange(n_groups), p=w0, size=n_nodes)
    X0 = np.zeros((n_nodes, 2), dtype=np.float64)
    for group_id in range(n_groups):
        group_size = np.sum(z0 == group_id)
        X0[z0 == group_id, :] = (sigmas[group_id] * rng.randn(group_size, 2) +
                                    mus[group_id])
    X.append(X0)
    z.append(z0)

    for t in range(1, ceil(n_time_steps / 2)):
        zt = np.zeros(n_nodes, dtype=np.int)
        for group_id in range(n_groups):
            group_mask = z[t - 1] == group_id
            zt[group_mask] = rng.choice(np.arange(n_groups), p=wt[group_id, :],
                                        size=np.sum(group_mask))

        Xt = np.zeros((n_nodes, 2), dtype=np.float64)
        for group_id in range(n_groups):
            group_mask = zt == group_id
            group_size = np.sum(group_mask)
            Xt[group_mask, :] = (
                sigmas[group_id] * rng.randn(group_size, 2) + (
                    lmbda * mus[group_id] + (1 - lmbda) * X[t-1][group_mask, :])
            )

        X.append(Xt)
        z.append(zt)

    # merge into two clusters (n_groups -> 2)
    wt_merge = 1. / pairwise_distances(mus, new_mus)
    wt_merge /= wt_merge.sum(axis=1).reshape(-1, 1)
    zt = np.zeros(n_nodes, dtype=np.int)
    for group_id in range(n_groups):
        group_mask = z[-1] == group_id
        group_size = np.sum(group_mask)
        zt[group_mask] = rng.choice(n_groups + np.arange(n_groups_new),
                                    p=wt_merge[group_id, :], size=group_size)

    Xt = np.zeros((n_nodes, 2), dtype=np.float64)
    for group_id in range(n_groups_new):
        group_mask = zt == group_id + n_groups
        group_size = np.sum(group_mask)
        Xt[group_mask, :] = (
            sigmas[group_id + n_groups_new] * rng.randn(group_size, 2) +
                lmbda * new_mus[group_id] +
                    (1 - lmbda) * X[t - 1][group_mask,:]
        )
    z.append(zt)
    X.append(Xt)

    # run the process forward in time
    with np.errstate(divide='ignore'):
        wt = 1. / pairwise_distances(new_mus)

    diag_indices = np.diag_indices_from(wt)
    wt[diag_indices] = 0
    wt[diag_indices] = 8 * np.max(wt, axis=1)

    wt = wt / wt.sum(axis=1).reshape(-1, 1)
    for t in range(ceil(n_time_steps / 2) + 1, n_time_steps):
        zt = np.zeros(n_nodes, dtype=np.int)
        for group_id in range(n_groups_new):
            group_mask = z[t-1] == group_id + n_groups
            group_size = np.sum(group_mask)
            zt[group_mask] = rng.choice(n_groups + np.arange(n_groups_new),
                                        p=wt[group_id, :], size=group_size)

        Xt = np.zeros((n_nodes, 2), dtype=np.float64)
        for group_id in range(n_groups_new):
            group_mask = zt == group_id + n_groups
            group_size = np.sum(zt == group_id + n_groups)
            Xt[group_mask, :] = (
                sigmas[group_id + n_groups_new] * rng.randn(group_size, 2) +
                    lmbda * new_mus[group_id] +
                        (1 - lmbda) * X[t-1][group_mask, :]
            )

        z.append(zt)
        X.append(Xt)

    X = np.stack(X, axis=0)
    z = np.vstack(z)

    # generate radii if necessary
    if is_directed:
        norms = 1. / np.linalg.norm(X[0], axis=1)
        norms /= np.max(norms)
        radii = rng.dirichlet(100 * norms)
        intercept = np.array([0.3, 0.7])
    else:
        radii = None

    Y, _ = network_from_dynamic_latent_space(
          X, intercept=intercept, radii=radii, random_state=rng)

    return Y, X, z, intercept, radii


def simple_splitting_synthetic_undirected_network(n_nodes=120, n_time_steps=6,
                                                  intercept=1.0, lmbda=0.8,
                                                  sticky_const=20.,
                                                  sigma_shape=6, sigma_scale=20,
                                                  random_state=42):
    rng = check_random_state(random_state)

    # group locations
    mus = np.array([[-3.0, 0.],
                    [0., 3.0]])
    n_groups = mus.shape[0]

    # group spread
    sigmas = np.sqrt(1. / rng.gamma(shape=sigma_shape, scale=sigma_scale,
                                    size=3))

    # run data generating process
    X, z = [], []

    # t = 0
    z0 = np.zeros(n_nodes, dtype=np.int)
    X0 = sigmas[0] * rng.randn(n_nodes, 2)
    X.append(X0)
    z.append(z0)

    for t in range(1, ceil(n_time_steps / 2)):
        zt = np.zeros(n_nodes, dtype=np.int)
        Xt = sigmas[0] * rng.randn(n_nodes, 2) + (1 - lmbda) * X[t-1]
        X.append(Xt)
        z.append(zt)

    # split into two clusters
    # merge into two clusters (1 -> 2)
    wt_split = rng.dirichlet(np.repeat(10, n_groups))
    zt = rng.choice(np.arange(1, n_groups + 1),
                    p=wt_split, size=n_nodes)

    Xt = np.zeros((n_nodes, 2), dtype=np.float64)
    for group_id in range(1, 3):
        group_mask = zt == group_id
        group_size = np.sum(group_mask)
        Xt[group_mask, :] = (
            sigmas[group_id] * rng.randn(group_size, 2) +
                lmbda * mus[group_id - 1] +
                    (1 - lmbda) * X[t - 1][group_mask,:]
        )
    z.append(zt)
    X.append(Xt)

    # run the process forward in time
    with np.errstate(divide='ignore'):
        wt = 1. / pairwise_distances(mus)

    diag_indices = np.diag_indices_from(wt)
    wt[diag_indices] = 0
    wt[diag_indices] = 8 * np.max(wt, axis=1)
    wt = wt / wt.sum(axis=1).reshape(-1, 1)
    for t in range(ceil(n_time_steps / 2) + 1, n_time_steps):
        zt = np.zeros(n_nodes, dtype=np.int)
        for group_id in range(1, 3):
            group_mask = z[t-1] == group_id
            group_size = np.sum(group_mask)
            zt[group_mask] = rng.choice(np.arange(1, 3),
                                        p=wt[(group_id-1), :], size=group_size)

        Xt = np.zeros((n_nodes, 2), dtype=np.float64)
        for group_id in range(1, 3):
            group_mask = zt == group_id
            group_size = np.sum(zt == group_id)
            Xt[group_mask, :] = (
                sigmas[group_id] * rng.randn(group_size, 2) +
                    lmbda * mus[group_id - 1] +
                        (1 - lmbda) * X[t-1][group_mask, :]
            )

        z.append(zt)
        X.append(Xt)

    X = np.stack(X, axis=0)
    z = np.vstack(z)

    # generate radii if necessary
    if is_directed:
        norms = 1. / np.linalg.norm(X[0])
        norms /= np.max(norms)
        radii = rng.dirichlet(100 * norms)
    else:
        radii = None

    Y, _ = network_from_dynamic_latent_space(
        X, intercept=intercept, radii=radii, random_state=rng)

    return Y, X, z, intercept, radii


def splitting_synthetic_network(n_nodes=120, n_time_steps=6,
                                intercept=1.0, lmbda=0.8,
                                sticky_const=20.,
                                sigma_shape=5, sigma_scale=0.25,
                                is_directed=False,
                                random_state=42):
    rng = check_random_state(random_state)

    # group locations
    mus = np.array([[-3.0, 0.],
                    [0, 3.0],
                    [1.5, -1.5]])
    n_groups = mus.shape[0]

    if is_directed:
        mus /= 100.

    # group spread
    sigmas = np.sqrt(1. / rng.gamma(shape=sigma_shape, scale=sigma_scale,
                                    size=3))
    # run data generating process
    X, z = [], []

    # t = 0
    z0 = rng.choice(np.arange(2), size=n_nodes)
    X0 = np.zeros((n_nodes, 2), dtype=np.float64)
    for group_id in range(2):
        group_size = np.sum(z0 == group_id)
        X0[z0 == group_id, :] = (sigmas[group_id] * rng.randn(group_size, 2) +
                                    mus[group_id])
    X.append(X0)
    z.append(z0)

    # set-up transition distribution
    with np.errstate(divide='ignore'):
        wt = 1. / pairwise_distances(mus)

    # only took necessary groups
    wt = wt[:2][:, :2]
    diag_indices = np.diag_indices_from(wt)
    wt[diag_indices] = 0
    wt[diag_indices] = sticky_const * np.max(wt, axis=1)
    wt /= wt.sum(axis=1).reshape(-1, 1)


    for t in range(1, ceil(n_time_steps / 2)):
        #zt = np.zeros(n_nodes, dtype=np.int)
        zt = np.zeros(n_nodes, dtype=np.int)
        for group_id in range(2):
            group_mask = z[t-1] == group_id
            group_size = np.sum(group_mask)
            zt[group_mask] = rng.choice(np.arange(2),
                                        p=wt[group_id, :], size=group_size)
        Xt = np.zeros((n_nodes, 2), dtype=np.float64)
        for group_id in range(2):
            group_mask = zt == group_id
            group_size = np.sum(group_mask)
            Xt[group_mask, :] = (
                sigmas[group_id] * rng.randn(group_size, 2) +
                    lmbda * mus[group_id] +
                        (1 - lmbda) * X[t - 1][group_mask,:]
            )
        z.append(zt)
        X.append(Xt)

    # split into two clusters
    # merge into two clusters (1 -> 2)
    wt_split = rng.dirichlet(np.repeat(10, 3))
    zt = rng.choice(np.arange(3),
                    p=wt_split, size=n_nodes)

    Xt = np.zeros((n_nodes, 2), dtype=np.float64)
    for group_id in range(3):
        group_mask = zt == group_id
        group_size = np.sum(group_mask)
        Xt[group_mask, :] = (
            sigmas[group_id] * rng.randn(group_size, 2) +
                lmbda * mus[group_id] +
                    (1 - lmbda) * X[t - 1][group_mask,:]
        )
    z.append(zt)
    X.append(Xt)

    # run the process forward in time
    with np.errstate(divide='ignore'):
        wt = 1. / pairwise_distances(mus)

    diag_indices = np.diag_indices_from(wt)
    wt[diag_indices] = 0
    wt[diag_indices] = sticky_const * np.max(wt, axis=1)
    wt = wt / wt.sum(axis=1).reshape(-1, 1)
    for t in range(ceil(n_time_steps / 2) + 1, n_time_steps):
        zt = np.zeros(n_nodes, dtype=np.int)
        for group_id in range(3):
            group_mask = z[t-1] == group_id
            group_size = np.sum(group_mask)
            zt[group_mask] = rng.choice(np.arange(3),
                                        p=wt[group_id, :], size=group_size)

        Xt = np.zeros((n_nodes, 2), dtype=np.float64)
        for group_id in range(3):
            group_mask = zt == group_id
            group_size = np.sum(zt == group_id)
            Xt[group_mask, :] = (
                sigmas[group_id] * rng.randn(group_size, 2) +
                    lmbda * mus[group_id] +
                        (1 - lmbda) * X[t-1][group_mask, :]
            )

        z.append(zt)
        X.append(Xt)

    X = np.stack(X, axis=0)
    z = np.vstack(z)

    # generate radii if necessary
    if is_directed:
        norms = 1. / np.linalg.norm(X[0])
        norms /= np.max(norms)
        radii = rng.dirichlet(100 * norms)
    else:
        radii = None

    Y, _ = network_from_dynamic_latent_space(
        X, intercept=intercept, radii=radii, random_state=rng)

    return Y, X, z, intercept, radii


def synthetic_dynamic_network(n_nodes=120, n_time_steps=9,
                              intercept=1.0, lmbda=0.8, sticky_const=20.,
                              sigma_shape=6, sigma_scale=20, is_directed=False,
                              random_state=42):
    """Split from 2 -> 6 and then merge from 6 -> 4"""
    rng = check_random_state(random_state)

    time_chunks = ceil(n_time_steps / 3)

    # group locations
    if is_directed:
        all_mus = np.array([[-1.5, -2/3.],
                            [1.5, 2/3.],
                            [-3, 0],
                            [3, 0],
                            [-1.0, 0.],
                            [1.0, 0.],
                            [0, 2.0],
                            [0, -2.0]]) / 100.
    else:
        #all_mus = np.array([[-2, 0.],
        #                    [2, 0.],
        #                    [-6, 0.],
        #                    [6, 0.],
        #                    [-2, 0],
        #                    [2, 0],
        #                    [0, 4.0],
        #                    [0, -4.0]])
        #all_mus = np.array([[-1.5, 0.],
        #                    [1.5, 0.],
        #                    [-5, 0],
        #                    [5, 0],
        #                    [-1.5, 0.],
        #                    [1.5, 0.],
        #                    [0, 5.0],
        #                    [0, -5.0]])
        all_mus = np.array([[-1.5, 0.],
                            [1.5, 0.],
                            [-3, 0],
                            [3, 0],
                            [-1.5, 0.],
                            [1.5, 0.],
                            [0, 2.0],
                            [0, -2.0]])

    n_groups_total = all_mus.shape[0]

    # group spread
    if is_directed:
        sigma_scale = 1e5
        sigma_shape = 13
    sigmas = np.sqrt(1. / rng.gamma(shape=sigma_shape, scale=sigma_scale,
                                    size=n_groups_total))


    # initial groups
    mus = all_mus[:2].copy()
    n_groups = mus.shape[0]

    # sample initial distribution
    w0 = rng.dirichlet(np.repeat(10, n_groups))  # E[p] = 1 / n_groups

    # set-up transition distribution
    with np.errstate(divide='ignore'):
        wt = 1. / pairwise_distances(mus)

    # calculate self-transition probabilities
    diag_indices = np.diag_indices_from(wt)
    wt[diag_indices] = 0
    wt[diag_indices] = sticky_const * np.max(wt, axis=1)
    wt /= wt.sum(axis=1).reshape(-1, 1)

    # run data generating process
    X, z = [], []

    # t = 0
    z0 = rng.choice(np.arange(n_groups), p=w0, size=n_nodes)
    X0 = np.zeros((n_nodes, 2), dtype=np.float64)
    for group_id in range(n_groups):
        group_size = np.sum(z0 == group_id)
        X0[z0 == group_id, :] = (sigmas[group_id] * rng.randn(group_size, 2) +
                                    mus[group_id])
    X.append(X0)
    z.append(z0)

    for t in range(1, time_chunks):
        zt = np.zeros(n_nodes, dtype=np.int)
        for group_id in range(n_groups):
            group_mask = z[t - 1] == group_id
            zt[group_mask] = rng.choice(np.arange(n_groups), p=wt[group_id, :],
                                        size=np.sum(group_mask))

        Xt = np.zeros((n_nodes, 2), dtype=np.float64)
        for group_id in range(n_groups):
            group_mask = zt == group_id
            group_size = np.sum(group_mask)
            Xt[group_mask, :] = (
                sigmas[group_id] * rng.randn(group_size, 2) + (
                    lmbda * mus[group_id] + (1 - lmbda) * X[t-1][group_mask, :])
            )

        X.append(Xt)
        z.append(zt)

    # split into 6 clusters (2 -> 6)
    old_mus = mus.copy()
    mus = all_mus[2:].copy()
    n_groups_new = mus.shape[0]
    with np.errstate(divide='ignore'):
        wt_merge = 1. / pairwise_distances(old_mus, mus)

    # self-transitions
    infinite_mask = ~np.isfinite(wt_merge)
    wt_merge[infinite_mask] = 0
    wt_merge[infinite_mask] = np.max(wt_merge, axis=1)
    wt_merge /= wt_merge.sum(axis=1).reshape(-1, 1)

    zt = np.zeros(n_nodes, dtype=np.int)
    for group_id in range(n_groups):
        group_mask = z[-1] == group_id
        group_size = np.sum(group_mask)
        zt[group_mask] = rng.choice(n_groups + np.arange(n_groups_new),
                                    p=wt_merge[group_id, :], size=group_size)

    Xt = np.zeros((n_nodes, 2), dtype=np.float64)
    for group_id in range(n_groups_new):
        group_mask = zt == group_id + n_groups
        group_size = np.sum(group_mask)
        Xt[group_mask, :] = (
            sigmas[group_id + n_groups] * rng.randn(group_size, 2) +
                lmbda * mus[group_id] +
                    (1 - lmbda) * X[t - 1][group_mask,:]
        )
    z.append(zt)
    X.append(Xt)

    # run the process forward in time
    with np.errstate(divide='ignore'):
        wt = 1. / pairwise_distances(mus)

    diag_indices = np.diag_indices_from(wt)
    wt[diag_indices] = 0
    wt[diag_indices] = sticky_const * np.max(wt, axis=1)
    wt = wt / wt.sum(axis=1).reshape(-1, 1)

    for t in range(time_chunks + 1, 2 * time_chunks):
        zt = np.zeros(n_nodes, dtype=np.int)
        for group_id in range(n_groups_new):
            group_mask = z[t-1] == group_id + n_groups
            group_size = np.sum(group_mask)
            zt[group_mask] = rng.choice(n_groups + np.arange(n_groups_new),
                                        p=wt[group_id, :], size=group_size)

        Xt = np.zeros((n_nodes, 2), dtype=np.float64)
        for group_id in range(n_groups_new):
            group_mask = zt == group_id + n_groups
            group_size = np.sum(zt == group_id + n_groups)
            Xt[group_mask, :] = (
                sigmas[group_id + n_groups] * rng.randn(group_size, 2) +
                    lmbda * mus[group_id] +
                        (1 - lmbda) * X[t-1][group_mask, :]
            )
        z.append(zt)
        X.append(Xt)

    # merge groups 6 -> 4
    old_mus = mus.copy()
    new_groups = [2, 3, 6, 7]
    mus = all_mus[new_groups].copy()

    with np.errstate(divide='ignore'):
        wt_merge = 1. / pairwise_distances(old_mus, mus)
    infinite_mask = ~np.isfinite(wt_merge)
    wt_merge[infinite_mask] = 0
    wt_merge[infinite_mask] = sticky_const * np.max(wt_merge, axis=1)[[0, 1, 4, 5]]
    wt_merge /= wt_merge.sum(axis=1).reshape(-1, 1)

    zt = np.zeros(n_nodes, dtype=np.int)
    for group_id in range(n_groups_new):
        group_mask = z[-1] == group_id + n_groups
        group_size = np.sum(group_mask)
        zt[group_mask] = rng.choice(new_groups,
                                    p=wt_merge[group_id, :], size=group_size)

    Xt = np.zeros((n_nodes, 2), dtype=np.float64)
    for group_id in [0, 1, 4, 5]:
        group_mask = zt == group_id + n_groups
        group_size = np.sum(group_mask)
        Xt[group_mask, :] = (
            sigmas[group_id + n_groups] * rng.randn(group_size, 2) +
                lmbda * old_mus[group_id] +
                    (1 - lmbda) * X[t - 1][group_mask,:]
        )
    z.append(zt)
    X.append(Xt)

    # run the process forward in time
    with np.errstate(divide='ignore'):
        wt = 1. / pairwise_distances(mus)

    diag_indices = np.diag_indices_from(wt)
    wt[diag_indices] = 0
    wt[diag_indices] = sticky_const * np.max(wt, axis=1)
    wt = wt / wt.sum(axis=1).reshape(-1, 1)

    for t in range(2 * time_chunks + 1, n_time_steps):
        zt = np.zeros(n_nodes, dtype=np.int)
        for idx, group_id in enumerate(new_groups):
            group_mask = z[t-1] == group_id
            group_size = np.sum(group_mask)
            zt[group_mask] = rng.choice(new_groups,
                                        p=wt[idx, :], size=group_size)

        Xt = np.zeros((n_nodes, 2), dtype=np.float64)
        for group_id in new_groups:
            group_mask = zt == group_id
            group_size = np.sum(zt == group_id)
            Xt[group_mask, :] = (
                sigmas[group_id] * rng.randn(group_size, 2) +
                    lmbda * all_mus[group_id] +
                        (1 - lmbda) * X[t-1][group_mask, :]
            )

        z.append(zt)
        X.append(Xt)

    X = np.stack(X, axis=0)
    z = np.vstack(z)

    # generate radii if necessary
    if is_directed:
        norms = 1. / np.linalg.norm(X[0], axis=1)
        norms /= np.max(norms)
        radii = rng.dirichlet(100 * norms)
        intercept = np.array([0.3, 0.7])
    else:
        radii = None

    Y, probas = network_from_dynamic_latent_space(
        X, intercept=intercept, radii=radii, random_state=rng)

    return Y, X, z, intercept, radii, probas

