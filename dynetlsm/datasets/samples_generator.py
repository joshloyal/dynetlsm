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


__all__ = ['network_from_dynamic_latent_space',
           'synthetic_directed_network',
           'synthetic_undirected_network',
           'merging_synthetic_network',
           'simple_splitting_synthetic_undirected_network',
           'synthetic_dynamic_network']


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

    # radii
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
