import numpy as np

from functools import lru_cache
from scipy.special import expit
from sklearn.utils import check_random_state

from .samples_generator import network_from_dynamic_latent_space


__all__ = ['make_lookup_table', 'detection_limit_simulation']


@lru_cache()
def make_lookup_table(
        n_samples=10000, low=0.1, high=2.5, n_bins=100, random_state=42):
    rng = check_random_state(random_state)

    sigma = 0.5
    intercept = 1.0
    ratio = np.zeros((n_bins, 4))
    mu = np.linspace(low, high, n_bins)
    for b, m in enumerate(mu):
        mus = m * np.array([[1, 0],
                             [-1, 0]])
        X = np.sqrt(sigma) * rng.randn(n_samples, 8)
        p_in, p_out = 0, 0
        for i in range(n_samples):
            x = X[i, :2] + mus[0]
            y = X[i, 2:4] + mus[0]
            x0 = X[i, 4:6] + mus[0]
            x1 = X[i, 6:] + mus[1]
            p_in += expit(intercept - np.sqrt(((x - x0) ** 2).sum()))
            p_out += expit(intercept - np.sqrt(((y - x1) ** 2).sum()))

        ratio[b] = np.array([
            p_out / p_in, m, p_in / n_samples, p_out / n_samples])

    return ratio


def detection_limit_simulation(
        n_nodes=120, n_time_steps=4, trans_proba=0.2, lmbda=0.8, r=0.5,
        random_state=42):
    rng = check_random_state(random_state)

    ratio = make_lookup_table()
    idx = np.argmin(np.abs(r - ratio[:, 0]))
    mu = ratio[idx, 1]
    sigma = 0.5
    intercept = 1.0
    mus = mu * np.array([[1, 0],
                         [-1, 0]])
    X, z = [], []
    z0 = rng.choice([0, 1], p=[0.5, 0.5], size=n_nodes)
    X0 = sigma * rng.randn(n_nodes, 2) + mus[z0]
    X.append(X0)
    z.append(z0)

    wt = np.array([[1 - trans_proba, trans_proba],
                   [trans_proba, 1 - trans_proba]])
    for t in range(1, n_time_steps):
        zt = np.zeros(n_nodes, dtype=np.int)
        for group_id in range(2):
            group_mask = z[t - 1] == group_id
            zt[group_mask] = rng.choice(np.arange(2), p=wt[group_id, :],
                                        size=np.sum(group_mask))

        Xt = np.zeros((n_nodes, 2), dtype=np.float64)
        for group_id in range(2):
            group_mask = zt == group_id
            group_count = np.sum(group_mask)
            Xt[group_mask, :] = (
                sigma * rng.randn(group_count, 2) + (
                    lmbda * mus[group_id] + (1 - lmbda) * X[t-1][group_mask, :])
            )

        X.append(Xt)
        z.append(zt)

    X = np.stack(X, axis=0)
    z = np.vstack(z)

    Y, probas = network_from_dynamic_latent_space(
        X, intercept=intercept, random_state=rng)

    return Y, X, z, probas, ratio[idx, 0], mus
