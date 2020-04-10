import numpy as np

from sklearn.utils import check_random_state
from .gaussian_likelihood_fast import compute_gaussian_likelihood
from .distributions import spherical_normal_log_pdf


def log_normalize(probas):
    max_proba = np.max(probas)
    probas -= max_proba
    np.exp(probas, probas)
    probas /= np.sum(probas)
    return probas


def sample_categorical(probas, rng):
    cdf = np.cumsum(probas)
    u = rng.uniform(0, cdf[-1])
    return np.sum(u > cdf)


def sample_labels_gibbs(X, mu, sigma, lmbda, w, random_state=None):
    random_state = check_random_state(random_state)

    n_time_steps, n_nodes, _ = X.shape
    n_components = sigma.shape[0]

    # initialize cluster statistics
    # NOTE: n[0, 0, k] stores transitions for initial distribution
    n = np.zeros((n_time_steps, n_components, n_components))
    resp = np.zeros((n_time_steps, n_nodes, n_components), dtype=np.int)
    nk = np.zeros((n_time_steps, n_components), dtype=np.int)

    # initialize labels
    z = np.zeros((n_time_steps, n_nodes), dtype=np.int)

    # store sample probabilities
    probas = np.zeros(n_components, dtype=np.float64)

    # sample labels for each node
    for t in range(n_time_steps):
        for i in range(n_nodes):
            # FIXME: using 1e-5 hack to avoid log of zero
            if t == 0:
                for k in range(n_components):
                    probas[k] = (np.log(w[0, 0, k] + 1e-5) +
                                 spherical_normal_log_pdf(X[t, i],
                                                          mu[k],
                                                          sigma[k]))
            else:
                for k in range(n_components):
                    probas[k] = (np.log(w[t, z[t-1, i], k] + 1e-5) +
                                 spherical_normal_log_pdf(
                                    X[t, i],
                                    lmbda * mu[k] - (1 - lmbda) * X[t-1, i],
                                    sigma[k]))

            # sample zt
            probas = log_normalize(probas)
            z[t, i] = sample_categorical(probas, random_state)

            # update statistics
            if t == 0:
                n[0, 0, z[t, i]] += 1
            else:
                n[t, z[t-1, i], z[t, i]] += 1
            resp[t, i, z[t, i]] = 1
            nk[t, z[t, i]] += 1

    return z, n, nk, resp


def sample_labels_block(X, mu, sigma, lmbda, w, random_state=None):
    random_state = check_random_state(random_state)

    n_time_steps, n_nodes, _ = X.shape
    n_components = sigma.shape[0]

    # initialize message passing variables
    bwds_msg = np.ones((n_time_steps, n_components),
                       dtype=np.float64)
    partial_marg = np.zeros((n_time_steps, n_components),
                            dtype=np.float64)

    # initialize cluster statistics
    # NOTE: n[0, 0, k] stores transitions for initial distribution
    n = np.zeros((n_time_steps, n_components, n_components))
    resp = np.zeros((n_time_steps, n_nodes, n_components), dtype=np.int)
    nk = np.zeros((n_time_steps, n_components), dtype=np.int)

    # initialize labels
    z = np.zeros((n_time_steps, n_nodes), dtype=np.int)

    # sample labels for each node
    for i in range(n_nodes):
        # calculate likelihood of X_t^i under all groups
        # n_time_steps x n_components
        likelihood = compute_gaussian_likelihood(X[:, i], mu, sigma,
                                                 lmbda, normalize=False)

        # calculate backwards messages and partial likelihoods
        # (phi_k * m_k)
        for t in range(n_time_steps - 1, 0, -1):
            partial_marg[t] = likelihood[t] * bwds_msg[t]
            bwds_msg[t-1] = np.dot(w[t], partial_marg[t].reshape(-1, 1)).ravel()

            # helps with underflow (could also divide by maximum)
            bwds_msg[t-1] /= np.sum(bwds_msg[t-1])
        partial_marg[0] = likelihood[0] * bwds_msg[0]

        # sample labels forward in time
        for t in range(n_time_steps):
            if t == 0:
                probas = w[0, 0] * partial_marg[0]
            else:
                probas = w[t, z[t-1, i]] * partial_marg[t]

            # sample zt
            z[t, i] = sample_categorical(probas, random_state)

            # update statistics
            if t == 0:
                n[0, 0, z[t, i]] += 1
            else:
                n[t, z[t-1, i], z[t, i]] += 1
            resp[t, i, z[t, i]] = 1
            nk[t, z[t, i]] += 1

    return z, n, nk, resp
