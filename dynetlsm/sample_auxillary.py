import numpy as np

from sklearn.utils import check_random_state


def sample_tables(n, beta, alpha_init, alpha, kappa, random_state=None):
    rng = check_random_state(random_state)

    n_time_steps, n_components, _ = n.shape
    m = np.zeros((n_time_steps, n_components, n_components),
                 dtype=np.int)

    # t = 0 does not include a spike
    probas = alpha_init * beta
    for k in range(n_components):
        x = rng.binomial(1, probas[k] / (probas[k] + np.arange(n[0, 0, k])))
        m[0, 0, k] = np.sum(x)

    # include spike for remaining time steps
    probas = alpha * beta + kappa * np.eye(n_components)
    for t in range(1, n_time_steps):
        for j in range(n_components):
            for k in range(n_components):
                x = rng.binomial(
                    1, probas[j, k] / (probas[j, k] + np.arange(n[t, j, k])))
                m[t, j, k] = np.sum(x)

    return m


def sample_mbar(m, beta, kappa=1.0, alpha=1.0, random_state=None):
    rng = check_random_state(random_state)
    n_time_steps, n_components, _ = m.shape

    # sample override variables for t = 1 ... T (do not include t = 0)
    w = np.zeros((n_time_steps - 1, n_components), dtype=np.float64)
    rho = kappa / (alpha + kappa)
    for t in range(n_time_steps - 1):
        for j in range(n_components):
            w[t, j] = rng.binomial(m[t + 1, j, j],
                                   rho / (rho + beta[j] * (1 - rho)))

    # mbar is determined by m and w
    m_bar = np.zeros((n_time_steps - 1, n_components, n_components),
                     dtype=np.float64)
    for t in range(n_time_steps - 1):
        m_bar[t] = m[t + 1] - np.diag(w[t])

    # NOTE: we have to add on the initial distribution transitions
    return np.sum(m_bar, axis=(0, 1)) + m[0, 0], w
