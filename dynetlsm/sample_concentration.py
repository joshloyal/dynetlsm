import numpy as np

from sklearn.utils import check_random_state


def sample_concentration_param(alpha, n_clusters, n_samples, prior_shape=1.0,
                               prior_scale=1.0, random_state=None):
    """Sample concentration parameters as in Escobar and West (1995)"""
    rng = check_random_state(random_state)

    # auxillary variable sampler
    eta = rng.beta(alpha + 1, n_samples)

    m_shape = prior_shape + n_clusters - 1
    m_scale = prior_scale - np.log(eta)

    log_odds = (m_shape / m_scale) * (1 / n_samples)
    mix_indicator = rng.binomial(1, log_odds / (1 + log_odds))
    m_shape = m_shape + 1 if mix_indicator else m_shape

    return rng.gamma(shape=m_shape, scale=1. / m_scale)
