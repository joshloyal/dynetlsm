import numpy as np
import scipy.linalg as linalg
import scipy.stats as stats

from scipy.stats import truncnorm
from scipy.special import gammaln
from sklearn.utils import check_random_state


SMALL_EPS = np.finfo('float64').tiny


def sample_categorical(probas, rng):
    """
    Sample from a Categorical(probas) distribution.
    """
    cdf = probas.cumsum()
    u = rng.rand(probas.shape[0])
    return (u < cdf).argmax()


def spherical_normal_log_pdf(x, mean, var):
    """Logarithm of the pdf of a spherical multivariate gaussian
    distribution."""
    n_features = mean.shape[0]
    sum_sq = np.sum((x - mean) ** 2)
    sum_sq *= 0.5 * (1. / var)
    return -0.5 * n_features * np.log(2 * np.pi * var) - sum_sq


def spherical_normal_pdf(x, mean, var):
    """Probability Density Function for a spherical multivariate
    gaussian distribution. Note that this is 10x faster than
    the more general scipy.stats.multivariate_normal.pdf
    """
    n_features = mean.shape[0]
    sum_sq = np.sum((x - mean) ** 2)
    sum_sq *= 0.5 * (1. / var)
    return (1 / (2 * np.pi * var) ** (n_features / 2)) * np.exp(-sum_sq)


def multivariate_t_log_pdf(x, df, mu0, S):
    """Logarithm of the pdf of a multivariate t distribution."""
    x = np.atleast_1d(x)
    mu0 = np.atleast_1d(mu0)

    p = x.shape[0]
    if not isinstance(S, np.ndarray):
        rss = np.sum((x - mu0) ** 2) / S
        log_var = p * np.log(np.sqrt(S))
    else:
        L = linalg.cholesky(S)
        LinvX = linalg.solve_triangular(L, x - mu0, trans=1)
        rss = np.sum(LinvX ** 2, axis=0)
        log_var = np.sum(np.log(np.diag(L)))
    logdt = (gammaln((p + df) / 2.) - (
        gammaln(df / 2.) + log_var +
        (p / 2.) * np.log(df * np.pi)) -
            0.5 * (df + p) * np.log1p(rss / df))

    return logdt


def multivariate_t_pdf(x, df, mu0, S):
    """
    Probability Density Function of a multivariate t-distribution with
    df degrees of freedom, location parameter mu0, and scale matrix S.
    """
    return np.exp(multivariate_t_log_pdf(x, df, mu0, S))


def truncated_normal(mean, var, lower=0, upper=1, size=1, random_state=None):
    std = np.sqrt(var)
    a = (lower - mean) / std
    b = (upper - mean) / std
    return truncnorm.rvs(a, b, size=size, loc=mean, scale=std,
                         random_state=random_state)


def truncated_normal_logpdf(x, mean, var, lower=0, upper=1):
    std = np.sqrt(var)
    a = (lower - mean) / std
    b = (upper - mean) / std
    return truncnorm.logpdf(x, a, b, loc=mean, scale=std)


def sample_dirichlet(alphas, random_state=None):
    """The numpy dirichlet sampler is numerically unstable and produces samples
    with zero entries. Clip these values before using the sample.
    """
    rng = check_random_state(random_state)
    if np.any(alphas <= 0.):
        alphas = np.clip(alphas, a_min=SMALL_EPS, a_max=None)
    return rng.dirichlet(alphas)


def dirichlet_logpdf(x, alphas):
    if np.any(alphas <= 0.):
        alphas = np.clip(alphas, a_min=SMALL_EPS, a_max=None)
    if np.any(x <= 0):
        x = np.clip(x, a_min=SMALL_EPS, a_max=None)
    return stats.dirichlet.logpdf(x, alphas)
