import numpy as np
import scipy.stats as stats

from math import floor, ceil

from statsmodels.regression.linear_model import yule_walker
from arviz.stats.diagnostics import geweke


def mean_detrend(x):
    return x - np.mean(x)


def xcorr(x, y, normed=True, detrend=mean_detrend, maxlags=10):
    Nx = len(x)
    if Nx != len(y):
        raise ValueError('x and y must be equal length')

    x = detrend(np.asarray(x))
    y = detrend(np.asarray(y))

    correls = np.correlate(x, y, mode='full')

    if normed:
        correls /= np.sqrt(np.dot(x, x) * np.dot(y, y))

    if maxlags is None:
        maxlags = Nx - 1

    if maxlags >= Nx or maxlags < 1:
        raise ValueError('maxlags must be None or strictly '
                         'postive < %d' % Nx)

    lags = np.arange(-maxlags, maxlags + 1)
    correls = correls[Nx - 1 - maxlags:Nx + maxlags]

    return lags, correls


def effective_n(x, lags=None, corr=None, maxlags=100):
    """Effective sample size."""
    if lags is None or corr is None:
        lags, corr = xcorr(x, x, maxlags=maxlags, normed=True)

    n_samples = x.shape[0]
    return n_samples / (1 + 2 * np.sum(corr[lags >= 1]))


def aic_ar(sigma, n, p):
    """AIC for an AR(p) model with n samples.
    Note: Assumes the series is de-meaned.
    """
    return 2 * n * np.log(sigma) + 2 * (p + 1)


def spec0_ar(sigma, coefs):
    return (sigma ** 2) / ((1 - np.sum(coefs)) ** 2)


def spectrum0_ar(x, max_order='auto'):
    """Calculates f(0) of the spectrum of x using an AR fit."""
    n_samples = x.shape[0]

    if np.allclose(np.var(x), 0.0):
        return 0., 0.

    if max_order == 'auto':
        max_order = floor(10 * np.log10(n_samples))

    # calculate f(0) and AIC for each AR(p) model
    results = np.zeros((max_order, 3))
    for p in range(1, max_order + 1):
        coefs, sigma = yule_walker(x, order=p, demean=True, method='unbiased')
        results[p-1] = [p, spec0_ar(sigma, coefs), aic_ar(sigma, n_samples, p)]

    # return result for model minimizing the AIC
    min_id = np.argmin(results[:, -1])
    order, var0 = results[min_id, :2]

    return var0 / n_samples, order


def geweke_corrected(x, first=0.1, last=0.5):
    """Calculate the z-score using Geweke's correction for autocorrelations."""
    n_samples = x.shape[0]

    # extract start and end chains
    x1 = x[:ceil(first * n_samples)]
    x2 = x[n_samples - floor(last * n_samples):]

    # calculate means
    x1_mean = np.mean(x1)
    x2_mean = np.mean(x2)

    # calculate variances
    x1_var, _ = spectrum0_ar(x1)
    x2_var, _ = spectrum0_ar(x2)

    # z score
    return (x1_mean - x2_mean) / np.sqrt(x1_var + x2_var)


def geweke_diag(x, first=0.1, last=0.5, n_burn=None, corrected=True):
    """Performs Geweke's diagnostic on a chain x.
    Note: ArviZ and PyMC3 do no correct for autocorrelation and use a naive
    z-score!
    """
    if n_burn is not None:
        x = x[n_burn:]

    if corrected:
        z_score = geweke_corrected(x, first=first, last=last)
    else:
        z_score = geweke(x, intervals=1, first=first, last=last)[0, 1]

    # calculate two-sided p-value
    p_val = 2 * (1 - stats.norm.cdf(np.abs(z_score)))

    return z_score, p_val
