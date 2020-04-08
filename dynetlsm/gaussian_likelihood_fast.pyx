# encoding: utf-8
# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: initializedcheck=False
from libc.math cimport log, exp, sqrt, M_PI

import numpy as np
cimport numpy as np


ctypedef np.npy_float64 DOUBLE


cdef DOUBLE spherical_normal_log_pdf(DOUBLE[:] x,
                                     DOUBLE[:] mean,
                                     double var) nogil:
    cdef int k = 0
    cdef int n_features = x.shape[0]
    cdef DOUBLE sum_sq = 0.0

    for k in range(n_features):
        sum_sq += (x[k] - mean[k]) ** 2
    sum_sq *= 0.5 * (1. / var)
    return -0.5 * n_features * log(2 * M_PI * var) - sum_sq


def compute_gaussian_likelihood(DOUBLE[:, :] X,
                                DOUBLE[:, :] mu,
                                DOUBLE[:] sigma,
                                double lmbda,
                                bint normalize=True):
    cdef int t, k, j = 0
    cdef int n_time_steps = X.shape[0]
    cdef int n_features = X.shape[1]
    cdef int n_components = sigma.shape[0]
    cdef DOUBLE[:]  muk = np.zeros(n_features, dtype=np.float64)
    cdef DOUBLE[:, :] loglik = np.zeros((n_time_steps, n_components), dtype=np.float64)

    for t in range(n_time_steps):
        for k in range(n_components):
            if t == 0:
                loglik[t, k] = spherical_normal_log_pdf(X[t], mu[k], sigma[k])
            else:
                for j in range(n_features):
                    muk[j] = lmbda * mu[k, j] + (1 - lmbda) * X[t-1, j]
                loglik[t, k] = spherical_normal_log_pdf(X[t], muk, sigma[k])

    if normalize:
        loglik -= np.max(loglik, axis=1).reshape(-1, 1)

    return np.exp(loglik)
