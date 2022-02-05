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
ctypedef np.npy_int64 INT

cdef inline double expit(double z):
    return 1. / (1. + exp(-z))



cdef double normal_pdf(DOUBLE[:] x,
                       DOUBLE[:] mean,
                       double var) nogil:
    cdef int k
    cdef int n_features = x.shape[0]
    cdef double sum_sq = 0.0

    for k in range(n_features):
        sum_sq += (x[k] - mean[k]) ** 2
    sum_sq *= 0.5 * (1. / var)

    return exp(-0.5 * n_features * log(2 * M_PI * var) - sum_sq)


cdef double mixture_normal_pdf(DOUBLE[:] x,
                               DOUBLE[:] x_prev,
                               DOUBLE[:] weights,
                               double lmbda,
                               DOUBLE[:, :] mean,
                               DOUBLE[:] sigma):
    cdef int k
    cdef int n_groups = mean.shape[0]
    cdef int n_features = mean.shape[1]
    cdef double res = 0
    cdef DOUBLE[:] mu = np.zeros(n_features, dtype=np.float64)

    for k in range(n_groups):
        for p in range(n_features):
            mu[p] = lmbda * mean[k, p] + (1 - lmbda) * x_prev[p]
        res += weights[k] * normal_pdf(x, mu, sigma[k])

    return res


def renormalize_weights(z, weights, means, sigmas):
    n_components = sigmas.shape[0]

    active_groups, z = np.unique(z, return_inverse=True)
    trans_w = weights[active_groups][:, active_groups]
    trans_w /= np.sum(trans_w, axis=1).reshape(-1, 1)

    mu = means[active_groups]
    sigma = sigmas[active_groups]

    return z, trans_w, mu, sigma


cdef inline double euclidean_distance(DOUBLE[:] x, DOUBLE[:] y) nogil:
    cdef int n_features = x.shape[0]
    cdef double d = 0.
    for k in range(n_features):
        d += (x[k] - y[k]) ** 2

    return sqrt(d)


def marginal_forecast(DOUBLE[:, :] x,
                      DOUBLE[:, :, :] x_prev,
                      np.ndarray[np.int64_t, ndim=2, mode='c'] z,
                      np.ndarray[double, ndim=3, mode='c'] trans_weights,
                      np.ndarray[double, ndim=3, mode='c'] mus,
                      np.ndarray[double, ndim=2, mode='c'] sigmas,
                      DOUBLE[:] intercepts,
                      DOUBLE[:] lmbdas,
                      bint renormalize=True):
    cdef int i, j, s = 0
    cdef int n_iter = x_prev.shape[0]
    cdef int n_nodes = x_prev.shape[1]

    cdef double dij, wij

    cdef np.ndarray[double, ndim=2, mode='c'] sum_w = np.zeros(
        (n_nodes, n_nodes))
    cdef np.ndarray[double, ndim=2, mode='c'] probas = np.zeros(
        (n_nodes, n_nodes))
    cdef np.ndarray[np.int64_t, ndim=1, mode='c'] zs
    cdef DOUBLE[:, :] weights, mean
    cdef DOUBLE[:] sigma

    for s in range(n_iter):
        if renormalize:
            zs, weights, mean, sigma = renormalize_weights(
                z[s], trans_weights[s], mus[s], sigmas[s])
        else:
            weights = trans_weights[s]
            mean = mus[s]
            sigma = sigmas[s]
            zs = z[s]

        for i in range(n_nodes):
            for j in range(i):
                dij = euclidean_distance(x[i], x[j])

                wij = mixture_normal_pdf(
                    x[i], x_prev[s, i], weights[zs[i]], lmbdas[s], mean, sigma)
                wij *= mixture_normal_pdf(
                    x[j], x_prev[s, j], weights[zs[j]], lmbdas[s], mean, sigma)
                probas[i, j] += wij * expit(intercepts[s] - dij) / n_iter
                sum_w[i, j] += wij / n_iter

    sum_w += sum_w.T
    sum_w[np.diag_indices(n_nodes)] = 1
    probas += probas.T
    probas /= sum_w

    return np.asarray(probas)
