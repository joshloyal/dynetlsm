# encoding: utf-8
# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

from libc.math cimport log, exp, sqrt

import numpy as np
cimport numpy as np


ctypedef np.npy_float64 DOUBLE
ctypedef np.npy_int64 INT


def partial_loglikelihood(DOUBLE[:, :] Y,
                          DOUBLE[:, :] X,
                          double intercept,
                          int node_id,
                          bint squared=False):
    cdef int i, d = 0
    cdef int n_nodes = Y.shape[0]
    cdef int n_features = X.shape[1]
    cdef double dist = 0
    cdef double eta = 0
    cdef double loglik  = 0

    for i in range(n_nodes):
        dist = 0
        eta = 0
        if i != node_id:
            for d in range(n_features):
                dist += (X[i, d] - X[node_id, d]) ** 2
            if squared:
                eta = intercept - dist
            else:
                eta = intercept - sqrt(dist)

            # in-case the network is undirected
            loglik += Y[node_id, i] * eta
            loglik -= log(1 + exp(eta))

    return loglik


def approx_partial_loglikelihood(DOUBLE[:, :] X,
                                 double intercept,
                                 INT[:, :] edges,
                                 INT[:] degrees,
                                 INT[:, :] control_nodes,
                                 int node_id,
                                 bint squared=False):
    cdef int j, d = 0
    cdef int n_nodes = X.shape[0]
    cdef int n_features = X.shape[1]
    cdef int n_control = control_nodes.shape[1]
    cdef int node_degree = degrees[node_id]
    cdef double dist = 0
    cdef double eta = 0
    cdef double loglik  = 0
    cdef double control = 0
    cdef double control_adj = (<double> (n_nodes - 1) / <double> n_control)

    # edges
    for j in range(node_degree):
        dist = 0
        eta = 0
        for d in range(n_features):
            dist += (X[edges[node_id, j], d] - X[node_id, d]) ** 2
        if squared:
            eta = intercept - dist
        else:
            eta = intercept - sqrt(dist)

        loglik += eta

    # control estimate
    for j in range(n_control):
        dist = 0
        eta = 0
        for d in range(n_features):
            dist += (X[control_nodes[node_id, j], d] - X[node_id, d]) ** 2
        if squared:
            eta = intercept - dist
        else:
            eta = intercept - sqrt(dist)

        control += log(1 + exp(eta))

    # add control estimate
    loglik -= control_adj * control

    return loglik


def approx_loglikelihood(DOUBLE[:, :] X,
                         double intercept,
                         INT[:, :] edges,
                         INT[:] degrees,
                         INT[:, :] control_nodes,
                         int node_id,
                         bint squared=False):
    pass
