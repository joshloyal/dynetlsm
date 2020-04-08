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

cdef inline double expit(double z):
    return 1. / (1. + exp(-z))


def directed_intercept_grad(DOUBLE[:, :, :] Y,
                            DOUBLE[:, :, :] dist,
                            DOUBLE[:] radii,
                            double intercept_in,
                            double intercept_out):
    cdef int i, j, t = 0
    cdef int n_time_steps = Y.shape[0]
    cdef int n_nodes = Y.shape[1]
    cdef double d_in, d_out, eta, step
    cdef double in_grad, out_grad = 0.

    for t in range(n_time_steps):
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    d_in = (1 - dist[t, i, j] / radii[j])
                    d_out = (1 - dist[t, i, j] / radii[i])
                    eta = intercept_in * d_in + intercept_out * d_out
                    step = Y[t, i, j] - expit(eta)

                    in_grad += d_in * step
                    out_grad += d_out * step

    return np.array([in_grad, out_grad])


def directed_partial_loglikelihood(DOUBLE[:, ::1] Y,
                                   DOUBLE[:, ::1] X,
                                   DOUBLE[:] radii,
                                   double intercept_in,
                                   double intercept_out,
                                   int node_id,
                                   bint squared=False):
    cdef int j, d = 0
    cdef int n_nodes = Y.shape[0]
    cdef int n_features = X.shape[1]
    cdef double dist = 0
    cdef double eta = 0
    cdef double loglik  = 0

    for j in range(n_nodes):
        dist = 0
        eta = 0
        if j != node_id:
            for d in range(n_features):
                dist += (X[j, d] - X[node_id, d]) ** 2

            if not squared:
                dist = sqrt(dist)

            # Y_ijt
            eta = intercept_in * (1 - dist / radii[j])
            eta += intercept_out * (1 - dist / radii[node_id])
            loglik += Y[node_id, j] * eta - log(1 + exp(eta))

            # Y_jit
            eta = intercept_in * (1 - dist / radii[node_id])
            eta += intercept_out * (1 - dist / radii[j])
            loglik += Y[j, node_id] * eta - log(1 + exp(eta))

    return loglik


def approx_directed_partial_loglikelihood(DOUBLE[:, :] X,
                                          DOUBLE[:] radii,
                                          INT[:, :] in_edges,
                                          INT[:, :] out_edges,
                                          INT[:, :] degree,
                                          INT[:, :] control_nodes_in,
                                          INT[:, :] control_nodes_out,
                                          double intercept_in,
                                          double intercept_out,
                                          int node_id,
                                          bint squared=False):
    cdef int j, d = 0
    cdef int in_degree = degree[node_id, 0]
    cdef int out_degree = degree[node_id, 1]
    cdef int n_nodes = X.shape[0]
    cdef int n_features = X.shape[1]
    cdef int n_control = control_nodes_in.shape[1]
    cdef double dist = 0
    cdef double eta = 0
    cdef double loglik  = 0
    cdef double control = 0
    cdef double n_control_edges = 0
    cdef double control_adj = 0

    # in edges
    for j in range(in_degree):
        dist = 0
        eta = 0
        for d in range(n_features):
            dist += (X[in_edges[node_id, j], d] - X[node_id, d]) ** 2

        if not squared:
            dist = sqrt(dist)

        eta = intercept_in * (1 - dist / radii[node_id])
        eta += intercept_out * (1 - dist / radii[in_edges[node_id, j]])
        loglik += eta - log(1 + exp(eta))

    # out edges
    for j in range(out_degree):
        dist = 0
        eta = 0
        for d in range(n_features):
            dist += (X[out_edges[node_id, j], d] - X[node_id, d]) ** 2

        if not squared:
            dist = sqrt(dist)

        eta = intercept_in * (1 - dist / radii[out_edges[node_id, j]])
        eta += intercept_out * (1 - dist / radii[node_id])
        loglik += eta - log(1 + exp(eta))

    # control estimates (in edges)
    for j in range(n_control):
        if control_nodes_in[node_id, j] == -1.0:
            break

        dist = 0
        eta = 0
        for d in range(n_features):
            dist += (X[control_nodes_in[node_id, j], d] - X[node_id, d]) ** 2

        if not squared:
            dist = sqrt(dist)

        eta = intercept_in * (1 - dist / radii[node_id])
        eta += intercept_out * (1 - dist / radii[control_nodes_in[node_id, j]])
        control += log(1 + exp(eta))

        n_control_edges += 1

    # add control estimate
    control_adj = (n_nodes - in_degree - 1) / n_control_edges
    loglik -= control_adj * control

    control = 0
    n_control_edges = 0
    for j in range(n_control):
        if control_nodes_in[node_id, j] == -1.0:
            break

        dist = 0
        eta = 0
        for d in range(n_features):
            dist += (X[control_nodes_out[node_id, j], d] - X[node_id, d]) ** 2

        if not squared:
            dist = sqrt(dist)

        eta = intercept_in * (1 - dist / radii[control_nodes_out[node_id, j]])
        eta += intercept_out * (1 - dist / radii[node_id])
        control += log(1 + exp(eta))

        n_control_edges += 1

    # add control estimate
    control_adj = (n_nodes - out_degree - 1) / n_control_edges
    loglik -= control_adj * control

    return loglik


def directed_network_loglikelihood_fast(DOUBLE[:, :, ::1] Y,
                                        DOUBLE[:, :, ::1] dist,
                                        DOUBLE[:] radii,
                                        double intercept_in,
                                        double intercept_out):
    cdef int i, j, t = 0
    cdef int n_time_steps = Y.shape[0]
    cdef int n_nodes = Y.shape[1]
    cdef double d_in, d_out, eta = 0.
    cdef double loglik = 0.

    for t in range(n_time_steps):
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    d_in = (1 - dist[t, i, j] / radii[j])
                    d_out = (1 - dist[t, i, j] / radii[i])
                    eta = intercept_in * d_in + intercept_out * d_out
                    loglik += Y[t, i, j] * eta - log(1 + exp(eta))

    return loglik


def approx_directed_network_loglikelihood(DOUBLE[:, :, :] X,
                                          DOUBLE[:] radii,
                                          INT[:, :, :] in_edges,
                                          INT[:, :, :] out_edges,
                                          INT[:, :, :] degree,
                                          INT[:, :, :] control_nodes,
                                          double intercept_in,
                                          double intercept_out,
                                          bint squared=False):
    cdef int i, j, t = 0
    cdef int n_time_steps = X.shape[0]
    cdef int n_nodes = X.shape[1]
    cdef int n_features = X.shape[2]
    cdef int n_control = control_nodes.shape[2]
    cdef double dist = 0.
    cdef double eta = 0.
    cdef double loglik = 0.

    cdef int out_degree = 0
    cdef double control = 0.
    cdef double n_control_edges = 0.
    cdef double control_adj = 0.

    for t in range(n_time_steps):
        for i in range(n_nodes):
            # out edges
            out_degree = degree[t, i, 1]
            for j in range(out_degree):
                dist = 0
                eta = 0
                for d in range(n_features):
                    dist += (X[t, out_edges[t, i, j], d] - X[t, i, d]) ** 2
                if not squared:
                    dist = sqrt(dist)

                eta = intercept_in * (1 - dist / radii[out_edges[t, i, j]])
                eta += intercept_out * (1 - dist / radii[i])
                loglik += eta - log(1 + exp(eta))

            # control estimate
            control = 0.
            n_control_edges = 0.
            for j in range(n_control):
                if control_nodes[t, i, j] == -1.0:
                    break

                dist = 0
                eta = 0
                for d in range(n_features):
                    dist += (X[t, control_nodes[t, i, j], d] - X[t, i, d]) ** 2
                if not squared:
                    dist = sqrt(dist)

                eta = intercept_in * (1 - dist / radii[control_nodes[t, i, j]])
                eta += intercept_out * (1 - dist / radii[i])
                control += log(1 + exp(eta))

                n_control_edges += 1

            control_adj = (n_nodes - out_degree - 1) / n_control_edges
            loglik -= control_adj * control

    return loglik


def directed_network_probas(DOUBLE[:, :, :] dist,
                            DOUBLE[:] radii,
                            double intercept_in,
                            double intercept_out):
    cdef int i, j, t = 0
    cdef int n_time_steps = dist.shape[0]
    cdef int n_nodes = dist.shape[1]
    cdef double d_in, d_out, eta
    cdef double loglik = 0.

    cdef DOUBLE[:, :, :] probas = np.zeros_like(dist)

    for t in range(n_time_steps):
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    d_in = (1 - dist[t, i, j] / radii[j])
                    d_out = (1 - dist[t, i, j] / radii[i])
                    eta = intercept_in * d_in + intercept_out * d_out
                    probas[t, i, j] = 1 / (1 + exp(-eta))

    return np.asarray(probas)
