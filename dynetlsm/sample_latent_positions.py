import numpy as np

from scipy.sparse import csgraph
from sklearn.utils import check_random_state

from .network_likelihoods import (
    partial_loglikelihood,
    directed_partial_loglikelihood,
    approx_directed_partial_loglikelihood
)


def sample_control_nodes(edge_list, n_samples=100, random_state=None):
    rng = check_random_state(random_state)
    n_nodes = len(edge_list)

    # TODO: n_samples can be a fraction of total number of nodes

    control_nodes = np.zeros((n_nodes, n_samples), dtype=np.int)
    for i in range(n_nodes):
        # stratify sample based one connections vs. non-connections
        n_connected = int(edge_list[i].shape[0] / n_nodes * n_samples)
        if edge_list[i].shape[0] > 0:
            n_connected = max(n_connected, 1)

        control_nodes[i, :n_connected] = rng.choice(edge_list[i],
                                                    size=n_connected,
                                                    replace=False)

        edges = set.difference(
            set(range(n_nodes)), edge_list[i].tolist() + [i])
        control_nodes[i, n_connected:] = rng.choice(
            list(edges), size=n_samples - n_connected, replace=False)

    return control_nodes


def sample_control_edges(Y, n_samples=100, random_state=None):
    n_time_steps, n_nodes, _ = Y.shape

    n_edges = int(0.5 * n_nodes * (n_nodes))
    edge_list = np.zeros((n_time_steps, n_edges, 2))
    for t in range(n_time_steps):
        triu_indices = np.triu_indices_from(Y[t])
        edge_list[t, 0] = triu_indices[0]
        edge_list[t, 1] = triu_indices[1]

        edges = np.where(Y[t][triu_indices] == 1)[0]
        non_edges = np.where(Y[t][triu_indices] == 0)[0]

    return edge_list, edges, non_edges


def case_control_init(Y, is_directed=False, n_samples=100):
    n_time_steps, n_nodes, _ = Y.shape

    # compute in-degree / out-degree of each node
    degree = np.zeros((n_time_steps, n_nodes, 2), dtype=np.int)
    for t in range(n_time_steps):
        degree[t, :, 0] = Y[t].sum(axis=0)  # in-degree
        degree[t, :, 1] = Y[t].sum(axis=1)  # out-degree

    # store indices of edges, i.e. Y_ijt = 1
    max_in_degree = int(np.max(degree[:, :, 0]))
    max_out_degree = int(np.max(degree[:, :, 1]))
    in_edges = np.zeros((n_time_steps, n_nodes, max_in_degree), dtype=np.int)
    out_edges = np.zeros((n_time_steps, n_nodes, max_out_degree), dtype=np.int)
    for t in range(n_time_steps):
        for i in range(n_nodes):
            indices = np.where(Y[t, i, :] == 1)[0]
            n_edges = indices.shape[0]
            if n_edges:
                out_edges[t, i, :n_edges] = indices

            indices = np.where(Y[t, :, i] == 1)[0]
            n_edges = indices.shape[0]
            if n_edges:
                in_edges[t, i, :n_edges] = indices

    # determine edges (Y_ijt = 1 or Y_jit = 1 for at least one time step)
    edge_list = []
    for i in range(n_nodes):
        mask = (np.logical_or(Y[:, i, :] == 1, Y[:, :, i] == 1)).astype(np.int)
        mask = mask.sum(axis=0)
        edge_list.append(np.unique(np.where(mask > 0)[0]))

    if is_directed:
        return degree, in_edges, out_edges, edge_list
    return degree[:, :, 0], in_edges, edge_list


def sample_latent_positions(Y, X, intercept, tau_sq, sigma_sq, samplers,
                            radii=None, is_directed=False, squared=False,
                            case_control_sampler=None, random_state=None):
    rng = check_random_state(random_state)
    n_time_steps, n_nodes, _ = Y.shape

    for t in range(n_time_steps):
        for j in range(n_nodes):
            def logp(x):
                X[t, j] = x
                if is_directed:
                    if case_control_sampler is not None:
                        loglik = approx_directed_partial_loglikelihood(
                            X[t],
                            radii=radii,
                            in_edges=case_control_sampler.in_edges_[t],
                            out_edges=case_control_sampler.out_edges_[t],
                            degree=case_control_sampler.degrees_[t],
                            control_nodes_in=(
                                case_control_sampler.control_nodes_in_[t]),
                            control_nodes_out=(
                                case_control_sampler.control_nodes_out_[t]),
                            intercept_in=intercept[0],
                            intercept_out=intercept[1],
                            node_id=j,
                            squared=squared)
                    else:
                        loglik = directed_partial_loglikelihood(
                                    Y[t], X[t],
                                    radii=radii,
                                    intercept_in=intercept[0],
                                    intercept_out=intercept[1],
                                    node_id=j,
                                    squared=squared)
                else:
                    loglik = partial_loglikelihood(Y[t], X[t],
                                                   intercept, j,
                                                   squared=squared)

                # prior
                if t == 0:
                    loglik -= 0.5 * np.sum(x * x) / tau_sq
                else:
                    diff = x - X[t-1, j]
                    loglik -= 0.5 * np.sum(diff * diff) / sigma_sq

                if t < (n_time_steps - 1):
                    diff = X[t+1, j] - x
                    loglik -= 0.5 * np.sum(diff * diff) / sigma_sq

                return loglik

            X[t, j] = samplers[t][j].step(X[t, j].copy(), logp, rng)

    return X


def sample_latent_positions_mixture(Y, X, intercept, mu, sigma, lmbda, z,
                                    samplers, radii=None, is_directed=False,
                                    squared=None, case_control_sampler=None,
                                    random_state=None):
    rng = check_random_state(random_state)
    n_time_steps, n_nodes, _ = Y.shape

    for t in range(n_time_steps):
        for j in range(n_nodes):
            def logp(x):
                X[t, j] = x
                if is_directed:
                    if case_control_sampler:
                        loglik = approx_directed_partial_loglikelihood(
                            X[t],
                            radii=radii,
                            in_edges=case_control_sampler.in_edges_[t],
                            out_edges=case_control_sampler.out_edges_[t],
                            degree=case_control_sampler.degrees_[t],
                            control_nodes_in=(
                                case_control_sampler.control_nodes_in_[t]),
                            control_nodes_out=(
                                case_control_sampler.control_nodes_out_[t]),
                            intercept_in=intercept[0],
                            intercept_out=intercept[1],
                            node_id=j,
                            squared=squared)
                    else:
                        loglik = directed_partial_loglikelihood(
                                    Y[t], X[t],
                                    radii=radii,
                                    intercept_in=intercept[0],
                                    intercept_out=intercept[1],
                                    node_id=j)
                else:
                    loglik = partial_loglikelihood(Y[t], X[t],
                                                   intercept, j)

                # prior P(X_t | X_{t-1})
                if t == 0:
                    diff = x - mu[z[t, j]]
                    loglik -= 0.5 * np.sum(diff * diff) / sigma[z[t, j]]
                else:
                    diff = x - (1 - lmbda) * X[t-1, j] - lmbda * mu[z[t, j]]
                    loglik -= 0.5 * np.sum(diff * diff) / sigma[z[t, j]]

                # prior P(X_{t+1} | X_t)
                if t < (n_time_steps - 1):
                    diff = (X[t+1, j] - (1 - lmbda) * x -
                            lmbda * mu[z[t+1, j]])
                    loglik -= 0.5 * np.sum(diff * diff) / sigma[z[t+1, j]]

                return loglik

            X[t, j] = samplers[t][j].step(X[t, j].copy(),
                                          logp, rng)

    return X
