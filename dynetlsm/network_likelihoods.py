import numpy as np

from .latent_space import calculate_distances
from .gaussian_likelihood_fast import compute_gaussian_likelihood
from .array_utils import triu_indices_from_3d, nondiag_indices_from_3d
from .directed_likelihoods_fast import (
    directed_network_loglikelihood_fast,
    directed_partial_loglikelihood, directed_intercept_grad,
    directed_network_probas,
    approx_directed_partial_loglikelihood,
    approx_directed_network_loglikelihood)
from .static_network_fast import partial_loglikelihood


# TODO: mask nan entries
def dynamic_network_loglikelihood_directed(Y, X,
                                           intercept_in, intercept_out, radii,
                                           squared=False, dist=None):
    dist = calculate_distances(X, squared=squared) if dist is None else dist

    return directed_network_loglikelihood_fast(Y, dist, radii,
                                               intercept_in, intercept_out)


# TODO: mask nan entries
def dynamic_network_loglikelihood_undirected(Y, X, intercept, squared=False,
                                             dist=None):
    dist = calculate_distances(X, squared=squared) if dist is None else dist

    triu_indices = triu_indices_from_3d(dist, k=1)
    eta = intercept - dist[triu_indices]

    return np.sum(Y[triu_indices] * eta - np.log(1 + np.exp(eta)))


def dynamic_network_loglikelihood(model, sample_id, dist=None):
    X = model.Xs_[sample_id]
    intercept = model.intercepts_[sample_id]
    radii = model.radiis_[sample_id] if model.is_directed else None
    if dist is None:
        dist = (None if model.case_control_sampler_ else
                calculate_distances(X, squared=False))

    if model.is_directed:
        if model.case_control_sampler_ is not None:
            loglik = approx_directed_network_loglikelihood(
                X,
                radii=radii,
                in_edges=model.case_control_sampler_.in_edges_,
                out_edges=model.case_control_sampler_.out_edges_,
                degree=model.case_control_sampler_.degrees_,
                control_nodes=model.case_control_sampler_.control_nodes_out_,
                intercept_in=intercept[0],
                intercept_out=intercept[1],
                squared=False)
        else:
            loglik = dynamic_network_loglikelihood_directed(
                model.Y_fit_, X,
                intercept_in=intercept[0],
                intercept_out=intercept[1],
                radii=radii, dist=dist)
    else:
        loglik = dynamic_network_loglikelihood_undirected(
            model.Y_fit_, X, intercept, dist=dist)

    return loglik
