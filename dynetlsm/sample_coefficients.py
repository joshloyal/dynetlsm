import numpy as np

from sklearn.utils import check_random_state

from .network_likelihoods import (
    dynamic_network_loglikelihood_directed,
    dynamic_network_loglikelihood_undirected,
    approx_directed_network_loglikelihood,
)


def sample_intercepts(Y, X, intercepts, intercept_prior,
                      intercept_variance_prior, samplers, radii=None,
                      dist=None, is_directed=False, case_control_sampler=None,
                      squared=False, random_state=None):
    rng = check_random_state(random_state)

    if is_directed:
        # sample intercept_in
        def logp(x):
            if case_control_sampler is not None:
                # TODO: we do not cache distances here, decrease by
                #       factor of 2 if we do this
                loglik = approx_directed_network_loglikelihood(
                    X=X,
                    radii=radii,
                    in_edges=case_control_sampler.in_edges_,
                    out_edges=case_control_sampler.out_edges_,
                    degree=case_control_sampler.degrees_,
                    control_nodes=case_control_sampler.control_nodes_out_,
                    intercept_in=x[0],
                    intercept_out=intercepts[1],
                    squared=squared)
            else:
                loglik = dynamic_network_loglikelihood_directed(
                    Y, X,
                    intercept_in=x[0], intercept_out=intercepts[1],
                    radii=radii,
                    squared=squared,
                    dist=dist)
            loglik -= ((x[0] - intercept_prior[0]) ** 2 /
                       (2 * intercept_variance_prior))
            return loglik

        intercepts[0] = samplers[0].step(
                                np.array([intercepts[0]]), logp, rng)[0]

        # sample intercept_out
        def logp(x):
            if case_control_sampler is not None:
                # TODO: we do not cache distances here, decrease by
                #       factor of 2 if we do this
                loglik = approx_directed_network_loglikelihood(
                    X=X,
                    radii=radii,
                    in_edges=case_control_sampler.in_edges_,
                    out_edges=case_control_sampler.out_edges_,
                    degree=case_control_sampler.degrees_,
                    control_nodes=case_control_sampler.control_nodes_out_,
                    intercept_in=intercepts[0],
                    intercept_out=x[0],
                    squared=squared)
            else:
                loglik = dynamic_network_loglikelihood_directed(
                    Y, X,
                    intercept_in=intercepts[0], intercept_out=x[0],
                    radii=radii,
                    squared=squared,
                    dist=dist)
            loglik -= ((x[0] - intercept_prior[1]) ** 2 /
                       (2 * intercept_variance_prior))
            return loglik

        intercepts[1] = samplers[1].step(
                            np.array([intercepts[1]]), logp, rng)[0]
    else:
        def logp(x):
            loglik = dynamic_network_loglikelihood_undirected(Y, X,
                                                              intercept=x,
                                                              squared=squared,
                                                              dist=dist)
            loglik -= ((x - intercept_prior) ** 2 /
                       (2 * intercept_variance_prior))
            return loglik

        intercepts = samplers[0].step(intercepts, logp, rng)

    return intercepts


def sample_radii(Y, X, intercepts, radii, sampler, dist=None,
                 case_control_sampler=None, squared=False, random_state=None):
    rng = check_random_state(random_state)

    def logp(x):
        # NOTE: dirichlet prior (this is constant for alpha = 1.0
        if case_control_sampler:
            # TODO: we do not cache distances here, decrease by
            #       factor of 2 if we do this
            loglik = approx_directed_network_loglikelihood(
                        X=X,
                        radii=x,
                        in_edges=case_control_sampler.in_edges_,
                        out_edges=case_control_sampler.out_edges_,
                        degree=case_control_sampler.degrees_,
                        control_nodes=case_control_sampler.control_nodes_out_,
                        intercept_in=intercepts[0],
                        intercept_out=intercepts[1],
                        squared=squared)
        else:
            loglik = dynamic_network_loglikelihood_directed(
                       Y, X,
                       intercept_in=intercepts[0],
                       intercept_out=intercepts[1],
                       radii=x,
                       squared=squared,
                       dist=dist)

        return loglik

    return sampler.step(radii, logp, rng)
