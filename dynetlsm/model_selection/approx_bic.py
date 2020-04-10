import numpy as np

from ..network_likelihoods import compute_gaussian_likelihood
from ..network_likelihoods import dynamic_network_loglikelihood_undirected
from ..network_likelihoods import dynamic_network_loglikelihood_directed
from ..array_utils import nondiag_indices_from_3d


__all__ = ['select_bic']


class DynamicNetworkMixtureModel:
    def __init__(self, beta, init_weights, trans_weights, X, mu, sigma, lmbda,
                 z, intercept, radii=None):
        self.beta = beta
        self.init_weights = init_weights
        self.trans_weights = trans_weights
        self.X = X
        self.mu = mu
        self.sigma = sigma
        self.lmbda = lmbda
        self.z = z
        self.intercept = intercept
        self.radii = radii


def calculate_cluster_counts_t(model):
    n_burn = model.n_burn_

    z = model.zs_
    n_iter, n_time_steps, _ = z.shape
    n_burn = n_burn if n_burn is not None else 0

    counts = np.zeros((n_time_steps, int(n_iter - n_burn)), dtype=np.int)
    for t in range(n_time_steps):
        for i in range(n_iter - n_burn):
            n_clusters = np.unique(z[i + n_burn, t]).shape[0]
            counts[t, i] = n_clusters
    return counts


def calculate_cluster_counts(model):
    n_burn = model.n_burn_

    z = model.zs_
    n_iter = z.shape[0]
    n_burn = n_burn if n_burn is not None else 0

    counts = np.zeros(int(n_iter - n_burn), dtype=np.int)
    for i in range(n_iter - n_burn):
        n_clusters = np.unique(z[i + n_burn].ravel()).shape[0]
        counts[i] = n_clusters
    return counts


def latent_marginal_loglikelihood(X, init_w, trans_w, mu, sigma, lmbda):
    n_time_steps, n_nodes, _ = X.shape
    n_components = sigma.shape[0]

    loglik = 0.0
    for i in range(n_nodes):
        gauss_loglik = compute_gaussian_likelihood(X[:, i], mu, sigma, lmbda,
                                                   normalize=False)
        fwds_msg = init_w * gauss_loglik[0]
        c = np.sum(fwds_msg)
        loglik += np.log(c)
        fwds_msg /= c

        for t in range(1, n_time_steps):
            fwds_msg = (gauss_loglik[t] *
                        np.dot(trans_w[t].T, fwds_msg.reshape(-1, 1)).ravel())
            c = np.sum(fwds_msg)
            loglik += np.log(c)
            fwds_msg /= c

    return loglik


def select_bic(model):
    n_time_steps, n_nodes, _ = model.Y_fit_.shape
    n_burn = model.n_burn_

    # determine model sizes available in the posterior samples
    counts = calculate_cluster_counts(model)

    bic = []
    models = []
    for k in np.unique(counts):
        # determine MAP for model size k
        mask = counts != k
        map_id = np.ma.array(model.logps_[n_burn:], mask=mask).argmax() + n_burn

        # extract MAP estimators
        intercept = model.intercepts_[map_id]
        X = model.Xs_[map_id]
        mu = model.mus_[map_id]
        sigma = model.sigmas_[map_id]
        beta = model.betas_[map_id]
        weights = model.weights_[map_id]
        lmbda = model.lambdas_[map_id]
        radii = model.radiis_[map_id] if model.is_directed else None

        # re-normalize weights
        active_clusters = np.unique(model.zs_[map_id].ravel())
        active_mask = np.in1d(np.arange(model.n_components), active_clusters)

        beta = beta[active_clusters]
        beta /= beta.sum()

        init_w = weights[0, 0, active_clusters]
        init_w /= init_w.sum()

        trans_w = np.zeros((n_time_steps, k, k), dtype=np.float64)
        for t in range(1, n_time_steps):
            trans_w[t] = weights[t, active_clusters][:, active_clusters]
            trans_w[t] /= np.sum(trans_w[t], axis=1).reshape(-1, 1)

        # filter cluster components
        mu = mu[active_clusters]
        sigma = sigma[active_clusters]

        # BIC component for P(Y | X)
        if model.is_directed:
            loglik_k = dynamic_network_loglikelihood_directed(
                            model.Y_fit_, X,
                            intercept_in=intercept[0],
                            intercept_out=intercept[1],
                            radii=radii)
            bic_k = -2 * loglik_k

            n_params = 2 + n_nodes
            nondiag_indices = nondiag_indices_from_3d(model.Y_fit_)
            bic_k += n_params * np.log(np.sum(model.Y_fit_[nondiag_indices]))
        else:
            loglik_k = dynamic_network_loglikelihood_undirected(
                model.Y_fit_, X, intercept)
            bic_k = -2 * loglik_k
            bic_k += np.log(0.5 * (
                np.sum(model.Y_fit_) - np.einsum('ikk', model.Y_fit_).sum()))

        # BIC component for P(X | G) = P(X | mu, sigma, w)
        bic_k -= 2 * latent_marginal_loglikelihood(
            X, init_w, trans_w, mu, sigma, lmbda)

        n_params = ((model.n_features + 1) * k +        # cluster params
                    (k - 1) +                           # beta
                    (k - 1) +                           # init_weights
                    (n_time_steps - 1) * k * (k - 1))   # trans_weights
        bic_k += n_params * np.log(n_nodes * n_time_steps)

        model_k = DynamicNetworkMixtureModel(init_weights=init_w,
                                             trans_weights=trans_w,
                                             beta=beta,
                                             X=X, mu=mu, sigma=sigma,
                                             lmbda=lmbda,
                                             z=model.zs_[map_id],
                                             intercept=intercept,
                                             radii=radii)
        bic.append([k, bic_k, loglik_k, map_id])
        models.append(model_k)

    return np.array(bic), models, counts
