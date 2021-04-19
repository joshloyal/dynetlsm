import numpy as np
import scipy.stats as stats
import itertools

from math import ceil

from scipy.optimize import minimize
from scipy.special import expit
from sklearn.utils import check_random_state, check_array
from tqdm import tqdm

from .array_utils import (
    triu_indices_from_3d, diag_indices_from_3d, nondiag_indices_from_3d)
from .case_control_likelihood import DirectedCaseControlSampler
from .distributions import (
    truncated_normal, truncated_normal_logpdf,
    sample_dirichlet, dirichlet_logpdf)
from .lsm import DynamicNetworkLSM
from .imputer import SimpleNetworkImputer
from .latent_space import calculate_distances, longitudinal_kmeans
from .label_utils import calculate_posterior_cooccurrence, renormalize_weights
from .label_utils import calculate_posterior_group_counts
from .metrics import network_auc
from .metropolis import Metropolis
from .model_selection import select_bic, minimize_posterior_expected_vi
from .network_likelihoods import (
    approx_directed_network_loglikelihood,
    dynamic_network_loglikelihood_directed,
    dynamic_network_loglikelihood_undirected,
    directed_network_probas
)
from .procrustes import longitudinal_procrustes_rotation
from .sample_coefficients import sample_intercepts, sample_radii
from .sample_labels import sample_labels_gibbs, sample_labels_block_lpcm
from .sample_latent_positions import sample_latent_positions_mixture


SMALL_EPS = np.finfo('float64').tiny


__all__ = ['DynamicNetworkLPCM']


def init_sampler(Y, is_directed=False,
                 n_iter=100, n_features=2, n_components=10,
                 lambda_init=0.9, sample_missing=False,
                 n_control=None, n_resample_control=None,
                 random_state=None):
    """Initialize the the HDP-LPCM sampler."""
    n_time_steps, n_nodes, _ = Y.shape

    # initialize latent space parameters
    if is_directed:
        dynamic_emb = DynamicNetworkLSM(
            n_iter=500,
            n_features=n_features,
            tune=250, burn=250,
            sigma_sq=0.001,
            tau_sq='auto',
            step_size_X=0.0075,
            is_directed=is_directed,
            n_control=n_control,
            n_resample_control=n_resample_control,
            random_state=random_state).fit(Y)
    else:
        dynamic_emb = DynamicNetworkLSM(
            n_iter=500,
            n_features=n_features,
            tune=250, burn=250,
            sigma_sq=0.1,
            tau_sq=2.0,
            step_size_X=0.1,
            is_directed=is_directed,
            random_state=random_state).fit(Y)

    # imputed Y
    Y_fit = Y.copy()
    if sample_missing:
        nan_mask = Y == -1
        Y_fit[nan_mask] = dynamic_emb.probas_[nan_mask] > 0.5

    # latent positions
    Xs = np.zeros((n_iter, n_time_steps, n_nodes, n_features),
                  dtype=np.float64)
    Xs[0] = dynamic_emb.X_

    # intercept
    if is_directed:
        intercepts = np.zeros((n_iter, 2), dtype=np.float64)
    else:
        intercepts = np.zeros((n_iter, 1), dtype=np.float64)
    intercepts[0] = dynamic_emb.intercept_

    # radii
    if is_directed:
        radiis = np.zeros((n_iter, n_nodes), dtype=np.float64)
        radiis[0] = dynamic_emb.radii_
    else:
        radiis = None

    # cluster parameters
    zs = np.zeros((n_iter, n_time_steps, n_nodes), dtype=np.int)
    mus = np.zeros((n_iter, n_components, n_features),
                   dtype=np.float64)
    sigmas = np.zeros((n_iter, n_components), dtype=np.float64)

    # k-means initialization
    mus[0], sigmas[0], zs[0] = longitudinal_kmeans(Xs[0],
                                                   n_clusters=n_components,
                                                   random_state=random_state)

    # intialize initial distribution to empirical distrbution due to k-means
    init_weights = np.zeros((n_iter, n_components), dtype=np.float64)
    resp = np.zeros((n_nodes, n_components))
    resp[np.arange(n_nodes), zs[0]] = 1
    nk = resp.sum(axis=0)
    init_weights[0] = nk / n_nodes

    # blending coefficient set to initial value
    lambdas = np.zeros((n_iter, 1), dtype=np.float64)
    lambdas[0] = lambda_init

    # initialize transition distributions by sampling from the uniform prior
    rng = check_random_state(random_state)
    trans_weights = np.zeros((n_iter, n_components, n_components), dtype=np.float64)
    for k in range(n_components):
        trans_weights[0, k] = (1. / n_components) * np.ones(n_components)

    return (Xs, intercepts, mus, sigmas, zs, init_weights,
            trans_weights, lambdas, radiis, Y_fit)


class DynamicNetworkLPCM(object):
    def __init__(self,
                 n_features=2,
                 n_components=5,
                 is_directed=False,
                 selection_type='map',
                 n_iter=5000,
                 tune=2500,
                 tune_interval=100,
                 burn=2500,
                 thin=None,
                 intercept_prior='auto',
                 intercept_variance_prior=2,
                 mean_variance_prior='auto',
                 a=2.0,
                 b='auto',
                 lambda_prior=0.9,
                 lambda_variance_prior=0.01,
                 dirichlet_prior='uniform',
                 sigma_prior_std=4.0,
                 mean_variance_prior_std=4.0,
                 step_size_X='auto',
                 step_size_intercept=0.1,
                 step_size_radii=175000,
                 n_control=None,
                 n_resample_control=100,
                 copy=True,
                 random_state=None):
        self.n_iter = n_iter
        self.is_directed = is_directed
        self.selection_type = selection_type
        self.n_features = n_features
        self.n_components = n_components
        self.dirichlet_prior = dirichlet_prior
        self.step_size_X = step_size_X
        self.intercept_prior = intercept_prior
        self.intercept_variance_prior = intercept_variance_prior
        self.step_size_intercept = step_size_intercept
        self.mean_variance_prior = mean_variance_prior
        self.a = a
        self.b = b
        self.lambda_prior = lambda_prior
        self.lambda_variance_prior = lambda_variance_prior
        self.mean_variance_prior_std = mean_variance_prior_std
        self.sigma_prior_std = sigma_prior_std
        self.step_size_radii = step_size_radii
        self.tune = tune
        self.tune_interval = tune_interval
        self.burn = burn
        self.thin = thin
        self.n_control = n_control
        self.n_resample_control = n_resample_control
        self.copy = copy
        self.random_state = random_state

    @property
    def n_burn_(self):
        n_burn = 0
        if self.burn is not None:
            n_burn += self.burn
        if self.tune is not None:
            n_burn += self.tune

        return ceil(n_burn / self.thin) if self.thin else n_burn

    @property
    def distances_(self):
        """Distance matrix between latent positions,
        shape (n_time_steps, n_nodes, n_nodes)
        """
        if not hasattr(self, 'X_'):
            raise ValueError('Model not fit.')
        return calculate_distances(self.X_)

    @property
    def probas_(self):
        """Estimated connection probability matrix,
        shape (n_time_steps, n_nodes, n_nodes).
        """
        if not hasattr(self, 'X_'):
            raise ValueError('Model not fit.')

        if self.is_directed:
            probas = directed_network_probas(self.distances_,
                                             self.radii_,
                                             self.intercept_[0],
                                             self.intercept_[1])
        else:
            probas = expit(self.intercept_ - self.distances_)

        # set diagonals to zero (assuming no self-loops)
        probas[diag_indices_from_3d(probas)] = 0.0

        return probas

    @property
    def forecast_probas_(self):
        """Simple plug-in estimate of one-step-ahead probabilities based on
        the MAP estimate."""
        ws = self.trans_weight_[self.z_[-1]]

        X_ahead = np.zeros((self.Y_fit_.shape[1], self.n_features))
        for g in range(self.n_components):
            X_ahead += ws[:, g].reshape(-1, 1) * (self.lambda_ * self.mu_[g] +
                (1 - self.lambda_) * self.X_[-1])

        return expit(self.intercept_ - calculate_distances(X_ahead))

    @property
    def auc_(self):
        """In-sample AUC of the final estimated model."""
        # FIXME: This should mask nan values
        if not hasattr(self, 'X_'):
            raise ValueError('Model not fit.')
        return network_auc(self.Y_fit_, self.probas_,
                           is_directed=self.is_directed,
                           nan_mask=self.nan_mask_)

    def fit(self, Y):
        """Sample from the posterior of the HDP-LPCM based on an observed
        network Y.

        Parameters
        ----------
        Y : array-like, shape (n_time_steps, n_nodes, n_nodes)
            The training dynamic network. The networks should be represented
            as binary directed or undirected adjacency matrices. For example,
            Y[0] is an array of shape (n_nodes, n_nodes) corresponding to the
            adjacency matrix of the network at time zero. Currently,
            weighted networks are not supported. The network should be stored
            as ``dtype=np.float64``.

        Returns
        -------
        self : DynamicNetworkHDPLPCM
            Fitted estimator.
        """
        n_time_steps, n_nodes, _ = Y.shape
        rng = check_random_state(self.random_state)

        Y = check_array(Y, dtype=np.float64,
                        force_all_finite='allow-nan',
                        ensure_2d=False, allow_nd=True, copy=self.copy)

        # determine whether to sample missing values
        # note that imputation happens in the non-mixutre model during init
        self.nan_mask_ = None
        if self.is_directed:
            nondiag_indices = nondiag_indices_from_3d(Y)
            self.nan_mask_ = Y[nondiag_indices] == -1
        else:
            triu_indices = triu_indices_from_3d(Y, k=1)
            self.nan_mask_ = Y[triu_indices] == -1
        sample_missing = np.any(self.nan_mask_)
        if sample_missing:
            self.Y_fit_ = SimpleNetworkImputer(
                strategy='random', missing_value=-1).fit_transform(Y)
        else:
            self.Y_fit_ = Y

        # update n_iter with iterations including burn-in and tuning
        if self.burn is not None:
            self.n_iter += self.burn
        if self.tune is not None:
            self.n_iter += self.tune

        (self.Xs_, self.intercepts_, self.mus_, self.sigmas_, self.zs_,
         self.init_weights_, self.trans_weights_, self.lambdas_,
         self.radiis_, self.Y_fit_) = init_sampler(
                         Y, is_directed=self.is_directed,
                         n_iter=self.n_iter,
                         n_features=self.n_features,
                         n_components=self.n_components,
                         lambda_init=self.lambda_prior,
                         sample_missing=sample_missing,
                         n_control=self.n_control,
                         n_resample_control=self.n_resample_control,
                         random_state=rng)

        if self.dirichlet_prior == 'uniform':
            self.dirichlet_prior_ = 1.
        else:
            self.dirichlet_prior_ = 1. / self.n_components

        # store missing value imputation counts
        if sample_missing:
            self.missings_ = np.zeros(self.nan_mask_.sum())

        if self.step_size_X == 'auto':
            self.step_size_X = 0.01 if self.is_directed else 0.1

        # initialize case-control likelihood sampler
        self.case_control_sampler_ = None
        if self.n_control is not None:
            if not self.is_directed:
                raise ValueError('The case-control likelihood currently only '
                                 'supported for directed networks.')

            self.case_control_sampler_ = DirectedCaseControlSampler(
                                            n_control=self.n_control,
                                            n_resample=self.n_resample_control,
                                            random_state=rng)
            self.case_control_sampler_.init(self.Y_fit_)

        # init metropolis samplers
        self.latent_samplers = []
        for t in range(n_time_steps):
            self.latent_samplers.append(
                [Metropolis(step_size=self.step_size_X,
                            tune=self.tune,
                            tune_interval=self.tune_interval,
                            proposal_type='random_walk') for
                 _ in range(n_nodes)])

        if self.is_directed:
            self.intercept_samplers = [
                Metropolis(step_size=self.step_size_intercept,
                           tune=self.tune, proposal_type='random_walk') for
                _ in range(2)]
        else:
            self.intercept_samplers = [
                Metropolis(step_size=self.step_size_intercept, tune=self.tune,
                           proposal_type='random_walk')]

        if self.is_directed:
            self.radii_sampler = Metropolis(step_size=self.step_size_radii,
                                            tune=self.tune,
                                            proposal_type='dirichlet')

        # initialize hyper-parameters
        if self.intercept_prior == 'auto':
            self.intercept_prior = self.intercepts_[0]

        # initialize hyper-priors

        # NOTE: This sets the scale of the latent space
        # tau_sq ~ Inv-Gamma(a0/2, b0/2) hyper-prior chosen so that
        #
        # E(tau_sq) = mean_variance_prior
        # sqrt(Var(tau_sq)) = mean_variance_prior_std * E(tau_sq)
        if self.mean_variance_prior == 'auto':
            if self.is_directed:
                self.mean_variance_prior_ = (
                    2 * (1. / n_nodes) ** (2. / self.n_features))
            else:
                # for n_nodes = 100, tau_sq = 2.0 is good, so use this scale
                self.mean_variance_prior_ = (
                    (n_nodes ** (2. / self.n_features)) / 50.)
        else:
            self.mean_variance_prior_ = self.mean_variance_prior

        if self.mean_variance_prior_std is not None:
            self.a0_ = (self.mean_variance_prior_std ** 2 + 2) * 2
            self.b0_ = (self.a0_ - 2) * self.mean_variance_prior_ * 2

        # NOTE: This sets the scale of the cluster sizes. Default is to assume
        # there is no cluster structure.
        # b ~ Gamma(c/2, d/2) hyper-prior chossen so that
        #
        # E(b) = b
        # sqrt(Var(b)) = sigma_prior_std * E(b)
        if self.b == 'auto':
            if self.is_directed:
                # Mode(sigma**2) = (1 / n_nodes) * (1 / self.n_components)
                self.b_ = (self.a + 2) * self.mean_variance_prior_
            else:
                # Mode(sigma ** 2) = n_nodes / self.n_components
                self.b_ = (self.a + 2) * self.mean_variance_prior_
        else:
            self.b_ = self.b

        if self.sigma_prior_std is not None:
            self.d0_ = (self.sigma_prior_std ** 2 / self.b_) * 2
            self.c0_ = self.b_ * self.d0_

        # record log-probability of each sample
        self.logps_ = np.zeros(self.n_iter, dtype=np.float64)
        if self.is_directed:
            self.logps_[0] = self.logp(
                self.Xs_[0], self.intercepts_[0],
                self.mus_[0], self.sigmas_[0], self.zs_[0],
                self.init_weights_[0], self.trans_weights_[0],
                self.lambdas_[0], radii=self.radiis_[0])
        else:
            self.logps_[0] = self.logp(
                self.Xs_[0], self.intercepts_[0],
                self.mus_[0], self.sigmas_[0], self.zs_[0],
                self.init_weights_[0], self.trans_weights_[0],
                self.lambdas_[0])
        self.logp_ = self.logps_[0]

        return self._fit(Y, rng)

    def _fit(self, Y, random_state):
        rng = check_random_state(random_state)
        n_time_steps, n_nodes, _ = self.Y_fit_.shape

        if self.is_directed:
            nondiag_indices = nondiag_indices_from_3d(Y)
        else:
            triu_indices = triu_indices_from_3d(Y, k=1)
        sample_missing = np.any(self.nan_mask_)

        for it in tqdm(range(1, self.n_iter)):
            # copy over previous samples
            X = self.Xs_[it - 1].copy()
            intercept = self.intercepts_[it - 1].copy()
            z = self.zs_[it - 1].copy()
            mu = self.mus_[it - 1].copy()
            sigma = self.sigmas_[it - 1].copy()
            init_weights = self.init_weights_[it - 1].copy()
            trans_weights = self.trans_weights_[it - 1].copy()
            lmbda = self.lambdas_[it - 1].copy()
            radii = self.radiis_[it - 1].copy() if self.is_directed else None

            # re-sample control group if necessary
            if self.case_control_sampler_ is not None:
                self.case_control_sampler_.resample()

            # sample latent positions
            X = sample_latent_positions_mixture(
                    self.Y_fit_, X,
                    intercept=intercept,
                    mu=mu, sigma=sigma, lmbda=lmbda,
                    z=z, radii=radii,
                    samplers=self.latent_samplers,
                    is_directed=self.is_directed,
                    squared=False,
                    case_control_sampler=self.case_control_sampler_,
                    random_state=rng)

            # center latent space across time
            X -= np.mean(X, axis=(0, 1))

            # cache new distances
            dist = (None if self.case_control_sampler_ else
                    calculate_distances(X, squared=False))

            # sample intercepts
            intercept = sample_intercepts(
                self.Y_fit_, X, intercept,
                intercept_prior=self.intercept_prior,
                intercept_variance_prior=self.intercept_variance_prior,
                samplers=self.intercept_samplers, radii=radii,
                dist=dist, is_directed=self.is_directed,
                case_control_sampler=self.case_control_sampler_,
                squared=False, random_state=rng)

            # sample radii for directed networks
            if self.is_directed:
                radii = sample_radii(
                            self.Y_fit_, X, intercepts=intercept, radii=radii,
                            sampler=self.radii_sampler, dist=dist,
                            case_control_sampler=self.case_control_sampler_,
                            squared=False, random_state=rng)

            # block sample labels
            z, n, nk, resp = sample_labels_block_lpcm(
                X, mu, sigma, lmbda, init_weights, trans_weights,
                random_state=rng)

            # sample initial distribution (w0)
            init_weights = sample_dirichlet(
                 self.dirichlet_prior_ + nk[0], random_state=rng)

            # sample transition distributions (w)
            for k in range(self.n_components):
                trans_weights[k] = sample_dirichlet(
                     self.dirichlet_prior_ + n[1:, k].sum(axis=0), random_state=rng)

            # sample cluster means
            for k in range(self.n_components):
                pk = 1 / self.mean_variance_prior_
                mk = np.zeros(self.n_features)
                for t in range(n_time_steps):
                    if nk[t, k] > 0:
                        cluster_mask = resp[t, :, k].astype(bool)
                        if t == 0:
                            pk += nk[0, k] / sigma[k]
                            mk += (1 / sigma[k]) * np.sum(X[t, cluster_mask],
                                                          axis=0)
                        else:
                            pk += (lmbda ** 2 / sigma[k]) * nk[t, k]
                            mk += (lmbda / sigma[k]) * np.sum(
                                X[t, cluster_mask] -
                                (1 - lmbda) * X[t-1, cluster_mask], axis=0)

                pk = 1 / pk
                mk *= pk
                mu[k] = rng.multivariate_normal(
                    mean=mk, cov=pk * np.eye(self.n_features))

            # sample cluster variances
            for k in range(self.n_components):
                ak = 0.5 * (np.sum(nk[:, k]) * self.n_features + self.a)
                bk = 0.5 * self.b_
                for t in range(n_time_steps):
                    if nk[t, k] > 0:
                        cluster_mask = resp[t, :, k].astype(bool)
                        if t == 0:
                            bk += 0.5 * np.sum(
                                (X[t, cluster_mask] - mu[k]) ** 2)
                        else:
                            bk += 0.5 * np.sum((
                                X[t, cluster_mask] -
                                (1 - lmbda) * X[t-1, cluster_mask] -
                                lmbda * mu[k]) ** 2)
                sigma[k] = 1. / rng.gamma(shape=ak, scale=1. / bk)

            # sample lambda
            ml = 0.0
            sl = 1.0 / self.lambda_variance_prior
            for t in range(1, n_time_steps):
                ml_diff = (mu[z[t]] - X[t-1]) / sigma[z[t]].reshape(-1, 1)
                X_diff = X[t] - X[t-1]
                ml += np.sum(ml_diff * X_diff)
                ml_diff = (
                    (mu[z[t]] - X[t-1]) / np.sqrt(sigma[z[t]].reshape(-1, 1)))
                sl += np.sum(ml_diff ** 2)
            sl = 1. / sl
            ml += self.lambda_prior / self.lambda_variance_prior
            ml *= sl
            lmbda = truncated_normal(mean=ml,
                                     var=sl,
                                     random_state=rng)

            # re-sample hyperparameters
            if self.mean_variance_prior_std is not None:
                b = 0.5 * self.b0_
                for k in range(self.n_components):
                    b += 0.5 * np.sum(mu[k] ** 2)
                a = 0.5 * (self.a0_ + self.n_components)
                self.mean_variance_prior_ = 1 / rng.gamma(shape=a,
                                                          scale=1. / b,
                                                          size=1)

            # sample sigma priors
            if self.sigma_prior_std is not None:
                scale = 0.5 * self.d0_
                for k in range(self.n_components):
                    scale += 0.5 * (1. / sigma[k])
                shape = 0.5 * (self.c0_ + self.n_components * self.a)
                self.b_ = rng.gamma(shape=shape, scale=1. / scale)

            # sample missing data
            if sample_missing:
                # calculate pij for missing edges and sample from Bern(pij)
                # This is sampling everything! just sample missing...
                if self.is_directed:
                    probas = directed_network_probas(dist, radii,
                                                     intercept[0],
                                                     intercept[1])
                    probas = probas[nondiag_indices][self.nan_mask_]

                    Y_new = np.zeros_like(self.Y_fit_)
                    Y_new[nondiag_indices] = self.Y_fit_[nondiag_indices]
                    Y_new[nondiag_indices][self.nan_mask_] = rng.binomial(1, probas)
                    self.Y_fit_ = Y_new
                else:
                    eta = intercept - dist[triu_indices][self.nan_mask_]
                    Y_new = np.zeros_like(self.Y_fit_)
                    Y_new[triu_indices] = self.Y_fit_[triu_indices]

                    y_ij = rng.binomial(1, expit(eta))
                    Y_new[triu_indices][self.nan_mask_] = y_ij
                    self.Y_fit_ = Y_new + np.transpose(Y_new, axes=(0, 2, 1))

                    if it > self.n_burn_:
                        self.missings_ += y_ij

            # store sample
            self.Xs_[it] = X
            self.intercepts_[it] = intercept
            self.mus_[it] = mu
            self.sigmas_[it] = sigma
            self.zs_[it] = z
            self.init_weights_[it] = init_weights
            self.trans_weights_[it] = trans_weights
            self.lambdas_[it] = lmbda
            if self.is_directed:
                self.radiis_[it] = radii

            # calculat log-joint
            if self.is_directed:
                self.logps_[it] = self.logp(
                    X, intercept, mu, sigma, z, init_weights, trans_weights,
                    lmbda, radii=radii, dist=dist)
            else:
                self.logps_[it] = self.logp(
                    X, intercept, mu, sigma, z, init_weights, trans_weights,
                    lmbda, dist=dist)

        # apply thinning if necessary
        if self.thin is not None:
            self.Xs_ = self.Xs_[::self.thin]
            self.intercepts_ = self.intercepts_[::self.thin]
            self.mus_ = self.mus_[::self.thin]
            self.sigmas_ = self.sigmas_[::self.thin]
            self.zs_ = self.zs_[::self.thin]
            self.init_weights_ = self.init_weights_[::self.thin]
            self.trans_weights_ = self.trans_weights_[::self.thin]
            self.lambdas_ = self.lambdas_[::self.thin]
            self.logps_ = self.logps_[::self.thin]
            if self.is_directed:
                self.radiis_ = self.radiis_[::self.thin]

        # perform model selection
        n_burn = self.n_burn_

        # calculate coocurrence probabilities
        self._calculate_posterior_cooccurrences()

        # Store posterior mode estimates
        if self.selection_type == 'map':
            best_id = np.argmax(self.logps_[n_burn:])
        else:
            best_id = minimize_posterior_expected_vi(self)

        self.logp_ = self.logps_[best_id]
        self.X_ = self.Xs_[best_id]
        self.intercept_ = self.intercepts_[best_id]
        self.lambda_ = self.lambdas_[best_id]
        if self.is_directed:
            self.radii_ = self.radiis_[best_id]
        self.z_ = self.zs_[best_id]
        self.init_weight_ = self.init_weights_[best_id]
        self.trans_weight_ = self.trans_weights_[best_id]
        self.mu_ = self.mus_[best_id]
        self.sigma_ = self.sigmas_[best_id]
        self.selected_id_ = best_id

        # Procrustes: rotate to reference position (best model)
        for idx in range(self.Xs_.shape[0]):
            # NOTE: Means should be rotated as well.. How to do this
            #       since they are constant over time?
            self.Xs_[idx] = longitudinal_procrustes_rotation(
                self.X_, self.Xs_[idx])

        # store posterior means
        self.X_mean_ = self.Xs_[n_burn:].mean(axis=0)
        self.lambda_mean_ = self.lambdas_[n_burn:].mean(axis=0)
        self.intercepts_mean_ = self.intercepts_[n_burn:].mean(axis=0)
        if self.is_directed:
            self.radii_mean_ = self.radiis_[n_burn:].mean(axis=0)

        if sample_missing:
            self.missings_ /= (self.n_iter - self.n_burn_)

        return self

    def _calculate_posterior_cooccurrences(self):
        n_time_steps, n_nodes, _ = self.Y_fit_.shape

        self.cooccurrence_probas_ = np.zeros((n_time_steps, n_nodes, n_nodes))
        for t in range(n_time_steps):
            self.cooccurrence_probas_[t] = calculate_posterior_cooccurrence(
                self, t=t)

    def logp(self, X, intercept, mu, sigma, z, init_weights, trans_weights,
             lmbda, radii=None, dist=None):
        n_time_steps, n_nodes, _ = X.shape

        # initial distribution (w0) log-likelihood
        loglik = dirichlet_logpdf(
            init_weights, self.dirichlet_prior_ * np.ones(self.n_components))

        # transition probabilities (w) log-likelihood
        for k in range(self.n_components):
            loglik += dirichlet_logpdf(
                trans_weights[k], self.dirichlet_prior_ * np.ones(self.n_components))

        # log-likelihood of each node's label markov chain
        for i in range(n_nodes):
            loglik += np.log(init_weights[z[0, i]])
            for t in range(1, n_time_steps):
                loglik += np.log(trans_weights[z[t-1, i], z[t, i]])

        # network log-likelihood
        if self.is_directed:
            if self.case_control_sampler_ is not None:
                loglik += approx_directed_network_loglikelihood(
                    X,
                    radii=radii,
                    in_edges=self.case_control_sampler_.in_edges_,
                    out_edges=self.case_control_sampler_.out_edges_,
                    degree=self.case_control_sampler_.degrees_,
                    control_nodes=self.case_control_sampler_.control_nodes_out_,
                    intercept_in=intercept[0],
                    intercept_out=intercept[1],
                    squared=False)
            else:
                loglik += dynamic_network_loglikelihood_directed(
                                self.Y_fit_, X,
                                intercept_in=intercept[0],
                                intercept_out=intercept[1],
                                radii=radii, dist=dist)
        else:
            loglik += dynamic_network_loglikelihood_undirected(self.Y_fit_, X,
                                                               intercept,
                                                               dist=dist)

        # intercept prior
        if self.is_directed:
            diff = intercept - self.intercept_prior
            loglik -= np.sum(0.5 * (diff * diff) /
                             self.intercept_variance_prior)
        else:
            diff = intercept - self.intercept_prior
            loglik -= 0.5 * (diff * diff) / self.intercept_variance_prior

        # latent position log-likelihood
        for t in range(n_time_steps):
            if t == 0:
                diff = X[t] - mu[z[t]]
            else:
                diff = X[t] - (1 - lmbda) * X[t-1] - lmbda * mu[z[t]]
            loglik += np.sum(-0.5 * np.log(sigma[z[t]]) -
                             0.5 * np.sum(diff * diff, axis=1) / sigma[z[t]])

        # cluster means log-likelihood
        for k in range(self.n_components):
            loglik -= 0.5 * np.sum(mu[k] ** 2) / self.mean_variance_prior_

        # cluster sigmas log-likelihood
        loglik += np.sum(-(0.5 * self.a + 1) * np.log(sigma[z]) -
                         (0.5 * self.b_ / sigma[z]))

        # lambda log-likelihood
        loglik += truncated_normal_logpdf(lmbda,
                                          mean=self.lambda_prior,
                                          var=self.lambda_variance_prior)

        # constant dirichlet normalizing factor (should cache this...)
        if self.is_directed:
            loglik += stats.dirichlet.logpdf(radii, np.ones(n_nodes))

        # hyperprior loglik
        if self.mean_variance_prior_std is not None:
            loglik += (-(0.5 * self.a0_ + 1) *
                       np.log(self.mean_variance_prior_) -
                       (0.5 * self.b0_ / self.mean_variance_prior_))
        if self.sigma_prior_std is not None:
            loglik += (self.c0_ - 1) * np.log(self.b_) - self.d0_ * self.b_

        return loglik

    def delete_traces(self):
        """Delete stored traces. Useful for storage, since the traces
        can take up a lot of space on disk.
        """
        del self.Xs_
        del self.intercepts_
        del self.zs_
        del self.mus_
        del self.sigmas_
        del self.init_weights_
        del self.trans_weights_
        del self.lambdas_
        del self.logps_

        if self.is_directed:
            del self.radiis_
