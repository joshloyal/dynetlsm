import numpy as np
import scipy.stats as stats

from scipy.optimize import minimize
from scipy.special import expit
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state, check_array
from tqdm import tqdm

from .case_control_likelihood import DirectedCaseControlSampler
from .static_network_fast import partial_loglikelihood
from .latent_space import calculate_distances, generalized_mds
from .latent_space import initialize_radii
from .sample_latent_positions import sample_latent_positions
from .sample_coefficients import sample_intercepts, sample_radii
from .metropolis import Metropolis
from .metrics import network_auc
from .network_likelihoods import (
    dynamic_network_loglikelihood_directed,
    dynamic_network_loglikelihood_undirected,
    directed_network_probas,
    directed_partial_loglikelihood,
    approx_directed_partial_loglikelihood,
    approx_directed_network_loglikelihood,
    directed_intercept_grad
)
from .procrustes import longitudinal_procrustes_rotation
from .imputer import SimpleNetworkImputer
from .array_utils import (
    triu_indices_from_3d, diag_indices_from_3d, nondiag_indices_from_3d)


__all__ = ['DynamicNetworkLSM']


def undirected_intercept_grad(Y, X, intercept, squared=False, dist=None):
    dist = calculate_distances(X, squared=squared) if dist is None else dist
    eta = intercept - dist
    grad = Y - np.exp(eta) / (1 + np.exp(eta))
    return 0.5 * (np.sum(grad) - np.einsum('ikk', grad).sum())


def scale_grad(Y, X, intercept, scale, squared=False, dist=None):
    dist = calculate_distances(X, squared=squared) if dist is None else dist
    scaled_dist = np.exp(scale) * dist
    eta = intercept - scaled_dist
    grad = -scaled_dist * (Y - np.exp(eta) / (1 + np.exp(eta)))
    return np.sum(grad) - np.einsum('ikk', grad).sum()


def scale_intercept_mle(Y, X, squared=False, tol=1e-4):
    """Compute the conditional MLE of the intercept term."""
    dist = calculate_distances(X, squared=squared)

    def logp(x):
        scale, intercept = x[0], x[1]
        scaled_dist = np.exp(scale) * dist
        return -dynamic_network_loglikelihood_undirected(Y, X, intercept,
                                                         squared=squared,
                                                         dist=scaled_dist)

    def grad(x):
        scale, intercept = x[0], x[1]
        scaled_dist = np.exp(scale) * dist
        return -np.array([scale_grad(Y, X, intercept, scale, squared=squared,
                                     dist=dist),
                          undirected_intercept_grad(Y, X, intercept,
                                                    squared=squared,
                                                    dist=scaled_dist)])

    result = minimize(fun=logp, x0=np.array([0.0, 1.0]),
                      method='BFGS', jac=grad, tol=tol)

    return result.x[0], result.x[1]


def directed_intercept_mle(Y, X, radii, intercept_init=None, squared=False,
                           tol=1e-4):
    """Conditional MLE for intercept_in and intercept_out"""
    dist = calculate_distances(X, squared=squared)

    def logp(x):
        intercept_in, intercept_out = x[0], x[1]
        return -dynamic_network_loglikelihood_directed(Y, X,
                                                       intercept_in,
                                                       intercept_out,
                                                       radii,
                                                       squared=squared,
                                                       dist=dist)


    def grad(x):
        intercept_in, intercept_out = x[0], x[1]
        return -directed_intercept_grad(Y, dist=dist, radii=radii,
                                        intercept_in=intercept_in,
                                        intercept_out=intercept_out)

    x0 = intercept_init if intercept_init is not None else np.array([0.0, 0.0])
    result = minimize(fun=logp, x0=x0,
                      method='BFGS', jac=grad, tol=tol)

    return result.x[0], result.x[1]


class DynamicNetworkLatentSpace(BaseEstimator):
    def __init__(self,
                 n_iter=2500,
                 is_directed=False,
                 squared=False,
                 n_features=2,
                 tau_sq='auto',
                 sigma_sq=0.001,
                 step_size_X=0.0075,
                 step_size_intercept=0.1,
                 intercept_prior='auto',
                 intercept_variance_prior=2.0,
                 step_size_radii=175000,
                 tune=500,
                 burn=1000,
                 n_control=None,
                 n_resample_control=100,
                 copy=True,
                 random_state=None):
        self.n_iter = n_iter
        self.is_directed = is_directed
        self.squared = squared
        self.n_features = n_features
        self.tau_sq = tau_sq
        self.sigma_sq = sigma_sq
        self.step_size_X = step_size_X
        self.intercept_prior = intercept_prior
        self.intercept_variance_prior = intercept_variance_prior
        self.step_size_intercept = step_size_intercept
        self.step_size_radii = step_size_radii
        self.tune = tune
        self.burn = burn
        self.n_control = n_control
        self.n_resample_control = n_resample_control
        self.copy = copy
        self.random_state = random_state

    @property
    def distances_(self):
        if not hasattr(self, 'X_'):
            raise ValueError('Model not fit.')
        return calculate_distances(self.X_, squared=self.squared)

    @property
    def probas_(self):
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
    def auc_(self):
        ## FIXME: This should mask nan values
        if not hasattr(self, 'X_'):
            raise ValueError('Model not fit.')
        return network_auc(self.Y_fit_, self.probas_,
                           is_directed=self.is_directed)

    def fit(self, Y):
        n_time_steps, n_nodes, _ = Y.shape
        rng = check_random_state(self.random_state)

        Y = check_array(Y, dtype=np.float64,
                        force_all_finite='allow-nan',
                        ensure_2d=False, allow_nd=True, copy=self.copy)

        # XXX: self.Y_fit_ could point to Y, which may be changed...
        # perform imputation and sample missing edges
        if self.is_directed:
            nondiag_indices = nondiag_indices_from_3d(Y)
            nan_mask = np.isnan(Y[nondiag_indices])
        else:
            triu_indices = triu_indices_from_3d(Y, k=1)
            nan_mask = np.isnan(Y[triu_indices])
        sample_missing = np.any(nan_mask)
        if sample_missing:
            self.Y_fit_ = SimpleNetworkImputer().fit_transform(Y)
        else:
            self.Y_fit_ = Y

        n_iter_procrustes = 0
        if self.tune is not None:
            self.n_iter += self.tune
            n_iter_procrustes += self.tune
        if self.burn is not None:
            self.n_iter += self.burn
            n_iter_procrustes += self.burn

        # initialize variables
        self.Xs_ = np.zeros(
            (self.n_iter, n_time_steps, n_nodes, self.n_features),
            dtype=np.float64
        )

        if self.is_directed:
            # intercept = [beta_in, beta_out]
            self.intercepts_ = np.zeros((self.n_iter, 2), dtype=np.float64)

            # radii
            self.radiis_ = np.zeros((self.n_iter, n_nodes), dtype=np.float64)
        else:
            self.intercepts_ = np.zeros((self.n_iter, 1), dtype=np.float64)

        # initialize latent positions through GMDS
        X = generalized_mds(self.Y_fit_, n_features=self.n_features,
                            is_directed=self.is_directed,
                            random_state=rng)

        if self.is_directed:
            # initialize radii
            radii = initialize_radii(self.Y_fit_)
            intercept_in, intercept_out = directed_intercept_mle(
                                                self.Y_fit_,
                                                X, radii,
                                                squared=self.squared)
            intercept = np.array([intercept_in, intercept_out])
        else:
            scale, intercept = scale_intercept_mle(self.Y_fit_, X,
                                                   squared=self.squared)
            intercept = np.array([intercept])

            # rescale initial latent positions
            X *= np.exp(scale)

        # center latent space across time
        X -= np.mean(X, axis=(0, 1))

        if self.tau_sq == 'auto':
            self.tau_sq = np.mean(X[0] * X[0])

        if self.intercept_prior == 'auto':
            self.intercept_prior = intercept.copy()

        # store initial values
        self.Xs_[0] = X
        self.intercepts_[0] = intercept

        if self.is_directed:
            self.radiis_[0] = radii

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

        # init log-posteriors
        self.logps_ = np.zeros(self.n_iter, dtype=np.float64)

        if self.is_directed:
            self.logps_[0] = self.logp(self.Y_fit_, X, intercept, radii=radii)
        else:
            self.logps_[0] = self.logp(self.Y_fit_, X, intercept)

        # set current MAP estimates
        self.logp_ = self.logps_[0]
        self.X_ = X
        self.intercept_ = intercept
        if self.is_directed:
            self.radii_ = radii

        # initialize metropolis samplers
        self.latent_samplers = []
        for t in range(n_time_steps):
            self.latent_samplers.append(
                [Metropolis(step_size=self.step_size_X, tune=self.tune,
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
                                            tune=None,
                                            proposal_type='dirichlet')

        for it in tqdm(range(1, self.n_iter)):
            X = self.Xs_[it - 1].copy()
            intercept = self.intercepts_[it - 1].copy()
            radii = self.radiis_[it - 1].copy() if self.is_directed else None

            # re-sample control group if necessary
            if self.case_control_sampler_ is not None:
                self.case_control_sampler_.resample()

            X = sample_latent_positions(
                    self.Y_fit_, X, intercept=intercept,
                    radii=radii,
                    tau_sq=self.tau_sq,
                    sigma_sq=self.sigma_sq,
                    samplers=self.latent_samplers,
                    is_directed=self.is_directed,
                    squared=self.squared,
                    case_control_sampler=self.case_control_sampler_,
                    random_state=rng)

            # perform a procrustes transformation
            if it > n_iter_procrustes:
                prev_map = np.argmax(self.logps_[:(n_iter_procrustes+1)])
                X_ref = self.Xs_[prev_map]
                X = longitudinal_procrustes_rotation(X_ref, X)

            # center latent space across time
            X -= np.mean(X, axis=(0, 1))

            # cache new distances
            dist = (None if self.case_control_sampler_ else
                        calculate_distances(X, squared=self.squared))

            # sample intercepts
            intercept = sample_intercepts(
                        self.Y_fit_, X, intercept,
                        intercept_prior=self.intercept_prior,
                        intercept_variance_prior=self.intercept_variance_prior,
                        samplers=self.intercept_samplers, radii=radii,
                        dist=dist, is_directed=self.is_directed,
                        case_control_sampler=self.case_control_sampler_,
                        squared=self.squared, random_state=rng)

            # sample radii for directed networks
            if self.is_directed:
                radii = sample_radii(
                            self.Y_fit_, X, intercepts=intercept, radii=radii,
                            sampler=self.radii_sampler, dist=dist,
                            case_control_sampler=self.case_control_sampler_,
                            squared=self.squared, random_state=rng)

            # sample missing edges
            if sample_missing:
                # calculate pij for missing edges and sample from Bern(pij)
                # This is sampling everything! just sample missing...
                if self.is_directed:
                    probas = directed_network_probas(dist, radii,
                                                     intercept[0],
                                                     intercept[1])
                    probas = probas[nondiag_indices][nan_mask]

                    Y_new = np.zeros_like(self.Y_fit_)
                    Y_new[nondiag_indices] = self.Y_fit_[nondiag_indices]
                    Y_new[nondiag_indices][nan_mask] = rng.binomial(1, probas)
                    self.Y_fit_ = Y_new
                else:
                    eta = intercept - dist[triu_indices][nan_mask]

                    Y_new = np.zeros_like(self.Y_fit_)
                    Y_new[triu_indices] = self.Y_fit_[triu_indices]
                    Y_new[triu_indices][nan_mask] = rng.binomial(1, expit(eta))
                    self.Y_fit_ = Y_new + np.transpose(Y_new, axes=(0, 2, 1))


            # check if current MAP
            if self.is_directed:
                self.logps_[it] = self.logp(self.Y_fit_, X, intercept,
                                            radii=radii, dist=dist)
            else:
                self.logps_[it] = self.logp(self.Y_fit_, X, intercept, dist)

            if self.tune and (it == (self.tune + self.burn)):
                self.logp_ = self.logps_[it]
                self.X_ = X
                self.intercept_ = intercept
                if self.is_directed:
                    self.radii_ = radii

            elif self.logps_[it] > self.logp_:
                self.logp_ = self.logps_[it]
                self.X_ = X
                self.intercept_ = intercept
                if self.is_directed:
                    self.radii_ = radii

            # save samples
            self.Xs_[it] = X
            self.intercepts_[it] = intercept
            if self.is_directed:
                self.radiis_[it] = radii

        return self

    def logp(self, Y, X, intercept, radii=None, dist=None):
        n_time_steps, n_nodes, _ = Y.shape

        # network log-likelihood
        if self.is_directed:
            if self.case_control_sampler_ is not None:
                loglik = approx_directed_network_loglikelihood(
                    X,
                    radii=radii,
                    in_edges=self.case_control_sampler_.in_edges_,
                    out_edges=self.case_control_sampler_.out_edges_,
                    degree=self.case_control_sampler_.degrees_,
                    control_nodes=self.case_control_sampler_.control_nodes_out_,
                    intercept_in=intercept[0],
                    intercept_out=intercept[1],
                    squared=self.squared)
            else:
                loglik = dynamic_network_loglikelihood_directed(Y, X,
                                                                intercept_in=intercept[0],
                                                                intercept_out=intercept[1],
                                                                radii=radii,
                                                                squared=self.squared,
                                                                dist=dist)
        else:
            loglik = dynamic_network_loglikelihood_undirected(Y, X, intercept,
                                                              squared=self.squared,
                                                              dist=dist)

        # latent space priors
        for t in range(n_time_steps):
            if t == 0:
                loglik -= np.sum(0.5 * np.sum(X[t] * X[t], axis=1) / self.tau_sq)
            else:
                diff = X[t] - X[t-1]
                loglik -= np.sum(0.5 * np.sum(diff * diff, axis=1) / self.sigma_sq)

        # intercept prior
        if self.is_directed:
            diff = intercept - self.intercept_prior
            loglik -= np.sum(0.5 * (diff * diff) / self.intercept_variance_prior)
        else:
            diff = intercept[0] - self.intercept_prior
            loglik -= 0.5 * (diff * diff) / self.intercept_variance_prior

        return loglik
