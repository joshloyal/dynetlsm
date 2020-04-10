import numpy as np
import scipy.stats as stats

from scipy.optimize import minimize
from scipy.special import expit
from sklearn.utils import check_random_state, check_array
from tqdm import tqdm

from .array_utils import (
    triu_indices_from_3d, diag_indices_from_3d, nondiag_indices_from_3d)
from .case_control_likelihood import DirectedCaseControlSampler
from .imputer import SimpleNetworkImputer
from .latent_space import calculate_distances, generalized_mds
from .latent_space import initialize_radii
from .metrics import network_auc
from .metropolis import Metropolis
from .network_likelihoods import (
    dynamic_network_loglikelihood_directed,
    dynamic_network_loglikelihood_undirected,
    directed_network_probas,
    approx_directed_network_loglikelihood,
    directed_intercept_grad
)
from .procrustes import longitudinal_procrustes_rotation
from .sample_latent_positions import sample_latent_positions
from .sample_coefficients import sample_intercepts, sample_radii


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


class DynamicNetworkLSM(object):
    """A latent space model for dynamic networks [1].

    Originally proposed by Hoff et. al. [2], latent space models (LSM) embed a
    network's nodes within a Euclidean latent space. Closeness in the latent
    space increases the probability that two actors form an edge in the
    observed network. The LSM's popularity stems from the intuitive meaning
    of its embeddings and its ability to naturally incorporate desirable
    sociological features such as homophily, reciprocity, and transitivity.

    Sewell and Chen [1] extended the original latent space model to the
    dynamic network setting by allowing the nodes to move around the latent
    space according to Markovian dynamics.

    Parameters
    ----------
    n_features : int (default=2)
        The number of latent features. This is the dimension of the Euclidean
        latent space.

    is_directed : bool (default=False)
        Whether the network is directed or undirected.

    n_iter : int (default=5000)
        Number of iterations after tuning and burn-in to run the
        Markov chain Monte Carlo (MCMC) sampler. Total number of iterations
        is equal to `tune + burn + n_iter`.

    tune : int (default=2500)
        Number of iterations used to tune the step sizes of the
        metropolis-hastings samplers.

    tune_interval : int (default=100)
        Number of iterations to wait before adjusting the random-walk
        step sizes during the tuning phase.

    burn : int (default=2500)
        Number of iterations used for burn-in after the tuning phase.

    intercept_prior : float or str, optional (default='auto')
        The mean of the normal prior placed on the intercept parameter.
        If 'auto' the prior mean is set to the value inferred during
        initialization (MAP).

    intercept_variance_prior : float, optional (default=2)
        The variance of the normal prior placed on the intercept parameter

    tau_sq : float (default=2.0)
        The initial latent positions have a normal(0, tau_sq) prior. This
        controls the size of the latent space.

    sigma_sq : float (default=0.1)
        At subsequent time points, the latent positions are drawn from a
        X_i(t+1) ~ normal(X_it, sigma_sq) distribution. sigma_sq controls
        the step size of the latent positions. Larger values means that
        the latent positions can move further at each time step.

    step_size_X : float or str, optional (default=0.1)
        Initial step size of the random-walk metropolis sampler for the latent
        positions.

    step_size_intercept : float, optional (default=0.1)
        Initial step size for the random-walk metropolis sampler for the
        intercepts.

    step_size_radii : float, optional (default=175000)
        Initial step size for the metropolis-hastings sampler with a
        Dirichlet(step_size_radii * radii) proposal used to sample the radii
        parameters in the case of directed networks. Larger values correspond
        to smaller step sizes.

    n_control : int or None, optional (default=None)
        Number of nodes to sample when using the case-control likelihood. This
        is still experimental. Allows the algorithm to scale linearly instead
        of quadratically. If None then the case-control likelihood is not used.
        Only implemented for directed networks.

    n_resample_control : int, optional (default=1000)
        The number of iterations to wait before re-sampling the control nodes
        when the case-control likelihood is utilized.

    copy : bool, optional (default=True)
        Whether to copy the dynamic network when manipulating the network
        internally.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Attributes
    ----------
    X_ : array-like, shape (n_time_steps, n_nodes, n_features)
        MAP estimate of the latent positions.

    intercept_ : array-like, shape (1,) or (2,)
        MAP estimate of the intercepts. If `is_directed` is False a
        single intercept is estimated. Otherwise, two intercepts are estimated
        for incoming and outgoing connections, respectively.

    radii_ : array-like, shape (n_nodes,)
        MAP estimates of the social radius of each node. Only available if
        `is_directed`=True.

    Xs_ : array-like, shape (n_iter, n_time_steps, n_nodes, n_features)
        Posterior samples of the latent positions.


    intercepts_ :  array-like, shape (n_iter, 1) or (n_iter, 2)
        Posterior samples of the intercepts.

    radiis_ :  array-like, shape (n_iter, n_nodes)
        Posterior samples of the social radii.

    Examples
    --------

    >>> from dynetlsm import DynamicNetworkLSM
    >>> from dynetlsm.datasets import load_monks
    >>> Y, _, _ = load_monks(is_directed=False)
    >>> Y.shape
    (3, 18, 18)
    >>> model = DynamicNetworkLSM(n_iter=250, burn=250, tune=250).fit(Y)

    References
    ----------
    [1] Sewell, Daniel K., and Chen, Yuguo (2016). "Latent space models for
        dynamic networks". Journal of the American Statistical Association,
        110(512):1646-1657.
    [2] Hoff, P.D., Raftery, A. E., and Handcock, M.S. (2002). "Latent
        space approaches to social network analysis". Journal of the
        American Statistical Association, 97(460):1090-1098.
    """
    def __init__(self,
                 n_features=2,
                 is_directed=False,
                 n_iter=5000,
                 tune=2500,
                 tune_interval=100,
                 burn=2500,
                 intercept_prior='auto',
                 intercept_variance_prior=2.0,
                 tau_sq=2.0,
                 sigma_sq=0.1,
                 step_size_X=0.1,
                 step_size_intercept=0.1,
                 step_size_radii=175000,
                 n_control=None,
                 n_resample_control=100,
                 copy=True,
                 random_state=None):
        self.n_iter = n_iter
        self.is_directed = is_directed
        self.n_features = n_features
        self.tau_sq = tau_sq
        self.sigma_sq = sigma_sq
        self.step_size_X = step_size_X
        self.intercept_prior = intercept_prior
        self.intercept_variance_prior = intercept_variance_prior
        self.step_size_intercept = step_size_intercept
        self.step_size_radii = step_size_radii
        self.tune = tune
        self.tune_interval = tune_interval
        self.burn = burn
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

        return n_burn

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
    def auc_(self):
        """In-sample AUC of the final estimated model."""
        # FIXME: This should mask nan values
        if not hasattr(self, 'X_'):
            raise ValueError('Model not fit.')
        return network_auc(self.Y_fit_, self.probas_,
                           is_directed=self.is_directed)

    def fit(self, Y):
        """Sample from the posterior of the latent space model for dynamic
        networks based on an observed dynamic network Y.

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
        self : DynamicNetworkLSM
            Fitted estimator.
        """
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
                                                squared=False)
            intercept = np.array([intercept_in, intercept_out])
        else:
            scale, intercept = scale_intercept_mle(self.Y_fit_, X,
                                                   squared=False)
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
                            tune_interval=self.tune_interval,
                            proposal_type='random_walk') for
                 _ in range(n_nodes)])

        if self.is_directed:
            self.intercept_samplers = [
                Metropolis(step_size=self.step_size_intercept,
                           tune=self.tune, tune_interval=self.tune_interval,
                           proposal_type='random_walk') for _ in range(2)]
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
                    squared=False,
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
                    squared=False)
            else:
                loglik = dynamic_network_loglikelihood_directed(
                    Y, X,
                    intercept_in=intercept[0],
                    intercept_out=intercept[1],
                    radii=radii,
                    squared=False,
                    dist=dist)
        else:
            loglik = dynamic_network_loglikelihood_undirected(
                Y, X, intercept,
                squared=False,
                dist=dist)

        # latent space priors
        for t in range(n_time_steps):
            if t == 0:
                loglik -= np.sum(
                    0.5 * np.sum(X[t] * X[t], axis=1) / self.tau_sq)
            else:
                diff = X[t] - X[t-1]
                loglik -= np.sum(
                    0.5 * np.sum(diff * diff, axis=1) / self.sigma_sq)

        # intercept prior
        if self.is_directed:
            diff = intercept - self.intercept_prior
            loglik -= (
                np.sum(0.5 * (diff * diff) / self.intercept_variance_prior))
        else:
            diff = intercept[0] - self.intercept_prior
            loglik -= 0.5 * (diff * diff) / self.intercept_variance_prior

        return loglik
