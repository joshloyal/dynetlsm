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
from .sample_auxillary import sample_tables, sample_mbar
from .sample_coefficients import sample_intercepts, sample_radii
from .sample_concentration import sample_concentration_param
from .sample_labels import sample_labels_block, sample_labels_gibbs
from .sample_latent_positions import sample_latent_positions_mixture
from .trace_utils import geweke_diag


SMALL_EPS = np.finfo('float64').tiny


__all__ = ['DynamicNetworkHDPLPCM']


def init_sampler(Y, is_directed=False,
                 n_iter=100, n_features=2, n_components=10,
                 gamma=1.0, alpha=1.0, kappa=4.0,
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
    if sample_missing:
        nan_mask = np.isnan(Y)
        Y[nan_mask] = dynamic_emb.probas_[nan_mask] > 0.5

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
    weights = np.zeros((n_iter, n_time_steps, n_components, n_components),
                       dtype=np.float64)
    resp = np.zeros((n_nodes, n_components))
    resp[np.arange(n_nodes), zs[0]] = 1
    nk = resp.sum(axis=0)
    weights[0, 0, 0] = nk / n_nodes  # convenction store w0 in w[0, 0]

    # blending coefficient set to initial value
    lambdas = np.zeros((n_iter, 1), dtype=np.float64)
    lambdas[0] = lambda_init

    # initialize beta by sampling from the prior
    rng = check_random_state(random_state)
    betas = np.zeros((n_iter, n_components), dtype=np.float64)
    betas[0] = rng.dirichlet(np.repeat(gamma / n_components, n_components))

    # initialize transition distributions by sampling from beta prior
    dir_alpha = alpha * betas[0]
    for t in range(1, n_time_steps):
        for k in range(n_components):
            wtk = rng.dirichlet(dir_alpha + kappa * np.eye(n_components)[k])
            weights[0, t, k] = wtk

    return Xs, intercepts, mus, sigmas, zs, betas, weights, lambdas, radiis, Y


class DynamicNetworkHDPLPCM(object):
    """The HDP Latent Position Clustering Model (HDP-LPCM) [1].

    The hierarchal Dirichlet process latent position clustering model
    (HDP-LPCM) is a Bayesian nonparametric model for inferring community
    structure in dynamic (time-varying) networks.

    Based on the latent space model of Hoff et. al. [2], the HDP-LPCM embeds
    the nodes of the network in a latent Euclidean space. The probability of two
    nodes forming an edge in the network is proportional to the node's distance
    in this space. Community structure in the network is a natural result of
    nodes clustering within this space. To infer this structure, the
    distribution of the nodes is assumed to follow a Gaussian mixture model.

    The dynamics of the network are the result of nodes moving around the
    latent space. These movements are assumed to be Markovian in time.
    The communities still result from nodes clustering together; however,
    nodes may move between clusters over-time. In addition, new clusters
    may form or old clusters may die out as time progresses. Such dynamics
    are naturally modeled with an autoregressive hidden Markov model (AR-HMM).
    The number of hidden states (or communities) is inferred nonparametrically
    with a sticky hierarchical Dirichlet process (HDP).

    Parameters
    ----------
    n_features : int (default=2)
        The number of latent features. This is the dimension of the Euclidean
        latent space.

    n_components : int (default=10)
        An upper-bound on the number of latent communities. This is the number
        of components used by the weak-limit approximation to the HDP.
        The number of estimated communities may be much smaller.

    is_directed : bool (default=False)
        Whether the network is directed or undirected.

    selection_type : {'vi', 'bic', 'map'} (default='vi')
        String describing the model selection method. Must be one of::

            'vi': select the model that minimizes the posterior expected
                  variation of information (VI),
            'bic': select the model that minimizes the Bayesian information
                   criterion (BIC),
            'map': select the maximum a-posterior (MAP) estimate,

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

    thin : int or None (default=None)
        Thinning interval. Every `thin` samples are stored. If `None` then
        no thinning is performed. To save memory if the number of iteration is
        large.

    gamma : float, optional (default=1.0)
        Concentration parameter of the top-level Dirichlet proccess (DP)
        in the hierarchical Dirichlet process (HDP). Higher values put
        more mass on a larger number of communities.

    alpha_init : float, optional (default=1.0)
        Concentration parameter for the DP governing the initial distribution
        of the hidden Markov model. Larger values put more mass on a
        larger number of initial communities.

    alpha : float, optional (default=1.0)
        Concentration parameter for the DPs governing the transition
        distributions. Larger values put more mass on a larger number of
        communities.

    kappa : float, optional (default=4.0)
        Stickyness parameter of the sticky-HDP. Larger values put more
        mass on self-transitions, i.e., nodes are more likely to remain
        in the same group over time.

    intercept_prior : float or str, optional (default='auto')
        The mean of the normal prior placed on the intercept parameter.
        If 'auto' the prior mean is set to the value inferred during
        initialization.

    intercept_variance_prior : float, optional (default=2)
        The variance of the normal prior placed on the intercept parameter

    mean_variance_prior : float or str, optional (default='auto')
        The cluster means have a normal(0, tau_sq) prior, where
        E[tau_sq] = mean_variance_prior. Larger values increase the size of
        the latent space. If 'auto' then prior parameters are chosen such that
        sqrt(Var(tau_sq)) = mean_variance_prior_std * E(tau_sq).

    a : float, optional (default=2)
        The cluster variances have a InvGamma(a/2, b/2) prior.
        This is the shape parameter a.

    b : float or str, optional (default='auto')
        The cluster variances have a InvGamma(a/2, b/2) prior.
        This is the scale parameter b.

    lambda_prior : float, optional (default=0.9)
        This value must be between 0 and 1. The blending coefficient has a
        normal(lambda_prior, lambda_variance_prior) prior truncated to
        the range (0, 1). This is the prior mean. Larger values result in the
        dynamics of the nodes being more influenced by their assigned clusters.

    lambda_variance_prior : float, optional (default=0.01)
        The blending coefficient has a
        normal(lambda_prior, lambda_variance_prior) prior truncated to
        the range (0, 1). This is the prior variance. Larger values result in
        the dynamics of the nodes being more influenced by their assigned
        clusters.

    sigma_prior_std : float, optional (default=4.0)
        The standard deviation of the prior on the cluster shapes. Used
        to select reasonable priors.

    mean_variance_prior_std : float, optional (default='auto')
        The standard deviation for the prior on tau_sq. Used to automatically
        select reasonable priors.

    step_size_X : float or str, optional (default='auto')
        Initial step size of the random-walk metropolis sampler for the latent
        postilions.

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
        The latent position of each node at every time-step.

    intercept_ : array-like, shape (1,) or (2,)
        The intercept of the model. If `is_directed` is False a single intercept
        is estimated. Otherwise, two intercepts are estimated for incoming and
        outgoing connections, respectively.

    radii_ : array-like, shape (n_nodes,)
        The social radius of each node. Only available if `is_directed`=True.

    z_ : array-like, shape (n_time_steps, n_nodes)
        The cluster assignments of each node for every time-step.

    mu_ : array-like, shape (n_components, n_features)
        The mean of each mixture component.

    sigma_ : array-like, shape (n_components,)
        The variance of each mixture component.

    lambda_ : float
        The estimated blending coefficient.

    init_weight_ : array-like, shape (n_components,)
        The initial distribution over mixture components.

    trans_weights_ : array-like,
                     shape (n_time_steps - 1, n_components, n_components)
        The transition probabilities between mixture components at
        each time-step.

    beta_ : array-like, shape (n_components,)
        Inferred expected transition probabilities, i.e., the transition
        probabilities are drawn from a DP(alpha * beta) distribution.

    coocurrence_probabilities_ : array-like,
                                 shape (n_time_steps, n_nodes, n_nodes)
        The posterior cooccurrence probabilities at each time step. This
        is the probability that node i and node j are in the same community
        at time t.

    Xs_ :  array-like, shape (n_iter, n_time_steps, n_nodes, n_features)
        Posterior samples of the latent positions.

    intercepts_ : array-like, shape (n_iter, 1) or (n_iter, 2)
        Posterior samples of the intercept parameter.

    radiis_ : array-like, shape (n_iter, n_nodes)
        Posterior samples of each node's social radius.

    lambdas_ : array-like, shape (n_iter,)
        Posterior samples of the blending coefficient.

    Examples
    --------

    >>> from dynetlsm import DynamicNetworkHDPLPCM
    >>> from dynetlsm.datasets import load_monks
    >>> Y, _, _ = load_monks(is_directed=False)
    >>> Y.shape
    (3, 18, 18)
    >>> model = DynamicNetworkHDPLPCM(n_iter=250, burn=250, tune=250,
    ...                               n_features=2, n_components=10).fit(Y)
    >>> model.X_.shape
    (3, 18, 2)

    References
    ----------
    [1] Loyal, Joshua D., and Chen, Yuguo (2020). "A Bayesian nonparametric
        latent space approach to modeling evolving communities in dynamic
        networks", arXiv:2003.07404.
    [2] Hoff, P.D., Raftery, A. E., and Handcock, M.S. (2002). "Latent
        space approaches to social network analysis". Journal of the
        American Statistical Association, 97(460):1090-1098.
    """
    def __init__(self,
                 n_features=2,
                 n_components=10,
                 is_directed=False,
                 selection_type='vi',
                 n_iter=5000,
                 tune=2500,
                 tune_interval=100,
                 burn=2500,
                 thin=None,
                 gamma=1.0,
                 alpha_init=1.0,
                 alpha=1.0,
                 kappa=4.0,
                 intercept_prior='auto',
                 intercept_variance_prior=2,
                 mean_variance_prior='auto',
                 a=2.0,
                 b='auto',
                 lambda_prior=0.9,
                 lambda_variance_prior=0.01,
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
        self.n_features = n_features
        self.n_components = n_components
        self.step_size_X = step_size_X
        self.intercept_prior = intercept_prior
        self.intercept_variance_prior = intercept_variance_prior
        self.step_size_intercept = step_size_intercept
        self.mean_variance_prior = mean_variance_prior
        self.a = a
        self.b = b
        self.alpha_init = alpha_init
        self.alpha = alpha
        self.gamma = gamma
        self.kappa = kappa
        self.lambda_prior = lambda_prior
        self.lambda_variance_prior = lambda_variance_prior
        self.mean_variance_prior_std = mean_variance_prior_std
        self.sigma_prior_std = sigma_prior_std
        self.step_size_radii = step_size_radii
        self.tune = tune
        self.tune_interval = tune_interval
        self.burn = burn
        self.thin = thin
        self.selection_type = selection_type
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
    def auc_(self):
        """In-sample AUC of the final estimated model."""
        # FIXME: This should mask nan values
        if not hasattr(self, 'X_'):
            raise ValueError('Model not fit.')
        return network_auc(self.Y_fit_, self.probas_,
                           is_directed=self.is_directed)

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

        # update n_iter with iterations including burn-in and tuning
        if self.burn is not None:
            self.n_iter += self.burn
        if self.tune is not None:
            self.n_iter += self.tune

        (self.Xs_, self.intercepts_, self.mus_, self.sigmas_, self.zs_,
         self.betas_, self.weights_, self.lambdas_,
         self.radiis_, self.Y_fit_) = init_sampler(
                         Y, is_directed=self.is_directed,
                         n_iter=self.n_iter,
                         n_features=self.n_features,
                         n_components=self.n_components,
                         lambda_init=self.lambda_prior,
                         gamma=self.gamma,
                         alpha=self.alpha,
                         sample_missing=sample_missing,
                         n_control=self.n_control,
                         n_resample_control=self.n_resample_control,
                         random_state=rng)

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
                self.weights_[0], self.betas_[0],
                self.lambdas_[0], radii=self.radiis_[0])
        else:
            self.logps_[0] = self.logp(
                self.Xs_[0], self.intercepts_[0],
                self.mus_[0], self.sigmas_[0], self.zs_[0],
                self.weights_[0], self.betas_[0],
                self.lambdas_[0])
        self.logp_ = self.logps_[0]

        return self._fit(Y, rng)

    def _fit(self, Y, random_state):
        rng = check_random_state(random_state)
        n_time_steps, n_nodes, _ = self.Y_fit_.shape

        if self.is_directed:
            nondiag_indices = nondiag_indices_from_3d(Y)
            nan_mask = np.isnan(Y[nondiag_indices])
        else:
            triu_indices = triu_indices_from_3d(Y, k=1)
            nan_mask = np.isnan(Y[triu_indices])
        sample_missing = np.any(nan_mask)

        for it in tqdm(range(1, self.n_iter)):
            # copy over previous samples
            X = self.Xs_[it - 1].copy()
            intercept = self.intercepts_[it - 1].copy()
            z = self.zs_[it - 1].copy()
            mu = self.mus_[it - 1].copy()
            sigma = self.sigmas_[it - 1].copy()
            weights = self.weights_[it - 1].copy()
            beta = self.betas_[it - 1].copy()
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
            z, n, nk, resp = sample_labels_block(X, mu, sigma, lmbda, weights,
                                                 random_state=rng)

            # sample auxiliary variables
            m = sample_tables(n, beta, self.alpha_init, self.alpha, self.kappa,
                              random_state=rng)
            m_bar, w = sample_mbar(m, beta, kappa=self.kappa, alpha=self.alpha,
                                   random_state=rng)

            # sample global transition distribution (beta)
            beta = rng.dirichlet((self.gamma / self.n_components) + m_bar)

            # sample initial distribution (w0)
            weights[0, 0] = sample_dirichlet(self.alpha_init * beta + nk[0],
                                             random_state=rng)

            # sample transition distributions (w)
            probas = self.alpha * beta + self.kappa * np.eye(self.n_components)
            for t in range(1, n_time_steps):
                for k in range(self.n_components):
                    weights[t, k] = sample_dirichlet(probas[k] + n[t, k],
                                                     random_state=rng)

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

            # sample lambda (this is the problem!)
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

            # hyper-parameter samplers

            # sample gamma
            self.gamma = sample_concentration_param(
                            self.gamma,
                            n_clusters=np.sum(m_bar > 0),
                            n_samples=np.sum(m_bar),
                            prior_shape=1.0,
                            prior_scale=0.1,
                            random_state=rng)

            # sample concentration parameter of the initial distribution
            # auxillary sampler of Escobar and West (1995)
            # with k = m_{00.}
            # NOTE: a single group in the HDP results in the the same
            #       auxillary sampler as in the DP case.
            self.alpha_init = sample_concentration_param(
                                self.alpha_init,
                                n_clusters=np.sum(m[0, 0]),
                                n_samples=n_nodes,
                                prior_shape=1.0,
                                prior_scale=1.0,
                                random_state=rng)

            # sample alpha + kappa
            ak_shape, ak_scale = 5, .1
            alpha_kappa = self.alpha + self.kappa

            n_dot = np.sum(n[1:], axis=2)
            valid_indices = n_dot > 0
            valid_n_dot = n_dot[valid_indices]
            s = rng.binomial(
                    1, p=(valid_n_dot / (valid_n_dot + alpha_kappa)))
            r = rng.beta(alpha_kappa + 1, valid_n_dot)

            shape = (ak_shape +
                     np.sum(m[1:], axis=2)[valid_indices].sum() -
                     np.sum(s))
            scale = ak_scale - np.sum(np.log(r))
            alpha_kappa = rng.gamma(shape=shape, scale=1. / scale)

            # sample rho
            # prior mean ~ 0.8, highly skewed to a high stickiness
            # NOTE: other option was 10, 1
            rho_a, rho_b = 8, 2
            n_success = np.sum(w)
            rho = rng.beta(a=rho_a + n_success,
                           b=np.sum(m[1:]) - n_success + rho_b)

            self.kappa = alpha_kappa * rho
            self.alpha = alpha_kappa - self.kappa

            # sample missing data
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

            # store sample
            self.Xs_[it] = X
            self.intercepts_[it] = intercept
            self.mus_[it] = mu
            self.sigmas_[it] = sigma
            self.zs_[it] = z
            self.betas_[it] = beta
            self.weights_[it] = weights
            self.lambdas_[it] = lmbda
            if self.is_directed:
                self.radiis_[it] = radii

            # set current MAP
            if self.is_directed:
                self.logps_[it] = self.logp(X, intercept, mu, sigma, z, weights,
                                            beta, lmbda, radii=radii, dist=dist)
            else:
                self.logps_[it] = self.logp(X, intercept, mu, sigma, z, weights,
                                            beta, lmbda, dist=dist)

        # apply thinning if necessary
        if self.thin is not None:
            self.Xs_ = self.Xs_[::self.thin]
            self.intercepts_ = self.intercepts_[::self.thin]
            self.mus_ = self.mus_[::self.thin]
            self.sigmas_ = self.sigmas_[::self.thin]
            self.zs_ = self.zs_[::self.thin]
            self.betas_ = self.betas_[::self.thin]
            self.weights_ = self.weights_[::self.thin]
            self.lambdas_ = self.lambdas_[::self.thin]
            self.logps_ = self.logps_[::self.thin]
            if self.is_directed:
                self.radiis_ = self.radiis_[::self.thin]

        # perform model selection
        n_burn = self.n_burn_

        # store statistics for BIC or MAP model selection
        self.bic_, self.models_, self.counts_ = select_bic(self)

        # calculate coocurrence probabilities
        self._calculate_posterior_cooccurrences()

        # perform model selection
        if self.selection_type == 'vi':
            best_id = minimize_posterior_expected_vi(self)
            self.logp_ = self.logps_[best_id]
            self.X_ = self.Xs_[best_id]
            self.intercept_ = self.intercepts_[best_id]
            self.lambda_ = self.lambdas_[best_id]
            if self.is_directed:
                self.radii_ = self.radiis_[best_id]

            z, beta, init_w, trans_w, mu, sigma = renormalize_weights(
                self, sample_id=best_id)
            self.z_ = z
            self.beta_ = beta
            self.init_weights_ = init_w
            self.trans_weights_ = trans_w
            self.mu_ = mu
            self.sigma_ = sigma
            self.selected_id_ = best_id
        else:
            if self.selection_type == 'bic':
                model_id = np.argmin(self.bic_[:, 1])
                self.best_k_ = int(self.bic_[model_id, 0])
            elif self.selection_type == 'map':
                self.best_k_ = np.argmax(np.bincount(self.counts_))
                model_id = np.argwhere(self.bic_[:, 0] == self.best_k_)[0, 0]
            else:
                raise ValueError('Selection type not recognized')

            self.logp_ = self.logps_[int(self.bic_[model_id, 3])]
            self.X_ = self.models_[model_id].X
            self.intercept_ = self.models_[model_id].intercept
            self.mu_ = self.models_[model_id].mu
            self.sigma_ = self.models_[model_id].sigma
            if self.is_directed:
                self.radii_ = self.models_[model_id].radii

            # return_inverse relabels to start at zero
            _, temp_z = np.unique(self.models_[model_id].z.ravel(),
                                  return_inverse=True)
            self.z_ = temp_z.reshape(n_time_steps, n_nodes)
            self.beta_ = self.models_[model_id].beta
            self.init_weights_ = self.models_[model_id].init_weights
            self.trans_weights_ = self.models_[model_id].trans_weights
            self.lambda_ = self.models_[model_id].lmbda

        # Procrustes: rotate to reference position (best model)
        for idx in range(self.Xs_.shape[0]):
            # NOTE: Means should be rotated as well.. How to do this
            #       since they are constant over time?
            self.Xs_[idx] = longitudinal_procrustes_rotation(self.X_,
                                                             self.Xs_[idx])

        # store posterior means
        self.X_mean_ = self.Xs_[n_burn:].mean(axis=0)
        self.lambda_mean_ = self.lambdas_[n_burn:].mean(axis=0)
        self.intercepts_mean_ = self.intercepts_[n_burn:].mean(axis=0)
        if self.is_directed:
            self.radii_mean_ = self.radiis_[n_burn:].mean(axis=0)

        # store posterior group count probabilities
        self.posterior_group_ids_, self.posterior_group_counts_ = [], []
        for t in range(n_time_steps):
            index, counts = calculate_posterior_group_counts(self, t=t)
            self.posterior_group_ids_.append(index)
            self.posterior_group_counts_.append(counts)

        # store Gweke's diagnostic
        self.logp_geweke_ = geweke_diag(self.logps_, n_burn=n_burn)
        self.lambda_geweke_ = geweke_diag(self.lambdas_[:, 0], n_burn=n_burn)

        if self.is_directed:
            self.intercept_in_geweke_ = geweke_diag(
                self.intercepts_[:, 0], n_burn=n_burn)
            self.intercept_out_geweke_ = geweke_diag(
                self.intercepts_[:, 1], n_burn=n_burn)
        else:
            self.intercept_geweke_ = geweke_diag(
                self.intercepts_[:, 0], n_burn=n_burn)

        return self

    def _calculate_posterior_cooccurrences(self):
        n_time_steps, n_nodes, _ = self.Y_fit_.shape

        self.cooccurrence_probas_ = np.zeros((n_time_steps, n_nodes, n_nodes))
        for t in range(n_time_steps):
            self.cooccurrence_probas_[t] = calculate_posterior_cooccurrence(
                self, t=t)

    def logp(self, X, intercept, mu, sigma, z, weights, beta, lmbda, radii=None,
             dist=None):
        n_time_steps, n_nodes, _ = X.shape

        # beta log-likelihood
        loglik = dirichlet_logpdf(beta,
                                  np.repeat(self.gamma / self.n_components,
                                            self.n_components))

        # initial distribution (w0) log-likelihood
        loglik += dirichlet_logpdf(weights[0, 0], self.alpha_init * beta)

        # transition probabilities (w) log-likelihood
        deltas = self.kappa * np.eye(self.n_components)
        for t in range(1, n_time_steps):
            for k in range(self.n_components):
                loglik += dirichlet_logpdf(weights[t, k],
                                           self.alpha * beta + deltas[k])

        # log-likelihood of each node's label markov chain
        for i in range(n_nodes):
            loglik += np.log(weights[0, 0, z[0, i]])
            for t in range(1, n_time_steps):
                loglik += np.log(weights[t, z[t-1, i], z[t, i]])

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

    def set_best_model(self, selection_type='bic'):
        n_time_steps, n_nodes, _ = self.Y_fit_.shape

        self.selection_type = selection_type

        if selection_type == 'bic':
            model_id = np.argmin(self.bic_[:, 1])
            self.best_k_ = int(self.bic_[model_id, 0])
        elif self.selection_type == 'map':
            self.best_k_ = np.argmax(np.bincount(self.counts_))
            model_id = np.argwhere(self.bic_[:, 0] == self.best_k_)[0, 0]
        else:
            raise ValueError('Selection type not recognized')

        self.logp_ = self.logps_[int(self.bic_[model_id, 3])]
        self.X_ = self.models_[model_id].X
        self.intercept_ = self.models_[model_id].intercept
        self.mu_ = self.models_[model_id].mu
        self.sigma_ = self.models_[model_id].sigma
        if self.is_directed:
            self.radii_ = self.models_[model_id].radii

        # return_inverse relabels to start at zero
        _, temp_z = np.unique(self.models_[model_id].z.ravel(),
                              return_inverse=True)
        self.z_ = temp_z.reshape(n_time_steps, n_nodes)
        self.beta_ = self.models_[model_id].beta
        self.init_weights_ = self.models_[model_id].init_weights
        self.trans_weights_ = self.models_[model_id].trans_weights
        self.lambda_ = self.models_[model_id].lmbda

        return self

    def delete_traces(self):
        """Delete stored traces. Useful for storage, since the traces
        can take up a lot of space on disk.
        """
        del self.Xs_
        del self.intercepts_
        del self.zs_
        del self.mus_
        del self.sigmas_
        del self.weights_
        del self.betas_
        del self.lambdas_
        del self.logps_

        if self.is_directed:
            del self.radiis_
