import numpy as np
import scipy.cluster.hierarchy as hc

from scipy.spatial.distance import squareform

from .model_selection.approx_bic import calculate_cluster_counts
from .model_selection.approx_bic import calculate_cluster_counts_t


def renormalize_weights(model, sample_id):
    # re-normalize weights
    active_groups = np.unique(model.zs_[sample_id].ravel())
    active_mask = np.in1d(np.arange(model.n_components), active_groups)
    n_groups = active_groups.shape[0]

    beta = model.betas_[sample_id, active_groups]
    beta /= beta.sum()

    weights = model.weights_[sample_id]
    init_w = weights[0, 0, active_groups]
    init_w /= init_w.sum()

    n_time_steps, n_nodes, _ = model.Y_fit_.shape
    trans_w = np.zeros((n_time_steps, n_groups, n_groups), dtype=np.float64)
    for t in range(1, n_time_steps):
        trans_w[t] = weights[t, active_groups][:, active_groups]
        trans_w[t] /= np.sum(trans_w[t], axis=1).reshape(-1, 1)

    # return_inverse relabels z to start at zero
    _, temp_z = np.unique(model.zs_[sample_id].ravel(), return_inverse=True)
    z = temp_z.reshape(n_time_steps, n_nodes)

    # relabel mu and sigma as well
    mu = model.mus_[sample_id, active_groups]
    sigma = model.sigmas_[sample_id, active_groups]

    return z, beta, init_w, trans_w, mu, sigma


def calculate_cooccurrence_matrix(z, n_groups=None):
    if n_groups is None:
        n_groups = np.unqiue(z).shape[0]

    # dummy encode group membership
    indicator = np.eye(n_groups)[z]

    return np.dot(indicator, indicator.T)


def calculate_posterior_cooccurrence(model, t=0):
    # determine burn in samples
    n_burn = model.n_burn_

    n_nodes = model.Y_fit_.shape[1]
    cooccurrence_proba = np.zeros((n_nodes, n_nodes))
    n_iter = 0
    for z in model.zs_[n_burn:, t]:
        n_iter += 1
        cooccurrence_proba += calculate_cooccurrence_matrix(
                                z, n_groups=model.n_components)

    return cooccurrence_proba / n_iter


def cluster_posterior_coocurrence(model, t=0, threshold=0.5):
    cooccurence_proba = model.cooccurrence_probas_[t]

    # hierarchical clustering with average linkage
    linkage = hc.linkage(squareform(1. - cooccurence_proba), method='average',
                         optimal_ordering=True)

    return hc.fcluster(linkage, t=threshold, criterion='distance') - 1


def calculate_posterior_group_counts(model, t=0):
    counts = calculate_cluster_counts_t(model)[t]

    freq = np.bincount(counts)
    index = np.where(freq != 0)[0]
    freq = freq[index]

    return index, freq
