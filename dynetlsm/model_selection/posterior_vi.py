import numpy as np

from ..network_likelihoods import dynamic_network_loglikelihood


__all__ = ['posterior_expected_vi', 'time_averaged_posterior_expected_vi',
           'minimize_posterior_expected_vi']


def nonvectorized_posterior_expected_vi(labels, cooccurrence_proba):
    """non-vectorized expected VI used for testing"""
    vi = 0.
    n_samples = labels.shape[0]
    for i in range(n_samples):
        ind = labels == labels[i]
        vi += np.log2(np.sum(ind))
        vi -= 2 * np.log2(np.sum(ind * cooccurrence_proba[i, :]))
        vi += np.log2(np.sum(cooccurrence_proba[i, :]))

    return vi / n_samples


def posterior_expected_vi(labels, cooccurrence_proba):
    """Lower-bound to the posterior expectation of the VI."""
    vi = 0.

    # number of samples and groups
    n_samples = labels.shape[0]
    n_groups = labels.max() + 1

    # cluster counts and membership indicators
    resp = np.zeros((n_samples, n_groups))
    resp[np.arange(n_samples), labels] = 1
    nk = np.sum(resp, axis=0)

    # VI calculation
    nonzero_mask = nk != 0  # (labels may be non-contiguous)
    vi += np.sum(nk[nonzero_mask] * np.log2(nk[nonzero_mask]))
    vi -= 2 * np.log2(
        np.sum(cooccurrence_proba * resp[:, labels].T,  axis=1)).sum()
    vi += np.log2(np.sum(cooccurrence_proba, axis=1)).sum()

    return vi / n_samples


def time_averaged_posterior_expected_vi(labels, cooccurrence_proba):
    """Lower-bound to the time averaged posterior expected VI."""
    vi = 0.
    n_time_steps = labels.shape[0]
    for t in range(n_time_steps):
        vi += posterior_expected_vi(labels[t], cooccurrence_proba[t])

    return vi / n_time_steps


def minimize_posterior_expected_vi(model):
    # determine how many samples to burn
    n_burn = model.n_burn_

    # calculated expected VI for the partitions explored by the markov chain
    n_samples = model.zs_.shape[0]
    sample_ids = np.arange(n_burn, n_samples)
    vis = np.zeros(sample_ids.shape[0])
    for i, idx in enumerate(sample_ids):
        vis[i] = time_averaged_posterior_expected_vi(
            model.zs_[idx], model.cooccurrence_probas_)

    # check for ties
    min_ids = np.where(vis == vis.min())[0]
    if min_ids.shape[0] > 1:
        # choose the configuration with the highest logliklihood log(p(Y | X))
        best_id, best_loglik = None, -np.inf
        for min_id in min_ids:
            loglik = dynamic_network_loglikelihood(
                model, sample_id=sample_ids[min_id])
            if loglik > best_loglik:
                best_id = sample_ids[min_id]
                best_loglik = loglik
    else:
        best_id = sample_ids[min_ids[0]]

    return best_id
