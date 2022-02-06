import glob
import os
import plac

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.utils import check_random_state
from sklearn.metrics import adjusted_rand_score, roc_auc_score

from dynetlsm import DynamicNetworkHDPLPCM, DynamicNetworkLPCM
from dynetlsm.datasets import detection_limit_simulation
from dynetlsm.model_selection.approx_bic import calculate_cluster_counts
from dynetlsm.model_selection import minimize_posterior_expected_vi
from dynetlsm.model_selection import train_test_split
from dynetlsm.metrics import variation_of_information, out_of_sample_auc
from dynetlsm.network_statistics import density, modularity


# group seperation ratio
ratio = 0.5
# ratio = 0.1, 0.25, 0.5, 0.7, 0.8, 0.9

# tranisition probability
trans_proba = 0.1
# trans_proba = 0.1, 0.2, 0.4

out_dir = 'results_ratio_{}'.format(trans_proba, ratio)


# create a directory to store the results
if not os.path.exists(out_dir):
    os.mkdir(out_dir)


def counts_per_time_step(z):
    n_time_steps = z.shape[0]
    group_counts = np.zeros(n_time_steps, dtype=np.int)
    for t in range(n_time_steps):
        group_counts[t] = np.unique(z[t]).shape[0]

    return group_counts

def posterior_per_time_step(model):
    n_time_steps = model.Y_fit_.shape[0]
    probas = np.zeros((n_time_steps, model.n_components + 1))
    for t in range(n_time_steps):
        freq = model.posterior_group_counts_[t]
        index = model.posterior_group_ids_[t]
        probas[t, index] = freq / freq.sum()

    return probas


def benchmark_single(n_iter=10000, burn=5000, tune=1000,
                     outfile_name='benchmark',
                     ratio=0.5, trans_proba=0.2,
                     random_state=None):
    random_state = check_random_state(random_state)

    # generate simulated networks
    Y, X, z, probas, r, _ = detection_limit_simulation(
        r=ratio, trans_proba=trans_proba, random_state=random_state)


    # fit HDP-LPCM
    model = DynamicNetworkHDPLPCM(n_iter=n_iter,
                                  burn=burn,
                                  tune=tune,
                                  tune_interval=1000,
                                  is_directed=False,
                                  selection_type='vi',
                                  n_components=5,
                                  random_state=random_state).fit(Y)

    # MAP: number of clusters per time point
    map_counts = counts_per_time_step(model.z_)

    # Posterior group count probabilities
    probas = posterior_per_time_step(model)
    results = pd.DataFrame(probas)

    # create dataframe of results
    results['map_counts'] = map_counts

    # Variation of Information
    results['vi'] = variation_of_information(
        z.ravel(), model.z_.ravel())
    vi = 0.
    for t in range(Y.shape[0]):
        vi_t = variation_of_information(z[t], model.z_[t])
        results['vi_{}'.format(t)] = vi_t
        vi += vi_t
    results['vi_avg'] = vi / Y.shape[0]


    # adjusted rand index
    results['rand_index'] = adjusted_rand_score(
        z.ravel(), model.z_.ravel())
    adj_rand = 0.
    for t in range(Y.shape[0]):
        adj_t = adjusted_rand_score(z[t], model.z_[t])
        results['rand_{}'.format(t)] = adj_t
        adj_rand += adj_t
    results['rand_avg'] = adj_rand / Y.shape[0]

    # info about simulated networks
    results['ratio'] = r

    results.to_csv(outfile_name, index=False)


# run for 20 different networks
for i in range(20):
    benchmark_single(
        n_iter=35000, burn=10000, tune=5000, random_state=i,
        ratio=ratio, trans_proba=trans_proba,
        outfile_name=os.path.join(
            out_dir, 'benchmark_{}.csv'.format(i)))
