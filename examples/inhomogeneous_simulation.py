"""
Runs the time-inhomogeneous simulations found in the
paper 'A Bayesian nonparametric latent space approach to modeling evolving
communities in dynamic networks' by Joshua Loyal and Yuguo Chen
"""
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
from dynetlsm.datasets import inhomogeneous_simulation
from dynetlsm.model_selection.approx_bic import calculate_cluster_counts
from dynetlsm.model_selection import minimize_posterior_expected_vi
from dynetlsm.model_selection import train_test_split
from dynetlsm.metrics import variation_of_information, out_of_sample_auc
from dynetlsm.network_statistics import density, modularity


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
                     sim_type='hard',
                     random_state=None):
    random_state = check_random_state(random_state)

    Y, X, z, intercept, _, _, _, _ = inhomogeneous_simulation(
        n_nodes=120, random_state=random_state, simulation_type=sim_type)

    Y, Y_ahead = Y[:9], Y[9]
    Y_obs, test_indices = train_test_split(
        Y, test_size=0.1, random_state=random_state)

    # fit HDP-LPCM
    model = DynamicNetworkHDPLPCM(n_iter=n_iter,
                                  burn=burn,
                                  tune=tune,
                                  tune_interval=1000,
                                  is_directed=False,
                                  selection_type='vi',
                                  n_components=10,
                                  random_state=random_state).fit(Y_obs)

    # MAP: number of clusters per time point
    map_counts = counts_per_time_step(model.z_)

    # Posterior group count probabilities
    probas = posterior_per_time_step(model)
    results = pd.DataFrame(probas)

    # create dataframe of results
    results['map_counts'] = map_counts

    # goodness-of-fit metrics for MAP
    results['insample_auc'] = model.auc_
    results['outsample_auc'] = out_of_sample_auc(
        Y, model.missings_, test_indices)

    indices = np.tril_indices(Y.shape[1], k=-1)

    pred_probas = model.forecast_probas_marginalized_[indices]
    results['onestep_auc'] = roc_auc_score(
        Y_ahead[indices], pred_probas)

    # Variation of Information
    results['vi'] = variation_of_information(
        z[:9].ravel(), model.z_.ravel())
    vi = 0.
    for t in range(Y.shape[0]):
        vi_t = variation_of_information(z[t], model.z_[t])
        results['vi_{}'.format(t)] = vi_t
        vi += vi_t
    results['vi_avg'] = vi / Y.shape[0]

    # adjusted rand index
    results['rand_index'] = adjusted_rand_score(
        z[:9].ravel(), model.z_.ravel())
    adj_rand = 0.
    for t in range(Y.shape[0]):
        adj_t = adjusted_rand_score(z[t], model.z_[t])
        results['rand_{}'.format(t)] = adj_t
        adj_rand += adj_t
    results['rand_avg'] = adj_rand / Y.shape[0]

    results.to_csv(outfile_name, index=False)


# NOTE: This is meant to be run in parallel on a computer cluster
n_reps = 50
sim_type = 'easy'
out_dir = 'results'

# create a directory to store the results
if not os.path.exists(out_dir):
    os.mkdir(out_dir)


for i in range(n_reps):
    benchmark_single(
        n_iter=35000, burn=10000, tune=5000, random_state=i,
        sim_type=sim_type,
        outfile_name=os.path.join(
            out_dir, 'benchmark_{}.csv'.format(i)))


# calculate median metric values
n_time_steps = 9
n_groups = 10

n_files = len(glob.glob('results/*'))
stat_names = ['insample_auc', 'outsample_auc', 'onestep_auc', 'vi',
              'rand_index', 'vi_avg', 'rand_avg']
data = np.zeros((n_files, len(stat_names)))
for i, file_name in enumerate(glob.glob('results/*')):
    df = pd.read_csv(file_name)
    data[i] = df.loc[0, stat_names].values

data = pd.DataFrame(data, columns=stat_names)
print('Median Metrics:')
print(data.median(axis=0))
print('Metrics SD:')
print(data.std(axis=0))

# plot posterior boxplots
data = {'probas': [], 'cluster_number': [], 't': []}
for file_name in glob.glob('results/*'):
    df = pd.read_csv(file_name)
    for t in range(n_time_steps):
        for i in range(1, n_groups):
            data['probas'].append(df.iloc[t, i])
            data['cluster_number'].append(i)
            data['t'].append(t + 1)

data = pd.DataFrame(data)

plt.rc('font', family='sans-serif', size=16)
g = sns.catplot(x='cluster_number', y='probas', col='t',
                col_wrap=3, kind='box', data=data)

for ax in g.axes:
    ax.set_ylabel('posterior probability')
    ax.set_xlabel('# of groups')

g.fig.tight_layout()

plt.savefig('cluster_posterior.png', dpi=300)

# clear figure
plt.clf()

# plot selected number of groups for each simulation
data = np.zeros((n_time_steps, n_groups), dtype=np.int)
for sim_id, file_name in enumerate(glob.glob('results/*')):
    df = pd.read_csv(file_name)
    for t in range(n_time_steps):
        data[t, df.iloc[t, n_groups + 1] - 1] +=1

data = pd.DataFrame(data, columns=range(1, n_groups + 1), index=range(1, n_time_steps + 1))
mask = data.values == 0

g = sns.heatmap(data, annot=True, cmap="Blues", cbar=False, mask=mask)
g.set_xlabel('# of groups')
g.set_ylabel('t')
plt.savefig('num_clusters.png', dpi=300)
