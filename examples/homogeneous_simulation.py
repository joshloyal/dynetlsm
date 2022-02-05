"""
Runs the time-homogeneous simulations found in the
paper 'A Bayesian nonparametric latent space approach to modeling evolving
communities in dynamic networks' by Joshua Loyal and Yuguo Chen
"""

import glob
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.utils import check_random_state
from sklearn.metrics import adjusted_rand_score

from dynetlsm import DynamicNetworkHDPLPCM
from dynetlsm.datasets import synthetic_static_community_dynamic_network
from dynetlsm.model_selection.approx_bic import calculate_cluster_counts
from dynetlsm.metrics import variation_of_information
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
                     random_state=None):
    random_state = check_random_state(random_state)

    # generate simulated networks
    Y, X, z, intercept, _, _ = synthetic_static_community_dynamic_network(
        n_time_steps=6, n_nodes=120, random_state=random_state)

    # fit HDP-LPCM
    model = DynamicNetworkHDPLPCM(n_iter=n_iter,
                                  burn=burn,
                                  tune=tune,
                                  tune_interval=1000,
                                  is_directed=False,
                                  selection_type='vi',
                                  n_components=10,
                                  random_state=random_state).fit(Y)

    # MAP: number of clusters per time point
    map_counts = counts_per_time_step(model.z_)

    # Posterior group count probabilities
    probas = posterior_per_time_step(model)

    # create dataframe of results
    results = pd.DataFrame(probas)
    results['map_counts'] = map_counts

    # goodness-of-fit metrics for MAP
    results['insample_auc'] = model.auc_
    results['vi'] = variation_of_information(z.ravel(), model.z_.ravel())

    # time average VI
    vi = 0.
    for t in range(Y.shape[0]):
        vi += variation_of_information(z[t], model.z_[t])
    results['vi_avg'] = vi / Y.shape[0]

    results['rand_index'] = adjusted_rand_score(z.ravel(), model.z_.ravel())

    # time average rand
    adj_rand = 0.
    for t in range(Y.shape[0]):
        adj_rand += adjusted_rand_score(z[t], model.z_[t])
    results['rand_avg'] = adj_rand / Y.shape[0]

    # info about simulated networks
    results['modularity'] = modularity(Y, z)
    results['density'] = density(Y)

    results.to_csv(outfile_name, index=False)


# NOTE: This is meant to be run in parallel on a computer cluster!
n_reps = 50
out_dir = 'results'

# create a directory to store the results
if not os.path.exists('results'):
    os.mkdir(out_dir)

for i in range(n_reps):
    benchmark_single(n_iter=35000, burn=10000, tune=5000, random_state=i,
                     outfile_name=os.path.join(
                        out_dir, 'benchmark_{}.csv'.format(i)))

# calculate median metric values
n_time_steps = 6
n_groups = 10

n_files = len(glob.glob('results/*'))
stat_names = ['insample_auc', 'vi_avg', 'rand_avg']
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
