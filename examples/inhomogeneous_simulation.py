"""
Runs the time-inhomogeneous simulations found in the
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
from dynetlsm.datasets import synthetic_dynamic_network
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
    Y, X, z, intercept, _, _ = synthetic_dynamic_network(
        n_time_steps=9, n_nodes=120, is_directed=False,
        lmbda=0.8, intercept=1.0, sticky_const=20,
        random_state=random_state)

    # fit HDP-LPCM
    model = DynamicNetworkHDPLPCM(n_iter=n_iter,
                                  burn=burn,
                                  tune=tune,
                                  tune_interval=1000,
                                  is_directed=False,
                                  n_components=10,
                                  selection_type='vi',
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

    # clustering results
    results['vi'] = variation_of_information(z.ravel(), model.z_.ravel())
    results['vi_t1'] = variation_of_information(z[:3].ravel(), model.z_[:3].ravel())
    results['vi_t2'] = variation_of_information(z[3:6].ravel(), model.z_[3:6].ravel())
    results['vi_t3'] = variation_of_information(z[6:].ravel(), model.z_[6:].ravel())

    # time average VI
    vi = 0.
    for t in range(Y.shape[0]):
        vi += variation_of_information(z[t], model.z_[t])
    results['vi_avg'] = vi / Y.shape[0]

    results['rand_index'] = adjusted_rand_score(z.ravel(), model.z_.ravel())
    results['rand_t1'] = adjusted_rand_score(z[:3].ravel(), model.z_[:3].ravel())
    results['rand_t2'] = adjusted_rand_score(z[3:6].ravel(), model.z_[3:6].ravel())
    results['rand_t3'] = adjusted_rand_score(z[6:].ravel(), model.z_[6:].ravel())

    # time average rand
    adj_rand = 0.
    for t in range(Y.shape[0]):
        adj_rand += adjusted_rand_score(z[t], model.z_[t])
    results['rand_avg'] = adj_rand / Y.shape[0]

    # info about simulated networks
    results['modularity'] = modularity(Y, z)
    results['modularity_t1'] = modularity(Y[:3], z[:3])
    results['modularity_t2'] = modularity(Y[3:6], z[3:6])
    results['modularity_t3'] = modularity(Y[6:], z[6:])

    results['density'] = density(Y)
    results['density_t1'] = density(Y[:3])
    results['density_t2'] = density(Y[3:6])
    results['density_t3'] = density(Y[6:])

    results.to_csv(outfile_name + '.csv', index=False)


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
n_time_steps = 9
n_groups = 10

n_files = len(glob.glob('results/*'))
stat_names = ['insample_auc', 'vi_avg', 'vi_t1', 'vi_t2', 'vi_t3',
              'rand_avg', 'rand_t1', 'rand_t2', 'rand_t3']
data = np.zeros((n_files, len(stat_names)))
for i, file_name in enumerate(glob.glob('results/*')):
    df = pd.read_csv(file_name)
    data[i] = df.loc[0, stat_names].values

data = pd.DataFrame(data, columns=stat_names)
print('Median Metrics:')
print(data.median(axis=0))
print('Metrics SD:')
print(data.std(axis=0))

data = data[['vi_avg', 'vi_t1', 'vi_t2', 'vi_t3',
             'rand_avg', 'rand_t1', 'rand_t2', 'rand_t3']]
data.columns = ['Avg. VI', 'VI (t = 1 - 3)', 'VI (t = 4 - 6)', 'VI (t = 7 - 9)',
                'Avg. ARI', 'ARI (t = 1 - 3)', 'ARI (t = 4 - 6)',
                'ARI (t = 7 - 9)']

# boxplots of the metrics
plt.rc('font', family='sans-serif', size=24)
fig, ax = plt.subplots(figsize=(10, 6))
g = sns.boxplot(x='variable', y='value',
                data=pd.melt(data), fliersize=2.0, ax=ax)
g.tick_params(labelsize=8)
sns.despine()
ax.set(xlabel='', ylabel='')
plt.savefig('performance_plot.png', dpi=300)

# clear figure
plt.clf()

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

data = pd.DataFrame(data,
    columns=range(1, n_groups + 1), index=range(1, n_time_steps + 1))
mask = data.values == 0

g = sns.heatmap(data, annot=True, cmap="Blues", cbar=False, mask=mask)
g.set_xlabel('# of groups')
g.set_ylabel('t')
plt.savefig('num_clusters.png', dpi=300)
