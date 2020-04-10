"""
Runs the analysis of the Sampson's monastery network found in the
paper 'A Bayesian nonparametric latent space approach to modeling evolving
communities in dynamic networks' by Joshua Loyal and Yuguo Chen
"""

from dynetlsm import DynamicNetworkHDPLPCM
from dynetlsm.datasets import load_monks
from dynetlsm.plots import (
    plot_traces,
    plot_posterior_counts,
    alluvial_plot,
    plot_latent_space
)


# Load Sampson's monastery network
Y, labels, names = load_monks(dynamic=True, is_directed=False)

# Fit HDP-LPCM
model = DynamicNetworkHDPLPCM(n_iter=165000,
                              tune=15000,
                              burn=20000,
                              tune_interval=1000,
                              random_state=42,
                              selection_type='vi',
                              is_directed=False).fit(Y)

# Trace plots
fig, ax = plot_traces(model, figsize=(10, 12))
fig.savefig('sampson_monks_traces.png', dpi=300)

# posterior group counts
for t in range(Y.shape[0]):
    fig, ax = plot_posterior_counts(model, figsize=(8, 6), fontsize=18,
                                    ticksize=18, t=t, normalize=True,
                                    bar_width=0.25, include_title=False)
    ax.set_xticks(range(0, 10))
    ax.set_xlim(0, 9)
    fig.savefig('sampson_monks_posterior_counts_t{}.png'.format(t), dpi=300)

# alluvial diagram
fig, ax = alluvial_plot(model.z_, figsize=(10, 5))
fig.savefig('sampson_monks_alluvial.png', dpi=300)

# latent space visualizations
for t in range(Y.shape[0]):
    fig, ax = plot_latent_space(
        model, figsize=(10, 12), t=t,
        node_size=100,
        linewidth=1.0,
        mutation_scale=30,
        connectionstyle='arc3,rad=0.2',
        title_text=None,
        plot_group_sigma=True,
        node_names=names,
        node_textsize=10,
        repel_strength=0.3,
        number_nodes=True, border=1.0)
    fig.savefig('sampson_monks_latent_space_t{}.png'.format(t), dpi=300)
