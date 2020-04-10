"""
Runs the analysis of the GoT character interactions network found in the
paper 'A Bayesian nonparametric latent space approach to modeling evolving
communities in dynamic networks' by Joshua Loyal and Yuguo Chen
"""

from dynetlsm import DynamicNetworkHDPLPCM
from dynetlsm.datasets import load_got
from dynetlsm.plots import (
    plot_traces,
    alluvial_plot,
    plot_latent_space
)


# Load GoT character interaction networks
Y, names = load_got(seasons=[1,2,3,4], weight_min=10)

# Fit HDP-LPCM
# NOTE: This will take days to sample!
model = DynamicNetworkHDPLPCM(n_iter=400000,
                              tune=50000,
                              burn=50000,
                              tune_interval=1000,
                              random_state=42,
                              n_components=25,
                              selection_type='vi',
                              is_directed=False).fit(Y)

# Trace plots
fig, ax = plot_traces(model, figsize=(10, 12))
fig.savefig('GoT_traces.png', dpi=300)

# alluvial diagram
fig, ax = alluvial_plot(model.z_, figsize=(10, 5))
fig.savefig('GoT_alluvial.png', dpi=300)

# latent space visualizations
for t in range(Y.shape[0]):
    fig, ax = plot_latent_space(
        model, figsize=(30, 30), t=t,
        textsize=50,
        node_size=500,
        mutation_scale=20,
        linewidth=1.0,
        connectionstyle='arc3,rad=0.2',
        title_text=None,
        plot_group_sigma=True,
        node_names=names,
        node_textsize=15,
        repel_strength=0.3,
        mask_groups=[5], # NOTE: this may not be background on other settings!
        only_show_connected=True,
        size_cutoff=2,
        number_nodes=True,
        border=3.0)
    fig.savefig('GoT_latent_space_t{}.png'.format(t), dpi=300)
