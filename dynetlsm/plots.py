import numbers
import itertools

import arviz as az
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import networkx as nx
import pyvis.network as pyvis
import scipy.linalg as linalg
import scipy.sparse as sp
import scipy.cluster.hierarchy as hc
import seaborn as sns

from matplotlib import gridspec
from matplotlib.colors import to_hex
from matplotlib.patches import Ellipse, Rectangle, FancyArrowPatch
from matplotlib.ticker import MaxNLocator

from scipy.special import expit
from scipy.interpolate import CubicSpline
from scipy.spatial.distance import squareform

from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    pairwise_distances, roc_auc_score, adjusted_rand_score)


from .latent_space import calculate_distances
from .metrics import network_auc
from .array_utils import triu_indices_from_3d, nondiag_indices_from
from .network_likelihoods import directed_network_probas
from .trace_utils import effective_n
from .model_selection.approx_bic import calculate_cluster_counts
from .model_selection.approx_bic import calculate_cluster_counts_t
from .network_statistics import connected_nodes
from .text_utils import repel_labels


__all__ = ['animate_latent_space',
           'animate_dynamic_graph',
           'draw_arrow',
           'plot_network_pyvis',
           'plot_network_embedding',
           'plot_group_embedding',
           'plot_trajectories',
           'plot_latent_space',
           'plot_probability_matrix',
           'plot_traces',
           'plot_poserior_counts',
           'plot_model_selection',
           'plot_model_parameters',
           'plot_distance_comparison',
           'plot_adjacency_matrix',
           'alluvial_plot']


def flatten(l):
    return [item for sublist in l for item in sublist]


def get_color20():
    colors = np.asarray([to_hex(c) for c in plt.cm.get_cmap('tab20').colors])

    # the most common case is the need for two colors. The first two do
    # not have a lot of contrast so swap them
    colors[1], colors[2] = colors[2], colors[1]

    return colors

def get_husl(n_groups):
    colors = sns.color_palette('husl', n_colors=n_groups)
    return np.asarray([to_hex(c) for c in colors])


def get_colors(labels):
    # integer encode labels
    encoder = LabelEncoder().fit(labels)
    n_groups = encoder.classes_.shape[0]

    return get_color20() if n_groups <= 20 else get_husl(n_groups)


def normal_contour(mean, cov, n_std=2, ax=None, animated=False, **kwargs):
    if cov.shape[0] != 2:
        raise ValueError('Only for bivariate normal densities.')

    eigenvalues, eigenvectors = linalg.eigh(cov)

    # sort the eigenvalues and eigenvectors in descending order
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # determine the angle of rotation
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))

    if ax is None:
        ax = plt.gca()

    if isinstance(n_std, numbers.Integral):
        # the diameter of the ellipse is twice the square root of the evalues
        width, height = 2 * n_std * np.sqrt(eigenvalues)
        ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle,
                          animated=animated, **kwargs)
        ax.add_artist(ellipse)

        return ellipse

    ellipses = []
    for std in n_std:
        width, height = 2 * std * np.sqrt(eigenvalues)
        ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle,
                          animated=animated, **kwargs)

        ax.add_artist(ellipse)
        ellipses.append(ellipse)

    return ellipses


def animate_latent_space(X, labels=None, centers=None, stds=None,
                         radii=None,
                         figsize=(8, 6), marker_size=50, border=0.1,
                         interval=1000, square=True, **kwargs):
    """Animate Latent Space."""
    n_time_steps, n_nodes, _ = X.shape

    fig, ax = plt.subplots(figsize=figsize)

    scatter = ax.scatter([], [], edgecolors='w', s=marker_size, **kwargs)
    scatter_centers = ax.scatter([], [], edgecolor='k', s=2 * marker_size,
                                 marker='*', **kwargs)
    if square:
        ax.set_aspect('equal', 'box')

    colors = get_colors(labels.ravel()) if labels is not None else None

    if labels is None:
        labels = np.zeros((n_time_steps, n_nodes), dtype=np.int)

    has_centers = centers is not None
    has_stds = stds is not None
    if not has_centers:
        centers = [None] * X.shape[0]
    elif centers.ndim == 2:
        centers = np.repeat(np.expand_dims(centers, axis=0), X.shape[0], axis=0)

    if has_stds:
        stds = [normal_contour((0, 0), std * np.eye(2), n_std=[1, 2],
                               ax=ax, fill=None, visible=True,
                               animated=False) for std in stds]

    def init():
        scatter.set_offsets([])

        if has_centers:
            scatter_centers.set_offsets([])

        xy_min = np.min(X, axis=(0, 1)) - border
        xy_max = np.max(X, axis=(0, 1)) + border

        ax.set_xlim(xy_min[0], xy_max[0])
        ax.set_ylim(xy_min[1], xy_max[1])

        elements = [scatter]
        if has_centers:
            elements.append(scatter_centers)
        if has_stds:
            elements = elements + flatten(stds)

        return elements

    def frame_iter():
        for t in range(X.shape[0]):
            yield (t, X[t], labels[t], centers[t])

    def animate(frames):
        t, Xt, zt, mut = frames
        scatter.set_offsets(Xt)
        if radii is not None:
            scatter.set_sizes(radii / radii.min() * 10)
        scatter.set_color(colors[zt])
        scatter.set_edgecolors('w')

        if mut is not None:
            center_ids = np.unique(zt)
            scatter_centers.set_offsets(mut[center_ids])
            scatter_centers.set_color(colors[center_ids])
            scatter_centers.set_edgecolor('k')

        if stds is not None:
            center_ids = np.unique(zt)
            for k, ellipses in enumerate(stds):
                if k in center_ids:
                    for e in ellipses:
                        e.set_visible(True)
                        e.set_center(mut[k])
                        e.set_color(colors[k])
                else:
                    for e in ellipses:
                        e.set_visible(False)

        ax.set_title("t = {}".format(t + 1))

        elements = [scatter]
        if has_centers:
            elements.append(scatter_centers)
        if has_stds:
            elements = elements + flatten(stds)

        return elements

    return animation.FuncAnimation(fig,
                                   func=animate,
                                   frames=frame_iter,
                                   init_func=init,
                                   interval=interval,
                                   save_count=X.shape[0],
                                   blit=True)


def animate_dynamic_graph(Y, labels, is_directed=False, change_order=False,
                          figsize=(6, 6),
                          node_size=80, interval=1000):
    """Animate Dynamic Network (Circular Layout)"""
    fig, ax = plt.subplots(figsize=figsize)

    order = np.argsort(labels[0])
    G0 = nx.from_numpy_array(Y[0][order][:, order])
    layout = nx.circular_layout(G0)
    colors = get_color20()

    def animate(t):
        ax.clear()

        if change_order:
            order = np.argsort(labels[t])
        else:
            order = np.argsort(labels[0])

        Gt = nx.from_numpy_array(Y[t][order][:, order])
        zt = labels[t]
        nx.draw(Gt,
                pos=layout,
                node_color=colors[zt[order]],
                edge_color='gray',
                edgecolors='white',
                node_size=node_size,
                ax=ax)

        ax.set_title("t = {}".format(t + 1))

    return animation.FuncAnimation(fig,
                                   func=animate,
                                   frames=Y.shape[0],
                                   interval=interval,
                                   save_count=Y.shape[0])


def plot_network_pyvis(Y, labels=None, output_name='network_vis.html',
                       is_directed=False, in_notebook=False, names=None,
                       height="550px", width="100%", **kwargs):
    """Use the pyvis plotting library to display a network."""
    network = pyvis.Network(height=height, width=width, notebook=in_notebook,
                            directed=is_directed, **kwargs)

    # import graph
    if is_directed:
        nx_graph = nx.from_numpy_array(Y, create_using=nx.DiGraph)
    else:
        nx_graph = nx.from_numpy_array(Y)

    network.from_nx(nx_graph)

    if labels is not None:
        # integer encode labels
        encoder = LabelEncoder().fit(labels)
        n_groups = encoder.classes_.shape[0]

        # add node colors
        if n_groups <= 20:
            colors = get_color20()
        else:
            colors = sns.color_palette('husl', n_colors=n_groups)
            colors = np.asarray([to_hex(c) for c in colors])

        for node in network.nodes:
            if names is not None:
                node['label'] = names[node['id']]
            node['color'] = colors[encoder.transform([labels[node['id']]])[0]]

    # display
    network.barnes_hut()

    return network.show(output_name)

def plot_latent_space(X, means=None, sigmas=None, labels=None, figsize=(10, 8)):
    fig, ax = plt.subplots(figsize=figsize)

    if labels is not None:
        # filter for clusters with points actually assigned
        unique_labels = np.unique(labels)
        means = means[unique_labels]
        sigmas = sigmas[unique_labels]

        # transform labels to intergers 0-19
        encoder = LabelEncoder().fit(labels)
        z = encoder.transform(labels)

        colors = get_color20()
        ax.scatter(X[:, 0], X[:, 1], c=colors[z], edgecolor='k', alpha=0.8)

        for k, cluster_id in enumerate(unique_labels):
            ax.scatter(means[k, 0], means[k, 1], edgecolor='k', s=200,
                       marker='*',
                       color=colors[encoder.transform([cluster_id])])
            normal_contour(means[k], sigmas[k] * np.eye(2),
                           n_std=[1, 2], ax=ax, fill=None,
                           color=colors[encoder.transform([cluster_id])[0]])
    else:
        ax.scatter(X[:, 0], X[:, 1], edgecolor='k', alpha=0.8)

    return ax


def plot_probability_matrix(probas, z, figsize=(10, 6),
                            is_adj=False,
                            is_directed=False, in_notebook=True):

    fig, ax = plt.subplots(figsize=figsize)

    probas = probas.copy()
    z = z.copy()

    # re-order in terms of partitioning
    order = np.argsort(z)
    probas = probas[order, :][:, order]

    # mask upper diagonal for undirected graphs
    mask = None
    #if not is_directed:
    #    mask = np.zeros_like(probas)
    #    mask[np.triu_indices_from(mask)] = True

    with sns.axes_style('white'):
        #ax.set_title('t = {}'.format(t+1))

        sns.heatmap(probas, #cmap='rocket_r',
                    cmap='Blues',mask=mask,
                    yticklabels=False, xticklabels=False,
                    vmin=0.0, vmax=1.0, ax=ax,
                    #cbar_ax=plt.subplot(gs[nrows-1, :]),
                    cbar_kws={"orientation": "horizontal"})
        #else:
        #    sns.heatmap(probas[t], cmap='rocket_r', mask=mask,
        #                yticklabels=False, xticklabels=False,
        #                vmin=0.0, vmax=1.0, ax=ax, cbar=False)

    return fig, ax


def plot_traces(model, figsize=(10, 6), maxlags=100, fontsize=8):
    fig = plt.figure(figsize=figsize)
    colors = get_color20()

    if model.is_directed:
        gs = gridspec.GridSpec(nrows=3, ncols=3)
        ax = np.array([[plt.subplot(gs[0, :]), None, None],
                       [plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1]),
                        plt.subplot(gs[1, 2])],
                       [plt.subplot(gs[2, 0]), plt.subplot(gs[2, 1]),
                        plt.subplot(gs[2, 2])]])
    else:
        gs = gridspec.GridSpec(nrows=2, ncols=3)
        ax = np.array([[plt.subplot(gs[0, :]), None, None],
                       [plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1]),
                        plt.subplot(gs[1, 2])]])

    # change fontsize of graph
    for a in ax.ravel():
        if a is not None:
            [l.set_fontsize(fontsize) for l in a.get_yticklabels()]
            [l.set_fontsize(fontsize) for l in a.get_xticklabels()]

    ax[0, 0].plot(model.logps_, c=colors[0])
    ax[0, 0].set_title('Unnormalized Log Posterior', fontsize=fontsize)

    # draw a line indicating end of burn-in
    n_burn = model.n_burn_
    if n_burn > 0:
        ax[0, 0].axvline(n_burn, linestyle='--',  color='k', alpha=0.7)
        ax[0, 0].annotate(' burn-in', (n_burn, np.min(model.logps_)), alpha=0.7)



    if model.is_directed:
        sns.kdeplot(model.intercepts_[n_burn:, 0].ravel(), ax=ax[1, 0],
                    shade=True, color=colors[1])
        ax[1, 0].set_title(r'Intercept $\beta_{in}$', fontsize=fontsize)
        ax[1, 1].plot(model.intercepts_[:, 0], c=colors[1])

        x = model.intercepts_[:, 0].ravel()[n_burn:]
        lags, corr, _, _ = ax[1, 2].acorr(x - np.mean(x), maxlags=maxlags,
                                          normed=True,  usevlines=True,
                                          alpha=0.5)

        ax[1, 2].set_xlim((0, maxlags))
        ax[1, 2].text(0.5, 0.9, 'ESS = {:.2f}'.format(effective_n(x, lags, corr)),
                      fontsize=8,
                      horizontalalignment='left',
                      verticalalignment='center',
                      transform=ax[1, 2].transAxes)

        sns.kdeplot(model.intercepts_[n_burn:, 1].ravel(), ax=ax[2, 0],
                    shade=True, color=colors[2])
        ax[2, 0].set_title(r'Intercept $\beta_{out}$', fontsize=fontsize)
        ax[2, 1].plot(model.intercepts_[:, 1], c=colors[2])

        x = model.intercepts_[:, 1].ravel()[n_burn:]
        lags, corr, _, _ = ax[2, 2].acorr(x - np.mean(x), maxlags=maxlags,
                                          normed=True,  usevlines=True,
                                          alpha=0.5)

        ax[2, 2].set_xlim((0, maxlags))
        ax[2, 2].text(0.5, 0.9, 'ESS = {:.2f}'.format(effective_n(x, lags, corr)),
                      fontsize=8,
                      horizontalalignment='left',
                      verticalalignment='center',
                      transform=ax[2, 2].transAxes)
    else:
        sns.kdeplot(model.intercepts_[n_burn:].ravel(), ax=ax[1, 0], shade=True, color=colors[1])
        ax[1, 0].set_title(r'Intercept $\beta_0$', fontsize=fontsize)
        ax[1, 1].plot(model.intercepts_, c=colors[1])

        x = model.intercepts_.ravel()[n_burn:]
        lags, corr, _, _ = ax[1, 2].acorr(x - np.mean(x), maxlags=maxlags,
                                          normed=True,  usevlines=True,
                                          alpha=0.5)
        ax[1, 2].set_xlim((0, maxlags))
        ax[1, 2].text(0.5, 0.9, 'ESS = {:.2f}'.format(effective_n(x, lags, corr)),
                      fontsize=8,
                      horizontalalignment='left',
                      verticalalignment='center',
                      transform=ax[1, 2].transAxes)


def plot_traces_mixture(model, figsize=(10, 12), maxlags=100, fontsize=8):
    fig = plt.figure(figsize=figsize)
    colors = get_color20()

    if model.is_directed:
        gs = gridspec.GridSpec(nrows=4, ncols=3)
        ax = np.array([[plt.subplot(gs[0, :]), None, None],
                       [plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1]),
                        plt.subplot(gs[1, 2])],
                       [plt.subplot(gs[2, 0]), plt.subplot(gs[2, 1]),
                        plt.subplot(gs[2, 2])],
                       [plt.subplot(gs[3, 0]), plt.subplot(gs[3, 1]),
                        plt.subplot(gs[3, 2])]])
    else:
        gs = gridspec.GridSpec(nrows=3, ncols=3)
        ax = np.array([[plt.subplot(gs[0, :]), None, None],
                       [plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1]),
                        plt.subplot(gs[1, 2])],
                       [plt.subplot(gs[2, 0]), plt.subplot(gs[2, 1]),
                        plt.subplot(gs[2, 2])]])

    # change fontsize of graph
    for a in ax.ravel():
        if a is not None:
            [l.set_fontsize(fontsize) for l in a.get_yticklabels()]
            [l.set_fontsize(fontsize) for l in a.get_xticklabels()]

    ax[0, 0].plot(model.logps_, c=colors[0])
    ax[0, 0].set_title('Unnormalized Log Posterior', fontsize=fontsize)

    # draw a line indicating end of burn-in
    n_burn = model.n_burn_
    if n_burn > 0:
        ax[0, 0].axvline(n_burn, linestyle='--',  color='k', alpha=0.7)
        ax[0, 0].annotate(' burn-in', (n_burn, np.min(model.logps_)), alpha=0.7)

    if model.is_directed:
        sns.kdeplot(model.intercepts_[n_burn:, 0].ravel(), ax=ax[1, 0],
                    shade=True, color=colors[1])
        ax[1, 0].set_title(r'Intercept $\beta_{in}$', fontsize=fontsize)
        ax[1, 1].plot(model.intercepts_[:, 0], c=colors[1])

        x = model.intercepts_[n_burn:, 0].ravel()
        lags, corr, _, _ = ax[1, 2].acorr(x - np.mean(x), maxlags=maxlags,
                                          normed=True,  usevlines=True,
                                          alpha=0.5)
        ax[1, 2].set_xlim((0, maxlags))
        ax[1, 2].text(0.5, 0.9, 'ESS = {:.2f}'.format(effective_n(x, lags, corr)),
                      fontsize=8,
                      horizontalalignment='left',
                      verticalalignment='center',
                      transform=ax[1, 2].transAxes)

        sns.kdeplot(model.intercepts_[n_burn:, 1].ravel(), ax=ax[2, 0],
                    shade=True, color=colors[2])
        ax[2, 0].set_title(r'Intercept $\beta_{out}$', fontsize=fontsize)
        ax[2, 1].plot(model.intercepts_[:, 1], c=colors[2])

        x = model.intercepts_[n_burn:, 1].ravel()
        lags, corr, _, _ = ax[2, 2].acorr(x - np.mean(x), maxlags=maxlags,
                                          normed=True,  usevlines=True,
                                          alpha=0.5)
        ax[2, 2].set_xlim((0, maxlags))
        ax[2, 2].text(0.5, 0.9, 'ESS = {:.2f}'.format(effective_n(x, lags, corr)),
                      fontsize=8,
                      horizontalalignment='left',
                      verticalalignment='center',
                      transform=ax[2, 2].transAxes)

        sns.kdeplot(model.lambdas_[n_burn:].ravel(), ax=ax[3, 0],
                    shade=True, color=colors[4])
        ax[3, 0].set_title(r'Blending Coefficient $\lambda$', fontsize=fontsize)
        ax[3, 1].plot(model.lambdas_, c=colors[4])

        x = model.lambdas_.ravel()[n_burn:]
        lags, corr, _, _ = ax[3, 2].acorr(x - np.mean(x), maxlags=maxlags,
                                          normed=True, usevlines=True,
                                          alpha=0.5)
        ax[3, 2].set_xlim((0, maxlags))
        ax[3, 2].text(0.5, 0.9, 'ESS = {:.2f}'.format(effective_n(x, lags, corr)),
                      fontsize=8,
                      horizontalalignment='left',
                      verticalalignment='center',
                      transform=ax[3, 2].transAxes)
    else:
        sns.kdeplot(model.intercepts_[n_burn:].ravel(), ax=ax[1, 0],
                    shade=True, color=colors[1])
        ax[1, 0].set_title(r'Intercept $\beta_0$', fontsize=fontsize)
        ax[1, 1].plot(model.intercepts_, c=colors[1])

        x = model.intercepts_.ravel()[n_burn:]
        lags, corr, _, _ = ax[1, 2].acorr(x - np.mean(x), maxlags=maxlags,
                                          normed=True,  usevlines=True,
                                          alpha=0.5)
        ax[1, 2].set_xlim((0, maxlags))
        ax[1, 2].text(0.5, 0.9, 'ESS = {:.2f}'.format(effective_n(x, lags, corr)),
                      fontsize=8,
                      horizontalalignment='left',
                      verticalalignment='center',
                      transform=ax[1, 2].transAxes)

        sns.kdeplot(model.lambdas_[n_burn:].ravel(), ax=ax[2, 0], shade=True,
                    color=colors[2])
        ax[2, 0].set_title(r'Blending Coefficient $\lambda$', fontsize=fontsize)
        ax[2, 1].plot(model.lambdas_, c=colors[2])

        x = model.lambdas_.ravel()[n_burn:]
        lags, corr, _, _ = ax[2, 2].acorr(x - np.mean(x), maxlags=maxlags,
                                          normed=True, usevlines=True,
                                          alpha=0.5)
        ax[2, 2].set_xlim((0, maxlags))
        ax[2, 2].text(0.5, 0.9, 'ESS = {:.2f}'.format(effective_n(x, lags, corr)),
                      fontsize=8,
                      horizontalalignment='left',
                      verticalalignment='center',
                      transform=ax[2, 2].transAxes)

    return fig, ax


def plot_posterior_counts(model, t=0, bar_width=0.25, normalize=False,
                         fontsize=16, ticksize=14, figsize=(10, 6),
                         include_title=True):
    fig, ax = plt.subplots(figsize=figsize)

    freq = model.posterior_group_counts_[t]
    index = model.posterior_group_ids_[t]
    if normalize:
        freq = freq / freq.sum()

    ax.bar(index, freq, width=bar_width)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xticks(index.astype(int))
    ax.tick_params(labelsize=ticksize)

    if normalize:
        ax.set_ylabel('Posterior probability', fontsize=fontsize)
    else:
        ax.set_ylabel('# of samples')
    ax.set_xlabel('Number of groups', fontsize=fontsize)

    if t != 'all' and include_title:
        ax.set_title('t = {}'.format(t + 1), fontsize=fontsize)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    return fig, ax


def plot_model_selection(model, figsize=(8, 6), include_bic=True, maxlags=100,
                         bar_width=0.5, normalize=False):
    freq = np.bincount(model.counts_)
    index = np.where(freq != 0)[0]
    freq = freq[index]

    if include_bic:
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=figsize, sharex=True)

        # BIC plot
        ax[0].plot(model.bic_[:, 0].astype(np.int), model.bic_[:, 1], '--ko')
        ax[0].set_ylabel('Approximate BIC')

        # posterior distribution of model sizes
        ax[1].bar(index, freq, width=bar_width)
        ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
        ax[1].set_xticks(index.astype(int))

        ax[1].set_ylabel('# of samples')
        ax[1].set_xlabel('# of groups')
    else:
        fig, ax = plt.subplots(figsize=figsize)

        if normalize:
            freq = freq / freq.sum()

        ax.bar(index, freq, width=bar_width)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xticks(index.astype(int))

        if normalize:
            ax.set_ylabel('Posterior probability')
        else:
            ax.set_ylabel('# of samples')
        ax.set_xlabel('Number of groups')

        n_burn = model.n_burn_
        lambdas = model.lambdas_[n_burn:].ravel()
        lambda_post = np.mean(lambdas)

        n_samples = effective_n(lambdas, maxlags=maxlags)
        lambda_std = np.std(lambdas) / np.sqrt(n_samples)
        ax.text(0.8, 0.9,
                r'$\lambda = {:.3f} \pm {:.3f}$'.format(lambda_post, lambda_std),
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transAxes,
                fontsize=16)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    return fig, ax


def plot_model_parameters(model, figsize=(10, 8), fontsize=8,
                          param_fontsize=8, zero_threshold=1e-3,
                          include_likelihood_params=True,
                          text_map=None):
    n_time_steps = model.Y_fit_.shape[0]

    fig = plt.figure(figsize=figsize)

    ncols = 2 if n_time_steps == 3 else 3
    nrows = ((n_time_steps - 1) // ncols) + 1
    height_ratios = [.2] + [.8 / (nrows - 1)] * (nrows - 1)
    gs = gridspec.GridSpec(nrows=nrows, ncols=ncols,
                           height_ratios=height_ratios)

    # form axes
    ax = [plt.subplot(gs[i, j]) for i in range(nrows) for j in range(ncols)]
    ax = np.array(ax).reshape(nrows, ncols)

    # set fontsizes
    for a in ax.ravel():
        if a is not None:
            [l.set_fontsize(fontsize) for l in a.get_yticklabels()]
            [l.set_fontsize(fontsize) for l in a.get_xticklabels()]

    if include_likelihood_params:
        # plot parameters
        ax[0, 0].annotate(r'$\lambda =$ {:.2f}'.format(model.lambda_[0]),
                          (0.6, 0.3), fontsize=12)

        if model.is_directed:
            ax[0, 0].annotate(r'$\beta_{in} =$' + ' {:.2f}'.format(model.intercept_[0]),
                              (0.6, 0.9), fontsize=12)
            ax[0, 0].annotate(r'$\beta_{out} =$' + ' {:.2f}'.format(model.intercept_[1]),
                              (0.6, 0.6), fontsize=12)
        else:
            ax[0, 0].annotate(r'$\beta_0 =$ {:.2f}'.format(model.intercept_[0]),
                              (0.5, 0.6), fontsize=12)
        ax[0, 0].annotate(r'AUC = {:.2f}'.format(model.auc_),
                          (0, 0.6), fontsize=12)

        ax[0, 0].axis('off')

    param_start = 1 if include_likelihood_params else 0
    # beta plot
    beta = model.beta_.reshape(1, -1).copy()
    beta[beta < zero_threshold] = 0.0
    sns.heatmap(beta, cmap='rocket_r', linewidths=10.0, square=True,
                cbar=False, yticklabels=False, vmin=0.0, vmax=1.0, annot=True,
                annot_kws={"fontsize" : 8}, ax=ax[0, param_start],
                xticklabels=text_map if text_map else 'auto')
    ax[0, param_start].set_title(r'$\beta$')

    # init_w plot

    w = model.init_weights_.reshape(1, -1).copy()
    w[w < zero_threshold] = 0.0
    active_clusters = np.unique(model.z_[0])
    mask = np.ones_like(w)
    mask[:, active_clusters] = 0.0

    sns.heatmap(w, cmap='rocket_r', linewidths=10.0, square=True,
                cbar=False, vmin=0.0, vmax=1.0, yticklabels=False,
                annot=True, annot_kws={"fontsize" : 8}, mask=mask,
                ax=ax[0, param_start + 1],
                xticklabels=text_map if text_map else 'auto')
    ax[0, param_start + 1].set_title(r'$p(z_0)$')

    if not include_likelihood_params and ncols == 3:
        ax[0, 2].axis('off')

    # plot remaining transition weights
    for t in range(1, n_time_steps):
        row_id = ((t - 1) // ncols) + 1
        col_id = (t - 1) % ncols

        w = model.trans_weights_[t].copy()
        w[w < zero_threshold] = 0.0
        active_clusters = np.unique(model.z_[t])
        mask = np.ones_like(w)
        ind = np.array(list(itertools.product(active_clusters, active_clusters)))
        mask[ind[:, 0], ind[:, 1]] = 0.0

        sns.heatmap(w, cmap='rocket_r', linewidths=10.0, square=True,
                    cbar=False, vmin=0.0, vmax=1.0, mask=mask,
                    annot=True, annot_kws={"fontsize" : param_fontsize},
                    xticklabels=text_map if text_map else 'auto',
                    yticklabels=text_map if text_map else 'auto',
                    ax=ax[row_id, col_id])
        ax[row_id, col_id].set_title(r'$p(z_{0} \, | \, z_{1})$'.format(t, t-1),
                                     fontsize=fontsize)
        ax[row_id, col_id].set_ylabel('$z_{}$'.format(t-1), fontsize=fontsize)
        ax[row_id, col_id].set_xlabel('$z_{}$'.format(t), fontsize=fontsize)

    # turn the axes off for the remaining plots
    last_col = (n_time_steps - 1) % ncols
    if last_col < ncols:
        for i in range(last_col, ncols):
            ax[nrows-1, i].axis('off')

    return fig, ax

def plot_clusters_over_time(model):
    n_time_steps = model.Y_fit_.shape[0]
    n_clusters = np.unique(model.z_.ravel()).shape[0]

    cluster_counts = np.zeros((n_time_steps, n_clusters), dtype=np.float64)
    for t in range(n_time_steps):
        pass


def plot_distance_comparison(model, X_true, intercept, radii=None,
                             z_true=None,
                             log_scale=True, figsize=(10, 6)):
    if not hasattr(model, 'X_'):
        raise ValueError('Model not fit.')

    dist_fit, dist_true = [], []

    triu_indices = triu_indices_from_3d(model.Y_fit_, k=1)
    dist_fit = model.distances_[triu_indices]
    dist_true = calculate_distances(X_true)[triu_indices]

    if model.is_directed:
        dist = calculate_distances(X_true)
        proba_true = directed_network_probas(dist, radii,
                                             intercept[0], intercept[1])
    else:
        proba_true = expit(intercept - calculate_distances(X_true))
    auc_true = network_auc(model.Y_fit_, proba_true)

    fig, ax = plt.subplots(figsize=(10, 6))

    ratio = np.log(dist_fit / dist_true) if log_scale else dist_fit / dist_true
    sns.distplot(ratio, hist_kws={'edgecolor': 'white'},
                 ax=ax)

    if log_scale:
        ax.set_xlabel(r'$\log(\hat{d}_{tij}/d_{tij})$')
    else:
        ax.set_xlabel(r'$\hat{d}_{tij}/d_{tij}$')

    ax.annotate('True AUC = {:.3f}'.format(auc_true), (-3, 1.5))
    ax.annotate('Fitted AUC = {:.3f}'.format(model.auc_), (-3, 1.4))

    if z_true is not None:
        rand_score = adjusted_rand_score(z_true.ravel(), model.z_.ravel())
        ax.annotate('Adjusted Rand = {:.2f}'.format(rand_score), (-3, 1.3))

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


def draw_edge(x1, x2, ax, is_directed=False, **kwargs):
    if is_directed:
        dx = x2 - x1
        ax.arrow(x1[0], x1[1], dx[0], dx[1], **kwargs)
    else:
        ax.plot([x1[0], x2[0]], [x1[1], x2[1]], **kwargs)


def arrow_patch(x1, x2, source_size, target_size, ax, **kwargs):
    shrink_source = np.sqrt(source_size) / 2
    shrink_target = np.sqrt(target_size) / 2

    arrow = FancyArrowPatch(x1, x2,
                            shrinkA=shrink_source,
                            shrinkB=shrink_target,
                            **kwargs)

    ax.add_patch(arrow)

def draw_trajectory(X, z, target_size, color, ax, mask_groups, alpha=1.0):
    shrink_target = np.sqrt(target_size) / 2
    for t in range(X.shape[0] - 1):
        if z[t + 1] not in mask_groups and z[t] not in mask_groups:
            arrow = FancyArrowPatch(X[t], X[t + 1],
                                    shrinkB=shrink_target,
                                    arrowstyle='-|>',
                                    mutation_scale=10,
                                    color=color[t+1],
                                    alpha=alpha,
                                    zorder=1)
            ax.add_patch(arrow)


def plot_network_embedding(model, t=0, estimate_type='best',
                           only_show_connected=False,
                           figsize=(10, 6), border=0.1,
                           head_width=0.003, linewidth=0.001, text_map=None,
                           node_size=40, center_size=300,
                           alpha=0.8, title_text='auto',
                           arrowstyle='-|>', connectionstyle=None,
                           mutation_scale=10, number_nodes=False,
                           group_title_offset=0, textsize=10, size_cutoff=1,
                           plot_group_sigma=False, mask_groups=None,
                           node_names=None, use_radii=True,
                           node_textsize=10, repel_strength=0.5,
                           group_id=None, colors=None, sample_id=None):

    fig, ax = plt.subplots(figsize=figsize)

    if mask_groups is not None:
        mask_groups = np.asarray(mask_groups)

    if only_show_connected:
        mask = connected_nodes(model.Y_fit_[t],
                               is_directed=model.is_directed,
                               size_cutoff=size_cutoff)
    else:
        mask = np.full(model.Y_fit_.shape[1], True)

    if sample_id is not None:
        z = model.zs_[sample_id]
        X = model.Xs_[sample_id]
        mu = model.mus_[sample_id]
        sigma = model.sigmas_[sample_id]
        if model.is_directed:
            radii = model.radiis_[sample_id]
    elif estimate_type == 'best':
        z = model.z_
        X = model.X_
        mu = model.mu_
        sigma = model.sigma_
        if model.is_directed:
            radii = model.radii_
    else:
        z = model.z_
        X = model.X_mean_
        mu = model.mu_
        sigma = model.sigma_
        if model.is_directed:
            radii = model.radii_mean_

    encoder = LabelEncoder().fit(z.ravel())
    colors = get_colors(z.ravel()) if colors is None else np.asarray(colors)

    #xy_min = np.min(model.X_, axis=(0, 1)) - border
    #xy_max = np.max(model.X_, axis=(0, 1)) + border
    xy_min = np.min(X[t, mask], axis=0)
    xy_max = np.max(X[t, mask], axis=0)

    for ts in range(model.Y_fit_.shape[0]):
        if only_show_connected:
            mask_t = connected_nodes(model.Y_fit_[ts],
                                   is_directed=model.is_directed,
                                   size_cutoff=size_cutoff)
        else:
            mask_t = np.arange(model.Y_fit_.shape[1])

        xy_min = np.minimum(xy_min, np.min(X[ts, mask_t], axis=0))
        xy_max = np.maximum(xy_max, np.max(X[ts, mask_t], axis=0))

    xy_min -= border
    xy_max += border

    ax.set_aspect('equal', 'box')
    ax.axis('off')
    ax.set_xlim(xy_min[0], xy_max[0])
    ax.set_ylim(xy_min[1], xy_max[1])

    if model.is_directed:
        row, col = nondiag_indices_from(model.Y_fit_[t])
    else:
        row, col = np.triu_indices_from(model.Y_fit_[t])

    if model.is_directed and use_radii:
        sizes = radii / radii.min() * node_size
    else:
        sizes = node_size

    for i, j in zip(row, col):
        if model.Y_fit_[t, i, j] == 1.0:
            x1 = X[t, i]
            x2 = X[t, j]

            if model.is_directed:
                arrow_patch(x1, x2, sizes[i], sizes[j], ax,
                           color=colors[encoder.transform([z[t, i]])[0]],
                           alpha=alpha,
                           connectionstyle=connectionstyle,
                           linewidth=linewidth,
                           mutation_scale=mutation_scale,
                           arrowstyle=arrowstyle,
                           zorder=1)
            else:
                arrow_patch(x1, x2, sizes, sizes, ax,
                           color=colors[encoder.transform([z[t, i]])[0]],
                           alpha=alpha,
                           connectionstyle=connectionstyle,
                           linewidth=linewidth,
                           mutation_scale=mutation_scale,
                           arrowstyle='-',
                           zorder=1)

    ax.scatter(X[t, mask, 0], X[t, mask, 1],
               c=colors[encoder.transform(z[t, mask])],
               alpha=alpha,
               edgecolor='white',
               s=sizes,
               zorder=2)

    # label nodes
    if number_nodes:
        #for i in range(X.shape[1]):
            #if mask[i]:
            #    ax.annotate(node_names[i] + ' ({})'.format(i) if node_names is not None else str(i),
            #                (X[t, i, 0], X[t, i, 1]),
            #                size=node_textsize,
            #                alpha=0.9,
            #                xycoords='data',
            #                zorder=3)
        repel_labels(X[t], node_names, datasize=sizes, k=repel_strength,
                     textsize=node_textsize, mask=mask, ax=ax)


    # annotate group number
    for k in np.unique(z[t, mask]):
        if mask_groups is None or k not in mask_groups:
            if estimate_type == 'mean':
                muk = np.mean(X[t, z[t] == k], axis=0)
            else:
                muk = mu[k]

            ax.annotate(str(k + 1) if text_map is None else text_map[k],
                        (muk[0] + group_title_offset, muk[1]),
                        bbox=dict(boxstyle='round', alpha=0.3,
                                  color=colors[encoder.transform([k])[0]]),
                        size=textsize,
                        xycoords='data',
                        zorder=2)

            # also plot cluster center
            ax.scatter(muk[0], muk[1],
                        #color=colors[encoder.transform([k])[0]],
                        color='k',
                        s=center_size,
                        marker='P',
                        alpha=0.8,
                        zorder=2)

            if plot_group_sigma:
                normal_contour(muk, sigma[k] * np.eye(2),
                               n_std=[1, 2], ax=ax,
                               linewidth=3 * linewidth,
                               linestyle='--',
                               facecolor=colors[encoder.transform([k])[0]],
                               #color=colors[encoder.transform([k])[0]],
                               edgecolor='k',
                               alpha=0.15,
                               zorder=1)

    if title_text == 'auto':
        ax.set_title('t = {}'.format(t + 1), size=80)
    elif title_text:
        ax.set_title(title_text)

    return fig, ax


def plot_group_embedding(model, t=0, figsize=(10, 6), border=2,
                         linewidth=0.5, node_size=50,
                         text_map=None, title_text=None,
                         colors=None, alpha=0.5,
                         arrowstyle='-|>', connectionstyle=None,
                         mutation_scale=10,
                         only_show_connected=True):
    # NOTE: only works for undirected graphs!
    fig, ax = plt.subplots(figsize=figsize)

    # extract stored graph
    Y = model.Y_fit_[t]

    if only_show_connected:
        connected_ids = np.where(Y.sum(axis=0) > 0)[0]
        freq = np.bincount(model.z_[t][connected_ids])
    else:
        freq = np.bincount(model.z_[t])

    # only plot groups with assigned nodes
    active_groups = np.where(freq > 0)[0]

    # NOTE: should set limit aver any group that were ever  active
    total_freq = np.zeros(model.n_components)
    for ts in range(model.Y_fit_.shape[0]):
        if only_show_connected:
            connected_ids_t = np.where(model.Y_fit_[ts].sum(axis=0) > 0)[0]
            freq_t = np.bincount(model.z_[ts][connected_ids_t])
        else:
            freq_t = np.bincount(model.z_[ts])
        index_t = np.where(freq_t > 0)[0]
        total_freq[index_t] += freq_t[index_t]
    ever_active_groups = np.where(total_freq > 0)[0]

    xy_min = np.min(model.mu_[ever_active_groups], axis=0) - border
    xy_max = np.max(model.mu_[ever_active_groups], axis=0) + border

    ax.set_aspect('equal', 'box')
    ax.axis('off')
    ax.set_xlim(xy_min[0], xy_max[0])
    ax.set_ylim(xy_min[1], xy_max[1])

    # draw edges
    sizes = node_size * freq[active_groups]
    colors = get_colors(model.z_.ravel())

    group_combos = itertools.product(active_groups, active_groups)
    for i, j in group_combos:
        group_mask_i = model.z_[t] == i
        group_mask_j = model.z_[t] == j
        weight_ij = Y[group_mask_i][:, group_mask_j].sum()
        if not model.is_directed:
            if i < j and weight_ij > 0:
                draw_edge(model.mu_[i], model.mu_[j], ax,
                          is_directed=False,
                          linewidth=linewidth * weight_ij,
                          alpha=0.3,
                          color='gray',
                          zorder=1)
        else:
            if i != j and weight_ij > 0:
                arrow_patch(model.mu_[i], model.mu_[j], sizes[i], sizes[j], ax,
                           color=colors[i],
                           alpha=alpha,
                           connectionstyle=connectionstyle,
                           linewidth=linewidth * weight_ij,
                           mutation_scale=mutation_scale,
                           arrowstyle=arrowstyle,
                           zorder=1)

    # draw group centers
    ax.scatter(model.mu_[active_groups, 0], model.mu_[active_groups, 1],
               c=colors[active_groups], s=node_size * freq[active_groups],
               edgecolor='white', zorder=2)

    # annotate groups
    for k in active_groups:
        ax.annotate(str(k) if text_map is None else text_map[k],
                    (model.mu_[k, 0], model.mu_[k, 1]),
                    xycoords='data',
                    bbox=dict(boxstyle='round', alpha=0.3, color=colors[k]))

    if title_text:
        ax.set_title(title_text)
    else:
        ax.set_title('t = {}'.format(t))

    return fig, ax


def plot_trajectories(model, only_show_connected=True, figsize=(10, 6), colors=None,
                      plot_group_sigma=True, linewidth=0.5, h_margin=0, v_margin=0,
                      node_size=100,
                      center_size=300,
                      mask_groups=None, number_nodes=False,
                      include_ids=None, highlight_ids=None,
                      node_names=None):
    fig, ax = plt.subplots(figsize=figsize)

    if mask_groups is not None:
        mask_groups = np.asarray(mask_groups)

    z = model.z_
    X = model.X_
    mu = model.mu_
    sigma = model.sigma_

    encoder = LabelEncoder().fit(z.ravel())
    colors = get_colors(z.ravel()) if colors is None else np.asarray(colors)

    xy_min, xy_max = np.inf, -np.inf
    node_masks = []
    for ts in range(model.Y_fit_.shape[0]):
        if only_show_connected:
            node_masks.append(connected_nodes(model.Y_fit_[ts],
                                              is_directed=model.is_directed,
                                              size_cutoff=2))
        else:
            node_masks.append(np.arange(model.Y_fit_.shape[1]))

        xy_min = np.minimum(xy_min, np.min(X[ts, node_masks[ts]], axis=0))
        xy_max = np.maximum(xy_max, np.max(X[ts, node_masks[ts]], axis=0))

    ax.set_aspect('equal', 'box')
    ax.axis('off')

    ax.set_xlim(xy_min[0] - h_margin, xy_max[0] + h_margin)
    ax.set_ylim(xy_min[1] - v_margin, xy_max[1] + v_margin)

    if highlight_ids is None:
        highlight_ids = np.arange(X.shape[1])
    else:
        highlight_ids = np.asarray(highlight_ids)

    for i in range(X.shape[1]):
        if i in highlight_ids:
            alpha = 1.0
        elif include_ids is not None and i in include_ids:
            alpha = 0.3
        else:
            alpha = 0.0

        draw_trajectory(X[:, i, :],
                        z=z[:, i],
                        target_size=node_size,
                        mask_groups=mask_groups,
                        alpha=alpha,
                        color=colors[encoder.transform(model.z_[:, i])],
                        ax=ax)

    for t in range(X.shape[0]):
        highlight_mask = np.in1d(np.arange(X.shape[1]), highlight_ids)
        ax.scatter(X[t, highlight_mask, 0], X[t, highlight_mask, 1],
                   c=colors[encoder.transform(z[t, highlight_mask])],
                   alpha=1.0,
                   edgecolor='white',
                   s=node_size,
                   zorder=2)

        #if np.any(~highlight_mask):
        #    ax.scatter(X[t, ~highlight_mask, 0], X[t, ~highlight_mask, 1],
        #               c=colors[encoder.transform(z[t, ~highlight_mask])],
        #               alpha=0.3,
        #               edgecolor='white',
        #               s=node_size,
        #               zorder=2)
        if include_ids is not None:
            ax.scatter(X[t, include_ids, 0], X[t, include_ids, 1],
                       c=colors[encoder.transform(z[t, include_ids])],
                       alpha=0.3,
                       edgecolor='white',
                       s=node_size,
                       zorder=2)


        if number_nodes:
            for i in range(X.shape[1]):
                # find first time in non-masked group
                if mask_groups is not None:
                    t_start = np.where(~np.in1d(z[:, i], mask_groups))[0]
                    t_start = t_start[0] if len(t_start) else 0
                else:
                    t_start = 0

                if t == t_start:
                    label_str = node_names[i] + ' ({})'.format(i) if node_names is not None else str(i)
                else:
                    label_str = str(i)
                ax.annotate(label_str,
                            xy=(X[t, i, 0], X[t, i, 1]),
                            alpha = 0.9 if i in highlight_ids else 0.0,
                            xycoords='data')

    if plot_group_sigma:
        for k in np.unique(z.ravel()):
            if k not in mask_groups:
                muk = mu[k]

                # also plot cluster center
                ax.scatter(muk[0], muk[1],
                            color=colors[encoder.transform([k])[0]],
                            s=center_size,
                            marker='+',
                            alpha=0.8,
                            zorder=1)

                normal_contour(muk, sigma[k] * np.eye(2),
                               n_std=[1, 2], ax=ax,
                               linewidth=3 * linewidth,
                               linestyle='--',
                               fill=colors[encoder.transform([k])[0]],
                               color=colors[encoder.transform([k])[0]],
                               alpha=0.1,
                               zorder=1)

    return fig, ax


def transition_freqs(z0, z1, n_groups):
    counts = np.zeros((n_groups, n_groups))

    # get unique values
    groups0 = np.unique(z0)
    groups1 = np.unique(z1)

    n_groups0 = groups0.shape[0]
    n_groups1 = groups1.shape[0]

    for group_id in groups0:
        mask = z0 == group_id
        group_count = np.sum(mask)
        freqs = np.bincount(z1[z0 == group_id])
        index = np.where(freqs != 0)[0]
        counts[group_id, index] = freqs[index]

    with np.errstate(invalid='ignore', divide='ignore'):
        freq_from = counts / counts.sum(axis=1).reshape(-1, 1)
        freq_to = counts / counts.sum(axis=0)

    return freq_from, freq_to


def alluvial_plot(z, figsize=(10, 6), margin=0.01, rec_width=0.01, alpha=0.5,
                  edgecolor='black', text_map=None):
    fig, ax = plt.subplots(figsize=figsize)

    n_time_steps, n_nodes = z.shape

    # plot dimensions
    canvas_height = 1.0  # canvas is 1.0 w x 1.0 h
    spacing = 1.0 / n_time_steps  # spacing between columns

    # integer encode labels (make them contiguous)
    encoder = LabelEncoder().fit(z.ravel())
    z = encoder.transform(z.ravel()).reshape(n_time_steps, n_nodes)
    n_groups = encoder.classes_.shape[0]

    if n_groups <= 20:
        colors = get_color20()
    else:
        colors = sns.color_palette('husl', n_colors=n_groups)

    # determine height of group partitions
    rec_heights = np.zeros((n_time_steps, n_groups, 2), dtype=np.float64)
    for t in range(n_time_steps):
        # groups and their frequency activate at time t
        group_ids = np.unique(z[t])
        n_groups_t = group_ids.shape[0]
        group_freq = np.bincount(z[t]) / n_nodes
        column_height = canvas_height - (n_groups_t + 1) * margin

        rec_height_prev = margin
        for group_id in group_ids:
            rec_height = column_height * group_freq[group_id]
            rec_heights[t, group_id, 0] = rec_height_prev  # start
            rec_heights[t, group_id, 1] = rec_height + rec_height_prev  # end
            rec_height_prev += rec_height + margin
    rec_extents = rec_heights[:, :, 1] - rec_heights[:, :, 0]

    # actually draw the figure
    for t in range(n_time_steps):
        # draw rectangles
        group_ids = np.unique(z[t])
        for group_id in group_ids:
            rec = Rectangle((spacing * t, rec_heights[t, group_id, 0]),
                            width=rec_width,
                            height=rec_extents[t, group_id],
                            facecolor=colors[group_id], alpha=alpha,
                            edgecolor=edgecolor)
            ax.add_patch(rec)

            # add labels
            ax.annotate(str(group_id + 1) if text_map is None else text_map[group_id],
                        (spacing * t + rec_width * 2,
                         rec_heights[t, group_id, 0] + rec_extents[t, group_id]/2.),
                        bbox=dict(boxstyle='round', alpha=0.3, color=colors[group_id]),
                        xycoords='data')

        # draw flow lines
        if t < n_time_steps - 1:
            # divide each group acording to where they came and going to
            freq_from, freq_to = transition_freqs(z[t], z[t + 1], n_groups)

            # x-values for spline fit and plotting
            x_start = spacing * t + rec_width
            x_end = spacing * (t + 1)
            x = np.array([x_start, (x_start + x_end) / 2., x_end])
            x_curve = np.linspace(x_start, x_end, 100)

            # track where to group heights end
            height_end = rec_heights[t + 1, :, 0].copy()
            for group_id_from in group_ids:
                y_start = rec_heights[t, group_id_from, 0]
                y_end = 0
                groups_to = np.where(freq_from[group_id_from, :] != 0)[0]
                for group_id_to in groups_to:
                    # bottom curve
                    y_end = height_end[group_id_to]
                    y = np.array([y_start, (y_start + y_end) / 2., y_end])
                    f_bottom = CubicSpline(x, y, bc_type='clamped')

                    # top curve
                    y_end = ((freq_to[group_id_from, group_id_to] *
                                rec_extents[t + 1, group_id_to]) +
                                height_end[group_id_to])
                    y_start = ((freq_from[group_id_from, group_id_to] *
                                rec_extents[t, group_id_from]) +
                                y_start)
                    y = np.array([y_start, (y_start + y_end) / 2., y_end])
                    f_top = CubicSpline(x, y, bc_type='clamped')

                    # update height_end
                    height_end[group_id_to] = y_end

                    # plot flow line
                    ax.fill_between(x_curve, f_bottom(x_curve), f_top(x_curve),
                                    alpha=0.25, color=colors[group_id_from],
                                    edgecolor=edgecolor)
    ax.axis('off')
    ax.set_xlim(0, 1)

    return fig, ax


def plot_posterior_cooccurrence(model, t=0, label_type='map', threshold=0.5,
                                colors=None, cmap='rocket',
                                mask_threshold=None, sample_id=None):

    # calculate coocurrence probabilities
    cooccurence_proba = model.cooccurrence_probas_[t]

    # hierarchical clustering with average linkage
    linkage = hc.linkage(squareform(1. - cooccurence_proba), method='average',
                         optimal_ordering=True)

    if label_type == 'linkage':
        z = hc.fcluster(linkage, t=threshold, criterion='distance') - 1
    else:
        if sample_id is None:
            z = model.z_[t]
        else:
            z = model.zs_[sample_id, t]

    encoder = LabelEncoder().fit(z)
    colors = get_colors(z) if colors is None else colors

    mask = cooccurence_proba <= mask_threshold if mask_threshold is not None else None
    cg = sns.clustermap(cooccurence_proba,
                        row_linkage=linkage, col_linkage=linkage,
                        row_colors=colors[encoder.transform(z)],
                        cmap=cmap,
                        mask=mask)

    # remove redundant side dendogram
    cg.ax_row_dendrogram.set_visible(False)

    return z


def plot_adjacency_matrix_from_model(model, t=0, figsize=(8, 6)):
    Y = model.Y_fit_[t].copy()
    z = model.z_[t].copy()

    return plot_adjacency_matrix(Y, z, figsize=figsize)


def plot_adjacency_matrix(Y, z, figsize=(8, 6)):
    fig, ax = plt.subplots(figsize=figsize)

    Y = Y.copy()
    z = z.copy()

    # re-order in terms of partitioning
    order = np.argsort(z)
    Y = Y[order, :][:, order]

    # plot binary matrix
    ax.imshow(Y, cmap=plt.cm.Blues)

    # draw lines partitioning groups
    group_ids = np.unique(z)
    for k in group_ids:
        idx = np.where(z[order] == k)[0][-1]
        if k < group_ids[-1]: # don't draw last line
            plt.vlines(idx + 0.5, 0, Y.shape[0], linewidth=0.5)

        if k < group_ids[-1]: # don't draw last line
            plt.hlines(idx + 0.5, 0, Y.shape[0], linewidth=0.5)

    ax.set_xlim(0, Y.shape[0] - 0.5)
    ax.set_ylim(0, Y.shape[0] - 0.5)

    return fig, ax
