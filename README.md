[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/joshloyal/dynetlsm/blob/master/LICENSE)
[![Travis](https://travis-ci.com/joshloyal/dynetlsm.svg?token=gTKqq3zSsip89mhYVQPZ&branch=master)](https://travis-ci.com/joshloyal/dynetlsm)
[![AppVeyor](https://ci.appveyor.com/api/projects/status/github/joshloyal/dynetlsm)](https://ci.appveyor.com/project/joshloyal/dynetlsm/history)

# DynetLSM: latent space models for dynamic networks

*Author: [Joshua D. Loyal](https://joshloyal.github.io/)*

This package provides an interface for learning and inference in latent
space models for dynamic networks. Inference is performed using
blocked Metropolis-Hastings within Gibbs sampling.

The primary method implemented in this package is the hierarchical Dirichlet
process latent postilion clustering model (HDP-LPCM) described in
"A Bayesian nonparametric latent space approach to modeling evolving communities in
dynamic networks" (Link: [arXiv:2003.07404](https://arxiv.org/abs/2003.07404)).

BibTeX reference to cite, if you use this package:
```bibtex
@article{loyal2020hdplpcm,
    title = {A Bayesian nonparametric latent space approach to modeling evolving communities in dynamic networks},
    author = {Loyal, Joshua Daniel and Chen, Yuguo},
    journal = {arXiv preprint arXiv:2003.07404},
    year = {2020},
}
```

Dependencies
------------
DynetLSM requires:

- Python (>= 3.7)

and the requirements highlighted in [requirements.txt](requirements.txt).

Installation
------------
You need a working installation of numpy and scipy to install DynetLSM. If you have a working installation of numpy and scipy, the easiest way to install dynetlsm is using ``pip``:

```
pip install -U dynetlsm
```

If you prefer, you can clone the repository and run the setup.py file. Use the following commands to get the copy from GitHub and install all the dependencies:

```
git clone https://github.com/joshloyal/dynetlsm.git
cd dynetlsm
pip install .
```

Or install using pip and GitHub:

```
pip install -U git+https://github.com/joshloyal/dynetlsm.git
```

Background
----------

### Latent Space Models

#### Static Networks
Latent space models (LSMs) are a powerful approach to modeling network data. One is often interested in inferring properties of nodes in a network based on their connectivity patterns. Originally proposed by Hoff et al. (2002)<sup>[[1]](#References)</sup>, LSMs learn a latent embedding for each node that captures the similarity between them. This package focuses on embeddings within a Euclidean space so that the log-odds of forming an edge between two nodes is inversely proportional to the distance between their latent positions. In other words, nodes that are close together in the latent space are more likely to form a connection in the observed network. The generative model is as follows:

1. For each node, sample a node's latent position from a Gaussian distribution:

<p align="center">
<img src="/images/static_lsm_prior.png" alt="latent positions prior" width="200">
</p>

2. For each edge, sample a connection from a Bernoulli distribution:

<p align="center">
<img src="/images/static_lsm.png" alt="static lsm" width="400">
</p>

#### Dynamic Networks
For dynamic (time-varying) networks, one is also interested in determining how properties of the nodes change over time. LSMs can also accomplish this task. Sarkar and Moore (2005)<sup>[[2]](#References)</sup> and Sewell and Chen (2015)<sup>[[3]](#References)</sup> proposed propagating the latent positions through time with a Gaussian random-walk Markovian process. Based on these latent positions, the edges in the network form in the same way as the static case. The generative process is as follows:

1. For `t = 1`, sample a node's initial latent position from a Gaussian distribution:

<p align="center">
<img src="/images/dynamic_lsm_initial.png" alt="lsm initial position prior" width="200">
</p>

2. For `t = 2, ..., T`, a node's latent position follows a Gaussian random walk:

<p align="center">
<img src="/images/dynamic_lsm_rw.png" alt="lsm dynamic random walk" width="200">


3. For each edge, sample a connection from a Bernoulli distribution:

<p align="center">
<img src="/images/dynamic_lsm.png" alt="dynamic lsm" width="400">
</p>


### Latent Position Clustering Models


#### Static Networks
Determining the high-level community structure of a network is another important task in network analysis. Community structure was incorporated into LSMs by Handcock et al. (2007)<sup>[[4]](#References)</sup> with their latent position clustering model (LPCM). Intuitively, the LPCM posits that communities are the result of clustering within the latent space. This clustering is incorporated in the LSM framework by assuming the latent positions are drawn from a Gaussian mixture model, i.e,

<p align="center">
<img src="/images/lpcm.png" alt="latent position clustering model" width="225">
</p>

The LPCM relates the latent positions to the probability of forming an edge in the same way as the original LSM. In practice, one interprets nodes that share the same mixture component as belonging to the same community.

#### Dynamic Networks
Inferring a network's community structure is especially difficult for dynamic networks because the number of communities may change over time. If one assumes that the number of communities is fixed, then the model of Sewell and Chen (2017)<sup>[[5]](#References)</sup> is able to infer a dynamic network's community structure by propagating each node's mixture assignment through time with a autoregressive hidden Markov model (AR-HMM). However, the assumption of a static number of communities is at odds with many real-world dynamic networks. It is often the case that the number of communities evolves over time.

To solve the problem of inferring evolving community structures in dynamic networks, Loyal and Chen (2020)<sup>[[6]](#References)</sup> proposed using a sticky hierarchical Dirichlet process hidden Markov model (HDP-HMM) with time-inhomogeneous transition probabilities in conjunction with the LPCM . For this reason, the model is called the hierarchical Dirichlet process latent position clustering model (HDP-LPCM). Under the HDP-LPCM, a node's latent community label propagate through time according to iid HDP-HMMs. Unlike previous models, this allows the HDP-LPCM to create and delete communities over-time as well as infer the number of the communities from the data. The generative model is as follows:

1. Draw the time-varying transition probabilities from a sticky-HDP:

<p align="center">
<img src="/images/hdp.png" alt="sticky-hdp prior" width="500">
</p>

2. For `t = 1, ..., T`, propagate a node's latent community label through time according to an HMM:

<p align="center">
<img src="/images/dynamic_label.png" alt="latent label hmm" width="150">
</p>

3. For `t = 1`, sample a node's initial latent position from its assigned Gaussian mixture component:

<p align="center">
<img src="/images/dynamic_lpcm_initial.png" alt="hdp-lpcm initial positions" width="225">
</p>

4. For `t = 2, ..., T`, sample a node's latent position as a mixture between its previous position and its assigned Gaussian mixture component:

<p align="center">
<img src="/images/dynamic_lpcm_rw.png" alt="hdp-lpcm mixture random walk" width="400">
</p>

5. For each edge, sample a connection from a Bernoulli distribution
:
<p align="center">
<img src="/images/dynamic_lsm.png" alt="hdp-lpcm" width="400">
</p>


Example
-------
DynetLSM exposes two classes for working with latent space models for dynamic networks:

* `DynamicNetworkLSM`:  Interface for learning the LSM in Sewell and Chen (2015)<sup>[[3]](#References)</sup>,
* `DynamicNetworkHDPLPCM`: Interface for learning the HDP-LPCM in Loyal and Chen (2020)<sup>[[6]](#References)</sup>.

To understand the merits of both approaches, we provide an example using a synthetic dynamic network which contains two communities at `t = 1` and four communities at `t = 2`. We can generate the data as follows:
```python
from dynetlsm.datasets import simple_splitting_dynamic_network

# Y : ndarray, shape (2, 50, 50)
#   The adjacency matrices at each time point
# labels : ndarray, shape  (2, 50)
#   The true community labels of the nodes at each time point.
Y, labels = simple_splitting_dynamic_network(n_nodes=50, n_time_steps=2)
```

To fit a dynamic LSM with a 2-dimensional latent space, we initialize the sampler and call `fit`:
```python
from dynetlsm import DynamicNetworkLSM

lsm = DynamicNetworkLSM(n_iter=5000, burn=2500, tune=2500,
                        n_features=2, random_state=42)
lsm.fit(Y)
```

To assess the convergence of the algorithm, we visualize the traces:
```python
from dynetlsm.plots import plot_traces

plot_traces(lsm)
```

<p align="center">
<img src="/images/lsm_traces.png" alt="Traces of the LSM model" width="400">
</p>

We can then visualize the latent space embeddings:
```python
import matplotlib.pyplot as plt

from dynetlsm.plots import plot_latent_space


axes = plt.subplots(ncols=2, nrows=1, figsize=(10, 6))
for t, ax in enumerate(axes.flat):
    plot_latent_space(lsm, t=t, connectionstyle=None, number_nodes=False,
                      linewidth=0.1, node_size=200, border=0.2, ax=ax)
```

<p align="center">
<img src="/images/lsm_latent_space.png" alt="Latent Space of the LSM model" width="400">
</p>

Although the LSM's embedding places nodes that share many connections close together, the true community structure of the network is not apparent. This is easily remedied by applying the HDP-LPCM. As before, we initialize the model and call `fit`:
```python

from dynetlsm import DynamicNetworkHDPLPCM

lpcm = DynamicNetworkHDPLPCM(n_iter=5000, burn=2500, tune=2500,
                             n_features=2, n_components=10, random_state=42)
lpcm.fit(Y)
```

Once again, we assess the convergence of the algorithm by visualizing the traces:
```python
from dynetlsm.plots import plot_traces

plot_traces(lpcm)
```

<p align="center">
<img src="/images/hdp_lpcm_traces.png" alt="Traces of the HDP-LPCM" width="400">
</p>

We can then visualize the latent space embeddings as well as the components of the inferred Gaussian mixture:
```python
import matplotlib.pyplot as plt

from dynetlsm.plots import plot_latent_space


fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(10, 6))
for t, ax in enumerate(axes.flat):
    plot_latent_space(lpcm, t=t, connectionstyle=None,
                      number_nodes=False, border=1.2, linewidth=0.2,
                      center_size=100, node_size=100, ax=ax)
```

<p align="center">
<img src="/images/hdp_lpcm_latent_space.png" alt="Latent Space of the HDP-LPCM" width="500">
</p>
The HDP-LPCM infers an embedding that makes the community structure of the network apparent. Furthermore, the HDP-LPCM correctly infers that the two communities split off into four communities at the second time point. To better visualize this behavior, one can display an alluvial diagram of the label assignments over time:

```python
from dynetlsm.plots import alluvial_plot

alluvial_plot(lpcm.z_)
```

<p align="center">
<img src="/images/alluvial_diagram.png" alt="Alluvial Diagram of the HDP-LPCM" width="300">
</p>

From this diagram, one can see that group 1 primarily splits off into group 3, while group 2 primarily splits off into group 4.

Simulation Studies and Real-Data Applications
---------------------------------------------
This package includes the simulation studies and real-data applications found in Loyal and Chen (2020)<sup>[[6]](#References)</sup>:

* A synthetic dynamic network with a time-homogeneous community structure: ([here](/examples/homogeneous_simulation.py)).
* A synthetic dynamic network with a time-inhomogeneous community structure: ([here](/examples/inhomogeneous_simulation.py)).
* Sampson's monastery network: ([here](/examples/sampson_monks.py)).
* A dynamic network constructed from international military alliances during the first three decades of the Cold War (1950 - 1979): ([here](/examples/military_alliances.py)).
* A dynamic network constructed from character interactions in the first four seasons of the Game of Thrones television series: ([here](/examples/GoT.py)).

We also provide a few [jupyter notebooks](/notebooks) that demonstrate the use of this package.

References
----------

[1]: Hoff, P. D., Raftery, A. E., and Handcock, M. S. (2002). Latent space approaches to social network analysis. *Journal of the American Statistical Association*, 97(460):1090-1098.

[2]: Sarkar, P. and Moore, A. W. (2006). Dynamic social network analysis using latent space models. pages 1145-1152.

[3]: Sewell, D. K. and Chen, Y. (2015). Latent space models for dynamic networks. *Journal of the American Statistical Association*, 110(512):1646-1657.

[4]: Handcock, M. S., Raftery, A. E., and Tantrum, J. M. (2007). Model-based clustering of social networks. *Journal of the Royal Statistical Society A*, 170(2):301-354.

[5]: Sewell, D. K. and Chen, Y. (2017). Latent space approaches to community detection in dynamic networks. *Bayesian Analysis*, 12(2):351-377.

[6]: Loyal, J. D. and Chen, Y. (2020). A Bayesian nonparametric latent space approach to modeling evolving communities in dynamic networks. arXiv preprint arXiv:2003.07404.

