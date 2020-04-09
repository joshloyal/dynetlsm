# dynetLSM: Bayesian inference for latent space models of dynamic networks

Background
----------

Latent space models (LSMs) are a powerful approach to modeling network data. One is often interested in inferring properties of nodes in a network based on their connectivity patterns. Originally proposed by Hoff et al. (2002), LSMs learn a latent embedding for each node that captures the similarity between them. This package focuses on embeddings within a Euclidean space so that the log-odds of forming an edge between two nodes is inversely proportional to the distance between their latent positions. In other words, nodes that are close together in the latent space are more likely to form a connection in the observed network. The generative model is as follows:

1. For each node, we sample a node's latent position from a Gaussian distribution:

<p align="center">
<img src="/images/static_lsm_prior.png" alt="latent positions prior" width="200">
</p>

2. For each edge, we sample a connection from a Bernoulli distribution:

<p align="center">
<img src="/images/static_lsm.png" alt="static lsm" width="400">
</p>

For dynamic (time-varying) networks, one is also interested in determining how properties of the nodes change over time. LSMs can also accomplish this task. Sarkar and Moore (2005) and Sewell and Chen (2015) proposed to allow the latent positions to evolve over time through a Gaussian random-walk Markovian process. Based on these latent positions, the edges in the network form in the same way as the static case. The generative process is as follows:

1. For `t = 1`, sample a node's initial latent position from a Gaussian distribution:

<p align="center">
<img src="/images/dynamic_lsm_initial.png" alt="latent positions prior" width="200">
</p>

2. For `t = 2, ..., T`, a node's latent position follows a Gaussian random walk:

<p align="center">
<img src="/images/dynamic_lsm_rw.png" alt="latent positions prior" width="200">
<

3. For each edge, we sample a connection from a Bernoulli distribution:

<p align="center">
<img src="/images/dynamic_lsm.png" alt="static lsm" width="400">
</p>

Determining the high-level community structure of a network is another important task of network analysis.

Example
-------

```python
from dynetlsm import DynamicNetworkLSM

lsm_model = DynamicNetworkLSM(n_features=2)
lsm_model.fit(Y)
```

```python

from dynetlsm import DynamicNetworkHDPLPCM

lpcm_model = DynamicNetworkHDPLPCM(n_features=2, n_components=10)
lpcm_model.fit(Y)
```


Installation
------------

Dependencies
------------
dynetlsm requires:

- Python (>= 2.7 or >= 3.4)
- NumPy (>= 1.8.2)
- SciPy (>= 0.13.3)
- Scikit-learn (>=0.17)

Additionally, to run examples, you need matplotlib(>=2.0.0).

Installation
------------
You need a working installation of numpy and scipy to install dynetlsm. If you have a working installation of numpy and scipy, the easiest way to install dynetlsm is using ``pip``::

```
pip install -U dynetlsm
```

If you prefer, you can clone the repository and run the setup.py file. Use the following commands to get the copy from GitHub and install all the dependencies::

```
git clone https://github.com/joshloyal/dynetlsm.git
cd dynetlsm
pip install .
```

Or install using pip and GitHub::

```
pip install -U git+https://github.com/joshloyal/dynetlsm.git
```


References:
-----------
