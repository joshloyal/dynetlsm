# DynetLSM: Bayesian inference for latent space models of dynamic networks

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

Background
----------

### Latent Space Models

#### Static Networks
Latent space models (LSMs) are a powerful approach to modeling network data. One is often interested in inferring properties of nodes in a network based on their connectivity patterns. Originally proposed by Hoff et al. (2002), LSMs learn a latent embedding for each node that captures the similarity between them. This package focuses on embeddings within a Euclidean space so that the log-odds of forming an edge between two nodes is inversely proportional to the distance between their latent positions. In other words, nodes that are close together in the latent space are more likely to form a connection in the observed network. The generative model is as follows:

1. For each node, sample a node's latent position from a Gaussian distribution:

<p align="center">
<img src="/images/static_lsm_prior.png" alt="latent positions prior" width="200">
</p>

2. For each edge, sample a connection from a Bernoulli distribution:

<p align="center">
<img src="/images/static_lsm.png" alt="static lsm" width="400">
</p>

#### Dynamic Networks
For dynamic (time-varying) networks, one is also interested in determining how properties of the nodes change over time. LSMs can also accomplish this task. Sarkar and Moore (2005) and Sewell and Chen (2015) proposed propagating the latent positions through time with a Gaussian random-walk Markovian process. Based on these latent positions, the edges in the network form in the same way as the static case. The generative process is as follows:

1. For `t = 1`, sample a node's initial latent position from a Gaussian distribution:

<p align="center">
<img src="/images/dynamic_lsm_initial.png" alt="latent positions prior" width="200">
</p>

2. For `t = 2, ..., T`, a node's latent position follows a Gaussian random walk:

<p align="center">
<img src="/images/dynamic_lsm_rw.png" alt="latent positions prior" width="200">


3. For each edge, sample a connection from a Bernoulli distribution:

<p align="center">
<img src="/images/dynamic_lsm.png" alt="static lsm" width="400">
</p>


### Latent Position Clustering Models


#### Static Networks
Determining the high-level community structure of a network is another important task in network analysis. Community structure was incorporated into LSMs by Handcock et al. (2007) with their latent position clustering model (LPCM). Intuitively, the LPCM posits that communities are the result of clustering within the latent space. This clustering is incorporated in the LSM framework by assuming the latent positions are drawn from a Gaussian mixture model, i.e,

<p align="center">
<img src="/images/lpcm.png" alt="latent positions prior" width="225">
</p>

The LPCM relates the latent positions to the probability of forming an edge in the same way as the original LSM. In practice, one interprets nodes that share the same mixture component as belonging to the same community.

#### Dynamic Networks
Inferring a network's community structure is especially difficult for dynamic networks because the number of communities may change over time. If one assumes the number of communities is fixed, then the model of Sewell and Chen (2017) is able to infer a dynamic network's community structure by propagating each nodes mixture assignment through time with a autoregressive hidden Markov model (AR-HMM). However, the assumption of a static number of communities is at odds with many real-world dynamic networks. To solve this problem, Loyal and Chen (2020) proposed using a sticky hierarchical Dirichlet process hidden Markov model (HDP-HMM) with time-inhomogeneous transition probabilities in conjunction with the LPCM to infer evolving community structures in dynamic networks. This model is deemed the hierarchical Dirichlet process latent position clustering model (HDP-LPCM). According to the HDP-LPCM, the latent community labels propagate through time according to iid HDP-HMMs. Unlike previous models, this allows the HDP-LPCM to create or delete communities over-time as well as infer the number of the communities from the data. The generative model is as follows:

1. Draw the time-varying transition probabilities from a sticky-HDP:

<p align="center">
<img src="/images/hdp.png" alt="latent positions prior" width="500">
</p>

2. For `t = 1, ..., T`, propagate a node's latent community label through time according to an HMM:

<p align="center">
<img src="/images/dynamic_label.png" alt="latent positions prior" width="150">
</p>

3. For `t = 1`, sample a node's initial latent position from its assigned Gaussian mixture component:

<p align="center">
<img src="/images/dynamic_lpcm_initial.png" alt="latent positions prior" width="225">
</p>

4. For `t = 2, ..., T`, sample a node's latent position as a mixture between its previous position and its assigned Gaussian mixture component:

<p align="center">
<img src="/images/dynamic_lpcm_rw.png" alt="latent positions prior" width="400">
</p>

5. For each edge, sample a connection from a Bernoulli distribution
:
<p align="center">
<img src="/images/dynamic_lsm.png" alt="static lsm" width="400">
</p>


Example
-------
DynetLSM exposes two classes for working with latent space models for dynamic networks:

* `DynamicNetworkLSM`:  Interface for learning the LSM in Sewell and Chen (2015),
* `DynamicNetworkHDPLPCM`: Interface for learning the HDP-LPCM in Loyal and Chen (2020).

To understand the merits of both approaches, we provide an example using a synthetic dynamic network which contains two communities at `t = 1` and four communities at `t = 2`. We can generate the data as follows:
```python
from dynetlsm.datasets import simple_splitting_dynamic_network

# Y : ndarray, shape (2, 50, 50)
#   The adjacency matrices at each time point
# labels : ndarray, shape  (2, 50) c
#   The true community labels of the nodes at each time point.
Y, labels = simple_splitting_dynamic_network(n_nodes=50, n_time_steps=2)
```

```python
from dynetlsm import DynamicNetworkLSM

lsm = DynamicNetworkLSM(n_features=2)
lsm.fit(Y)
```

```python

from dynetlsm import DynamicNetworkHDPLPCM

lpcm = DynamicNetworkHDPLPCM(n_features=2, n_components=10)
lpcm.fit(Y)
```




References:
-----------
