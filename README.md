dynetLSM
========
Bayesian inference for latent space models of dynamic networks

Background
----------

Latent space models (LSMs) are a powerful approach to modeling network data. The basic idea is to embed a network into a latent Euclidean space such that closeness in the latent space increases the probability that two nodes form an edge in the observed network. In other words, we associate a latent . The generative model is as follows:

1. For each node:

2. For each edge:

<p align="center">
<img src="/images/static_lsm.pgn" alt="static_lsm" width="300">
</p>

In the dynamic setting we


To infer group structure we


Example
-------

```python

from dynetlsm import DynamicNetworkHDPLPCM

model = DynamicNetworkHDPLPCM(n_features=2, n_components=10)
model.fit(Y)
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
