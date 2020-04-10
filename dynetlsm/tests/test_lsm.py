from dynetlsm import DynamicNetworkLSM
from dynetlsm.datasets import simple_splitting_dynamic_network


def test_lsm_smoke():
    Y, labels = simple_splitting_dynamic_network(
        n_nodes=50, n_time_steps=2, random_state=42)

    lsm = DynamicNetworkLSM(n_iter=250, burn=250, tune=250,
                            n_features=2, random_state=123)
    lsm.fit(Y)

    assert lsm.X_.shape == (2, 50, 2)
