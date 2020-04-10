from dynetlsm import DynamicNetworkHDPLPCM
from dynetlsm.datasets import simple_splitting_dynamic_network


def test_hdp_lpcm_smoke():
    Y, labels = simple_splitting_dynamic_network(
        n_nodes=50, n_time_steps=2, random_state=42)

    lpcm = DynamicNetworkHDPLPCM(n_iter=250, burn=250, tune=250,
                                 n_features=2, n_components=10,
                                 random_state=123)
    lpcm.fit(Y)

    assert lpcm.X_.shape == (2, 50, 2)
    assert lpcm.z_.shape == (2, 50)
