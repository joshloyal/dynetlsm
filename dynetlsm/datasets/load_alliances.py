import numpy as np
import pandas as pd
import networkx as nx

from os.path import dirname, join


__all__ = ['load_alliances']


def load_alliances(min_degree=1, directed=False, remove_periphery=True):
    module_path = dirname(__file__)
    file_path = join(module_path, 'raw_data', 'military_alliances')

    n_nodes = 180
    n_years = 6
    Y = np.zeros((n_years, n_nodes, n_nodes))

    if directed:
        file_fmt = 'directed_network_{}.npy'
    else:
        file_fmt = 'network_{}.npy'

    for t, year in enumerate(range(1950, 1980, 5)):
        Y[t] = np.loadtxt(join(file_path, file_fmt.format(year)))

    # binarize network
    Y = (Y > 0).astype(np.float64)

    # symmetrize network
    if not directed:
        for t in range(Y.shape[0]):
            Y[t] = (Y[t] + Y[t].T) / 2.
        Y = (Y > 0).astype(np.float64)

    if remove_periphery:
        for t in range(Y.shape[0]):
            G = nx.from_numpy_array(Y[t])
            core_id = np.asarray(list(nx.core_number(G).values()))
            mask = np.where(core_id <= 2)[0]
            Y[t, mask] = 0
            Y[t, :, mask] = 0

    # a country must be active for at least min_degree
    active_ids = np.where(
        (Y.sum(axis=(0, 1)) + Y.sum(axis=(0, 2))) >= min_degree)[0]
    Y = np.ascontiguousarray(Y[:, active_ids][:, :, active_ids])

    # load country names
    names = pd.read_csv(join(file_path, 'names.csv'))
    names = names.values.ravel()[active_ids]

    return np.ascontiguousarray(Y), names
