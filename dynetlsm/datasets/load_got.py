import glob

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp

from os.path import dirname, join

from sklearn.preprocessing import LabelEncoder


__all__ = ['load_got', 'load_got_edgelists']


def network_from_edgelist(edgelist, n_nodes):
    data = np.ones(edgelist.shape[0])
    Y = sp.coo_matrix((data, (edgelist[:, 0], edgelist[:, 1])),
                      shape=(n_nodes, n_nodes)).toarray()

    # symmetriz and binarize network
    Y += Y.T
    Y[Y > 0] = 1

    return Y


def load_got_edgelists():
    module_path = dirname(__file__)
    file_path = join(module_path, 'raw_data', 'got')

    # load edge-lists into one dataframe
    data = pd.concat([
        pd.read_csv(file_name,
                    names=['source', 'target', 'weight', 'season'], skiprows=1)
        for file_name in glob.glob(join(file_path, 'got-s*-edges.csv'))])

    # aggregate multiple edges into a single edge with a weight
    data = data.groupby(['source', 'target', 'season'],
                        as_index=False).agg({'weight': 'sum'})

    return data


def load_got(seasons=None, weight_min=None):
    data = load_got_edgelists()

    if seasons is not None:
        data.query('season == {}'.format(seasons), inplace=True)

    if weight_min is not None:
        data.query('weight >= {}'.format(weight_min), inplace=True)

    # assign integer label ids
    encoder = LabelEncoder().fit(data[['source', 'target']].values.ravel())
    data.loc[:, 'source'] = encoder.transform(data['source'])
    data.loc[:, 'target'] = encoder.transform(data['target'])

    n_seasons = data['season'].unique().shape[0]
    n_nodes = encoder.classes_.shape[0]
    Y = np.zeros((n_seasons, n_nodes, n_nodes))
    for season_id in range(1, n_seasons + 1):
        season_data = data[data['season'] == season_id]
        edgelist = season_data[['source', 'target']].values
        Y[season_id - 1] = network_from_edgelist(edgelist, n_nodes=n_nodes)

    return Y, encoder.classes_
