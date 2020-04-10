import numpy as np
import pandas as pd

from os.path import dirname, join
from sklearn.preprocessing import LabelEncoder


__all__ = ['load_monks']


def load_monks(dynamic=True, is_directed=True, include_waverers=False,
               encode_labels=True):
    """Loads Sampson's Monastery Network (1968)."""
    if dynamic:
        return load_dynamic_monks(encode_labels, include_waverers,
                                  is_directed=is_directed)
    else:
        return load_static_monks(encode_labels, include_waverers,
                                 is_directed=is_directed)


def load_dynamic_monks(encode_labels=True, include_waverers=False,
                       is_directed=True):
    module_path = dirname(__file__)

    n_time_steps = 3
    Y = np.empty((n_time_steps, 18, 18), dtype=np.float64)

    for t in range(n_time_steps):
        Y[t] = np.loadtxt(join(module_path, 'raw_data',
                               'sampson_{}.npy'.format(t)))
    # load groups
    file_name = ('sampson_groups_waverers.txt' if include_waverers else
                 'sampson_groups.txt')

    with open(join(module_path, 'raw_data', file_name)) as f:
        groups = np.array([l.rstrip('\n') for l in f.readlines()])

    if encode_labels:
        groups = LabelEncoder().fit_transform(groups)

    with open(join(module_path, 'raw_data', 'sampson_names.txt')) as f:
        names = np.array([l.rstrip('\n') for l in f.readlines()])

    if not is_directed:
        Y += Y.transpose((0, 2, 1))
        Y = (Y > 0).astype(np.float64)

    return Y, np.repeat(groups.reshape(1, -1), n_time_steps, axis=0), names


def load_static_monks(encode_labels=True, include_waverers=False,
                      is_directed=True):
    module_path = dirname(__file__)

    Y = np.loadtxt(join(module_path, 'raw_data', 'sampson.npy'))

    # load groups
    file_name = ('sampson_groups_waverers.txt' if include_waverers else
                 'sampson_groups.txt')
    with open(join(module_path, 'raw_data', file_name)) as f:
        groups = np.array([l.rstrip('\n') for l in f.readlines()])

    if encode_labels:
        groups = LabelEncoder().fit_transform(groups)

    if not is_directed:
        Y += Y.transpose((0, 2, 1))
        Y = (Y > 0).astype(np.float64)

    return Y, groups
