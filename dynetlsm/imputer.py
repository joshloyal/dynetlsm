import numpy as np
import scipy.stats as stats

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array, check_random_state


from dynetlsm.network_statistics import density


class SimpleNetworkImputer(BaseEstimator, TransformerMixin):
    """Impute missing values in the network by the most frequent value over
    all time points and edges.
    """
    def __init__(self, missing_value=-1, strategy='most_frequent',
                 random_state=123, copy=True):
        self.missing_value = missing_value
        self.strategy = strategy
        self.copy = copy
        self.random_state = random_state

    def _validate_input(self, Y):
        allowed_strategies = {'most_frequent', 'random'}
        if self.strategy not in allowed_strategies:
            raise ValueError("Can only use these strategies: {0} "
                             " got strategy='{1}".format(allowed_strategies,
                                                         self.strategy))
        Y = check_array(Y, dtype=np.float64,
                        force_all_finite='allow-nan',
                        ensure_2d=False, allow_nd=True, copy=self.copy)

        return Y

    def fit(self, Y):
        Y = self._validate_input(Y)

        # statistics are calculated per time point
        n_time_steps = Y.shape[0]
        self.statistics_ = np.empty(n_time_steps)
        for t in range(n_time_steps):
            nan_mask = Y[t] == self.missing_value
            if not np.any(nan_mask):
                self.statistics_[t] = 0.0
            else:
                if self.strategy == 'most_frequent':
                    mode = stats.mode(Y[t][~nan_mask].ravel())
                    self.statistics_[t] = mode[0][0]
                elif self.strategy == 'random':
                    n_nodes = Y.shape[1]
                    self.statistics_[t] = (
                        Y[t][~nan_mask].sum() / (n_nodes * (n_nodes - 1)))

        return self

    def transform(self, Y):
        check_is_fitted(self, 'statistics_')

        Y = self._validate_input(Y)

        if Y.shape[0] != self.statistics_.shape[0]:
            raise ValueError("Y has %d time steps, expected %d"
                             % (Y.shape[0], self.statistics_.shape[0]))

        rng = check_random_state(self.random_state)
        for t in range(Y.shape[0]):

            if self.strategy == 'random':
                indices = np.triu_indices(Y.shape[1], k=1)
                y_vec = Y[t][indices]
                nan_mask = y_vec == self.missing_value
                y_vec[nan_mask] = rng.choice([0, 1],
                    p=[1 - self.statistics_[t], self.statistics_[t]],
                    size=np.sum(nan_mask))
                Y[t][indices] = y_vec
                indices = np.tril_indices(Y.shape[1], k=-1)
                Y[t][indices] = 0
                Y[t] += Y[t].T
            else:
                nan_mask = Y[t] == self.missing_value
                Y[t][nan_mask] = self.statistics_[t]

        return Y
