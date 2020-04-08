import numpy as np
import scipy.stats as stats

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array


class SimpleNetworkImputer(BaseEstimator, TransformerMixin):
    """Impute missing values in the network by the most frequent value over
    all time points and edges.
    """
    def __init__(self, missing_value=np.nan, strategy='most_frequent',
                 copy=True):
        self.missing_value = missing_value
        self.strategy = strategy
        self.copy = copy

    def _validate_input(self, Y):
        allowed_strategies = {'most_frequent'}
        if self.strategy not in allowed_strategies:
            raise ValueError("Can only use these strategies: {0} "
                             " got strategy='{1}".format(allowed_strategies,
                                                         self.strategy))
        Y = check_array(Y, dtype=np.float64, warn_on_dtype=True,
                        force_all_finite='allow-nan',
                        ensure_2d=False, allow_nd=True, copy=self.copy)

        return Y

    def fit(self, Y):
        Y = self._validate_input(Y)

        # statistics are calculated per time point
        n_time_steps = Y.shape[0]
        self.statistics_ = np.empty(n_time_steps)
        for t in range(n_time_steps):
            nan_mask = np.isnan(Y[t])
            if not np.any(nan_mask):
                self.statistics_[t] = 0.0
            else:
                if self.strategy == 'most_frequent':
                    mode = stats.mode(Y[t][nan_mask].ravel())
                    self.statistics_[t] = mode[0][0]

        return self

    def transform(self, Y):
        check_is_fitted(self, 'statistics_')

        Y = self._validate_input(Y)

        if Y.shape[0] != self.statistics_.shape[0]:
            raise ValueError("Y has %d time steps, expected %d"
                             % (Y.shape[0], self.statistics_.shape[0]))

        for t in range(Y.shape[0]):
            nan_mask = np.isnan(Y[t])
            Y[t][nan_mask] = self.statistics_[t]

        return Y
