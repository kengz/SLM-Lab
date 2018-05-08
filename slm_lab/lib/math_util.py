'''
Math util
adopted from https://github.com/openai/baselines/blob/master/baselines/common/math_util.py
'''
import numpy as np
import scipy.signal


class Dataset:
    def __init__(self, data_map, deterministic=False, shuffle=True):
        self.data_map = data_map
        self.deterministic = deterministic
        self.enable_shuffle = shuffle
        self.n = next(iter(data_map.values())).shape[0]
        self._next_id = 0
        self.shuffle()

    def shuffle(self):
        if self.deterministic:
            return
        perm = np.arange(self.n)
        np.random.shuffle(perm)

        for key in self.data_map:
            self.data_map[key] = self.data_map[key][perm]

        self._next_id = 0

    def next_batch(self, batch_size):
        if self._next_id >= self.n and self.enable_shuffle:
            self.shuffle()

        cur_id = self._next_id
        cur_batch_size = min(batch_size, self.n - self._next_id)
        self._next_id += cur_batch_size

        data_map = dict()
        for key in self.data_map:
            data_map[key] = self.data_map[key][cur_id:cur_id + cur_batch_size]
        return data_map

    def iterate_once(self, batch_size):
        if self.enable_shuffle:
            self.shuffle()

        while self._next_id <= self.n - batch_size:
            yield self.next_batch(batch_size)
        self._next_id = 0

    def subset(self, num_elements, deterministic=True):
        data_map = dict()
        for key in self.data_map:
            data_map[key] = self.data_map[key][:num_elements]
        return Dataset(data_map, deterministic)


def discount(x, gamma):
    '''
    computes discounted sums along 0th dimension of x.
    @param {ndarray} x
    @param {float} gamma
    @returns {ndarray} y with same shape as x, satisfying y[t] = x[t] + gamma*x[t+1] + gamma^2*x[t+2] + ... + gamma^k x[t+k], where k = len(x) - t - 1
    '''
    assert x.ndim >= 1
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def explained_variance(y_pred, y):
    '''
    Computes fraction of variance that y_pred explains about y.
    @returns {float} explained_var = 1 - Var[y-y_pred] / Var[y]
    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero
    '''
    assert y.ndim == 1 and y_pred.ndim == 1
    vary = np.var(y)
    return np.nan if vary == 0 else 1 - np.var(y - y_pred) / vary


def explained_variance_2d(y_pred, y):
    assert y.ndim == 2 and y_pred.ndim == 2
    vary = np.var(y, axis=0)
    out = 1 - np.var(y - y_pred) / vary
    out[vary < 1e-10] = 0
    return out
