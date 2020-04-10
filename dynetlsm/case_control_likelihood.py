import abc
import numbers
import numpy as np

from sklearn.utils import check_random_state


class CaseControlSampler(abc.ABC):
    def __init__(self,
                 n_control=100,
                 n_resample=100,
                 random_state=None):
        self.n_control = n_control
        self.n_resample = n_resample
        self.random_state = random_state

        self.n_iter = 0

    @abc.abstractmethod
    def init(self, Y):
        pass

    @abc.abstractmethod
    def sample(self):
        pass

    def resample(self):
        if self.n_resample is not None and self.n_iter % self.n_resample == 0.:
            self.control_nodes_in_, self.control_nodes_out_ = self.sample()

        self.n_iter += 1

        return self.control_nodes_in_, self.control_nodes_out_


class DirectedCaseControlSampler(CaseControlSampler):
    def init(self, Y):
        n_time_steps, n_nodes, _ = Y.shape

        if isinstance(self.n_control, (numbers.Integral, np.integer)):
            self.n_control_ = self.n_control
        else:
            self.n_control_ = int(self.n_control * n_nodes)

        # compute in-degree / out-degree of each node
        self.degrees_ = np.zeros((n_time_steps, n_nodes, 2), dtype=np.int)
        for t in range(n_time_steps):
            self.degrees_[t, :, 0] = Y[t].sum(axis=0)  # in-degree
            self.degrees_[t, :, 1] = Y[t].sum(axis=1)  # out-degree

        # store indices of edges, i.e. Y_ijt = 1
        max_in_degree = int(np.max(self.degrees_[:, :, 0]))
        max_out_degree = int(np.max(self.degrees_[:, :, 1]))
        self.in_edges_ = np.zeros((n_time_steps, n_nodes, max_in_degree),
                                  dtype=np.int)
        self.out_edges_ = np.zeros((n_time_steps, n_nodes, max_out_degree),
                                   dtype=np.int)
        for t in range(n_time_steps):
            for i in range(n_nodes):
                indices = np.where(Y[t, i, :] == 1)[0]
                n_edges = indices.shape[0]
                if n_edges:
                    self.out_edges_[t, i, :n_edges] = indices

                indices = np.where(Y[t, :, i] == 1)[0]
                n_edges = indices.shape[0]
                if n_edges:
                    self.in_edges_[t, i, :n_edges] = indices

        self.control_nodes_in_, self.control_nodes_out_ = self.sample()
        self.n_iter += 1

        return self

    def sample(self):
        rng = check_random_state(self.random_state)
        n_time_steps, n_nodes, _ = self.out_edges_.shape

        control_nodes_out = np.full((n_time_steps, n_nodes, self.n_control_),
                                    -1.0, dtype=np.int)
        control_nodes_in = np.full((n_time_steps, n_nodes, self.n_control_),
                                   -1.0, dtype=np.int)
        for t in range(n_time_steps):
            for i in range(n_nodes):
                out_degree = self.degrees_[t, i, 1]
                n_zeros = n_nodes - out_degree - 1
                if n_zeros < self.n_control_:
                    n_sample = n_zeros
                else:
                    n_sample = self.n_control_

                edges = set.difference(
                    set(range(n_nodes)),
                    self.out_edges_[t, i, :out_degree].tolist() + [i])
                control_nodes_out[t, i, :n_sample] = rng.choice(list(edges),
                                                                size=n_sample,
                                                                replace=False)

                in_degree = self.degrees_[t, i, 0]
                n_zeros = n_nodes - in_degree - 1
                if n_zeros < self.n_control_:
                    n_sample = n_zeros
                else:
                    n_sample = self.n_control_

                edges = set.difference(
                    set(range(n_nodes)),
                    self.in_edges_[t, i, :in_degree].tolist() + [i])
                control_nodes_in[t, i, :n_sample] = rng.choice(list(edges),
                                                               size=n_sample,
                                                               replace=False)
        return control_nodes_in, control_nodes_out


class MissingDirectedCaseControlSampler(CaseControlSampler):
    def init(self, Y):
        n_time_steps, n_nodes, _ = Y.shape

        if isinstance(self.n_control, (numbers.Integral, np.integer)):
            self.n_control_ = self.n_control
        else:
            self.n_control_ = int(self.n_control * n_nodes)

        # compute in-degree / out-degree of each node
        self.degrees_ = np.zeros((n_time_steps, n_nodes, 2), dtype=np.int)
        for t in range(n_time_steps):
            self.degrees_[t, :, 0] = Y[t].sum(axis=0)  # in-degree
            self.degrees_[t, :, 1] = Y[t].sum(axis=1)  # out-degree

        # store indices of edges, i.e. Y_ijt = 1
        max_in_degree = int(np.max(self.degrees_[:, :, 0]))
        max_out_degree = int(np.max(self.degrees_[:, :, 1]))
        self.in_edges_ = np.zeros((n_time_steps, n_nodes, max_in_degree),
                                  dtype=np.int)
        self.out_edges_ = np.zeros((n_time_steps, n_nodes, max_out_degree),
                                   dtype=np.int)
        for t in range(n_time_steps):
            for i in range(n_nodes):
                indices = np.where(Y[t, i, :] == 1)[0]
                n_edges = indices.shape[0]
                if n_edges:
                    self.out_edges_[t, i, :n_edges] = indices

                indices = np.where(Y[t, :, i] == 1)[0]
                n_edges = indices.shape[0]
                if n_edges:
                    self.in_edges_[t, i, :n_edges] = indices

        # determine edges (Y_ijt = 1 or Y_jit = 1 for at least one time step)
        self.edge_list_ = []
        for i in range(n_nodes):
            mask = np.logical_or(Y[:, i, :] == 1, Y[:, :, i] == 1)
            mask = mask.sum(axis=0)
            self.edge_list_.append(np.unique(np.where(mask > 0)[0]))

        self.control_nodes_ = self.sample()
        self.n_iter += 1

        return self

    def sample(self):
        rng = check_random_state(self.random_state)
        n_nodes = len(self.edge_list_)

        # TODO: n_control_samples can be a fraction of total number of nodes

        control_nodes = np.zeros((n_nodes, self.n_control_), dtype=np.int)
        for i in range(n_nodes):
            # stratify sample based one connections vs. non-connections
            n_connected = int(self.edge_list_[i].shape[0] / n_nodes *
                              self.n_control_)
            if self.edge_list_[i].shape[0] > 0:
                n_connected = max(n_connected, 1)

            control_nodes[i, :n_connected] = rng.choice(self.edge_list_[i],
                                                        size=n_connected,
                                                        replace=False)

            edges = set.difference(
                set(range(n_nodes)), self.edge_list_[i].tolist() + [i])
            n_remaining = self.n_control_ - n_connected
            control_nodes[i, n_connected:] = rng.choice(list(edges),
                                                        size=n_remaining,
                                                        replace=False)

        return control_nodes
