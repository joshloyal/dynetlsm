import numpy as np
import scipy.stats as stats


def tune_step_size_random_walk(step_size, acc_rate):
    """Keep acceptance rate within 25% - 40% acceptance."""
    if acc_rate < 0.001:
        step_size *= 0.1
    elif acc_rate < 0.05:
        step_size *= 0.5
    elif acc_rate < 0.25:
        step_size *= 0.9
    elif acc_rate > 0.95:
        step_size *= 10.0
    elif acc_rate > 0.75:
        step_size *= 2.0
    elif acc_rate > 0.4:
        step_size *= 1.1

    return step_size


def tune_step_size_dirichlet(step_size, acc_rate):
    if acc_rate < 0.001:
        step_size *= 10.0
    elif acc_rate < 0.05:
        step_size *= 2
    elif acc_rate < 0.25:
        step_size *= 1.1
    elif acc_rate > 0.95:
        step_size *= 0.1
    elif acc_rate > 0.75:
        step_size *= 0.5
    elif acc_rate > 0.4:
        step_size *= 0.9

    return step_size


def random_walk_metropolis(x0, logp, step_size, random_state):
    n_features = x0.shape[0]

    # random walk proposal
    x = x0 + step_size * random_state.randn(n_features)

    # accept-reject
    accept_ratio = logp(x) - logp(x0)
    accepted = 1
    u = random_state.rand()
    if np.log(u) >= accept_ratio:
        x = x0
        accepted = 0

    return x, accepted, accept_ratio


def dirichlet_metropolis(x0, logp, step_size, random_state, reg=1e-5):
    n_nodes = x0.shape[0]

    # scaled dirichlet proposal
    x = random_state.dirichlet(step_size * x0)

    # occasionally draws are zero due to precision issues
    # add some regularization and re-normalize
    if np.any(x == 0.):
        x += reg
        x /= np.sum(x)

    # accept-reject
    accept_ratio = logp(x) - logp(x0)

    # dirichlet proposal
    accept_ratio += (stats.dirichlet.logpdf(x0, step_size * x) -
                     stats.dirichlet.logpdf(x, step_size * x0))

    accepted = 1
    u = random_state.rand()
    if np.log(u) >= accept_ratio:
        x = x0
        accepted = 0

    return x, accepted, accept_ratio


class Metropolis(object):
    def __init__(self, step_size=0.1, tune=500, tune_interval=100,
                 proposal_type='random_walk'):
        self.step_size = step_size
        self.tune = tune
        self.tune_interval = tune_interval
        self.proposal_type = proposal_type
        self.steps_until_tune = tune_interval
        self.n_accepted = 0
        self.n_steps = 0

    def step(self, x, logp, random_state):
        if self.proposal_type == 'dirichlet':
            x_new, accepted, _ = dirichlet_metropolis(x,
                                                      logp,
                                                      self.step_size,
                                                      random_state)
        elif self.proposal_type == 'random_walk':
            x_new, accepted, _ = random_walk_metropolis(x,
                                                        logp,
                                                        self.step_size,
                                                        random_state)
        else:
            raise ValueError("`proposal_type` must be in "
                             "{'random_walk', 'dirichlet'}, but got "
                             "{}".format(self.proposal_type))

        # track acceptance statistics for adaptation
        self.n_accepted += accepted
        self.n_steps += 1

        # tune step-sizes if necessary
        if self.tune is not None:
            self.tune_step_size()

        return x_new

    def tune_step_size(self):
        if (self.n_steps < self.tune and self.steps_until_tune == 0):
            # tune step size
            accept_rate = self.n_accepted / self.tune_interval

            if self.proposal_type == 'dirichlet':
                self.step_size = tune_step_size_dirichlet(self.step_size,
                                                          accept_rate)
            else:
                self.step_size = tune_step_size_random_walk(self.step_size,
                                                            accept_rate)
            self.n_accepted = 0
            self.steps_until_tune = self.tune_interval
        else:
            self.steps_until_tune -= 1
