import numpy as np
import pomegranate as impl

from .base import (BaseHMM, transition_matrix, start_probabilities, estimate_normal_distribution_params)


def _new_model(n_features, n_states, means, covars, topology):
    distributions = [impl.distributions.MultivariateGaussianDistribution(means[i], covars[i], diagonal=True) for i in range(n_states)]
    transitions = transition_matrix(n_states, topology)
    pi = start_probabilities(n_states, topology)

    # Create model and states
    model = impl.hmm.HiddenMarkovModel()
    states = [impl.base.State(distribution) for distribution in distributions]
    model.add_states(states)

    # Start probabilities
    for i, prob in enumerate(pi):
        if prob != 0.0:
            model.add_transition(model.start, states[i], prob)

    # Add transitions
    for i, row in enumerate(transitions):
        for j, prob in enumerate(row):
            if prob != 0.0:
                model.add_transition(states[i], states[j], prob)

    model.bake()
    return model


class GaussianHMM(BaseHMM):
    def __init__(self, n_states=10, n_training_iterations=10, training_threshold=1e-2, topology='left-to-right', verbose=False):
        super(GaussianHMM, self).__init__(n_states, n_training_iterations, training_threshold, topology, verbose)
        self.model_ = None

    def supports_parallel(self):
        # Training and predicting on multiple processes causes issues and exceptions.
        return False

    def _init(self, obs):
        assert self.n_features_
        means, covars = estimate_normal_distribution_params(obs, n_states=self.n_states)
        self.model_ = _new_model(self.n_features_, self.n_states, means, covars, self.topology)

    def _fit(self, obs):
        assert self.model_
        self.model_.train(obs, algorithm='baum-welch', max_iterations=self.n_training_iterations,
                          stop_threshold=self.training_threshold)

    def _loglikelihood(self, ob, method):
        if method != 'exact':
            raise ValueError('unknown method "%s"' % method)
        assert self.model_
        return self.model_.log_probability(ob)

    def _sample(self, n_samples, max_cycle_duration):
        assert self.model_
        return np.array(self.model_.sample(length=n_samples))
