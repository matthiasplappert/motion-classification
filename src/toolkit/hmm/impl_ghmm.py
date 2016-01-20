import ghmm as impl

from .base import (BaseHMM, transition_matrix, start_probabilities, estimate_normal_distribution_params)


def _sequence_set_from_list(l):
    # Conversion is similar to _sequence_from_data but here data is a list.
    unrolled = [matrix.ravel().tolist() for matrix in l]
    seq = impl.SequenceSet(impl.Float(), unrolled)
    return seq


def _sequence_from_matrix(m):
    # Conversion happens as follows: data is a n x m matrix, where n is the number of
    # samples and m is the number of features per sample. Multivariate data in ghmm is
    # represented as a single list, where the samples are unrolled. Hence the resulting
    # data has the following structure: [x_11, x_12, x_13, x21, x22, x23, ...] where m = 3.
    # Source: http://sourceforge.net/p/ghmm/mailman/message/20578788/
    unrolled = m.ravel().tolist()
    seq = impl.EmissionSequence(impl.Float(), unrolled)
    return seq


def _new_model(n_features, n_states, means, covars, topology):
    # Generate emissions
    emissions = []
    for i in range(n_states):
        emission = [means[i].tolist(), covars[i].ravel().tolist()]
        emissions.append(emission)

    # Create model
    domain = impl.Float()
    transitions = transition_matrix(n_states, topology).tolist()
    pi = start_probabilities(n_states, topology)
    distribution = impl.MultivariateGaussianDistribution(domain)
    model = impl.HMMFromMatrices(domain, distribution, transitions, emissions, pi)
    return model


class GaussianHMM(BaseHMM):
    def __init__(self, n_states=10, n_training_iterations=10, training_threshold=1e-2, topology='left-to-right', verbose=False):
        super(GaussianHMM, self).__init__(n_states, n_training_iterations, training_threshold, topology, verbose)
        self.model_ = None

    def _init(self, obs):
        assert self.n_features_
        means, covars = estimate_normal_distribution_params(obs, n_states=self.n_states)
        self.model_ = _new_model(self.n_features_, self.n_states, means, covars, self.topology)

    def _fit(self, obs):
        assert self.model_
        seq = _sequence_set_from_list(obs)
        self.model_.baumWelch(seq, self.n_training_iterations, self.training_threshold)

    def _loglikelihood(self, ob, method):
        if method != 'exact':
            raise ValueError('unknown method "%s"' % method)
        assert self.model_
        seq = _sequence_from_matrix(ob)
        return self.model_.loglikelihood(seq)

    def _sample(self, n_samples, max_cycle_duration):
        assert self.model_
        return self.model_.sampleSingle(n_samples)