import itertools
import logging

import numpy as np
from sklearn.mixture.gmm import log_multivariate_normal_density, sample_gaussian
from sklearn.utils.extmath import logsumexp
import hmmlearn.hmm as impl
import hmmlearn.fhmmc as fhmmc
from hmmlearn.utils import normalize

from .base import (BaseHMM, transition_matrix, start_probabilities, estimate_normal_distribution_params)


def _new_model(n_states, transition_init, means, covars, covar_type, topology, n_iter, thresh, verbose):
    # Generate transition matrix
    if transition_init == 'uniform':
        transitions = transition_matrix(n_states, topology, randomize=False)
        pi = start_probabilities(n_states, topology, randomize=False)
    elif transition_init == 'random':
        transitions = transition_matrix(n_states, topology, randomize=True)
        pi = start_probabilities(n_states, topology, randomize=True)
    else:
        raise ValueError('unknown initialization strategy %s' % transition_init)

    # Create a model that trains mean (m), covariance (c), transition probabilities (t).
    # Note: the probabilities for transmat and startprob will currently be replaced with a
    # very small number for all entries that are exactly zero.
    logging.info('creating HMM with n_states=%d, transition_init=%s, topology=%s, n_iter=%d, thresh=%f, covar_type=%s' % (n_states, transition_init, topology, n_iter, thresh, covar_type))
    logging.info('transmat:\n' + str(transitions))
    logging.info('pi:\n' + str(pi))
    logging.info('means:\n' + str(means))
    logging.info('covars:\n' + str(covars))
    model = impl.GaussianHMM(n_states, transmat=transitions, startprob=pi, covariance_type=covar_type, params='mct',
                             init_params='', verbose=verbose, n_iter=n_iter, thresh=thresh)
    if covar_type == 'diag':
        model.covars_ = [covar.diagonal() for covar in covars]
    else:
        model.covars_ = covars
    model.means_ = means
    model.verbose = True
    return model


def greedy_sample(chains, means, covars, n_samples=1, random_state=None, max_cycle_duration=0):
    assert means.shape[0] == covars.shape[0]

    # Greedy sample algorithm as described by Takano et al.
    n_chains = len(chains)
    n_features = means.shape[-1]
    states = np.zeros((n_samples, n_chains), dtype=int)
    for chain_idx, chain in enumerate(chains):
        states[0, chain_idx] = np.argmax(chain._log_startprob)
        t = 1
        while t < n_samples:
            prev_state = states[t - 1, chain_idx]

            # Stay in state until duration is over or until we reach the n_samples limit
            trans_prob_cycle = np.exp(chain._log_transmat[prev_state, prev_state])
            if trans_prob_cycle == 1.0:
                trans_prob_cycle -= np.finfo(float).eps
            assert 0.0 <= trans_prob_cycle < 1.0
            duration = int(np.floor(min(min(1.0 / (1.0 - trans_prob_cycle), n_samples-t), max_cycle_duration)))
            for d in xrange(duration):
                states[t + d, chain_idx] = prev_state
            t += duration
            if t >= n_samples:
                continue

            # Get argmax of transition probability of previous state but ignore transition from prev_state -> prev_state
            state = None
            for idx, val in enumerate(chain._log_transmat[prev_state]):
                if idx != prev_state and (state is None or val > chain._log_transmat[prev_state, state]):
                    state = idx
            assert state is not None
            assert state != prev_state
            states[t, chain_idx] = state
            t += 1

    obs = np.zeros((n_samples, n_features))
    for t in xrange(n_samples):
        state_combination = tuple(states[t])
        mean = means[state_combination]
        covar = covars[state_combination]
        obs[t] = sample_gaussian(mean, covar, 'diag', random_state=random_state)
    return obs


class GaussianHMM(BaseHMM):
    def __init__(self, n_states=10, n_training_iterations=10, training_threshold=1e-2, topology='left-to-right',
                 verbose=False, transition_init='uniform', emission_init='k-means', covar_type='diag'):
        super(GaussianHMM, self).__init__(n_states, n_training_iterations, training_threshold, topology, verbose,
                                          transition_init, emission_init, covar_type)
        self.model_ = None

    def _init(self, obs):
        randomize = True if self.emission_init == 'random' else False
        means, covars = estimate_normal_distribution_params(obs, n_states=self.n_states, covar_type=self.covar_type,
                                                            randomize=randomize)
        self.model_ = _new_model(self.n_states, self.transition_init, means, covars, self.covar_type, self.topology,
                                 self.n_training_iterations, self.training_threshold, self.verbose)

    def _fit(self, obs):
        assert self.model_
        self.model_.fit(obs)

    def _loglikelihood(self, ob, method):
        if method != 'exact':
            raise ValueError('unknown method "%s"' % method)
        assert self.model_
        return self.model_.score(ob)

    def _sample(self, n_samples, max_cycle_duration):
        assert self.model_
        return greedy_sample([self.model_], self.model_._means_, self.model_._covars_, n_samples=n_samples,
                             max_cycle_duration=max_cycle_duration)

# TODO: normalize seems to also do some sort of maximum thingy
# TODO: implement masking!
def _normalize_transmat(transmat):
    return normalize(np.maximum(transmat, 1e-20), axis=1)


def _normalize_startprob(startprob):
    return normalize(np.maximum(startprob, 1e-20))


class ExactGaussianFHMM(BaseHMM):
    def __init__(self, n_states=10, n_chains=2, n_training_iterations=10, training_threshold=1e-2,
                 topology='left-to-right', verbose=False, transition_init='uniform', emission_init='k-means',
                 covar_type='diag'):
        super(ExactGaussianFHMM, self).__init__(n_states, n_training_iterations, training_threshold, topology, verbose,
                                                transition_init, emission_init, covar_type)
        self.n_chains = n_chains

    def _init(self, obs):
        self.log_transmat = np.zeros((self.n_chains, self.n_states, self.n_states))
        self.log_startprob = np.zeros((self.n_chains, self.n_states))
        for chain_idx in xrange(self.n_chains):
            self.log_transmat[chain_idx] = np.log(_normalize_transmat(transition_matrix(self.n_states, self.topology)))
            self.log_startprob[chain_idx] = np.log(_normalize_startprob(start_probabilities(self.n_states, self.topology)))

        # Estimate covar
        _, covars = estimate_normal_distribution_params(obs, n_states=1, covar_type='full')
        self.covar = covars[0]

        # Estimate means. We estimate different means for each chain and each state. Each mean is divided by the number
        # of chains since means from each chain are summed to form the actual mean.
        means, _ = estimate_normal_distribution_params(obs, n_states=self.n_states * self.n_chains)
        self.means = (means.reshape((self.n_chains, self.n_states, self.n_features_)) / float(self.n_chains))

    def _do_forward_pass(self, framelogprob):
        n_observations = framelogprob.shape[0]
        state_combinations = [tuple(x) for x in list(itertools.product(np.arange(self.n_states), repeat=self.n_chains))]
        fwdlattice = np.zeros((n_observations, self.n_states ** self.n_chains))
        fhmmc._forward(n_observations, self.n_chains, self.n_states, state_combinations, self.log_startprob,
                       self.log_transmat, framelogprob, fwdlattice)
        return logsumexp(fwdlattice[-1]), fwdlattice

    def _do_backward_pass(self, framelogprob):
        n_observations = framelogprob.shape[0]
        state_combinations = [tuple(x) for x in list(itertools.product(np.arange(self.n_states), repeat=self.n_chains))]
        bwdlattice = np.zeros((n_observations, self.n_states ** self.n_chains))
        fhmmc._backward(n_observations, self.n_chains, self.n_states, state_combinations, self.log_startprob,
                        self.log_transmat, framelogprob, bwdlattice)
        return bwdlattice

    def _compute_logeta(self, framelogprob, fwdlattice, bwdlattice):
        n_observations = framelogprob.shape[0]
        state_combinations = [tuple(x) for x in list(itertools.product(np.arange(self.n_states), repeat=self.n_chains))]
        logeta = np.zeros((n_observations - 1, self.n_chains, self.n_states, self.n_states))
        fhmmc._compute_logeta(n_observations, self.n_chains, self.n_states, state_combinations, self.log_transmat,
                              framelogprob, fwdlattice, bwdlattice, logeta)

        # TODO: remove this validation eventually
        for t in xrange(n_observations - 1):
            for chain_idx in xrange(self.n_chains):
                assert np.allclose(np.sum(np.exp(logeta[t, chain_idx])), 1.0)

        return logeta

    def _do_mstep(self, stats):
        # Startprob and transmat
        for chain_idx in xrange(self.n_chains):
            self.log_startprob[chain_idx] = np.log(_normalize_startprob(stats['start'][chain_idx]))
            self.log_transmat[chain_idx] = np.log(_normalize_transmat(stats['trans'][chain_idx]))

        # Means
        print 'means equal per chain before', np.allclose(self.means[0], self.means[1])
        means = np.dot(stats['means_sum1'], np.linalg.pinv(stats['means_sum2'])).T
        means = means.reshape(self.means.shape)
        self.means = means
        print 'means equal per chain after', np.allclose(means[0], means[1])

        # Covariance
        covar1 = 1.0 / float(stats['T']) * (stats['covar_sum1'] - stats['covar_sum2'])
        # Alternative way of calculating covar
        covar2 = (-2.0 * stats['covar_sum']) / float(stats['T'])
        print 'covar1 == covar2', np.allclose(covar1, covar2)
        #self.covar = covar1

        assert np.allclose(self.covar.T, self.covar)  # ensure that covar is symmetric

    def _accumulate_sufficient_statistics(self, stats, seq, framelogprob, in_posteriors, fwdlattice, bwdlattice):
        n_observations, n_features = seq.shape
        partial_state_combinations = [list(x) for x in list(itertools.product(np.arange(self.n_states), repeat=self.n_chains - 1))]
        posteriors = in_posteriors.view()
        state_combination_shape = tuple([self.n_states for _ in xrange(self.n_chains)])
        posteriors.shape = (n_observations,) + state_combination_shape

        # Calculate posteriors for each time step and each chain (<S_t^(m)>)
        chain_posteriors = np.zeros((n_observations, self.n_chains, self.n_states))
        for t in xrange(n_observations):
            for chain_idx in xrange(self.n_chains):
                for state in xrange(self.n_states):
                    for partial_combination in partial_state_combinations:
                        state_combination = tuple(partial_combination[:chain_idx] + [state] + partial_combination[chain_idx:])
                        chain_posteriors[t, chain_idx, state] += posteriors[t][state_combination]
            # Ensure that posteriors for each chain sum to 1
            assert np.allclose(np.sum(chain_posteriors[t], axis=1), 1.0)

        chain_chain_posteriors = np.zeros((n_observations, self.n_chains, self.n_chains, self.n_states, self.n_states))
        assert self.n_chains == 2, 'This code currently only works for 2 chains'
        for t in xrange(n_observations):
            for chain0_idx in xrange(self.n_chains):
                for chain1_idx in xrange(self.n_chains):
                    for state0 in xrange(self.n_states):
                        for state1 in xrange(self.n_states):
                            # Now keep state0 and state1 fixed and vary the rest-in the case of only two chains, this
                            # however means that we cannot vary anything
                            chain_chain_posteriors[t, chain0_idx, chain1_idx, state0, state1] = posteriors[t, state0, state1]
                    assert np.allclose(np.sum(chain_chain_posteriors[t, chain0_idx, chain1_idx]), 1.0)

        combined_chain_chain_posteriors = np.zeros((n_observations, self.n_chains * self.n_states, self.n_chains * self.n_states))
        for t in xrange(n_observations):
            for chain0_idx in xrange(self.n_chains):
                for chain1_idx in xrange(self.n_chains):
                    post = chain_chain_posteriors[t, chain0_idx, chain1_idx]
                    assert post.shape == (self.n_states, self.n_states)
                    idx0 = chain0_idx * self.n_states
                    idx1 = chain1_idx * self.n_states
                    assert combined_chain_chain_posteriors[t, idx0:idx0+self.n_states, idx1:idx1+self.n_states].shape == (self.n_states, self.n_states)
                    assert np.allclose(combined_chain_chain_posteriors[t, idx0:idx0+self.n_states, idx1:idx1+self.n_states], 0)
                    combined_chain_chain_posteriors[t, idx0:idx0+self.n_states, idx1:idx1+self.n_states] = post

        #combined_chain_chain_posteriors = chain_chain_posteriors.reshape(n_observations, self.n_states * self.n_chains, self.n_states * self.n_chains)

        # Calculate posteriors for each time step and each chain combination (<S_t^(m),S_t^(n)'>)
        # TODO: this is super sketchy code right here
        # start = timeit.default_timer()
        # chain_chain_posteriors = np.zeros((n_observations, self.n_chains, self.n_chains, self.n_states, self.n_states))
        # steps = 0
        # skipped = 0
        # for t in xrange(n_observations):
        #     for chain0_idx in xrange(self.n_chains):
        #         for chain1_idx in xrange(self.n_chains):
        #             processed_state_combinations = set()  # TODO: is this correct?
        #             for i in xrange(self.n_states):
        #                 for j in xrange(self.n_states):
        #                     for state_combination in state_combinations:
        #                         actual_state_combination = list(state_combination)
        #                         actual_state_combination[chain0_idx] = i
        #                         actual_state_combination[chain1_idx] = j
        #                         actual_state_combination = tuple(actual_state_combination)
        #                         if actual_state_combination in processed_state_combinations:
        #                             # Skip states that we have already processed.
        #                             # TODO: make this efficient, right now we skip most states
        #                             skipped += 1
        #                             continue
        #                         steps += 1
        #                         processed_state_combinations.add(actual_state_combination)
        #                         chain_chain_posteriors[t, chain0_idx, chain1_idx, i, j] += posteriors[t][actual_state_combination]
        #                         if posteriors[t][actual_state_combination] == 0.0:
        #                             print '0.0!'
        # print('took %fs, %d steps (%d)' % ((timeit.default_timer() - start), steps, skipped))
        # print np.size(chain_chain_posteriors)

        # Update stats for start and trans
        stats['start'] += chain_posteriors[0]
        logeta = self._compute_logeta(framelogprob, fwdlattice, bwdlattice)
        for chain_idx in xrange(self.n_chains):
            # No need to normalize here since we'll do that later anyway
            stats['trans'][chain_idx] += np.exp(logsumexp(logeta[:, chain_idx], axis=0))

        # Update stats for means
        for t in xrange(n_observations):
            ob = seq[t].reshape(n_features, 1)
            post = chain_posteriors[t].flatten().reshape(1, self.n_chains * self.n_states)
            val1 = np.dot(ob, post)
            assert val1.shape == stats['means_sum1'].shape
            stats['means_sum1'] += val1

            val2 = combined_chain_chain_posteriors[t]
            assert val2.shape == stats['means_sum2'].shape
            stats['means_sum2'] += val2

        new_means = np.dot(stats['means_sum1'], np.linalg.pinv(stats['means_sum2'])).T.reshape(self.means.shape)

        # Alternative way of calculcating the covariance stats (seems to yield same (bad) results)
        for t in xrange(n_observations):
            ob = seq[t].reshape(n_features, 1)
            sum1 = np.zeros((self.n_features_, self.n_features_))
            for chain_idx in xrange(self.n_chains):
                post = chain_posteriors[t, chain_idx].reshape(self.n_states, 1)
                val1 = np.dot(np.dot(ob, post.T), self.means[chain_idx])
                assert val1.shape == sum1.shape
                sum1 += val1

            sum2 = np.zeros((self.n_features_, self.n_features_))
            for chain0_idx in xrange(self.n_chains):
                for chain1_idx in xrange(self.n_chains):
                    post = chain_chain_posteriors[t, chain1_idx, chain0_idx]
                    val2 = np.dot(np.dot(self.means[chain1_idx].T, post), self.means[chain0_idx])
                    assert val2.shape == sum2.shape
                    sum2 += val2

            covar_sum = sum1 - 0.5 * np.dot(ob, ob.T) - 0.5 * sum2
            assert covar_sum.shape == stats['covar_sum'].shape
            stats['covar_sum'] += covar_sum

        # Update covariance stats
        for t in xrange(n_observations):
            ob = seq[t].reshape(n_features, 1)
            val1 = np.dot(ob, ob.T)
            assert val1.shape == stats['covar_sum1'].shape
            stats['covar_sum1'] += val1

            tmp = np.zeros((n_features, 1))
            for chain_idx in xrange(self.n_chains):
                tmp += np.dot(new_means[chain_idx].T, chain_posteriors[t, chain_idx]).reshape((n_features, 1))
            val2 = np.dot(tmp, ob.T)
            assert val2.shape == stats['covar_sum2'].shape
            stats['covar_sum2'] += val2

        # Update bookkeeping stats
        stats['nobs'] += 1
        stats['T'] += n_observations

    def _initialize_sufficient_statistics(self):
        stats = {'nobs': 0,
                 'T': 0,
                 'start': np.zeros((self.n_chains, self.n_states)),
                 'trans': np.zeros((self.n_chains, self.n_states, self.n_states)),
                 'covar_sum1': np.zeros((self.n_features_, self.n_features_)),
                 'covar_sum2': np.zeros((self.n_features_, self.n_features_)),
                 'covar_sum': np.zeros((self.n_features_, self.n_features_)),  # TODO: remove eventually
                 'means_sum1': np.zeros((self.n_features_, self.n_states * self.n_chains)),
                 'means_sum2': np.zeros((self.n_states * self.n_chains, self.n_states * self.n_chains))}
        return stats

    def _compute_log_likelihood(self, seq):
        state_combinations = [tuple(x) for x in list(itertools.product(np.arange(self.n_states), repeat=self.n_chains))]
        n_state_combinations = self.n_states ** self.n_chains
        n_observations, n_features = seq.shape
        covars = np.array([self.covar for _ in xrange(n_state_combinations)])  # TODO: correct?!

        means = np.zeros((n_state_combinations, n_features))
        for idx, state_combination in enumerate(state_combinations):
            for chain_idx, state in enumerate(state_combination):
                means[idx] += self.means[chain_idx, state]

        framelogprob = log_multivariate_normal_density(seq, means, covars, covariance_type='full')
        return framelogprob

    def _fit(self, obs):
        prev_loglikelihood = None
        for iteration in xrange(self.n_training_iterations):
            stats = self._initialize_sufficient_statistics()
            curr_loglikelihood = 0
            for seq in obs:
                # Forward-backward pass and accumulate stats
                framelogprob = self._compute_log_likelihood(seq)
                lpr, fwdlattice = self._do_forward_pass(framelogprob)
                bwdlattice = self._do_backward_pass(framelogprob)
                gamma = fwdlattice + bwdlattice
                posteriors = np.exp(gamma.T - logsumexp(gamma, axis=1)).T
                assert np.allclose(np.sum(posteriors, axis=1), 1.0)  # posteriors must sum to 1 for each t
                curr_loglikelihood += lpr
                self._accumulate_sufficient_statistics(stats, seq, framelogprob, posteriors, fwdlattice, bwdlattice)

            # Test for convergence
            if prev_loglikelihood is not None:
                delta = curr_loglikelihood - prev_loglikelihood
                print ('%f (%f)' % (curr_loglikelihood, delta))
                assert delta >= -0.01  # Likelihood when training with Baum-Welch should grow monotonically
                if delta <= self.training_threshold:
                    break

            self._do_mstep(stats)
            prev_loglikelihood = curr_loglikelihood

    def _loglikelihood(self, ob, method):
        framelogprob = self._compute_log_likelihood(ob)
        loglikelihood, _ = self._do_forward_pass(framelogprob)
        return loglikelihood

    def _sample(self, n_samples, max_cycle_duration):
        raise NotImplementedError('not yet implemented')
        pass


class SequentialGaussianFHMM(BaseHMM):
    def __init__(self, n_states=10, n_chains=2, n_training_iterations=10, training_threshold=1e-2,
                 topology='left-to-right', verbose=False, transition_init='uniform', emission_init='k-means',
                 covar_type='diag'):
        super(SequentialGaussianFHMM, self).__init__(n_states, n_training_iterations, training_threshold, topology,
                                                     verbose, transition_init, emission_init, covar_type)
        self.n_chains = n_chains
        self.chains_ = None

    def _init(self, obs):
        # Initialize first chain
        randomize = True if self.emission_init == 'random' else False
        means, covars = estimate_normal_distribution_params(obs, n_states=self.n_states, randomize=randomize)
        chain = _new_model(self.n_states, self.transition_init, np.copy(means), np.copy(covars), self.covar_type,
                           self.topology, self.n_training_iterations, self.training_threshold, self.verbose)
        self.chains_ = [chain]

    def _fit(self, obs):
        assert self.chains_
        assert len(self.chains_) > 0

        # Re-use generated observations for later use when iterating over already trained chains. We do not need an
        # entry for the last chain since it never generates an observation.
        generated_obs_cache = [[] for _ in xrange(self.n_chains - 1)]

        # Train first chain
        self.chains_[0].fit(obs)

        # Train subsequent chains on the error
        for curr_chain_idx in xrange(1, self.n_chains):
            # Calculate residual error for each observation
            weight = 1.0 / float(self.n_chains)  # TODO: it's a bit unclear if this is correct
            err = []
            for ob_idx, ob in enumerate(obs):
                # Calculate residual error
                combined_generated_ob = self._generate_observation(ob, ob_idx, curr_chain_idx, weight, generated_obs_cache)
                curr_err = (1.0 / weight) * (ob - combined_generated_ob)
                assert curr_err.shape == ob.shape
                err.append(curr_err)
            assert len(err) == len(obs)

            # Create new chain
            randomize = True if self.emission_init == 'random' else False
            means, covars = estimate_normal_distribution_params(err, n_states=self.n_states, randomize=randomize)
            curr_chain = _new_model(self.n_states, self.transition_init, means, covars, self.covar_type, self.topology,
                                    self.n_training_iterations, self.training_threshold, self.verbose)
            self.chains_.append(curr_chain)

            # Fit chain on residual error
            curr_chain.fit(err)

        # Ensure that cache was filled properly and works as expected
        assert len(generated_obs_cache) == self.n_chains - 1
        for chain_cache in generated_obs_cache:
            assert len(chain_cache) == len(obs)

    def _loglikelihood_residual_approx(self, ob):
        scores = np.zeros(self.n_chains)

        # Re-use generated observations for later use when iterating over already trained chains. We do not need an
        # entry for the last chain since it never generates an observation.
        generated_obs_cache = [[] for _ in xrange(self.n_chains - 1)]

        for curr_chain_idx, curr_chain in enumerate(self.chains_):
            if curr_chain_idx == 0:
                scores[curr_chain_idx] = curr_chain.score(ob)
                continue

            # Calculate residual error for observation and calculate score on it
            weight = 1.0 / float(self.n_chains)  # TODO: it's a bit unclear if this is correct
            combined_generated_ob = self._generate_observation(ob, 0, curr_chain_idx, weight, generated_obs_cache)
            curr_err = (1.0 / weight) * (ob - combined_generated_ob)
            scores[curr_chain_idx] = curr_chain.score(curr_err)
        return logsumexp(scores)

    def _exact_loglikelihood(self, ob):
        log_transmat = np.zeros((self.n_chains, self.n_states, self.n_states))
        log_startprob = np.zeros((self.n_chains, self.n_states))
        for idx, chain in enumerate(self.chains_):
            log_transmat[idx] = chain._log_transmat
            log_startprob[idx] = chain._log_startprob

        n_state_combinations = self.n_states ** self.n_chains
        state_combinations = [tuple(x) for x in list(itertools.product(np.arange(self.n_states), repeat=self.n_chains))]
        n_observations = ob.shape[0]
        n_features = ob.shape[1]
        fwdlattice = np.zeros((n_observations, n_state_combinations))

        # Calculate means and covariances for all state combinations and calculate emission probabilities
        weight = (1.0 / float(self.n_chains))
        weight_squared = weight * weight
        covars = np.zeros((n_state_combinations, n_features))  # TODO: add support for all covariance types
        means = np.zeros((n_state_combinations, n_features))
        for idx, state_combination in enumerate(state_combinations):
            for chain_idx, state in enumerate(state_combination):
                chain = self.chains_[chain_idx]
                covars[idx] += chain._covars_[state]
                means[idx] += chain._means_[state]
            covars[idx] *= weight_squared
            means[idx] *= weight
        framelogprob = log_multivariate_normal_density(ob, means, covars, covariance_type='diag')  # TODO: add support for all covariance types

        # Run the forward algorithm
        fhmmc._forward(n_observations, self.n_chains, self.n_states, state_combinations, log_startprob, log_transmat,
                       framelogprob, fwdlattice)

        last_column = fwdlattice[-1]
        assert np.size(last_column) == n_state_combinations
        score = logsumexp(last_column)
        return score

    def _loglikelihood(self, ob, method):
        if method == 'exact':
            return self._exact_loglikelihood(ob)
        elif method == 'approx':
            return self._loglikelihood_residual_approx(ob)
        else:
            raise ValueError('unknown method "%s"' % method)

    def _sample(self, n_samples, max_cycle_duration):
        # Calculate means and covariances for all state combinations and calculate emission probabilities
        state_combinations = [tuple(x) for x in list(itertools.product(np.arange(self.n_states), repeat=self.n_chains))]
        state_combinations_shape = tuple([self.n_states for _ in xrange(self.n_chains)])
        weight = (1.0 / float(self.n_chains))
        weight_squared = weight * weight
        covars = np.zeros(state_combinations_shape + (self.n_features_,))  # TODO: add support for all covariance types
        means = np.zeros(state_combinations_shape + (self.n_features_,))
        for state_combination in state_combinations:
            for chain_idx, state in enumerate(state_combination):
                chain = self.chains_[chain_idx]
                covars[state_combination] += chain._covars_[state]
                means[state_combination] += chain._means_[state]
            covars[state_combination] *= weight_squared
            means[state_combination] *= weight

        obs = greedy_sample(self.chains_, means, covars, n_samples=n_samples, max_cycle_duration=max_cycle_duration)
        return obs

    def _generate_observation(self, ob, ob_idx, curr_chain_idx, weight, generated_obs_cache):
        combined_generated_ob = np.zeros(ob.shape)

        # Iterate over all chains up until (but excluding) the current chain and combine their generated
        # observations
        for prev_chain_idx in xrange(curr_chain_idx):
            prev_chain = self.chains_[prev_chain_idx]
            generated_ob = np.zeros(ob.shape)
            if ob_idx < len(generated_obs_cache[prev_chain_idx]):
                # Use cached value
                assert generated_obs_cache[prev_chain_idx][ob_idx].shape == ob.shape
                generated_ob = generated_obs_cache[prev_chain_idx][ob_idx]
            else:
                # Option a: gamma
                framelogprob = prev_chain._compute_log_likelihood(ob)
                logprob, fwdlattice = prev_chain._do_forward_pass(framelogprob)
                bwdlattice = prev_chain._do_backward_pass(framelogprob)
                fwdbwdlattice = fwdlattice + bwdlattice
                gamma = np.exp(fwdbwdlattice.T - logsumexp(fwdbwdlattice, axis=1)).T
                # TODO: this can probably be vectorized
                for t in xrange(ob.shape[0]):
                    for state in xrange(self.n_states):
                        generated_ob[t] += prev_chain.means_[state] * gamma[t][state]

                # Option b: Viterbi
                # _, state_seq = prev_chain.decode(ob)
                # assert np.size(state_seq) == generated_ob.shape[0]
                # for idx, state in enumerate(state_seq):
                #     generated_ob[idx] = prev_chain.means_[state]

                #  Option c: generated sequence
                # generated_ob = prev_chain.sample(ob.shape[0])[0]

                # Cache for future iterations
                generated_obs_cache[prev_chain_idx].append(generated_ob)

            combined_generated_ob += weight * generated_ob
        return combined_generated_ob
