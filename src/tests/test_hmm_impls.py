import unittest

import numpy as np

from sklearn.base import clone
from toolkit.hmm.impl_hmmlearn import GaussianHMM as HMMLearnGaussianHMM
from toolkit.hmm.impl_hmmlearn import SequentialGaussianFHMM
from toolkit.hmm.impl_pomegranate import GaussianHMM as PomegranateGaussianHMM


class TestHMMImplementations(unittest.TestCase):
    def setUp(self):
        n_states = 3
        self.hmms = [HMMLearnGaussianHMM(n_states=n_states),
                     PomegranateGaussianHMM(n_states=n_states),
                     SequentialGaussianFHMM(n_states=n_states, n_chains=2)]
        r = np.random.randn
        self.obs = [np.array([[-600 + r(), 100 + r()], [-300 + r(), 200 + r()], [0 + r(), 300 + r()]]) for _ in xrange(10)]

    def test_fit(self):
        for hmm in self.hmms:
            # Not all implementations support on-line learning. Workaround: create a clone of the model, fit it on
            # random data and use this loglikelihood as a baseline for a model that will actually be trained on the
            # observations.
            hmm_clone = clone(hmm)
            hmm_clone.fit([np.random.random((3, 2))])
            loglikelihood_before = hmm_clone.loglikelihood(self.obs[1])

            hmm.fit(self.obs)
            loglikelihood_after = hmm.loglikelihood(self.obs[1])
            self.assertGreater(loglikelihood_after, loglikelihood_before)

    def test_loglikelihood(self):
        for hmm in self.hmms:
            hmm.fit(self.obs)
            loglikelihood_pos = hmm.loglikelihood(self.obs[0])
            loglikelihood_neg = hmm.loglikelihood(np.array([[-100, 0.1], [-100, 500], [-100, -3000]]))
            self.assertGreater(np.abs(loglikelihood_pos - loglikelihood_neg), 1000)

    def test_sample(self):
        n_samples = 3
        n_features = self.obs[0].shape[1]
        for hmm in self.hmms:
            hmm.fit(self.obs)
            sample = hmm.sample(n_samples=n_samples, max_cycle_duration=0)
            sample = np.atleast_2d(sample)
            self.assertTupleEqual(sample.shape, (n_samples, n_features))
            self.assertTrue(np.allclose(sample[:, 0], self.obs[0][:, 0], rtol=20.0))
            self.assertTrue(np.allclose(sample[:, 1], self.obs[0][:, 1], rtol=20.0))

    def test_topology(self):
        topologies = ['left-to-right', 'left-to-right-cycle', 'bakis', 'full']
        for hmm in self.hmms:
            if hmm.__class__ == PomegranateGaussianHMM:
                continue
            if hmm.__class__ == SequentialGaussianFHMM:
                continue

            for topology in topologies:
                hmm.topology = topology
                hmm._init(self.obs)
                initial_transmat = np.round(hmm.model_.transmat_, decimals=5)
                initial_startprob = np.round(hmm.model_.startprob_, decimals=5)
                hmm.fit(self.obs)

                actual_transmat = np.round(hmm.model_.transmat_, decimals=5)
                actual_startprob = np.round(hmm.model_.startprob_, decimals=5)

                transmat_sum = np.sum(actual_transmat[initial_transmat == 0])
                startprob_sum = np.sum(actual_startprob[initial_startprob == 0])
                self.assertEqual(transmat_sum, 0)
                self.assertEqual(startprob_sum, 0)
                self.assertEqual(hmm.topology, topology)
