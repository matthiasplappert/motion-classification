import unittest

import numpy as np

from toolkit.hmm.base import BaseHMM
from toolkit.util import NotFittedError


class MockBaseHMM(BaseHMM):
    def __init__(self, n_states=10):
        super(MockBaseHMM, self).__init__(n_states, 10, 1e-2, 'left-to-right')
        self.init_callback = None
        self.fit_callback = None
        self.sample_callback = None
        self.loglikelihood_callback = None

    def _init(self, obs):
        if self.init_callback is not None:
            return self.init_callback(obs)
        return

    def _fit(self, obs):
        if self.fit_callback is not None:
            return self.fit_callback(obs)
        return

    def _sample(self, n_samples, max_cycle_duration):
        if self.sample_callback is not None:
            return self.sample_callback()
        return

    def _loglikelihood(self, ob, mode='exact'):
        if self.loglikelihood_callback is not None:
            return self.loglikelihood_callback(ob)
        return


class TestBaseHMM(unittest.TestCase):
    def setUp(self):
        self.hmm = MockBaseHMM()

    def test_invalid_fit(self):
        self.assertRaises(ValueError, self.hmm.fit, np.array([1, 1]))
        self.assertRaises(ValueError, self.hmm.fit, [])
        self.hmm.fit([np.array([1, 1])])
        self.assertRaises(ValueError, self.hmm.fit, [np.array([1, 1, 1])])

    def test_valid_fit(self):
        obs = [np.array([1, 1]), np.array([[1, 1], [2, 2]])]
        called = [False, False]

        def init(x):
            self.assertEqual(len(x), len(obs))
            self.assertEqual(np.ndim(x[0]), 2)
            called[0] = True

        def fit(x):
            self.assertEqual(len(x), len(obs))
            self.assertEqual(np.ndim(x[0]), 2)
            called[1] = True

        self.hmm.init_callback = init
        self.hmm.fit_callback = fit
        self.hmm.fit(obs)
        self.assertEqual(self.hmm.n_features_, 2)
        self.assertTrue(called[0])
        self.assertTrue(called[1])

        called[0], called[1] = False, False
        self.hmm.fit(obs)
        self.assertFalse(called[0])
        self.assertTrue(called[1])

    def test_valid_loglikelihood(self):
        ob = np.array([1, 1])
        called = [False]

        def loglikelihood(x):
            self.assertEqual(np.ndim(x), 2)
            called[0] = True

        self.hmm.loglikelihood_callback = loglikelihood
        self.hmm.fit([np.array([[1, 1], [2, 2]])])
        self.hmm.loglikelihood(ob)
        self.assertTrue(called[0])

    def test_invalid_loglikelihood(self):
        self.assertRaises(NotFittedError, self.hmm.loglikelihood, np.array([[1, 1], [2, 2]]))
        self.hmm.fit([np.array([[1, 1], [2, 2]])])
        self.assertRaises(ValueError, self.hmm.loglikelihood, np.array([[1, 1, 1], [2, 2, 2]]))

    def test_valid_sample(self):
        called = [False]

        def sample():
            called[0] = True

        self.hmm.sample_callback = sample
        self.hmm.fit([np.array([[1, 1], [2, 2]])])
        self.hmm.sample()

    def test_invalid_sample(self):
        self.assertRaises(NotFittedError, self.hmm.sample)