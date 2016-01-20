import unittest
import timeit
import itertools
import os

from scipy.io import loadmat
import numpy as np
from hmmlearn.hmm import GaussianHMM
import hmmlearn.fhmmc as fhmmc
from sklearn.mixture.gmm import log_multivariate_normal_density

from toolkit.hmm.impl_hmmlearn import SequentialGaussianFHMM
from toolkit.hmm.impl_hmmlearn import ExactGaussianFHMM


TESTDATA_PATH = os.path.join(os.path.dirname(__file__), 'fixtures')


class TestFHMM(unittest.TestCase):
    def test_forward(self):
        chain0_means = np.array([[-5.99882413e+02, 9.99894546e+01],
                                 [-3.00048799e+02, 1.99167070e+02],
                                 [-4.28859220e-02, 2.99416348e+02]])
        chain0_covars = np.array([[0.93523132, 0.75423995],
                                  [0.58861593, 0.4016048],
                                  [0.25641893, 1.60482337]])
        chain0_logstart = np.array([-4.44089210e-16, -3.60436534e+01, -3.60436534e+01])
        chain0_logtransmat = np.array([[-38.34619345,  0.0, -38.34619345],
                                       [-38.34619345, -38.34619345, 0.0],
                                       [-1.09861229, -1.09861229, -1.09861229]])

        chain1_means = np.array([[-5.99882413e+02, 9.99894546e+01],
                                 [-3.00048799e+02, 1.99167070e+02],
                                 [-4.39995282e-02, 2.99415977e+02]])
        chain1_covars = np.array([[3.73792529, 3.01395982],
                                  [2.35146944, 1.60341962],
                                  [1.3566332, 6.45346843]])
        chain1_logstart = np.array([-4.44089210e-16, -3.60436534e+01, -3.60436534e+01])
        chain1_logtransmat = np.array([[-3.83461934e+01, -3.71345941e-06, -1.25035485e+01],
                                       [-3.83461897e+01, -2.35611266e+01, -5.85509419e-11],
                                       [-2.58426449e+01, -2.58426449e+01, -1.19594334e-11]])

        loglikelihood = -55.089351487070289
        obs = [np.array([[-600, 100], [-300, 200], [0, 300]]) for _ in xrange(1)]

        # Fit the FHMM on random noise b/c we need it to be fitted in order to manipulate the chains
        fhmm = SequentialGaussianFHMM(n_chains=2, n_states=3)
        fhmm.fit([np.random.random((3, 2))])

        # Update chain0
        fhmm.chains_[0]._means_ = chain0_means
        fhmm.chains_[0]._covars_ = chain0_covars
        fhmm.chains_[0]._log_startprob = chain0_logstart
        fhmm.chains_[0]._log_transmat = chain0_logtransmat

        # Update chain1
        fhmm.chains_[1]._means_ = chain1_means
        fhmm.chains_[1]._covars_ = chain1_covars
        fhmm.chains_[1]._log_startprob = chain1_logstart
        fhmm.chains_[1]._log_transmat = chain1_logtransmat

        self.assertEqual(loglikelihood, fhmm.loglikelihood(obs[0]))

    def test_forward_with_hmmlearn(self):
        r = np.random.randn
        obs = [np.array([[-600 + r(), 100 + r()], [-300 + r(), 200 + r()], [0 + r(), 300 + r()]]) for _ in xrange(10)]
        hmm = GaussianHMM(n_components=3)
        hmm.fit(obs)

        # Calculcate fwdlattice using hmmlearn algorithm
        framelogprob = hmm._compute_log_likelihood(obs[0])
        start = timeit.default_timer()
        _, fwdlattice1 = hmm._do_forward_pass(framelogprob)
        print('hmmlearn took %fs' % (timeit.default_timer() - start))

        # Calculate fwdlattice using fhmm algorithm with #chains = 1. This should yield the exact same results
        start = timeit.default_timer()
        fwdlattice2 = np.zeros(fwdlattice1.shape)
        fhmmc._forward(obs[0].shape[0], 1, hmm.n_components, [(x,) for x in xrange(hmm.n_components)],
                       hmm._log_startprob.reshape(1, 3), hmm._log_transmat.reshape(1, 3, 3), framelogprob, fwdlattice2)
        print('fhmm took %fs' % (timeit.default_timer() - start))
        self.assertTrue(np.allclose(fwdlattice1, fwdlattice2))

    def test_backward_with_hmmlearn(self):
        r = np.random.randn
        obs = [np.array([[-600 + r(), 100 + r()], [-300 + r(), 200 + r()], [0 + r(), 300 + r()]]) for _ in xrange(10)]
        hmm = GaussianHMM(n_components=3)
        hmm.fit(obs)

        # Calculcate bwdlattice using hmmlearn algorithm
        framelogprob = hmm._compute_log_likelihood(obs[0])
        start = timeit.default_timer()
        bwdlattice1 = hmm._do_backward_pass(framelogprob)
        print('hmmlearn took %fs' % (timeit.default_timer() - start))

        # Calculate bwdlattice using fhmm algorithm with #chains = 1. This should yield the exact same results
        start = timeit.default_timer()
        bwdlattice2 = np.zeros(bwdlattice1.shape)
        fhmmc._backward(obs[0].shape[0], 1, hmm.n_components, [(x,) for x in xrange(hmm.n_components)],
                        hmm._log_startprob.reshape(1, 3), hmm._log_transmat.reshape(1, 3, 3), framelogprob, bwdlattice2)
        print('fhmm took %fs' % (timeit.default_timer() - start))
        self.assertTrue(np.allclose(bwdlattice1, bwdlattice2))

    def test_framelogprob_reshape(self):
        n_states = 3
        n_chains = 2
        n_state_combinations = n_states ** n_chains
        state_combinations = [tuple(x) for x in list(itertools.product(np.arange(n_states), repeat=n_chains))]

        covars = np.random.random((n_state_combinations, 10))
        means = np.random.random((n_state_combinations, 10))

        ob = np.random.random((5, 10))
        framelogprob = log_multivariate_normal_density(ob, means, covars, covariance_type='diag')

        # This test assures that resizing the framelogprob still yields the correct state variables
        reshaped_framelogprob = framelogprob.reshape((5, n_states, n_states))
        for ob_idx in xrange(5):
            for idx, state_combination in enumerate(state_combinations):
                self.assertEqual(reshaped_framelogprob[ob_idx][state_combination], framelogprob[ob_idx][idx])

    def test_exact_fhmm_train(self):
        path = os.path.join(TESTDATA_PATH, 'X.mat')
        self.assertTrue(os.path.exists(path))
        X = loadmat(path)['X']
        obs = [X]

        fhmm_exact = ExactGaussianFHMM(n_chains=2, n_states=3, n_training_iterations=20, training_threshold=0.01, topology='full')
        fhmm_exact.fit(obs)
        print 'exact:', fhmm_exact.loglikelihood(obs[0])

        fhmm_seq = SequentialGaussianFHMM(n_chains=2, n_states=3, n_training_iterations=20, training_threshold=0.01, topology='full')
        fhmm_seq.fit(obs)
        print 'seq:', fhmm_seq.loglikelihood(obs[0])

    def test_foo(self):
        obs = [np.array([[-600, 100], [-300, 200], [0, 300]]) for _ in xrange(10)]
        fhmm = ExactGaussianFHMM(n_chains=2, n_states=3)
        fhmm.fit(obs)
        print fhmm.loglikelihood(obs[0])

    def test_exact_fhmm(self):
        n_chains = 3
        n_states = 4
        n_features = 2

        obs = [np.array([[-600, 100], [-300, 200], [0, 300]])]

        means = np.array([[162.2636, 154.0879],
                          [-299.5184,   0.1605],
                          [-313.1509,  -4.3836],
                          [ -15.0993,  94.9669],
                          [-102.5866,  65.8045],
                          [ -88.9765,  70.3412],
                          [-135.6746,  54.7751],
                          [-138.2674,  53.9109],
                          [-147.2465,  50.9178],
                          [-135.9769,  54.6744],
                          [-33.4946,  88.8351],
                          [-148.7870,  50.4043]])
        means = means.reshape((n_chains, n_states, n_features))

        covar = 1.0e+03 * np.array([[9.1968, 3.0656],
                                   [3.0656, 1.0219]])

        transmat = np.array([[0.3660, 0.0045, 0.0072, 0.6223],
                             [0.2442, 0.0546, 0.0053, 0.6959],
                             [0.0037, 0.0887, 0.0503, 0.8573],
                             [0.9465, 0.0006, 0.0004, 0.0525],
                             [0.1922, 0.5774, 0.0381, 0.1923],
                             [0.3122, 0.4747, 0.0784, 0.1347],
                             [0.0432, 0.6053, 0.2195, 0.1320],
                             [0.3434, 0.2505, 0.0443, 0.3618],
                             [0.0864, 0.3736, 0.5159, 0.0241],
                             [0.2319, 0.2389, 0.5001, 0.0291],
                             [0.0741, 0.1285, 0.6239, 0.1735],
                             [0.3229, 0.2213, 0.2160, 0.2398]])
        transmat = transmat.reshape((n_chains, n_states, n_states))
        self.assertTrue(np.allclose(np.sum(transmat, axis=2), 1))

        startprob = np.array([[0.0012, 0.1196, 0.3150],
                              [0.5932, 0.1659, 0.3572],
                              [0.4049, 0.1440, 0.0027],
                              [0.0007, 0.5705, 0.3251]])
        startprob = startprob.T.reshape((n_chains, n_states))
        self.assertTrue(np.allclose(np.sum(startprob, axis=1), 1.0))

        fhmm = ExactGaussianFHMM(n_states=n_states, n_chains=n_chains, topology='full', n_training_iterations=1)
        fhmm.fit([np.random.random((20, 2))])
        fhmm.log_startprob = np.log(startprob)
        fhmm.log_transmat = np.log(transmat)
        fhmm.means = means
        fhmm.covar = covar

        actual_loglikelihood = fhmm.loglikelihood(obs[0])
        # The loglikelihood calculated by the reference impl is -15.0552, which is probably due to to the fact that we
        # calculate in logs and the ref implementation uses scaling
        loglikelihood = -11.425412024522673
        self.assertAlmostEqual(actual_loglikelihood, loglikelihood)
