import abc
import logging

from joblib import Parallel, delayed
import numpy as np
from sklearn.base import (BaseEstimator, ClassifierMixin, clone)
from sklearn.cluster import KMeans
from sklearn.covariance import EmpiricalCovariance

from ..util import (check_feature_array, check_is_fitted)


def distance(model1, model2, n_samples=100, loglikelihood_method='exact'):
    # Taken from Rabiner 1989 (p. 271)
    d12 = _non_symmetric_distance(model1, model2, n_samples, loglikelihood_method)
    d21 = _non_symmetric_distance(model2, model1, n_samples, loglikelihood_method)
    return (d12 + d21) / 2.0


def _non_symmetric_distance(model1, model2, n_samples, method):
    seq = model1.sample(n_samples)
    score1 = model1.loglikelihood(seq, method=method)
    score2 = model2.loglikelihood(seq, method=method)
    d = (score1 - score2) / float(n_samples)
    if d < 0.:
        logging.warn('distance measure is negative %f - using absolute value, but this indicates a problem' % d)
    return abs(d)


def estimate_normal_distribution_params(obs, n_states, covar_type='diag', randomize=False):
    """Estimates parameters (mean and covariance) for a number of states

    Parameters
    ----------
    obs : list of array-like, shape=(n_samples, n_features)
        A list of observations. The number of features must be equal across all observations.

    n_states : int
        The number of states of the HMM. For each state, a mean vector and a covariance matrix are estimated.

    covar_type : string, default: 'diag'
        The type of the covariance. Supported values are 'diag' for a diagonalized matrix and 'full' for a full
        covariance matrix.

    Returns
    -------
    means : array-like, shape=(n_states, n_features)
        The means for all states. Each row represents the means for all n_features.

    covars: array-like, shape=(n_states, n_features, n_features)
        The covars for all states.
    """
    if not isinstance(obs, list):
        raise ValueError('obs must be a list of sequences')
    if len(obs) == 0:
        raise ValueError('obs must contain at least one sequence')
    if covar_type not in ['diag', 'full']:
        raise ValueError('unknown covar_type %s' % covar_type)

    all_obs = np.concatenate(obs)
    n_features = all_obs.shape[1]

    if randomize:
        # No need to estimate anything, just use random stuff
        means = np.array([np.random.random(n_features) for _ in xrange(n_states)])
        covars = []
        for state in xrange(n_states):
            # Create random semi-definite matrix
            covar = np.random.random((n_features, n_features))
            covars.append(np.dot(covar, covar.T))
        covars = np.array(covars)
    else:
        # Estimate means
        kmeans = KMeans(n_clusters=n_states)
        kmeans.fit(all_obs)
        means = kmeans.cluster_centers_
        assert means.shape == (n_states, n_features)

        # Predict clusters and calculate covariances
        predict = kmeans.predict(all_obs)
        covars = []
        for state in xrange(n_states):
            indexes = np.where(predict == state)[0]
            state_obs = [all_obs[idx] for idx in indexes]
            assert len(state_obs) > 0
            covar_estimator = EmpiricalCovariance().fit(state_obs)
            covar = covar_estimator.covariance_
            covars.append(covar)
        covars = np.array(covars)

    # Constrain diagonal elements to be not smaller than 0.001 and make covar diagonal if requested
    if covar_type == 'diag':
        for state in xrange(n_states):
            covar = covars[state]
            diag = covar.diagonal().copy()
            covar[:, :] = 0.0
            diag[diag < 1e-4] = 1e-4
            np.fill_diagonal(covar, diag)

    assert means.shape == (n_states, n_features)
    assert covars.shape == (n_states, n_features, n_features)
    return means, covars


def transition_matrix(n_states, topology, randomize=False):
    transmat = np.zeros((n_states, n_states))
    if topology.startswith('left-to-right'):
        if topology.endswith('-full'):
            # In the full 'left-to-right' topology each state has a connection to every other state that is located on
            # its right and to itself. The transition matrix is therefore an upper triangular matrix.
            for i in range(n_states):
                transmat[i, i:] = 1.0 / float(n_states - i)
        elif topology.endswith('-1'):
            # Delta = 1, hence self transition + transition to next state is allowed.
            for i in range(n_states):
                if i == n_states-1:
                    transmat[i, i] = 1.0
                else:
                    transmat[i, i] = 0.5
                    transmat[i, i+1] = 0.5
        elif topology.endswith('-2'):
            # Delta = 2, hence self transition + transition to next state + transition to state after next state is
            # allowed.
            for i in range(n_states):
                if i == n_states-1:
                    transmat[i, i] = 1.0
                elif i == n_states-2:
                    transmat[i, i] = 0.5
                    transmat[i, i+1] = 0.5
                else:
                    transmat[i, i] = 1. / 3.
                    transmat[i, i+1] = 1. / 3.
                    transmat[i, i+2] = 1. / 3.
    elif topology == 'full':
        # In the 'full' topology each state is connect with every other state and itself.
        transmat[:, :] = 1.0 / float(n_states)
    else:
        raise ValueError('unknown topology %s' % topology)

    if randomize:
        transmat *= np.random.random(transmat.shape)
        sums = np.sum(transmat, axis=1)
        for row_idx in xrange(transmat.shape[0]):
            transmat[row_idx] /= sums[row_idx]

    assert np.allclose(np.sum(transmat, axis=1), np.ones(n_states))
    return transmat


def start_probabilities(n_states, topology, randomize=False):
    pi = np.zeros(n_states)
    if topology.startswith('left-to-right'):
        pi[0] = 1.0
    elif topology == 'full':
        pi[:] = 1.0 / float(n_states)
    else:
        raise ValueError('unknown topology %s' % topology)

    if randomize:
        pi *= np.random.random(pi.shape)
        pi /= np.sum(pi)

    assert np.isclose(np.sum(pi), 1.0)
    return pi


class BaseHMM(BaseEstimator):
    """Base class for all Hidden Markov Model implementations

    Parameters
    ----------
    n_states : int
        The number of states of the HMM.

    n_training_iterations : int, default: 10
        The maximum number of iterations that the training algorithm
        should perform.

    training_threshold: float, default: 1e-2
        The threshold after which the training is considered converged.

    topology: string, default: 'left-to-right'
        The topology of the HMM. Possible values are 'left-to-right', 'full' and 'linear'. See transition_matrix
        for details on the topologies.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, n_states, n_training_iterations, training_threshold, topology, verbose, transition_init, emission_init, covar_type):
        self.n_states = n_states
        self.n_training_iterations = n_training_iterations
        self.training_threshold = training_threshold
        self.topology = topology
        self.n_features_ = None
        self.verbose = verbose
        self.transition_init = transition_init
        self.emission_init = emission_init
        self.covar_type = covar_type

    def fit(self, obs):
        """Train HMM with the given observations

        Parameters
        ----------
        obs : list of array-like, shape=(n_samples, n_features)
            A list of observations. The number of features must be
            equal across all observations.
        """
        if not isinstance(obs, list):
            raise ValueError('obs must be a list of sequences')
        if len(obs) == 0:
            raise ValueError('obs must contain at least one sequence')

        n_features = self.n_features_
        if n_features is None:
            n_features = check_feature_array(obs[0]).shape[1]
        obs = [check_feature_array(ob, n_features=n_features) for ob in obs]

        if self.n_features_ is None:
            self.n_features_ = n_features
            self._init(obs)
        return self._fit(obs)

    def loglikelihood(self, ob, method='exact'):
        """Computes the loglikelihood of ob under the model

        Parameters
        ----------
        ob : array-like, shape=(n_samples, n_features)
            The observation. The number of features must be equal to the
            number of features that were used to train the model.

        Returns
        -------
        loglikelihood : float
            The loglikelihood of the observation under the model.
        """
        check_is_fitted(self, 'n_features_')
        ob = check_feature_array(ob, n_features=self.n_features_)
        return self._loglikelihood(ob, method)

    def sample(self, n_samples=1, max_cycle_duration=10000):
        """Samples an observation from the model

        Parameters
        ----------
        n_samples : int, default: 1
            The number of samples to generate.

        Returns
        -------
        ob : array-like, shape=(n_samples, n_features)
            The sampled observation.
        """
        check_is_fitted(self, 'n_features_')
        return self._sample(n_samples, max_cycle_duration)

    def supports_parallel(self):
        return True

    @abc.abstractmethod
    def _init(self, obs):
        return

    @abc.abstractmethod
    def _fit(self, obs):
        return

    @abc.abstractmethod
    def _loglikelihood(self, ob, method):
        return

    @abc.abstractmethod
    def _sample(self, n_samples, max_cycle_duration):
        return


class Classifier(BaseEstimator, ClassifierMixin):
    def __init__(self, model, n_jobs=-1):
        self.model = model
        self.n_jobs = n_jobs
        self.models_ = None
        self.n_labels_ = None

    def fit(self, X, y):
        assert isinstance(X, list)  #TODO: this should not be an assert
        assert len(y) > 0
        assert len(X) == len(y)

        # TODO: add support for fitting again after having already performed a fit
        self.n_labels_ = y.shape[1]
        self.models_ = []

        # Train one model per label. If no data is available for a given label, the model is set to None.
        models, data = [], []
        for idx in range(self.n_labels_):
            d = [X[i] for i in np.where(y[:, idx] == 1)[0]]
            if len(d) == 0:
                model = None
            else:
                model = clone(self.model)
            data.append(d)
            models.append(model)
        assert len(models) == len(data)
        n_jobs = self.n_jobs if self.model.supports_parallel() else 1
        self.models_ = Parallel(n_jobs=n_jobs)(delayed(_perform_fit)(models[i], data[i]) for i in range(len(models)))
        assert len(self.models_) == self.n_labels_

    def loglikelihoods(self, X, method='exact'):
        assert isinstance(X, list)  #TODO: this should not be an assert
        assert len(X) > 0
        check_is_fitted(self, 'models_', 'n_labels_')

        # Calculate loglikelihoods under each model for each observation.
        n_jobs = self.n_jobs if self.model.supports_parallel() else 1
        scores = Parallel(n_jobs=n_jobs)(delayed(_perform_loglikelihoods)(m, X, method) for m in self.models_)
        loglikelihoods = np.zeros((len(X), len(self.models_)))
        for column_idx, score in enumerate(scores):
            loglikelihoods[:, column_idx] = score
        return loglikelihoods

    def distances(self, n_samples=100, loglikelihood_method='exact'):
        # TODO: make this parallel
        check_is_fitted(self, 'models_', 'n_labels_')
        distances = np.zeros((self.n_labels_, self.n_labels_))
        for i in xrange(self.n_labels_):
            # Distance is a symmetric measure, hence the upper-right and lower-left triangular matrices are identical.
            # Speed things up a bit by only calculating the measure once.
            for j in xrange(i, self.n_labels_):
                if i == j:
                    distances[i, j] = 0
                else:
                    # Distance is symmetric
                    distances[i, j] = distance(self.models_[i], self.models_[j], n_samples=n_samples,
                                               loglikelihood_method=loglikelihood_method)
                    distances[j, i] = distances[i, j]
        return distances


def _perform_fit(model, data):
    if model is None:
        return None
    assert len(data) > 0
    model.fit(data)
    return model


def _perform_loglikelihoods(model, data, method):
    loglikelihoods = np.zeros(len(data))
    for i, d in enumerate(data):
        if model is None:
            # We do not have a model for this label since the training data did not contain it. Assume zero probability,
            # which is -inf on a logarithmic scale. Since -np.inf can cause all sorts of problems, use a very large
            # negative number.
            loglikelihoods[i] = np.finfo('float32').min
        else:
            loglikelihoods[i] = model.loglikelihood(d, method=method)
    return loglikelihoods