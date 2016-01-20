import abc

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier

from util import check_feature_array, check_multilabel_array, check_is_fitted


def decision_maker_from_args(args):
    decision_makers = []
    decision_maker_names = args.decision_maker
    allow_multiple = False
    if decision_maker_names is None:
        decision_maker_names = []
    elif type(decision_maker_names) == str:
        decision_maker_names = [decision_maker_names]
    elif type(decision_maker_names) == list:
        allow_multiple = True

    for decision_maker_name in decision_maker_names:
        if decision_maker_name == 'max':
            decision_maker = MaximumDecisionMaker()
        elif decision_maker_name in ('mean', 'median', 'zero'):
            decision_maker = CentralDecisionMaker(measure=decision_maker_name)
        elif decision_maker_name == 'decision-tree':
            decision_maker = DecisionTreeDecisionMaker()
        elif decision_maker_name == 'random-forest':
            decision_maker = RandomForestDecisionMaker()
        elif decision_maker_name == 'k-neighbors':
            decision_maker = KNeighborsDecisionMaker()
        elif decision_maker_name == 'log-regression':
            decision_maker = LogisticRegressionDecisionMaker()
        elif decision_maker_name == 'svm':
            decision_maker = SVMDecisionMaker()
        elif decision_maker_name == 'naive-bayes':
            decision_maker = NaiveBayesDecisionMaker()
        elif decision_maker_name == 'perceptron':
            decision_maker = PerceptronDecisionMaker()
        else:
            decision_maker = None
        assert decision_maker is not None
        decision_makers.append(decision_maker)

    for decision_maker in decision_makers:
        for arg_name in vars(args):
            split = arg_name.split('decision_maker_')
            if len(split) != 2:
                continue
            value = getattr(args, arg_name)
            setattr(decision_maker, split[1], value)
        if 'n_jobs' in vars(args):
            decision_maker.n_jobs = args.n_jobs

    if allow_multiple:
        return decision_makers
    else:
        if len(decision_makers) == 0:
            return None
        else:
            return decision_makers[0]


def add_decision_maker_parser_arguments(parser, allow_multiple=False):
    decision_makers = ['max', 'mean', 'median', 'zero', 'decision-tree', 'random-forest', 'k-neighbors',
                       'log-regression', 'svm', 'naive-bayes', 'perceptron']
    if allow_multiple:
        parser.add_argument('--decision-maker', choices=decision_makers, nargs='+', default=decision_makers)
    else:
        parser.add_argument('--decision-maker', choices=[None] + decision_makers, default=None)

    parser.add_argument('--decision-maker-penalty', choices=['l1', 'l2'], default='l2')
    parser.add_argument('--decision-maker-C', type=float, default=1.)
    parser.add_argument('--decision-maker-kernel', choices=['linear', 'poly', 'rbf', 'sigmoid'], default='linear')
    parser.add_argument('--decision-maker-gamma', type=float, default=0.)
    parser.add_argument('--decision-maker-max-depth', type=int, default=None)
    parser.add_argument('--decision-maker-splitter', choices=['best', 'random'], default='best')
    parser.add_argument('--decision-maker-criterion', choices=['gini', 'entropy'], default='gini')
    parser.add_argument('--decision-maker-n-estimators', type=int, default=10)
    parser.add_argument('--decision-bootstrap', type=bool, default=True)


class DecisionMaker(BaseEstimator, ClassifierMixin):
    """Base class for all DecisionMaker implementations. DecisionMakers are used to find a mapping from a multi-label
    classifier to the binary predictions.

    Concretely, a multi-label classifier calculates some sort of measure (e.g. the loglikelihood) for each possible
    label which it provides to the DecisionMaker instance. The DecisionMaker then attempts to find a good mapping from
    the scores (which can be any real number) to the binary predictions (where each value is either 1 or 0). Depending
    on the DecisionMaker, more than one prediction value per sample can be set to 1.

    To give an example, consider the following matrix of measures:
        [[ 100, -100,  200],
         [-300,  100, -300]]
    A decision maker would then output a matrix that looks somewhat like this:
        [[1, 0, 1],
         [0, 1, 0]]

    Notes
    ------
    If the DecisionMaker implements fit(self, X, y), fit will be called when training the classifier. This allows to
    implement supervised DecisionMakers that will be trained on the same training data that is used in the multi-label
    classifier.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.n_jobs = 1

    @abc.abstractmethod
    def predict(self, X):
        """Decide for each feature in each sample if it is on or off.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        predictions : array, shape (n_samples, n_features)
            For each sample, that is each row, the decision maker attempts to decide if the feature is on (encoded
            as 1) or off (encoded as 0).
        """
        pass


class MaximumDecisionMaker(DecisionMaker):
    def predict(self, X):
        """Decide for each feature in each sample if it is on or off. ExtremumDecisionMaker calculates the
        maximum for each row and turns on the feature with that extreme value.
        Exactly one feature per row is turned on.
        """
        X = check_feature_array(X)
        max_indexes = np.argmax(X, axis=1)
        predictions = np.zeros(X.shape, dtype=int)
        predictions[range(predictions.shape[0]), max_indexes] = 1
        return predictions


class CentralDecisionMaker(DecisionMaker):
    def __init__(self, measure='median', threshold=0.0):
        """CentralDecisionMaker uses some measure of central tendency to make decisions.

        Parameters
        ----------
        measure : string, default: 'median'
            The measure of central tendency to be used. Possible values are 'median', 'mean' and 'zero'.
        threshold : float, default: 0.0
            The threshold that is used to decide if a feature is on or off. The threshold can be used to introduce
            bias towards either class by increasing it (bias towards off) or decreasing it (bias towards on).
        """
        super(CentralDecisionMaker, self).__init__()
        if measure not in ['median', 'mean', 'zero']:
            raise ValueError('unknown measure %s' % measure)
        self.measure = measure
        self.threshold = threshold

    def predict(self, X):
        """Decide for each feature in each sample if it is on or off. The decision is made by the following simple
        calculation for each row x, where central_measure is the specified measure:
            x_scaled = x - central_measure
            predictions[x_scaled >= threshold] = 1
            predictions[x_scaled  < threshold] = 0
        """
        X = check_feature_array(X)

        central_measure = None
        if self.measure == 'median':
            central_measure = np.median(X, axis=1)
        elif self.measure == 'mean':
            central_measure = np.mean(X, axis=1)
        elif self.measure == 'zero':
            central_measure = 0.0
        assert central_measure is not None

        scaled_X = (X.T - central_measure).T
        predictions = np.zeros(scaled_X.shape, dtype=int)
        predictions[scaled_X >= self.threshold] = 1
        return predictions


class _MultiLabelDecisionMaker(DecisionMaker):
    def __init__(self):
        super(_MultiLabelDecisionMaker, self).__init__()
        self.model_ = None
        self.n_features_ = None

    def _init_model(self):
        raise NotImplementedError()

    def fit(self, X, y):
        """Fit the _MultiLabelDecisionMaker according to the given training data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and n_features is the number of features.
        y : array-like, shape (n_samples, n_features)
            Binary target values, where for each sample a feature is either on (encoded as 1) or off (encoded as 0).
        """
        X = check_feature_array(X)
        y = check_multilabel_array(y)
        if X.shape != y.shape:
            raise ValueError('X (shape=%s) and y (shape=%s) must have equal shapes' % (X.shape, y.shape))
        self.n_features_ = X.shape[1]
        self.model_ = self._init_model().fit(X, y)

    def predict(self, X):
        """Decide for each feature in each sample if it is on or off. _MultiLabelDecisionMaker uses a multi-label
        classifier to predict the multi-labels for all features. However, the classifier must first be trained by
        calling fit.
        """
        check_is_fitted(self, 'model_', 'n_features_')
        X = check_feature_array(X, self.n_features_)
        predictions = self.model_.predict(X)
        return predictions


class DecisionTreeDecisionMaker(_MultiLabelDecisionMaker):
    def __init__(self, splitter='best', criterion='gini', max_depth=None):
        super(DecisionTreeDecisionMaker, self).__init__()
        self.splitter = splitter
        self.criterion = criterion
        self.max_depth = max_depth

    def _init_model(self):
        return DecisionTreeClassifier(splitter=self.splitter, criterion=self.criterion, max_depth=self.max_depth)


class RandomForestDecisionMaker(_MultiLabelDecisionMaker):
    def __init__(self, n_estimators=10, bootstrap=True, criterion='gini', max_depth=None):
        super(RandomForestDecisionMaker, self).__init__()
        self.n_estimators = n_estimators
        self.bootstrap = bootstrap
        self.criterion = criterion
        self.max_depth = max_depth

    def _init_model(self):
        return RandomForestClassifier(n_estimators=self.n_estimators, bootstrap=self.bootstrap,
                                      criterion=self.criterion, max_depth=self.max_depth, n_jobs=self.n_jobs)


class KNeighborsDecisionMaker(_MultiLabelDecisionMaker):
    def _init_model(self):
        return KNeighborsClassifier()


class _BinaryRelevanceDecisionMaker(DecisionMaker):
    def __init__(self):
        super(_BinaryRelevanceDecisionMaker, self).__init__()
        self.model_ = None
        self.n_features_ = None

    def _init_model(self):
        raise NotImplementedError()

    def fit(self, X, y):
        X = check_feature_array(X)
        y = check_multilabel_array(y)
        if X.shape != y.shape:
            raise ValueError('X (shape=%s) and y (shape=%s) must have equal shapes' % (X.shape, y.shape))
        self.n_features_ = X.shape[1]
        self.model_ = OneVsRestClassifier(self._init_model(), n_jobs=self.n_jobs).fit(X, y)

    def predict(self, X):
        check_is_fitted(self, 'model_', 'n_features_')
        X = check_feature_array(X, self.n_features_)
        predictions = self.model_.predict(X)
        return predictions


class LogisticRegressionDecisionMaker(_BinaryRelevanceDecisionMaker):
    def __init__(self, penalty='l2', C=1.):
        super(LogisticRegressionDecisionMaker, self).__init__()
        self.penalty = penalty
        self.C = C

    def _init_model(self):
        return LogisticRegression(solver='liblinear', C=self.C, penalty=self.penalty)


class SVMDecisionMaker(_BinaryRelevanceDecisionMaker):
    def __init__(self, C=1., penalty='l2'):
        super(SVMDecisionMaker, self).__init__()
        self.C = C
        self.penalty = penalty

    def _init_model(self):
        return LinearSVC(C=self.C, dual=False, penalty=self.penalty, loss='squared_hinge')


class NaiveBayesDecisionMaker(_BinaryRelevanceDecisionMaker):
    def _init_model(self):
        return GaussianNB()


class PerceptronDecisionMaker(_BinaryRelevanceDecisionMaker):
    def _init_model(self):
        return Perceptron()
