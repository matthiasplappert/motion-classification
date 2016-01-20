# coding=utf8
from collections import namedtuple
from argparse import ArgumentParser
import timeit
import os
import logging

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import accuracy_score, coverage_error, label_ranking_average_precision_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import MDS
from sklearn.manifold import TSNE

from toolkit.hmm.base import Classifier as HMMClassifier
from toolkit.hmm.impl_hmmlearn import GaussianHMM as HMMLearnModel
from toolkit.hmm.impl_hmmlearn import ExactGaussianFHMM
from toolkit.hmm.impl_hmmlearn import SequentialGaussianFHMM
from toolkit.hmm.impl_pomegranate import GaussianHMM as PomegranateModel
from toolkit.dataset.base import StratifiedMultilabelKFold, load_manifest, shuffle
import toolkit.decision as decision
import toolkit.metrics as metrics
import toolkit.dataset.mmm as mmm
import toolkit.dataset.vicon as vicon


Dataset = namedtuple('Dataset', 'X y target_names groups')


def _plot_proto_symbol_space(coordinates, target_names, name, args):
    # Reduce to 2D so that we can plot it.
    coordinates_2d = TSNE().fit_transform(coordinates)

    n_samples = coordinates_2d.shape[0]
    x = coordinates_2d[:, 0]
    y = coordinates_2d[:, 1]
    colors = cm.rainbow(np.linspace(0, 1, n_samples))

    fig = plt.figure(1)
    plt.clf()
    ax = fig.add_subplot(111)
    dots = []
    for idx in xrange(n_samples):
        dots.append(ax.plot(x[idx], y[idx], "o", c=colors[idx], markersize=15)[0])
        ax.annotate(target_names[idx],  xy=(x[idx], y[idx]))
    lgd = ax.legend(dots, target_names, ncol=4, numpoints=1, loc='upper center', bbox_to_anchor=(0.5,-0.1))
    ax.grid('on')

    if args.output_dir is not None:
        path = os.path.join(args.output_dir, name + '.pdf')
        print('Saved plot to file "%s"' % path)
        fig.savefig(path, bbox_extra_artists=(lgd,), bbox_inches='tight')
    else:
        plt.show()


def run_kfold(dataset, args):
    n_samples = len(dataset.X)
    n_folds = args.n_folds

    print('\nValidating with %d-fold on %d samples ...' % (n_folds, n_samples))
    kfold = StratifiedMultilabelKFold(dataset.y, n_folds=n_folds, groups=dataset.groups)
    for rnd, (train, test) in enumerate(kfold):
        print('\n\n*** Validation Round %d ***' % (rnd+1))
        X_train = [dataset.X[i] for i in train]
        X_test = [dataset.X[i] for i in test]
        y_train, y_test = dataset.y[train], dataset.y[test]
        dataset_train = Dataset(X_train, y_train, dataset.target_names, None)
        dataset_test = Dataset(X_test, y_test, dataset.target_names, None)

        # Ensure that everything is fine before we spend a lot of time training
        n_features = X_train[0].shape[1]
        for curr in X_train:
            assert curr.shape[1] == n_features
        for curr in X_test:
            assert curr.shape[1] == n_features

        dataset_train, dataset_test = preprocess_datasets(dataset_train, dataset_test, args)
        classifier = get_classifier(args)
        if args.n_chains is not None:
            state_name = '%dx%d' % (args.n_states, args.n_chains)
        else:
            state_name = str(args.n_states)
        report_base_name = '%s-%s-%s-%s_kfold-%d' % (args.motion_type, args.model, args.topology, state_name, (rnd + 1))
        validate(classifier, dataset_train, dataset_test, args, report_base_name)


def validate(classifier, train, test, args, report_base_name):
    print('\nTraining classifier on %d samples ...' % len(train.X))
    start = timeit.default_timer()
    classifier.fit(train.X, train.y)
    stop = timeit.default_timer()
    print('Classifier trained, took %f seconds' % (stop - start))

    for method in args.loglikelihood_methods:
        report_name = report_base_name + '_' + method

        if args.calculate_distances:
            print('\nCalculating distances ...')
            start = timeit.default_timer()
            distances = classifier.distances(loglikelihood_method=method, n_samples=500)
            print('Distances calculated, took %f seconds' % (timeit.default_timer() - start))

            report = _generate_distance_reports(distances, target_names=train.target_names)
            _handle_report(report, report_name + '_distances', args)

            # Calculate proto symbol space
            #mds = MDS(n_components=5, dissimilarity='precomputed')
            #coordinates = mds.fit_transform(distances)
            #_plot_proto_symbol_space(coordinates, train.target_names, report_name + '_scatter', args)

        # Get loglikelihoods for train set
        print('\nValidating classifier on training set with %d samples ...' % len(train.X))
        loglikelihoods_train = _calculate_loglikelihoods(classifier, train.X, method)
        report = _generate_loglikelihood_reports(loglikelihoods_train, train.y, target_names=train.target_names)
        _handle_report(report, report_name + '_train_loglikelihoods', args)

        # Fit decision makers
        loglikelihoods_test = None
        for idx, decision_maker in enumerate(get_decision_makers(args)):
            if decision_maker is not None:
                name = args.decision_makers[idx]
                if hasattr(decision_maker, 'fit') and callable(getattr(decision_maker, 'fit')):
                    print('\nTraining decision maker %s on %d loglikelihoods ...' % (name, len(loglikelihoods_train)))
                    decision_maker.fit(loglikelihoods_train, train.y)
                    print('Decision maker trained, took %f seconds' % (stop - start))
                else:
                    print('\nUsing decision maker %s ...' % name)
                y_pred = _calculate_predictions(decision_maker, loglikelihoods_train)
                report = _generate_classification_reports(train.y, y_pred, target_names=train.target_names)
                _handle_report(report, report_name + '_train_classification_' + name, args)

            # Validate on test set
            print('\nValidating classifier on test set with %d samples ...' % len(test.X))
            if loglikelihoods_test is None:
                loglikelihoods_test = _calculate_loglikelihoods(classifier, test.X, method)
                report = _generate_loglikelihood_reports(loglikelihoods_test, test.y, target_names=test.target_names)
                _handle_report(report, report_name + '_test_loglikelihoods', args)
            if decision_maker is not None:
                y_pred = _calculate_predictions(decision_maker, loglikelihoods_test)
                report = _generate_classification_reports(test.y, y_pred, target_names=test.target_names)
                _handle_report(report, report_name + '_test_classification_' + name, args)


def _calculate_loglikelihoods(classifier, X, method):
    # Predict
    print('Calculating loglikelihoods (%s) ...' % method)
    start = timeit.default_timer()
    loglikelihoods = classifier.loglikelihoods(X, method=method)
    stop = timeit.default_timer()
    print('Calculation complete, took %f seconds\n' % (stop - start))
    return loglikelihoods


def _calculate_predictions(decision_maker, X):
    # Predict
    print('Predicting ...')
    start = timeit.default_timer()
    y_pred = decision_maker.predict(X)
    stop = timeit.default_timer()
    print('Prediction complete, took %f seconds\n' % (stop - start))
    return y_pred


def _handle_report(report, name, args):
    if args.output_dir is not None:
        path = os.path.join(args.output_dir, name + '.txt')
        with open(path, 'w') as f:
            f.write(report)
        print('Saved report to file "%s"' % path)
    else:
        print report


def _generate_distance_reports(distances, target_names=None):
    report = metrics.distances_report(distances, target_names=target_names)
    return report


def _generate_loglikelihood_reports(loglikelihoods, y, target_names=None):
    report = metrics.multilabel_loglikelihood_report(y, loglikelihoods)
    report += '\n\n'
    report += metrics.multilabel_loglikelihood_summary_report(y, loglikelihoods, target_names=target_names)
    return report


def _generate_classification_reports(y_true, y_pred, target_names=None):
    # Calculate additional stats
    total_accuracy = accuracy_score(y_true, y_pred)
    cov_error = coverage_error(y_true, y_pred)
    lrap = label_ranking_average_precision_score(y_true, y_pred)

    report = metrics.multilabel_prediction_report(y_true, y_pred)
    report += '\n\n'
    report += metrics.multilabel_classification_report(y_true, y_pred, target_names=target_names)
    report += '\n\n'
    report += 'coverage error:  %.3f' % cov_error
    report += '\n'
    report += 'LRAP:            %.3f' % lrap
    report += '\n'
    report += 'total accuracy:  %.3f' % total_accuracy
    return report


# def run_train_test(path_train, path_test, args):
#     print('Loading train data set "%s"...' % path_train)
#     X_train, y_train, tags_train, _ = dataset.load_manifest(path_train)
#
#     print('\nLoading test data set "%s" ...' % path_test)
#     X_test, y_test, tags_test, _ = dataset.load_manifest(path_test)
#
#     report_base_name = args.model + '_kfold_%d' % rnd
#     validate(X_train, y_train, X_test, y_test, report_base_name, target_names=tags_train)


def get_parser():
    all_features = []
    all_features.extend(mmm.FEATURE_NAMES)
    all_features.extend(vicon.FEATURE_NAMES)

    parser = ArgumentParser()
    parser.add_argument('dataset', help='path to the dataset')
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--permutation', type=str, default=None)

    parser.add_argument('--model', choices=['hmmlearn', 'pomegranate', 'fhmm-seq', 'fhmm-exact'], default='hmmlearn')
    parser.add_argument('--topology', choices=['left-to-right', 'left-to-right-cycle', 'bakis', 'full'], default='left-to-right')
    parser.add_argument('--decision-makers', nargs='*', choices=['min', 'max', 'mean', 'median', 'zero', 'decision-tree', 'random-forest'],
                        default=[None])
    parser.add_argument('--preprocessing', nargs='*', choices=['scale'], default=[None])
    parser.add_argument('--loglikelihood-methods', nargs='*', choices=['exact', 'approx'], default=['exact'])
    parser.add_argument('--features', nargs='*', choices=all_features, default=None)
    parser.add_argument('--motion-type', choices=['mmm', 'mmm-nlopt', 'vicon'], default='mmm-nlopt')

    parser.add_argument('--keep-groups', action='store_true')
    parser.add_argument('--calculate-distances', action='store_true')
    parser.add_argument('--disable-shuffle', action='store_true')
    parser.add_argument('--disable-cache', action='store_true')

    parser.add_argument('--n-folds', type=int, default=3)
    parser.add_argument('--n-iterations', type=int, default=10)
    parser.add_argument('--n-jobs', type=int, default=-1)
    parser.add_argument('--n-chains', type=int, default=2)
    parser.add_argument('--n-states', type=int, default=10)
    return parser


def get_classifier(args):
    if args.model == 'hmmlearn':
        model = HMMLearnModel()
    elif args.model == 'fhmm-exact':
        model = ExactGaussianFHMM(n_chains=args.n_chains)
    elif args.model == 'fhmm-seq':
        model = SequentialGaussianFHMM(n_chains=args.n_chains)
    elif args.model == 'pomegranate':
        model = PomegranateModel()
    else:
        model = None
    assert model is not None
    model.n_training_iterations = args.n_iterations
    model.n_states = args.n_states
    model.topology = args.topology
    return HMMClassifier(model, n_jobs=args.n_jobs)


def get_decision_makers(args):
    decision_makers = []
    for decision_maker_name in args.decision_makers:
        if decision_maker_name in ('min', 'max'):
            decision_maker = decision.ExtremumDecisionMaker(measure=decision_maker_name)
        elif decision_maker_name in ('mean', 'median', 'zero'):
            decision_maker = decision.CentralDecisionMaker(measure=decision_maker_name)
        elif decision_maker_name == 'decision-tree':
            decision_maker = decision.DecisionTreeDecisionMaker()
        elif decision_maker_name == 'random-forest':
            decision_maker = decision.RandomForestDecisionMaker()
        else:
            decision_maker = None
        decision_makers.append(decision_maker)
        # TODO: add support for cluster-based
    return decision_makers


def load_dataset(args):
    path = args.dataset
    keep_groups = args.keep_groups

    print('Loading data set "%s" ...' % path)
    print('  features: %s' % args.features)
    print('  type: %s' % args.motion_type)
    X, y, target_names, groups, _ = load_manifest(path, args.motion_type, feature_names=args.features, use_cache=not args.disable_cache)
    assert len(X) == len(y)
    if not keep_groups:
        groups = None

    if not args.disable_shuffle:
        if args.permutation is not None:
            permutation = np.load(open(args.permutation))
            X, y = [X[i] for i in permutation], y[permutation, :]
        else:
            X, y, groups, permutation = shuffle(X, y, groups=groups)
        print('Shuffling all data with permutation (keep_groups=%s):\n%s' % (keep_groups, permutation))
        if args.output_dir is not None:
            permutation_path = os.path.join(args.output_dir, 'permutation.npy')
            np.save(permutation_path, permutation)
            print('Saved permutation to file "%s"' % permutation_path)
    return Dataset(X, y, target_names, groups)


def preprocess_datasets(train, test, args):
    if 'scale' in args.preprocessing:
        print('Scaling features to range [-1,1] ...')
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(np.vstack(train.X))
        processed_train = Dataset([scaler.transform(X_curr) for X_curr in train.X], train.y, train.target_names, train.groups)
        processed_test = Dataset([scaler.transform(X_curr) for X_curr in test.X], test.y, test.target_names, test.groups)
    else:
        processed_train = train
        processed_test = test
    return processed_train, processed_test


def main(args):
    start = timeit.default_timer()

    # Validate that paths exist so that we don't need to check that whenever we use it
    if not os.path.exists(args.dataset):
        exit('data set at path "%s" does not exist' % args.dataset)
    if args.output_dir is not None and not os.path.isdir(args.output_dir):
        exit('--output-dir "%s" is not a directory' % args.output_dir)
    if 'fhmm' not in args.model:
        args.n_chains = None
    if args.permutation is not None and args.keep_groups:
        exit('using --permutation with --keep-groups is currently unsupported')

    if args.n_jobs != 1 and args.calculate_distances:
        print('Warning: There is currently a bug when using --calculating-distances and using --n-jobs >= 1. Falling back to 1 job!')
        args.n_jobs = 1

    # Load dataset and run kfold
    dataset = load_dataset(args)
    run_kfold(dataset, args)

    print('Total time: %fs' % (timeit.default_timer() - start))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main(get_parser().parse_args())
