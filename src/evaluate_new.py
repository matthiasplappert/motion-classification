# coding=utf8
from argparse import ArgumentParser
import timeit
import os
import logging
import pickle
import sys
import csv
import random
import itertools

import numpy as np
import sklearn.metrics as sk_metrics

from toolkit.hmm.base import Classifier as HMMClassifier
from toolkit.hmm.impl_hmmlearn import GaussianHMM
from toolkit.hmm.impl_hmmlearn import ExactGaussianFHMM
from toolkit.hmm.impl_hmmlearn import SequentialGaussianFHMM
import toolkit.dataset.base as data
import toolkit.decision as decision
import toolkit.metrics as metrics


def get_parser():
    parser = ArgumentParser()
    data.add_transformer_parser_arguments(parser)
    decision.add_decision_maker_parser_arguments(parser)

    parser.add_argument('action', choices=['model', 'features', 'hmm-hyperparameters', 'hmm-initialization', 'fhmm',
                                           'pca', 'end-to-end'])

    # action = features
    parser.add_argument('--measure', choices=['aicc', 'hmm-distance', 'wasserstein', 'mahalanobis'], default='aicc')

    # Dataset options
    parser.add_argument('dataset', type=str)
    parser.add_argument('--features', type=str, nargs='*', default=None)
    parser.add_argument('--permutation', type=str, default=None)
    parser.add_argument('--disable-shuffle', action='store_true')
    parser.add_argument('--transform-to-multiclass', action='store_true')

    # Model option
    parser.add_argument('--model', choices=['hmm', 'fhmm-seq', 'fhmm-exact'], default='hmm')
    parser.add_argument('--topology', choices=['left-to-right-full', 'left-to-right-1', 'left-to-right-2', 'full'],
                        default='left-to-right-1')
    parser.add_argument('--loglikelihood-method', choices=['exact', 'approx'], default='exact')
    parser.add_argument('--n-chains', type=int, default=2)
    parser.add_argument('--n-states', type=int, default=10)
    parser.add_argument('--n-training-iter', type=int, default=10)
    parser.add_argument('--covar-type', choices=['full', 'diag'], default='diag')
    parser.add_argument('--transition-init', choices=['random', 'uniform'], default='uniform')
    parser.add_argument('--emission-init', choices=['random', 'k-means'], default='k-means')

    # Evaluation options
    parser.add_argument('--n-iter', type=int, default=3)
    parser.add_argument('--n-jobs', type=int, default=1)
    parser.add_argument('--output-dir', type=str)
    parser.add_argument('--verbose', action='store_true')
    return parser


def get_classifier(args):
    if args.model == 'hmm':
        model = GaussianHMM()
    elif args.model == 'fhmm-exact':
        model = ExactGaussianFHMM(n_chains=args.n_chains)
    elif args.model == 'fhmm-seq':
        model = SequentialGaussianFHMM(n_chains=args.n_chains)
    else:
        model = None
    assert model is not None
    model.n_training_iterations = args.n_training_iter
    model.n_states = args.n_states
    model.topology = args.topology
    model.verbose = args.verbose
    model.transition_init = args.transition_init
    model.emission_init = args.emission_init
    model.covar_type = args.covar_type
    return HMMClassifier(model, n_jobs=args.n_jobs)


def save_results(path, y_true, y_pred, lls, prefix):
    np.savetxt(os.path.join(path, '%s_loglikelihoods.csv' % prefix), lls, delimiter=';', fmt='%f')
    np.savetxt(os.path.join(path, '%s_labels.csv' % prefix), y_true, delimiter=';', fmt='%d')
    if y_pred is not None:
        np.savetxt(os.path.join(path, '%s_predictions.csv' % prefix), y_pred, delimiter=';', fmt='%d')


def evaluate_end_to_end(dataset, iterator, args):
    # Things to evaluate
    feature_sets = [('normalized_root_rot_norm',),
                    ('normalized_root_pos', 'normalized_root_vel', 'normalized_extremity_pos', 'normalized_root_rot', 'normalized_root_rot_norm'),
                    ('normalized_extremity_pos', 'normalized_root_rot', 'normalized_root_rot_norm',),
                    ('normalized_root_pos', 'normalized_root_vel', 'normalized_com_pos', 'normalized_extremity_pos', 'normalized_root_rot', 'normalized_root_rot_norm'),
                    ('normalized_root_vel', 'normalized_extremity_pos', 'normalized_root_rot', 'normalized_root_rot_norm'),
                    ('normalized_root_pos', 'normalized_root_vel', 'normalized_com_pos', 'normalized_extremity_pos', 'normalized_root_rot', 'normalized_root_rot_norm', 'normalized_marker_vel_norm')]
    values = {'features': range(len(feature_sets)),
              'hyperparams': [('left-to-right-full', 5), ('left-to-right-1', 6), ('left-to-right-2', 5), ('full', 8)],
              'init': [('uniform', 'k-means', 'diag')],
              'model': [('hmm', None), ('fhmm-seq', 2)],
              'decision': ['all']}
    decision_makers = ['log-regression', 'svm', 'decision-tree', 'random-forest', 'zero', 'max']

    datasets = []
    print('selecting features ...')
    start = timeit.default_timer()
    for feature_set in feature_sets:
        features = _explode_features(feature_set)
        curr_dataset = dataset.dataset_from_feature_names(features)
        datasets.append(curr_dataset)
    dataset = None  # ensure that dataset is not usable hereinafter
    assert len(datasets) == len(feature_sets)
    print('done, took %fs' % (timeit.default_timer() - start))
    print('')

    # Save state
    output_dir = args.output_dir

    # Stats
    combinations = []
    total_accuracies = []
    precisions_mean = []
    precisions_std = []
    precisions_min = []
    precisions_max = []
    recalls_mean = []
    recalls_std = []
    recalls_min = []
    recalls_max = []
    fscores_mean = []
    fscores_std = []
    fscores_max = []
    fscores_min = []
    pos_ll_means = []
    pos_ll_stds = []
    neg_ll_means = []
    neg_ll_stds = []

    keys = values.keys()
    iterable_combinations = list(itertools.product(*values.values()))
    curr_step = 0
    for idx, combination in enumerate(iterable_combinations):
        print('(%.3d/%.3d) evaluating combination %s + decision makers ...' % (idx + 1, len(iterable_combinations), str(combination)))
        start = timeit.default_timer()

        curr_dataset = datasets[combination[keys.index('features')]]
        topology, n_states = combination[keys.index('hyperparams')]
        transition_init, emission_init, covar_type = combination[keys.index('init')]
        model, n_chains = combination[keys.index('model')]

        if output_dir is not None:
            curr_path = os.path.join(output_dir, '%.3d' % curr_step)
            os.mkdir(curr_path)
            args.output_dir = curr_path

        # Configure HMMs
        args.topology = topology
        args.n_states = n_states
        args.transition_init = transition_init
        args.emission_init = emission_init
        args.covar_type = covar_type
        args.model = model
        args.n_chains = n_chains
        args.decision_maker = None

        train_ll, train_y, test_ll, test_y = _evaluate_model(curr_dataset, iterator, args, print_results=False)
        assert len(train_ll) == len(train_y)
        assert len(train_ll) == len(test_ll)
        assert len(train_ll) == len(test_y)
        train_ll_combined = np.vstack(train_ll)
        test_ll_combined = np.vstack(test_ll)
        test_y_combined = np.vstack(test_y)
        assert train_ll_combined.shape == train_ll_combined.shape
        assert test_ll_combined.shape == test_y_combined.shape
        assert train_ll_combined.shape[0] > test_ll_combined.shape[0]  # just a sanity check so that the both are not confused

        n_samples, n_labels = test_ll_combined.shape
        curr_pos_ll_means = []
        curr_pos_ll_stds = []
        curr_neg_ll_means = []
        curr_neg_ll_stds = []
        for label_idx in xrange(n_labels):
            label_y = test_y_combined[:, label_idx]
            pos_indexes = np.where(label_y == 1)[0]
            neg_indexes = np.where(label_y == 0)[0]
            pos_ll = test_ll_combined[pos_indexes, label_idx]
            neg_ll = test_ll_combined[neg_indexes, label_idx]
            assert np.size(pos_ll) + np.size(neg_ll) == n_samples
            curr_pos_ll_means.append(np.mean(pos_ll))
            curr_pos_ll_stds.append(np.std(pos_ll))
            curr_neg_ll_means.append(np.mean(neg_ll))
            curr_neg_ll_stds.append(np.std(neg_ll))

        for name in decision_makers:
            args.decision_maker = name
            modified_combination = list(combination)
            modified_combination[keys.index('decision')] = name
            if name == 'svm':
                args.decision_maker_C = 1e-2
                args.decision_maker_penalty = 'l1'
            elif name == 'log-regression':
                args.decision_maker_C = 1e-3
                args.decision_maker_penalty = 'l1'
            elif name == 'decision-tree':
                args.decision_maker_criterion = 'entropy'
                args.decision_maker_max_depth = 15
            elif name == 'random-forest':
                args.decision_maker_criterion = 'entropy'
                args.decision_maker_n_estimators = 40
                args.decision_maker_max_depth = 15

            curr_preds = []
            for curr_train_ll, curr_train_y, curr_test_ll in zip(train_ll, train_y, test_ll):
                print('training decision maker')
                assert curr_train_ll.shape == curr_train_y.shape
                assert curr_train_ll.shape[0] > curr_test_ll.shape[0]
                decision_maker = decision.decision_maker_from_args(args)
                assert decision_maker is not None

                # Fit and predict using the decision maker
                if hasattr(decision_maker, 'fit') and callable(decision_maker.fit):
                    decision_maker.fit(curr_train_ll, curr_train_y)
                curr_preds.append(decision_maker.predict(curr_test_ll))
            print('')
            test_pred_combined = np.vstack(curr_preds)
            assert test_y_combined.shape == test_pred_combined.shape

            # Track everything
            combinations.append(modified_combination)
            total_accuracies.append(sk_metrics.accuracy_score(test_y_combined, test_pred_combined))
            precision, recall, fscore, _ = sk_metrics.precision_recall_fscore_support(test_y_combined, test_pred_combined)

            precisions_mean.append(np.mean(precision))
            precisions_std.append(np.std(precision))
            precisions_min.append(np.min(precision))
            precisions_max.append(np.max(precision))

            recalls_mean.append(np.mean(recall))
            recalls_std.append(np.std(recall))
            recalls_min.append(np.min(recall))
            recalls_max.append(np.max(recall))

            fscores_mean.append(np.mean(fscore))
            fscores_std.append(np.std(fscore))
            fscores_min.append(np.min(fscore))
            fscores_max.append(np.max(fscore))

            pos_ll_means.append(np.array(np.median(curr_pos_ll_means)))
            pos_ll_stds.append(np.array(np.median(curr_pos_ll_stds)))
            neg_ll_means.append(np.array(np.median(curr_neg_ll_means)))
            neg_ll_stds.append(np.array(np.median(curr_neg_ll_stds)))

            curr_step += 1
        print('done, took %fs' % (timeit.default_timer() - start))
        print('')
    assert len(combinations) == len(fscores_mean)
    assert len(combinations) == len(fscores_std)
    assert len(combinations) == len(fscores_min)
    assert len(combinations) == len(fscores_max)

    assert len(combinations) == len(precisions_mean)
    assert len(combinations) == len(precisions_std)
    assert len(combinations) == len(precisions_min)
    assert len(combinations) == len(precisions_max)

    assert len(combinations) == len(recalls_mean)
    assert len(combinations) == len(recalls_std)
    assert len(combinations) == len(recalls_min)
    assert len(combinations) == len(recalls_max)

    assert len(combinations) == len(pos_ll_means)
    assert len(combinations) == len(pos_ll_stds)
    assert len(combinations) == len(neg_ll_means)
    assert len(combinations) == len(neg_ll_stds)

    assert len(combinations) == len(total_accuracies)

    # Save results
    if output_dir is not None:
        filename = 'results.csv'
        with open(os.path.join(output_dir, filename), 'wb') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(['idx', 'combination', 'f1-score-mean', 'f1-score-std', 'f1-score-min', 'f1-score-max',
                             'precision-mean', 'precision-std', 'precision-min', 'precision-max',
                             'recall-mean', 'recall-std', 'recall-min', 'recall-max', 'total-accuracy',
                             'pos-ll-mean', 'pos-ll-std', 'neg-ll-mean', 'neg-ll-std'])
            for idx, d in enumerate(zip(combinations, fscores_mean, fscores_std, fscores_min, fscores_max,
                                        precisions_mean, precisions_std, precisions_min, precisions_max,
                                        recalls_mean, recalls_std, recalls_min, recalls_max,
                                        total_accuracies, pos_ll_means, pos_ll_stds, neg_ll_means, neg_ll_stds)):
                combination = ', '.join([str(x) for x in d[0]])
                new_data = ['%d' % idx] + list((combination, ) + d[1:])
                writer.writerow(new_data)
    print len(combinations)


def evaluate_model(dataset, iterator, args):
    # Select features
    if args.features is not None and args.features != dataset.feature_names:
        print('selecting features ...')
        features = _explode_features(args.features)
        start = timeit.default_timer()
        dataset = dataset.dataset_from_feature_names(features)
        print('done, took %fs' % (timeit.default_timer() - start))
    print('')

    _evaluate_model(dataset, iterator, args, print_results=True)


def _evaluate_model(dataset, iterator, args, print_results=False):
    loglikelihood_method = args.loglikelihood_method

    # Collect stats
    train_loglikelihoods = []
    train_predictions = []
    train_labels = []
    test_loglikelihoods = []
    test_predictions = []
    test_labels = []
    for rnd, (train_indexes, test_indexes) in enumerate(iterator):
        assert len(set(train_indexes).intersection(set(test_indexes))) == 0
        transformers = data.transformers_from_args(args)
        train, test = dataset.split_train_test(train_indexes, test_indexes, transformers)
        assert train.n_samples == len(train_indexes)
        assert test.n_samples == len(test_indexes)
        train_labels.append(train.y)
        test_labels.append(test.y)
        classifier = get_classifier(args)

        if print_results:
            print('evaluation round %d' % (rnd + 1))
            print('  train split: %s' % train_indexes)
            print('  test split:  %s' % test_indexes)
            print('  training classifier on training samples ...')
        start = timeit.default_timer()
        classifier.fit(train.X, train.y)
        stop = timeit.default_timer()
        if args.output_dir is not None:
            name = 'rnd%d_model.pkl' % (rnd+1)
            with open(os.path.join(args.output_dir, name), 'wb') as f:
                pickle.dump(classifier, f)
        if print_results:
            print('  done, took %fs' % (stop - start))

        if print_results:
            print('  computing %s loglikelihoods on train dataset ...' % loglikelihood_method)
        start = timeit.default_timer()
        train_ll = classifier.loglikelihoods(train.X, method=loglikelihood_method)
        train_loglikelihoods.append(train_ll)
        stop = timeit.default_timer()
        if print_results:
            print('  done, took %fs' % (stop - start))

        if print_results:
            print('  computing %s loglikelihoods on test dataset ...' % loglikelihood_method)
        start = timeit.default_timer()
        test_ll = classifier.loglikelihoods(test.X, method=loglikelihood_method)
        test_loglikelihoods.append(test_ll)
        stop = timeit.default_timer()
        if print_results:
            print('  done, took %fs' % (stop - start))

        decision_maker = decision.decision_maker_from_args(args)
        train_pred, test_pred = None, None
        if decision_maker is not None:
            if hasattr(decision_maker, 'fit') and callable(decision_maker.fit):
                if print_results:
                    print('  training decision maker %s on train loglikelihoods ...' % args.decision_maker)
                start = timeit.default_timer()
                decision_maker.fit(train_ll, train.y)
                stop = timeit.default_timer()
                if print_results:
                    print('  done, took %fs' % (stop - start))

            if print_results:
                print('  predicting labels on train dataset ...')
            start = timeit.default_timer()
            train_pred = decision_maker.predict(train_ll)
            train_predictions.append(train_pred)
            stop = timeit.default_timer()
            if print_results:
                print('  done, took %fs' % (stop - start))

            if print_results:
                print('  predicting labels on test dataset ...')
            start = timeit.default_timer()
            test_pred = decision_maker.predict(test_ll)
            test_predictions.append(test_pred)
            stop = timeit.default_timer()
            if print_results:
                print('  done, took %fs' % (stop - start))
        if print_results:
            print('')

        # Save round results
        if args.output_dir is not None:
            save_results(args.output_dir, train.y, train_pred, train_ll, prefix='rnd%d_train' % (rnd+1))
            save_results(args.output_dir, test.y, test_pred, test_ll, prefix='rnd%d_test' % (rnd+1))

    # Combine and save combined results
    train_y_combined = np.vstack(train_labels)
    train_ll_combined = np.vstack(train_loglikelihoods)
    train_pred_combined = np.vstack(train_predictions) if len(train_predictions) > 0 else None
    test_ll_combined = np.vstack(test_loglikelihoods)
    test_y_combined = np.vstack(test_labels)
    test_pred_combined = np.vstack(test_predictions) if len(test_predictions) > 0 else None
    if args.output_dir is not None:
        save_results(args.output_dir, train_y_combined, train_pred_combined, train_ll_combined, 'combined_train')
        save_results(args.output_dir, test_y_combined, test_pred_combined, test_ll_combined, 'combined_test')

    if print_results:
        # Print report
        label_names = dataset.unique_labels
        print('*** train dataset summary ***')
        print('')
        print(metrics.multilabel_loglikelihood_summary_report(train_y_combined, train_ll_combined, target_names=label_names))
        print('')
        if train_pred_combined is not None:
            print(metrics.multilabel_classification_report(train_y_combined, train_pred_combined, target_names=label_names))
            print('total accuracy: %.3f' % sk_metrics.accuracy_score(train_y_combined, train_pred_combined))
            print('')

        print('')
        print('*** test dataset summary ***')
        print('')
        print(metrics.multilabel_loglikelihood_summary_report(test_y_combined, test_ll_combined, target_names=label_names))
        print('')
        if test_pred_combined is not None:
            print(metrics.multilabel_classification_report(test_y_combined, test_pred_combined, target_names=label_names))
            print('total accuracy: %.3f' % sk_metrics.accuracy_score(test_y_combined, test_pred_combined))
            print('')

    return train_loglikelihoods, train_labels, test_loglikelihoods, test_labels


def _compute_measure(stats, curr_dataset, args):
    pos_ll_means = stats['pos_ll_means']
    pos_ll_stds = stats['pos_ll_stds']
    neg_ll_means = stats['neg_ll_means']
    neg_ll_stds = stats['neg_ll_stds']
    distance_means = stats['distance_means']
    k = curr_dataset.n_features
    n = curr_dataset.n_samples

    measure = None
    if args.measure == 'aicc':
        pos_median = np.median(pos_ll_means)
        aic = 2. * float(k) - 2. * pos_median
        measure = aic + float(2 * k * (k + 1)) / float(n - k - 1)
    elif args.measure == 'wasserstein' or args.measure == 'mahalanobis':
        n_labels = len(pos_ll_means)
        pos_vars = np.square(pos_ll_stds)
        neg_vars = np.square(neg_ll_stds)
        ds = []
        for idx in xrange(n_labels):
            m1 = pos_ll_means[idx]
            m2 = neg_ll_means[idx]
            var1 = pos_vars[idx]
            var2 = neg_vars[idx]
            if args.measure == 'wasserstein':
                d = np.sqrt(abs(m1 - m2) + (var1 + var2 - 2 * np.sqrt(var1 * var2)))
            else:
                d = np.square(m1 - m2) / (var1 + var2)
            ds.append(d)
        measure = np.median(ds) / float(k)
    elif args.measure == 'hmm-distance':
        lls = stats['combined_lls']
        ys = stats['combined_ys']
        f1s = []
        for idx in xrange(ys.shape[1]):
            curr_y = ys[:, idx]
            pos_lls = lls[:, idx][curr_y == 1]
            neg_lls = lls[:, idx][curr_y == 0]
            tp = np.sum(pos_lls >= 0.)
            fn = np.size(pos_lls) - tp
            tn = np.sum(neg_lls < 0.)
            fp = np.size(neg_lls) - tn
            precision = float(tp) / float(tp + fp) if tp + fp > 0 else 0.
            recall = float(tp) / float(tp + fn) if tp + fn > 0 else 0.
            f1 = 2. * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.
            f1s.append(f1)
        f1 = np.median(np.array(f1s))

        pos_mean = np.median(pos_ll_means)
        pos_std = np.median(pos_ll_stds)
        distance_median = np.median(distance_means[distance_means != 0.])  # ignore diagonal elements, which are always zero
        measure = (pos_mean - pos_std + distance_median) / float(k)
        measure *= np.square(f1)
    assert measure is not None
    return measure


def _select_best_measure_index(curr_measures, args):
    idx = None
    try:
        if args.measure == 'aicc':
            # The best score for AICc is the minimum.
            idx = np.nanargmin(curr_measures)
        elif args.measure in ['hmm-distance', 'wasserstein', 'mahalanobis']:
            # The best score for the l-d measure is the maximum.
            idx = np.nanargmax(curr_measures)
    except:
        idx = random.choice(range(len(curr_measures)))
    assert idx is not None
    return idx


def _explode_features(features):
    exploded_features = []
    for name in features:
        new_names = [name]
        if name == 'extremity_pos':
            new_names = ['left_hand_pos', 'right_hand_pos', 'left_foot_pos', 'right_foot_pos']
        elif name == 'extremity_vel':
            new_names = ['left_hand_vel', 'right_hand_vel', 'left_foot_vel', 'right_foot_vel']
        elif name == 'extremity_acc':
            new_names = ['left_hand_acc', 'right_hand_acc', 'left_foot_acc', 'right_foot_acc']
        elif name == 'extremity_vel_norm':
            new_names = ['left_hand_vel_norm', 'right_hand_vel_norm', 'left_foot_vel_norm', 'right_foot_vel_norm']
        elif name == 'extremity_acc_norm':
            new_names = ['left_hand_acc_norm', 'right_hand_acc_norm', 'left_foot_acc_norm', 'right_foot_acc_norm']
        elif name == 'normalized_extremity_pos':
            new_names = ['normalized_left_hand_pos', 'normalized_right_hand_pos', 'normalized_left_foot_pos', 'normalized_right_foot_pos']
        elif name == 'normalized_extremity_vel':
            new_names = ['normalized_left_hand_vel', 'normalized_right_hand_vel', 'normalized_left_foot_vel', 'normalized_right_foot_vel']
        elif name == 'normalized_extremity_acc':
            new_names = ['normalized_left_hand_acc', 'normalized_right_hand_acc', 'normalized_left_foot_acc', 'normalized_right_foot_acc']
        elif name == 'normalized_extremity_vel_norm':
            new_names = ['normalized_left_hand_vel_norm', 'normalized_right_hand_vel_norm', 'normalized_left_foot_vel_norm', 'normalized_right_foot_vel_norm']
        elif name == 'normalized_extremity_acc_norm':
            new_names = ['normalized_left_hand_acc_norm', 'normalized_right_hand_acc_norm', 'normalized_left_foot_acc_norm', 'normalized_right_foot_acc_norm']
        exploded_features.extend(new_names)
    return exploded_features


def evaluate_features(dataset, iterator, args):
    features = args.features
    measures = []
    feature_sets = []
    deleted_features = []
    n_features = []
    round_count = 0
    curr_step = 0
    prefix_fmt = '%.3d'

    total_steps = 1 + np.sum(range(2, len(features) + 1))
    print('evaluating %d features. %d iterations will be performed ...' % (len(features), total_steps))
    print('')

    # Start by training on all features
    curr_step += 1
    print('(%.3d/%.3d) all features: %s' % (curr_step, total_steps, ', '.join(features)))
    curr_dataset = dataset.dataset_from_feature_names(_explode_features(features))
    prefix = prefix_fmt % round_count
    try:
        ll_stats = _compute_averaged_pos_and_neg_lls(curr_dataset, iterator, prefix, args)
        measure = _compute_measure(ll_stats, curr_dataset, args)
    except:
        measure = np.nan
    _save_feature_results([measure], [curr_dataset.n_features], [features], ['-'], prefix, args)
    measures.append(measure)
    feature_sets.append(features[:])
    deleted_features.append('-')
    n_features.append(curr_dataset.n_features)
    print('measure with all features: %f' % measure)
    print('')

    # Start reducing features one by one
    while len(features) > 1:
        round_count += 1
        print('remaining features: %s' % (', '.join(features)))
        curr_measures = []
        curr_feature_sets = []
        curr_deleted_features = []
        curr_n_features = []
        prefix = prefix_fmt % round_count
        for feature_idx in xrange(len(features)):
            curr_step += 1
            start = timeit.default_timer()
            print('  (%.3d/%.3d) training classifier and computing stats WITHOUT %s ...' % (curr_step, total_steps, features[feature_idx]))
            curr_features = features[:]
            curr_deleted_features.append(curr_features[feature_idx])
            del curr_features[feature_idx]
            curr_feature_sets.append(curr_features)
            curr_dataset = dataset.dataset_from_feature_names(_explode_features(curr_features))
            sub_prefix = '%s_%.3d' % (prefix, feature_idx)
            try:
                ll_stats = _compute_averaged_pos_and_neg_lls(curr_dataset, iterator, sub_prefix, args)
                measure = _compute_measure(ll_stats, curr_dataset, args)
            except:
                measure = np.nan
            if np.isnan(measure):
                print('            measure: not computable')
            else:
                print('            measure: %f' % measure)
            print('            done, took %fs' % (timeit.default_timer() - start))
            curr_measures.append(measure)
            curr_n_features.append(curr_dataset.n_features)
        assert len(curr_measures) == len(features)
        assert len(curr_measures) == len(curr_feature_sets)
        assert len(curr_measures) == len(curr_deleted_features)
        assert len(curr_n_features) == len(curr_n_features)

        _save_feature_results(curr_measures, curr_n_features, curr_feature_sets, curr_deleted_features, prefix, args)

        # Select the *best* measure index and remove the corresponding feature. This seems counter-intuitive at first,
        # but makes sense: we left this feature out and the score was still the best compared to leaving any other
        # feature out -> the feature is the least important under the score.
        feature_idx = _select_best_measure_index(curr_measures, args)
        measure = curr_measures[feature_idx]
        deleted_feature = features[feature_idx]
        del features[feature_idx]
        measures.append(measure)
        feature_sets.append(features[:])
        deleted_features.append(deleted_feature)
        n_features.append(curr_n_features[feature_idx])
        print('deleted feature %s, new measure %f' % (deleted_feature, measure))
        print('')

    # Save final results
    prefix = '_final_'
    _save_feature_results(measures, n_features, feature_sets, deleted_features, prefix, args)

    best_idx = _select_best_measure_index(measures, args)
    print('best feature set with measure %f:' % measures[best_idx])
    print(', '.join(feature_sets[best_idx]))
    print('')
    print('detailed reports have been saved to the output directory')
    print('')


def _save_feature_results(measures, n_features, feature_sets, deleted_features, prefix, args):
    if args.output_dir is None:
        return
    assert len(measures) == len(feature_sets)
    assert len(measures) == len(deleted_features)

    filename = '%s_results.csv' % prefix
    with open(os.path.join(args.output_dir, filename), 'wb') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(['', 'idx', 'measure', 'n_features', 'deleted_feature', 'feature_set'])
        best_idx = _select_best_measure_index(measures, args)
        for idx, (measure, n_feature, feature_set, deleted_feature) in enumerate(zip(measures, n_features, feature_sets, deleted_features)):
            selected = '*' if best_idx == idx else ''
            writer.writerow([selected, '%d' % idx, '%f' % measure, n_feature, deleted_feature, ', '.join(feature_set)])


def _compute_averaged_pos_and_neg_lls(dataset, iterator, prefix, args, save_model=False, compute_distances=False):
    combined_lls = []
    combined_ys = []
    rounds = list(iterator)
    distances = []

    if args.measure == 'hmm-distance':
        compute_distances = True

    for rnd, (train_indexes, test_indexes) in enumerate(rounds):
        transformers = data.transformers_from_args(args)
        train, test = dataset.split_train_test(train_indexes, test_indexes, transformers)

        # Train classifier and save model
        classifier = get_classifier(args)
        classifier.fit(train.X, train.y)
        if save_model and args.output_dir is not None:
            filename = '%s_rnd%d_model.pkl' % (prefix, rnd + 1)
            with open(os.path.join(args.output_dir, filename), 'wb') as f:
                pickle.dump(classifier, f)

        # Calculate distances. I have no idea why, but having n_jobs > 1 is causing a deadlock. Again, I have no clue
        # how/why this could happen, hence this workaround.
        if compute_distances:
            old_jobs = args.n_jobs
            args.n_jobs = 1
            distances.append(classifier.distances(200, loglikelihood_method=args.loglikelihood_method))
            args.n_jobs = old_jobs
        else:
            d = np.zeros((dataset.n_labels, dataset.n_labels))
            distances.append(d)

        # Calculate likelihoods UNDER THE TEST SET (!!!).
        test_lls = classifier.loglikelihoods(test.X, method=args.loglikelihood_method)
        combined_lls.append(test_lls)
        combined_ys.append(test.y)

    combined_lls = np.vstack(combined_lls)
    combined_ys = np.vstack(combined_ys)
    assert combined_lls.shape == combined_ys.shape
    n_samples, n_labels = combined_lls.shape

    pos_ll_means = []
    pos_ll_stds = []
    neg_ll_means = []
    neg_ll_stds = []
    for label_idx in xrange(n_labels):
        label_lls = combined_lls[:, label_idx]
        curr_y = combined_ys[:, label_idx]
        pos_label_lls = label_lls[curr_y == 1]
        neg_label_lls = label_lls[curr_y == 0]
        assert np.size(pos_label_lls) + np.size(neg_label_lls) == n_samples
        pos_ll_means.append(np.mean(pos_label_lls))
        pos_ll_stds.append(np.std(pos_label_lls))
        neg_ll_means.append(np.mean(neg_label_lls))
        neg_ll_stds.append(np.std(neg_label_lls))
    pos_ll_means = np.array(pos_ll_means)
    pos_ll_stds = np.array(pos_ll_stds)
    neg_ll_means = np.array(neg_ll_means)
    neg_ll_stds = np.array(neg_ll_stds)
    assert pos_ll_means.shape == neg_ll_means.shape
    assert pos_ll_stds.shape == neg_ll_stds.shape
    assert pos_ll_means.shape == pos_ll_stds.shape
    assert pos_ll_means.shape == (dataset.n_labels,)

    # Calculate averaged distances
    averaged_distances = np.mean(np.array(distances), axis=0)
    assert averaged_distances.shape == (dataset.n_labels, dataset.n_labels)

    # Save likelihoods and distances
    if args.output_dir is not None:
        save_results(args.output_dir, combined_ys, None, combined_lls, prefix)
        if compute_distances:
            np.savetxt(os.path.join(args.output_dir, '%s_distances.csv' % prefix), averaged_distances, delimiter=';', fmt='%f')

    return {'pos_ll_means': pos_ll_means,
            'pos_ll_stds': pos_ll_stds,
            'neg_ll_means': neg_ll_means,
            'neg_ll_stds': neg_ll_stds,
            'distance_means': averaged_distances,
            'combined_lls': combined_lls,
            'combined_ys': combined_ys}


def evaluate_hyperparameters(dataset, iterator, args):
    # Select features
    if args.features is not None and args.features != dataset.feature_names:
        print('selecting features ...')
        features = _explode_features(args.features)
        start = timeit.default_timer()
        dataset = dataset.dataset_from_feature_names(features)
        print('done, took %fs' % (timeit.default_timer() - start))
    print('')

    states = range(3, 22 + 1)  # = [3,...,22]
    topologies = ['full', 'left-to-right-full', 'left-to-right-1', 'left-to-right-2']
    n_combinations = len(states) * len(topologies)

    curr_step = 0
    combinations = []
    measures = []
    for state in states:
        for topology in topologies:
            curr_step += 1
            prefix = '%.3d_%d_%s' % (curr_step, state, topology)
            print('(%.3d/%.3d) evaluating state=%d and topology=%s ...' % (curr_step, n_combinations, state, topology))
            start = timeit.default_timer()
            try:
                # Configure args from which the HMMs are created
                args.n_states = state
                args.topology = topology

                ll_stats = _compute_averaged_pos_and_neg_lls(dataset, iterator, prefix, args)
                measure = _compute_measure(ll_stats, dataset, args)
            except:
                measure = np.nan
            if measure is np.isnan(measure):
                print('measure: not computable')
            else:
                print('measure: %f' % measure)
            combinations.append((str(state), topology))
            measures.append(measure)
            print('done, took %fs' % (timeit.default_timer() - start))
            print('')

    best_idx = np.nanargmax(np.array(measures))  # get the argmax ignoring NaNs
    print('best combination with score %f: %s' % (measures[best_idx], ', '.join(combinations[best_idx])))
    print('detailed reports have been saved')

    # Save results
    assert len(combinations) == len(measures)
    if args.output_dir is not None:
        filename = '_results.csv'
        with open(os.path.join(args.output_dir, filename), 'wb') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(['', 'idx', 'measure', 'combination'])
            for idx, (measure, combination) in enumerate(zip(measures, combinations)):
                selected = '*' if best_idx == idx else ''
                writer.writerow([selected, '%d' % idx, '%f' % measure, ', '.join(combination)])


def evaluate_pca(dataset, iterator, args):
    # Select features
    if args.features is not None and args.features != dataset.feature_names:
        print('selecting features ...')
        features = _explode_features(args.features)
        start = timeit.default_timer()
        dataset = dataset.dataset_from_feature_names(features)
        print('done, took %fs' % (timeit.default_timer() - start))
    print('')

    pca_components = range(1, dataset.n_features)
    total_steps = len(pca_components)
    if 'pca' not in args.transformers:
        args.transformers.append('pca')

    curr_step = 0
    measures = []
    for n_components in pca_components:
        curr_step += 1
        prefix = '%.3d' % curr_step
        print('(%.3d/%.3d) evaluating with %d pca components ...' % (curr_step, total_steps, n_components))
        start = timeit.default_timer()
        try:
            args.pca_components = n_components

            ll_stats = _compute_averaged_pos_and_neg_lls(dataset, iterator, prefix, args)
            measure = _compute_measure(ll_stats, dataset, args)
        except:
            measure = np.nan
        if measure is np.isnan(measure):
            print('measure: not computable')
        else:
            print('measure: %f' % measure)

            # Correct score. The problem is that it is computed given the dataset, which has too many features.
            measure = (measure * float(dataset.n_features)) / float(n_components)
        measures.append(measure)
        print('done, took %fs' % (timeit.default_timer() - start))
        print('')
    assert len(pca_components) == len(measures)

    best_idx = np.nanargmax(np.array(measures))  # get the argmax ignoring NaNs
    print('best result with score %f: %d PCA components' % (measures[best_idx], pca_components[best_idx]))
    print('detailed reports have been saved')

    # Save results
    if args.output_dir is not None:
        filename = '_results.csv'
        with open(os.path.join(args.output_dir, filename), 'wb') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(['', 'idx', 'measure', 'components'])
            for idx, (measure, n_components) in enumerate(zip(measures, pca_components)):
                selected = '*' if best_idx == idx else ''
                writer.writerow([selected, '%d' % idx, '%f' % measure, '%d' % n_components])


def evaluate_fhmms(dataset, iterator, args):
    # Select features
    if args.features is not None and args.features != dataset.feature_names:
        print('selecting features ...')
        features = _explode_features(args.features)
        start = timeit.default_timer()
        dataset = dataset.dataset_from_feature_names(features)
        print('done, took %fs' % (timeit.default_timer() - start))
    print('')

    chains = [1, 2, 3, 4]
    total_steps = len(chains)

    curr_step = 0
    measures = []
    for chain in chains:
        curr_step += 1
        prefix = '%.3d_%d-chains' % (curr_step, chain)
        print('(%.3d/%.3d) evaluating n_chains=%d ...' % (curr_step, total_steps, chain))
        start = timeit.default_timer()
        old_loglikelihood_method = args.loglikelihood_method
        try:
            # Configure args from which the HMMs are created
            args.n_chains = chain
            if chain == 1:
                args.model = 'hmm'
                args.loglikelihood_method = 'exact'  # there's no approx loglikelihood method for HMMs
            else:
                args.model = 'fhmm-seq'

            ll_stats = _compute_averaged_pos_and_neg_lls(dataset, iterator, prefix, args, save_model=True, compute_distances=False)
            measure = _compute_measure(ll_stats, dataset, args)
        except:
            measure = np.nan
        args.loglikelihood_method = old_loglikelihood_method
        if measure is np.isnan(measure):
            print('measure: not computable')
        else:
            print('measure: %f' % measure)
        measures.append(measure)
        print('done, took %fs' % (timeit.default_timer() - start))
        print('')

    best_idx = np.nanargmax(np.array(measures))  # get the argmax ignoring NaNs
    print('best model with score %f: %d chains' % (measures[best_idx], chains[best_idx]))
    print('detailed reports have been saved')

    # Save results
    assert len(chains) == len(measures)
    if args.output_dir is not None:
        filename = '_results.csv'
        with open(os.path.join(args.output_dir, filename), 'wb') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(['', 'idx', 'measure', 'chains'])
            for idx, (measure, chain) in enumerate(zip(measures, chains)):
                selected = '*' if best_idx == idx else ''
                writer.writerow([selected, '%d' % idx, '%f' % measure, '%d' % chain])


def evaluate_initialization(dataset, iterator, args):
    # Select features
    if args.features is not None and args.features != dataset.feature_names:
        print('selecting features ...')
        features = _explode_features(args.features)
        start = timeit.default_timer()
        dataset = dataset.dataset_from_feature_names(features)
        print('done, took %fs' % (timeit.default_timer() - start))
    print('')

    transition_inits = ['random', 'uniform']
    emission_inits = ['random', 'k-means']
    covar_types = ['full', 'diag']

    n_combinations = len(transition_inits) * len(emission_inits) * len(covar_types)

    curr_step = 0
    combinations = []
    measures = []
    for transition_init in transition_inits:
        for emission_init in emission_inits:
            for covar_type in covar_types:
                curr_step += 1
                prefix = '%.3d_%s_%s_%s' % (curr_step, transition_init, emission_init, covar_type)
                print('(%.3d/%.3d) evaluating transition_init=%s, emission_init=%s and covar_type=%s ...' % (curr_step, n_combinations, transition_init, emission_init, covar_type))
                start = timeit.default_timer()
                try:
                    # Configure args from which the HMMs are created
                    args.transition_init = transition_init
                    args.emission_init = emission_init
                    args.covar_type = covar_type

                    ll_stats = _compute_averaged_pos_and_neg_lls(dataset, iterator, prefix, args)
                    measure = _compute_measure(ll_stats, dataset, args)
                except:
                    measure = np.nan
                if measure is np.isnan(measure):
                    print('measure: not computable')
                else:
                    print('measure: %f' % measure)
                combinations.append((transition_init, emission_init, covar_type))
                measures.append(measure)
                print('done, took %fs' % (timeit.default_timer() - start))
                print('')

    best_idx = np.nanargmax(np.array(measures))  # get the argmax ignoring NaNs
    print('best combination with score %f: %s' % (measures[best_idx], ', '.join(combinations[best_idx])))
    print('detailed reports have been saved')

    # Save results
    assert len(combinations) == len(measures)
    if args.output_dir is not None:
        filename = '_results.csv'
        with open(os.path.join(args.output_dir, filename), 'wb') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(['', 'idx', 'measure', 'combination'])
            for idx, (measure, combination) in enumerate(zip(measures, combinations)):
                selected = '*' if best_idx == idx else ''
                writer.writerow([selected, '%d' % idx, '%f' % measure, ', '.join(combination)])


def main(args):
    start_total = timeit.default_timer()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    # Validate that paths exist so that we don't need to check that whenever we use it
    if not os.path.exists(args.dataset):
        exit('data set at path "%s" does not exist' % args.dataset)
    if args.output_dir is not None and not os.path.isdir(args.output_dir):
        exit('--output-dir "%s" is not a directory' % args.output_dir)
    if 'fhmm' not in args.model:
        args.n_chains = None

    # Print command again to make it easier to re-produce later from the logs
    print('python ' + ' '.join(sys.argv))
    print('')

    print('args:')
    print(args)
    print('')

    # Load dataset
    print('loading data set "%s" ...' % args.dataset)
    start = timeit.default_timer()
    with open(args.dataset, 'rb') as f:
        dataset = pickle.load(f)
        if type(dataset) != data.Dataset:
            raise ValueError('invalid dataset')
    print('done, took %fs' % (timeit.default_timer() - start))
    if args.transform_to_multiclass:
        print('transforming to multi-class dataset ...')
        dataset = dataset.multiclass_dataset()
        print('done')
    print('')

    # Save labels
    if args.output_dir is not None:
        with open(os.path.join(args.output_dir, 'dataset_label_names.csv'), 'w') as f:
            if f is None:
                exit('cannot write to output dir')
            writer = csv.DictWriter(f, fieldnames=dataset.unique_labels)
            writer.writeheader()

    # Shuffle dataset
    if not args.disable_shuffle:
        permutation = None
        if args.permutation is not None:
            with open(args.permutation, 'rb') as f:
                permutation = np.load(f)
        permutation = dataset.shuffle(permutation)
        print('shuffled dataset with permutation:')
        print(permutation)
        print('')
        if args.output_dir is not None:
            np.save(os.path.join(args.output_dir, 'dataset_permutation.npy'), permutation)

    # Print overview
    print('dataset overview:')
    print('  samples:  %d' % dataset.n_samples)
    print('  labels:   %s' % ', '.join(dataset.unique_labels))
    print('  features: %s' % ', '.join(dataset.feature_names))
    print('')

    # Start evaluation
    iterator = data.StratifiedMultilabelKFold(dataset.y, n_folds=args.n_iter)
    if args.action == 'model':
        evaluate_model(dataset, iterator, args)
    elif args.action == 'features':
        evaluate_features(dataset, iterator, args)
    elif args.action == 'hmm-hyperparameters':
        evaluate_hyperparameters(dataset, iterator, args)
    elif args.action == 'hmm-initialization':
        evaluate_initialization(dataset, iterator, args)
    elif args.action =='fhmm':
        evaluate_fhmms(dataset, iterator, args)
    elif args.action == 'pca':
        evaluate_pca(dataset, iterator, args)
    elif args.action == 'end-to-end':
        evaluate_end_to_end(dataset, iterator, args)
    print('total time: %fs' % (timeit.default_timer() - start_total))


if __name__ == '__main__':
    main(get_parser().parse_args())
