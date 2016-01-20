from argparse import ArgumentParser
import timeit
import logging
import os
import sys
import itertools
import csv

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

import toolkit.decision as decision
import toolkit.dataset.base as data


def get_parser():
    parser = ArgumentParser()
    decision.add_decision_maker_parser_arguments(parser)
    data.add_transformer_parser_arguments(parser)

    # Dataset options
    parser.add_argument('classifier', choices=['log-regression', 'svm', 'decision-tree', 'random-forest'])
    parser.add_argument('dataset', type=str)

    # Evaluation options
    parser.add_argument('--output', type=str)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--n-jobs', type=int, default=1)
    parser.add_argument('--measure', choices=['f1', 'accuracy'], default='f1')
    return parser


def _compute_score(y_true, y_pred, args):
    assert y_true.shape == y_pred.shape
    score = None
    if args.measure == 'accuracy':
        score = accuracy_score(y_true, y_pred)
    elif args.measure == 'f1':
        score = f1_score(y_true, y_pred, average='macro')
    assert score is not None
    return score


def _select_best_score(scores, args):
    return np.nanargmax(np.array(scores))


def _evaluate_combinations(values, train_splits, test_splits, args):
    scores = []
    keys = values.keys()
    combinations = list(itertools.product(*values.values()))
    for combination in combinations:
        for idx, value in enumerate(combination):
            setattr(args, 'decision_maker_' + keys[idx], value)
        print('evaluating combination %s ...' % str(combination))
        start = timeit.default_timer()
        y_true = []
        y_pred = []
        for rnd, (train, test) in enumerate(zip(train_splits, test_splits)):
            decision_maker = decision.decision_maker_from_args(args)
            y_pred.append(evaluate_decision_maker(decision_maker, train, test, args))
            y_true.append(test[1])
        y_true = np.vstack(y_true)
        y_pred = np.vstack(y_pred)
        assert y_true.shape == y_pred.shape
        score = _compute_score(y_true, y_pred, args)
        scores.append(score)
        print('score: %f' % score)
        print('done, took %fs' % (timeit.default_timer() - start))
        print('')

    best_idx = _select_best_score(scores, args)
    print('best combination with score %f: %s' % (scores[best_idx], str(combinations[best_idx])))

    # Save results
    if args.output is not None:
        with open(args.output, 'wb') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(['', 'idx', 'score'] + keys)
            for idx, (score, combination) in enumerate(zip(scores, combinations)):
                selected = '*' if best_idx == idx else ''
                writer.writerow([selected, '%d' % idx, '%f' % score] + list(combination))


def evaluate_decision_tree(train_splits, test_splits, args):
    args.decision_maker = 'decision-tree'
    values = {'criterion': ['gini', 'entropy'],
              'max_depth': range(1, 41)}
    _evaluate_combinations(values, train_splits, test_splits, args)


def evaluate_random_forest(train_splits, test_splits, args):
    args.decision_maker = 'random-forest'
    values = {'criterion': ['gini', 'entropy'],
              'n_estimators': range(1, 101),
              'max_depth': [15]}
    _evaluate_combinations(values, train_splits, test_splits, args)


def evaluate_log_regression(train_splits, test_splits, args):
    args.decision_maker = 'log-regression'
    values = {'penalty': ['l1', 'l2'],
              'C': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5]}
    _evaluate_combinations(values, train_splits, test_splits, args)


def evaluate_svm(train_splits, test_splits, args):
    args.decision_maker = 'svm'
    #args.transformers = ['minmax-scaler']
    values = {'penalty': ['l1', 'l2'],
              'C': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5]}
    _evaluate_combinations(values, train_splits, test_splits, args)


def evaluate_decision_maker(decision_maker, train, test, args):
    train_X, train_y = train
    test_X, test_y = test

    transformers = data.transformers_from_args(args)
    for transformer in transformers:
        if hasattr(transformer, 'fit') and callable(transformer.fit):
            transformer.fit(train_X)
        train_X = transformer.transform(train_X)
        test_X = transformer.transform(test_X)

    if hasattr(decision_maker, 'fit') and callable(decision_maker.fit):
        decision_maker.fit(train_X, train_y)
    test_pred = decision_maker.predict(test_X)

    return test_pred


def main(args):
    start_total = timeit.default_timer()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    # Validate that paths exist so that we don't need to check that whenever we use it
    if not os.path.isdir(args.dataset):
        exit('data set at path "%s" does not exist' % args.dataset)

    # Print command again to make it easier to re-produce later from the logs
    print('python ' + ' '.join(sys.argv))
    print('')

    print(args)
    print('')

    # Load loglikelihoods
    print('loading data "%s" ...' % args.dataset)
    start = timeit.default_timer()
    rnd = 1
    train_data = []
    test_data = []
    while True:
        train_prefix = 'rnd%d_train_' % rnd
        if not os.path.exists(os.path.join(args.dataset, train_prefix + 'loglikelihoods.csv')):
            break
        train_lls = np.genfromtxt(os.path.join(args.dataset, train_prefix + 'loglikelihoods.csv'), delimiter=';',
                                  skip_header=1)
        if train_lls is None:
            exit('could not load train likelihoods')
        train_labels = np.genfromtxt(os.path.join(args.dataset, train_prefix + 'labels.csv'), delimiter=';',
                                     skip_header=1, dtype=int)
        if train_labels is None:
            exit('could not load train labels')
        if train_labels.shape != train_lls.shape:
            exit('incompatible train likelihoods and labels')
        train_data.append((train_lls, train_labels))

        test_prefix = 'rnd%d_test_' % rnd
        test_lls = np.genfromtxt(os.path.join(args.dataset, test_prefix + 'loglikelihoods.csv'), delimiter=';',
                                 skip_header=1)
        if test_lls is None:
            exit('could not load test likelihoods')
        test_labels = np.genfromtxt(os.path.join(args.dataset, test_prefix + 'labels.csv'), delimiter=';',
                                    skip_header=1, dtype=int)
        if test_labels is None:
            exit('could not load test labels')
        if test_labels.shape != test_lls.shape:
            exit('incompatible test likelihoods and labels')
        test_data.append((test_lls, test_labels))
        rnd += 1
    assert len(train_data) == len(test_data)
    if len(train_data) == 0:
        exit('did not find any data')
    print('done, took %fs' % (timeit.default_timer() - start))
    print('')

    # Print overview
    print('dataset overview:')
    print('  train: %s' % ', '.join([str(lls.shape) for (lls, _) in train_data]))
    print('  test:  %s' % ', '.join([str(lls.shape) for (lls, _) in test_data]))
    print('')

    function = None
    if args.classifier == 'log-regression':
        function = evaluate_log_regression
    elif args.classifier == 'svm':
        function = evaluate_svm
    elif args.classifier == 'decision-tree':
        function = evaluate_decision_tree
    elif args.classifier == 'random-forest':
        function = evaluate_random_forest
    assert function is not None
    function(train_data, test_data, args)

    print('total time: %fs' % (timeit.default_timer() - start_total))


if __name__ == '__main__':
    main(get_parser().parse_args())

