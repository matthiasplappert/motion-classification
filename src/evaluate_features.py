# coding=utf8
from collections import namedtuple
from argparse import ArgumentParser
import timeit
import os
import logging
from itertools import chain, combinations
import csv

import numpy as np
from sklearn.cross_validation import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

from toolkit.hmm.impl_hmmlearn import GaussianHMM
from toolkit.dataset.base import load_manifest
import toolkit.metrics as metrics
import toolkit.dataset.mmm as mmm
import toolkit.dataset.vicon as vicon


Dataset = namedtuple('Dataset', 'X y target_names groups lengths')


def get_parser():
    parser = ArgumentParser()
    parser.add_argument('dataset', help='path to the dataset')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--topology', choices=['left-to-right', 'left-to-right-cycle', 'bakis', 'full'],
                        default='left-to-right')
    parser.add_argument('--n-training-iterations', type=int, default=10)
    parser.add_argument('--n-iterations', type=int, default=10)
    parser.add_argument('--test-size', type=float, default=0.1)
    parser.add_argument('--n-states', type=int, default=10)
    parser.add_argument('--disable-cache', action='store_true')
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--preprocessing', nargs='*', choices=['scale'], default=['scale'])

    all_features = mmm.FEATURE_NAMES + vicon.FEATURE_NAMES
    parser.add_argument('--features', choices=all_features, nargs='+', default=all_features)

    return parser


def evaluate(X, args):
    enum = ShuffleSplit(len(X), n_iter=args.n_iterations, test_size=args.test_size)
    train_scores = []
    test_scores = []
    for train_index, test_index in enum:
        X_train = [X[idx] for idx in train_index]
        X_test = [X[idx] for idx in test_index]
        X_train, X_test = preprocess_datasets(X_train, X_test, args)
        model = GaussianHMM(n_states=args.n_states, n_training_iterations=args.n_training_iterations,
                            topology=args.topology)
        model.fit(X_train)
        train_scores.extend([model.loglikelihood(X_curr) for X_curr in X_train])
        test_scores.extend([model.loglikelihood(X_curr) for X_curr in X_test])

    train_scores_array = np.array(train_scores)
    train_mean = float(np.mean(train_scores_array))
    train_std = float(np.std(train_scores_array))
    test_scores_array = np.array(test_scores)
    test_mean = float(np.mean(test_scores_array))
    test_std = float(np.std(test_scores_array))
    return train_mean, train_std, test_mean, test_std


def load_dataset(path, motion_type, feature_names, args):
    print('Loading data set "%s" ...' % path)
    X, y, target_names, groups, lengths = load_manifest(path, motion_type, feature_names=feature_names,
                                                        use_cache=not args.disable_cache, normalize=args.normalize)
    assert len(X) == len(y)
    return Dataset(X, y, target_names, groups, lengths)


def feature_indexes_from_set(all_features, feature_set, lengths):
    indexes = []
    idx = 0
    for feature, length in zip(all_features, lengths):
        if feature in feature_set:
            indexes.extend(range(idx, idx + length))
        idx += length
    return indexes


def preprocess_datasets(X_train, X_test, args):
    if 'scale' in args.preprocessing:
        print('Scaling features to range [-1,1] ...')
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(np.vstack(X_train))
        X_train = [scaler.transform(X_curr) for X_curr in X_train]
        X_test = [scaler.transform(X_curr) for X_curr in X_test]
    return X_train, X_test


def main(args):
    start = timeit.default_timer()

    # Validate that paths exist so that we don't need to check that whenever we use it
    if not os.path.exists(args.dataset):
        exit('data set at path "%s" does not exist' % args.dataset)

    print('Arguments: %s\n' % args)

    # Load dataset and combine them
    all_features = args.features
    mmm_features = [feature for feature in all_features if feature in mmm.FEATURE_NAMES]
    vicon_features = [feature for feature in all_features if feature in vicon.FEATURE_NAMES]
    mmm_data = load_dataset(args.dataset, 'mmm-nlopt', mmm_features, args)
    vicon_data = load_dataset(args.dataset, 'vicon', vicon_features, args)
    assert mmm_data.y.shape == vicon_data.y.shape
    assert mmm_data.y.shape == (len(mmm_data.X), 1)  # assert that only one class is used per data set
    assert len(vicon_data.X) == len(mmm_data.X)
    lengths = mmm_data.lengths + vicon_data.lengths
    X = []
    for idx in xrange(len(mmm_data.X)):
        X.append(np.hstack((mmm_data.X[idx], vicon_data.X[idx])))
    assert len(mmm_data.X) == len(X)

    # Calculate power set of all features
    # (source: http://stackoverflow.com/questions/10342939/power-set-and-cartesian-product-of-a-set-python)
    all_features_power_set = []
    for z in chain.from_iterable(combinations(all_features, r) for r in range(len(all_features)+1)):
        if len(z) == 0:
            # Skip the empty set
            continue
        all_features_power_set.append(z)

    print('\nEvaluating %d feature combinations on %d samples ...' % (len(all_features_power_set), len(X)))
    train_means, train_stds = [], []
    test_means, test_stds = [], []
    for idx, feature_set in enumerate(all_features_power_set):
        print('(%.3d/%.3d): evaluating %s ...' % (idx+1, len(all_features_power_set), feature_set, ))
        current_set_start = timeit.default_timer()
        indexes = feature_indexes_from_set(all_features, feature_set, lengths)
        X_curr = [X_curr[:, indexes] for X_curr in X]  # right now this makes a copy of the sub array
        assert len(X_curr) == len(X)
        assert X_curr[0].shape[1] == len(indexes)
        train_mean, train_std, test_mean, test_std = evaluate(X_curr, args)
        print('train: %f +-%f' % (train_mean, train_std))
        print('test:  %f +-%f' % (test_mean, test_std))
        print('shape: %s' % str(X_curr[0].shape))
        print('done, took %fs\n' % (timeit.default_timer() - current_set_start))

        # Bookkeeping
        train_means.append(train_mean)
        train_stds.append(train_std)
        test_means.append(test_mean)
        test_stds.append(test_std)
    assert len(train_means) == len(train_stds) == len(test_means) == len(test_stds) == len(all_features_power_set)

    # Calculate best feature set and report
    best_train_idx = np.argmax(np.array(train_means))
    best_test_idx = np.argmax(np.array(test_means))
    print('Results:')
    print('total time: %fs' % (timeit.default_timer() - start))
    print('best feature set on train set with score %f: %s' % (train_means[best_train_idx], all_features_power_set[best_train_idx]))
    print('best feature set on test set with score %f: %s' % (test_means[best_test_idx], all_features_power_set[best_test_idx]))

    # Save results
    if args.output:
        print('\nSaving results to "%s" ...' % args.output)
        fieldnames = ['features', 'train_mean', 'train_std', 'test_mean', 'test_std']
        with open(args.output, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for features, train_mean, train_std, test_mean, test_std in zip(all_features_power_set, train_means,
                                                                            train_stds, test_means, test_stds):
                writer.writerow({'features': str(features),
                                 'train_mean': train_mean,
                                 'train_std': train_std,
                                 'test_mean': test_mean,
                                 'test_std': test_std})
        print('done')


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main(get_parser().parse_args())
