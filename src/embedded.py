from argparse import ArgumentParser
import timeit
import logging
import os
import pickle
import sys

import numpy as np

import toolkit.dataset.base as data
from toolkit.hmm.base import Classifier as HMMClassifier
from toolkit.hmm.impl_hmmlearn import GaussianHMM


def get_parser():
    parser = ArgumentParser()
    data.add_transformer_parser_arguments(parser)
    parser.add_argument('dataset', type=str)
    parser.add_argument('--features', type=str, nargs='*', default=None)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--n-jobs', type=int, default=1)
    return parser


def main(args):
    start_total = timeit.default_timer()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    # Validate that paths exist so that we don't need to check that whenever we use it
    if not os.path.exists(args.dataset):
        exit('data set at path "%s" does not exist' % args.dataset)

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
    if args.features is not None and args.features != dataset.feature_names:
        print('selecting features ...')
        features = args.features
        start = timeit.default_timer()
        dataset = dataset.dataset_from_feature_names(features)
        print('done, took %fs' % (timeit.default_timer() - start))
    print('')

    # Print overview
    print('dataset overview:')
    print('  samples:  %d' % dataset.n_samples)
    print('  labels:   %s' % ', '.join(dataset.unique_labels))
    print('  features: %s' % ', '.join(dataset.feature_names))
    print('')

    transformers = data.transformers_from_args(args)
    dataset = dataset.dataset_from_transformers(transformers)

    model = GaussianHMM()
    model.n_training_iterations = 10
    model.n_states = 6
    model.topology = 'left-to-right-1'
    model.verbose = args.verbose
    model.transition_init = 'uniform'
    model.emission_init = 'k-means'
    model.covar_type = 'diag'
    classifier = HMMClassifier(model, n_jobs=args.n_jobs)

    print('training classifier ...')
    start = timeit.default_timer()
    classifier.fit(dataset.X, dataset.y)
    print('done, took %fs' % (timeit.default_timer() - start))

    total_scores = np.zeros(len(dataset.feature_names))
    for idx, model in enumerate(classifier.models_):
        label_name = dataset.unique_labels[idx]
        print('important features for %s:' % label_name)
        mean_covar = np.mean(model.model_.covars_, axis=0)

        # Reduce to a single score per feature
        scores = np.zeros(len(dataset.feature_names))
        start_idx = 0
        for feature_idx, length in enumerate(dataset.feature_lengths):
            end_idx = start_idx + length
            print('from %d to %d' % (start_idx, end_idx))
            scores[feature_idx] = np.mean(mean_covar[start_idx:end_idx])
            start_idx += length

        total_scores += scores
        #
        # sorted_exploded_feature_names = exploded_feature_names[sorted_features_indexes]
        # sorted_feature_names = []
        # feature_scores = {}
        # for name_idx, exploded_name in enumerate(sorted_exploded_feature_names):
        #     name = exploded_name.split('*')[0]
        #     if name not in sorted_feature_names:
        #         sorted_feature_names.append(name)
        #         feature_scores[name] = 0
        #     feature_scores[name] += name_idx
        # for name, length in zip(dataset.feature_names, dataset.feature_lengths):
        #     feature_scores[name] /= length
        # print np.array(feature_scores.keys())[np.argsort(feature_scores.values())]
        # print('')
        #
        # if total_feature_scores is None:
        #     total_feature_scores = feature_scores
        # else:
        #     for k, v in feature_scores.iteritems():
        #         total_feature_scores[k] += v
    total_scores /= dataset.n_labels
    print('')
    print('total scores:')
    sorted_indexes = np.argsort(total_scores)
    sorted_names = np.array(dataset.feature_names)[sorted_indexes]
    sorted_scores = total_scores[sorted_indexes]
    for name, score in zip(sorted_names, sorted_scores):
        print('%s: %f' % (name, score))


if __name__ == '__main__':
    main(get_parser().parse_args())
