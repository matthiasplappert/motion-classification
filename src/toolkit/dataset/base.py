import os
import json
import pickle
import logging
from collections import deque

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

import mmm
import vicon
from ..util import check_feature_array


MOTION_TYPE_EXTENSIONS = {'mmm': '.xml', 'mmm-nlopt': '.xml', 'vicon': '.c3d'}


class Dataset(object):
    def __init__(self, X, y, unique_labels, feature_names, feature_lengths):
        if len(X) != y.shape[0]:
            raise ValueError('X and y must have matching dimensions')
        if len(unique_labels) != y.shape[1]:
            raise ValueError('unique_labels and y must have matching dimensions')
        if len(feature_names) != len(feature_lengths):
            raise ValueError('feature_names and feature_lengths must have matching dimensions')
        if not np.all(np.array([X_curr.shape[1] for X_curr in X]) == X[0].shape[1]):
            raise ValueError('all X elements must have equal number of features')
        if X[0].shape[1] != np.sum(feature_lengths):
            raise ValueError('X and feature_lengths must have matching dimensions')
        self.X = X
        self.y = y
        self.unique_labels = unique_labels
        self.feature_names = feature_names
        self.feature_lengths = feature_lengths

    def copy(self):
        X_copy = [np.copy(X_curr) for X_curr in self.X]
        return Dataset(X_copy, np.copy(self.y), self.unique_labels[:], self.feature_names[:], self.feature_lengths[:])

    @property
    def n_samples(self):
        return len(self.X)

    @property
    def n_features(self):
        return self.X[0].shape[1]

    @property
    def n_labels(self):
        return self.y.shape[1]

    def merge_with_dataset(self, other_dataset):
        if self.n_samples != other_dataset.n_samples:
            raise ValueError('number of samples must be equal')
        if not np.allclose(self.y, other_dataset.y):
            raise ValueError('ys must be equal')
        if self.unique_labels != other_dataset.unique_labels:
            raise ValueError('unique_labels must be equal')
        X_shapes = np.array([X_curr.shape[0] for X_curr in self.X])
        other_X_shapes = np.array([X_curr.shape[0] for X_curr in other_dataset.X])
        if not np.allclose(X_shapes, other_X_shapes):
            raise ValueError('all X elements must have equal length')

        self.X = [np.hstack((self.X[idx], other_dataset.X[idx])) for idx in xrange(self.n_samples)]
        self.feature_names = self.feature_names + other_dataset.feature_names
        self.feature_lengths = self.feature_lengths + other_dataset.feature_lengths

    def split_train_test(self, train_indexes, test_indexes, transformers=None):
        if len(set(train_indexes).intersection(set(test_indexes))) > 0:
            raise ValueError('train_indexes and test_indexes must be disjunct')
        if transformers is None:
            transformers = []
        if len(train_indexes) == 0:
            raise ValueError('train_indexes must at least contain one element')
        if len(test_indexes) == 0:
            raise ValueError('test_indexes must at least contain one element')

        X_train = [np.copy(self.X[idx]) for idx in train_indexes]
        y_train = self.y[train_indexes]
        feature_names = self.feature_names
        feature_lengths = self.feature_lengths
        if len(transformers) > 0:
            X_train, feature_names, feature_lengths = self._apply_transformers_to_X(X_train, transformers)
        train = Dataset(X_train, np.copy(y_train), self.unique_labels[:], feature_names[:], feature_lengths[:])

        X_test = [np.copy(self.X[idx]) for idx in test_indexes]
        for transformer in transformers:
            # Apply the previously fitted transform on the test dataset
            X_test = [transformer.transform(X_curr) for X_curr in X_test]
        y_test = self.y[test_indexes]
        test = Dataset(X_test, np.copy(y_test), self.unique_labels[:], feature_names[:], feature_lengths[:])

        return train, test

    def dataset_from_transformers(self, transformers):
        X_copy = [np.copy(X_curr) for X_curr in self.X]
        X_trans, feature_names, feature_lengths = self._apply_transformers_to_X(X_copy, transformers)
        return Dataset(X_trans, np.copy(self.y), self.unique_labels[:], feature_names[:], feature_lengths[:])

    def indexes_for_feature(self, name):
        start_idx = 0
        for curr_name, length in zip(self.feature_names, self.feature_lengths):
            if curr_name == name:
                return range(start_idx, start_idx + length)
            start_idx += length
        return None

    def dataset_from_feature_names(self, names):
        features_diff = set(names).difference(set(self.feature_names))
        if len(features_diff) > 0:
            raise ValueError('unknown features: %s' % features_diff)
        # Calculate start indexes and lengths
        location_data = {}
        start_idx = 0
        for name, length in zip(self.feature_names, self.feature_lengths):
            if name in names:
                location_data[name] = (start_idx, length)
            start_idx += length

        # Compute new dataset
        X = []
        lengths = []
        for idx, X_curr in enumerate(self.X):
            new_X = np.zeros((X_curr.shape[0], 0))
            for name in names:
                start_idx, length = location_data[name]
                if idx == 0:
                    lengths.append(length)
                end_idx = start_idx + length
                new_X = np.hstack((new_X, X_curr[:, start_idx:end_idx]))
            X.append(new_X)
        return Dataset(X, np.copy(self.y), self.unique_labels[:], names, lengths)

    # Transforms the dataset from a multi-label to a multi-class dataset by combining all unique label combinations
    # into a single class each
    def multiclass_dataset(self):
        label_combinations = []
        unique_labels_new = []
        for sample_idx in xrange(self.n_samples):
            combination = self.y[sample_idx, :].tolist()
            if combination in label_combinations:
                continue
            label_indexes = np.where(self.y[sample_idx, :] == 1)[0]
            label_combinations.append(combination)
            label_names = [self.unique_labels[label_idx] for label_idx in label_indexes]
            unique_labels_new.append(', '.join(label_names))

        n_label_combinations = len(label_combinations)
        y_new = np.zeros((self.n_samples, n_label_combinations), dtype=int)
        for sample_idx in xrange(self.n_samples):
            combination = self.y[sample_idx, :].tolist()
            combination_idx = label_combinations.index(combination)
            y_new[sample_idx, combination_idx] = 1

        dataset_new = self.copy()
        dataset_new.y = y_new
        dataset_new.unique_labels = unique_labels_new
        return dataset_new

    def _apply_transformers_to_X(self, X, transformers):
        for transformer in transformers:
            if not hasattr(transformer, 'fit') or not callable(transformer.fit):
                continue
            X_stacked = np.vstack(X)
            transformer.fit(X_stacked)
            X = [transformer.transform(X_curr) for X_curr in X]

        # The transforms may change the number of features (e.g. PCA). If this is the case, calculate generic
        # feature_names and feature_lengths
        n_features = X[0].shape[1]
        if n_features != self.n_features:
            feature_names = ['transformed_feature_%d' % idx for idx in xrange(n_features)]
            feature_lengths = [1 for idx in xrange(n_features)]
        else:
            feature_names = self.feature_names
            feature_lengths = self.feature_lengths

        return X, feature_names, feature_lengths

    def smooth_features(self, smoothable_feature_names=None, window_length=3):
        if smoothable_feature_names is None:
            smoothable_feature_names = self.feature_names
        smoothable_feature_names = set(smoothable_feature_names).intersection(set(self.feature_names))  # ignore unknown features
        if len(smoothable_feature_names) == 0:
            return
        logging.info('smoothing features %s' % smoothable_feature_names)

        # Calculate start indexes and lengths
        location_data = {}
        start_idx = 0
        for name, length in zip(self.feature_names, self.feature_lengths):
            if name in smoothable_feature_names:
                location_data[name] = (start_idx, length)
            start_idx += length

        # Perform smoothing
        for name in smoothable_feature_names:
            start_idx, length = location_data[name]
            end_idx = start_idx + length
            for X_curr in self.X:
                history = deque(maxlen=window_length)
                for frame_idx in xrange(X_curr.shape[0]):
                    history.append(X_curr[frame_idx, start_idx:end_idx])
                    X_curr[frame_idx, start_idx:end_idx] = np.mean(np.array(history), axis=0)

    def shuffle(self, permutation=None):
        if permutation is None:
            permutation = np.random.permutation(self.n_samples)
        self.X = [self.X[i] for i in permutation]
        self.y = self.y[permutation, :]
        return permutation


def transformers_from_args(args):
    transformers = []
    for transformer_name in args.transformers:
        transformer = None
        if transformer_name == 'minmax-scaler':
            transformer = MinMaxScaler(feature_range=(args.minmax_scaler_min, args.minmax_scaler_max), copy=False)
        elif transformer_name == 'pca':
            transformer = PCA(n_components=args.pca_components, copy=False)
        assert transformer is not None
        transformers.append(transformer)
    return transformers


def add_dataset_parser_arguments(parser):
    parser.add_argument('dataset', help='path to the dataset')
    parser.add_argument('--motion-type', choices=['mmm', 'mmm-nlopt', 'vicon'], default='mmm-nlopt')
    parser.add_argument('--features', nargs='+', default=[None],
                        help='the features to be used. defaults to all features of the selected motion type')
    default_smooth_features = set(mmm.FEATURE_NAMES).difference(mmm._POS_NAMES)
    parser.add_argument('--smooth-features', nargs='*', default=default_smooth_features)
    parser.add_argument('--smooth-window-len', type=int, default=3)
    parser.add_argument('--disable-normalization', action='store_true', help='disables dataset normalization')
    parser.add_argument('--disable-smoothing', action='store_true', help='disables dataset smoothing')


def add_transformer_parser_arguments(parser):
    parser.add_argument('--transformers', nargs='*', choices=['minmax-scaler', 'pca'], default=[])
    parser.add_argument('--minmax-scaler-min', default=-1)
    parser.add_argument('--minmax-scaler-max', default=1)
    parser.add_argument('--pca-components', default=5, type=int)


def load_dataset_from_args(args):
    path = args.dataset
    if not os.path.exists(path):
        raise RuntimeError('dataset at "%s" does not exist' % path)
    motion_type = args.motion_type
    features = args.features
    if path.endswith('.json'):
        if len(features) == 1 and features[0] is None:
            # Default: use all features from the appropriate set
            if motion_type == 'vicon':
                features = vicon.FEATURE_NAMES
            else:
                features = mmm.FEATURE_NAMES

        # Load manifest
        normalize = not args.disable_normalization
        X, y, unique_labels, _, lengths = load_manifest(path, motion_type, features, normalize)
        dataset = Dataset(X, y, unique_labels, features, lengths)
    else:
        logging.warn('loading pickled dataset: --disable-normalization and --motion-type are ignored')
        # Attempt to load pickled dataset
        with open(path, 'rb') as f:
            dataset = pickle.load(f)
            if type(dataset) != Dataset:
                raise RuntimeError('could not load dataset from pickled file "%s"' % path)
        if len(features) > 1 or features[0] is not None:
            dataset = dataset.dataset_from_feature_names(features)
            if len(set(dataset.feature_names).difference(set(features))) > 0:
                raise RuntimeError('dataset does not contain all requested features')

    smooth_features = args.smooth_features
    if not args.disable_smoothing and len(smooth_features) > 0:
        window_length = args.smooth_window_len
        dataset.smooth_features(smooth_features, window_length=window_length)
    return dataset


def load_motions(path, motion_type):
    if not os.path.exists(path):
        raise RuntimeError('path %s does not exist' % path)

    motions = []
    if os.path.isdir(path):
        files = sorted(os.listdir(path))  # sorting is important b/c on some systems this will be non-deterministic
        for f in files:
            motions.extend(load_motions(os.path.join(path, f), motion_type))
    else:
        ext = os.path.splitext(path)[1].lower()
        if ext != MOTION_TYPE_EXTENSIONS[motion_type]:
            # Skip files with wrong file extension
            return motions

        # Handle different motion types
        if motion_type == 'vicon':
            # Vicon
            logging.info('loading motion %s', path)
            motions.extend(vicon.parse_motions(path))
        elif motion_type == 'mmm' or motion_type == 'mmm-nlopt':
            # MMM
            has_nlopt_filename = 'nlopt' in path
            if motion_type == 'mmm' and not has_nlopt_filename:
                logging.info('loading motion %s', path)
                motions.extend(mmm.parse_motions(path))
            elif motion_type == 'mmm-nlopt' and has_nlopt_filename:
                logging.info('loading motion %s', path)
                motions.extend(mmm.parse_motions(path))
            else:
                pass  # Ignore the motion
        else:
            raise RuntimeError('unknown motion type: %s' % motion_type)
    return motions


# The above functions should not be used directly. Instead, use the load_dataset functions
def load_manifest(path, motion_type, feature_names=None, normalize=True):
    log_str = 'loading data set "%s": motion_type=%s, features=%s, normalization=%d'
    logging.info(log_str % (path, motion_type, feature_names, normalize))
    # If feature_names is None, use all available features
    if feature_names is None:
        if motion_type == 'mmm' or motion_type == 'mmm-nlopt':
            feature_names = mmm.FEATURE_NAMES
        elif motion_type == 'vicon':
            feature_names = vicon.FEATURE_NAMES

    # Load raw data
    labels, unique_labels, grouped_motions = [], [], []
    with open(path) as f:
        # TODO: error handling
        root = json.load(f)
        for item in root:
            item_path = os.path.join(os.path.dirname(path), item['path'])

            # Extract labels and ensure uniqueness
            item_labels = item['labels']
            labels.append(item_labels)
            for label in item_labels:
                if label not in unique_labels:
                    unique_labels.append(label)

            # Add motions
            m = load_motions(item_path, motion_type)
            grouped_motions.append(m)
    assert len(labels) == len(grouped_motions)

    # Count all motions
    n_motions = 0
    for motions in grouped_motions:
        n_motions += len(motions)

    # Transform into X and y
    X, y, n_labels, groups, lengths = [], [], len(unique_labels), [], None
    motion_count = 0
    for group_idx, motions in enumerate(grouped_motions):
        # Extract features and remember group position in X
        features = []
        for motion in motions:
            logging.info('(%.3d/%.3d) loading features ...' % (motion_count, n_motions))
            f, l = motion.features(feature_names, normalize=normalize)
            if lengths is None:
                lengths = l
            assert f is not None
            features.append(f)
            motion_count += 1
        group_start_idx = len(X)
        group_length = len(features)
        groups.append((group_start_idx, group_length))
        X.extend(features)

        # Encode labels as binary label vector
        item_labels = labels[group_idx]
        y_curr = np.zeros(n_labels)
        for label_idx, label in enumerate(unique_labels):
            if label in item_labels:
                y_curr[label_idx] = 1
        y.extend([y_curr for _ in motions])
    y = np.array(y, dtype=int)
    assert len(X) == n_motions
    assert len(X) == len(y)
    return X, y, unique_labels, groups, lengths


class StratifiedMultilabelKFold(object):
    def __init__(self, y, n_folds=3, groups=None):
        # TODO: validate y
        if groups is None:
            folds = stratified_multilabel_kfold(y, n_folds=n_folds)
        else:
            group_start_indexes = [idx for idx, _ in groups]
            group_folds = stratified_multilabel_kfold(y[group_start_indexes], n_folds=n_folds)

            folds = []
            for group_fold in group_folds:
                fold = []
                for group_idx in group_fold:
                    start_idx, length = groups[group_idx]
                    fold.extend(range(start_idx, start_idx + length))
                folds.append(fold)
        self.folds = folds

    def __iter__(self):
        n_folds = len(self.folds)
        for i in range(n_folds):
            test = self.folds[i]
            train = []
            for idx, fold in enumerate(self.folds):
                if idx == i:
                    continue
                train.extend(fold)
            yield (train, test)


def stratified_multilabel_kfold(y, n_folds=3):
    y = check_feature_array(y)  # TODO: validate that y is binary
    n_samples, n_labels = y.shape
    label_counts = np.sum(y, axis=0).astype(int)

    # Calculate desired counts for each fold and for each fold and each label
    desired_counts = np.zeros((n_folds, n_labels), dtype=int)
    for k in range(n_folds):
        for i in range(n_labels):
            desired_counts[k][i] = int(label_counts[i] / n_folds)  # TODO: rounding?

    # Iteratively calculate folds
    folds = [[] for _ in range(n_folds)]
    remaining_sample_indexes = range(n_samples)
    while len(remaining_sample_indexes) > 0:
        # Find argmin of label_counts where label_counts[label_idx] > 0
        label_idx = None
        for idx, count in enumerate(label_counts):
            if count == 0:
                continue
            if label_idx is None or count < label_counts[label_idx]:
                label_idx = idx
        assert label_idx is not None
        assert label_counts[label_idx] > 0

        # Iterate over all remaining samples and add those to a fold that have the label label_idx set. Mark those
        # samples and delete them later (we cannot delete them immediately since this would mutate the list while
        # iterating over it).
        marked_indexes = []
        for sample_idx in remaining_sample_indexes:
            sample = y[sample_idx]
            positive_label_indexes = np.where(sample == 1)[0]
            if label_idx not in positive_label_indexes:
                # Sample is not positive for our label_idx label, skip it
                continue

            # Add the sample to the fold with the "most" desire for samples with this label
            fold_idx = np.argmax(desired_counts[:, label_idx])  # TODO: paper suggests more sophisticated tie breaking
            folds[fold_idx].append(sample_idx)

            # Perform necessary bookkeeping
            desired_counts[fold_idx, positive_label_indexes] -= 1
            label_counts[positive_label_indexes] -= 1
            marked_indexes.append(sample_idx)
        remaining_sample_indexes = [idx for idx in remaining_sample_indexes if idx not in marked_indexes]
    return folds

