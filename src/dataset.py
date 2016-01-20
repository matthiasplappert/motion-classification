# coding=utf8
from collections import OrderedDict
from argparse import ArgumentParser
import pickle
import os
import logging
import json

import numpy as np
import scipy.io
import matplotlib.pyplot as plt

import toolkit.dataset.base as data
from toolkit.dataset.mmm import pose_matrix, rotation_matrix
from toolkit.util import pad_sequences


def get_parser():
    # TODO: this should really use subparsers
    parser = ArgumentParser()
    data.add_dataset_parser_arguments(parser)
    data.add_transformer_parser_arguments(parser)
    parser.add_argument('action', choices=['report', 'export', 'export-all', 'check', 'plot'], help='type of action')
    parser.add_argument('--format', choices=['pickle', 'matlab', 'matlab-stacked'], default='pickle')
    parser.add_argument('--output')
    parser.add_argument('--only-errors', action='store_true')
    parser.add_argument('--plot-labels', nargs='*', default=None)
    parser.add_argument('--verbose', action='store_true')
    return parser


def _color_str(input, color=None):
    # Source: https://www.siafoo.net/snippet/88
    if color == 'green':
        return '\033[0;32m%s\033[0;m' % input
    elif color == 'red':
        return '\033[0;31m%s\033[0;m' % input
    else:
        return input


def check(args):
    supported_types = OrderedDict()
    supported_types['mmm-nlopt'] = '_nlopt.xml'
    #supported_types['mmm'] = '.xml'
    supported_types['vicon'] = '.c3d'

    # Load the manifest manually
    path = args.dataset
    with open(path) as f:
        # TODO: error handling
        root = json.load(f)

        files = []
        for item in root:
            item_path = os.path.join(os.path.dirname(path), item['path'])
            for curr_file in os.listdir(item_path):
                file_path = os.path.join(item_path, curr_file)
                if os.path.isdir(file_path):
                    logging.warn('check does not perform a deep search - skipping folder %s', curr_file)
                    continue

                file_type = None
                name = None
                for possible_type in supported_types:
                    if supported_types[possible_type] in curr_file:
                        file_type = possible_type
                        name = curr_file[:len(curr_file)-len(supported_types[possible_type])]
                        break
                if file_type is None:
                    # Unknown type, skip
                    continue
                files.append({'path': file_path, 'name': curr_file, 'unique_name': os.path.join(item_path, name),
                              'type': file_type})

    # Group files by unique name
    files_by_unique_name = {}
    for curr_file in files:
        if curr_file['unique_name'] not in files_by_unique_name:
            files_by_unique_name[curr_file['unique_name']] = []
        files_by_unique_name[curr_file['unique_name']].append(curr_file)

    n_errors = 0
    n_motions = 0
    for unique_name in files_by_unique_name:
        files = files_by_unique_name[unique_name]
        n_files = len(files)
        types = []
        for curr_file in files:
            types.append(curr_file['type'])
        types.sort()

        # Check if all files are present
        files_okay = (n_files == len(types) and set(types) == set(supported_types))

        # Check if all files can be parsed and have all properties
        n_frames = []
        angular_momentum = []
        com = []
        frames_okay = True
        for curr_file in files:
            motions = data.load_motions(curr_file['path'], motion_type=curr_file['type'])
            assert len(motions) == 1  # we currently do only ever parse on motion per file

            # Check number of frames
            curr_n_frames = str(len(motions[0].frames))
            n_frames.append(curr_n_frames)
            if n_frames[-1] != curr_n_frames:
                frames_okay = False
        okay = files_okay and frames_okay
        if okay:
            color = None
        else:
            n_errors += 1
            color = 'red'
        n_motions += 1
        if not args.only_errors or not okay:
            info_str = '%s: %d files (%s), frames (%s)'
            print(_color_str(info_str % (unique_name, len(files), ', '.join(types), ', '.join(n_frames)), color))
    print('\nchecked %d motions, of which %d contain errors' % (n_motions, n_errors))


def plot(dataset, args):
    transformers = data.transformers_from_args(args)
    if len(transformers) > 0:
        dataset = dataset.dataset_from_transformers(transformers)

    plot_labels = args.plot_labels
    for features in dataset.X:
        n_samples, n_features = features.shape
        x = np.arange(n_samples)
        for feature_idx in xrange(n_features):
            feature = features[:, feature_idx]
            plt.plot(x, feature)
        if plot_labels is not None:
            if len(plot_labels) == n_features:
                plt.legend(plot_labels, loc='upper left')
            else:
                logging.warn('plot-labels must have length %d' % n_features)
        plt.xlabel('time steps')
        plt.ylabel('feature value')
        plt.show()


def report(dataset, args):
    print('')
    print('%d labels:' % len(dataset.unique_labels))
    for idx, label in enumerate(dataset.unique_labels):
        n_motions = int(np.sum(dataset.y[:, idx]))
        print('  %s: %d' % (label, n_motions))
    print('')

    label_combination_counts = []
    label_combinations = []
    for sample_idx in xrange(dataset.n_samples):
        combination = dataset.y[sample_idx, :].tolist()
        if combination not in label_combinations:
            label_combinations.append(combination)
            label_combination_counts.append(0)
        combination_idx = label_combinations.index(combination)
        label_combination_counts[combination_idx] += 1
    print('%d label combinations:' % len(label_combination_counts))
    for label_combination, count in zip(label_combinations, label_combination_counts):
        label_indexes = np.where(np.array(label_combination) == 1)[0]
        labels = [dataset.unique_labels[idx] for idx in label_indexes]
        print(' %s: %d' % (', '.join(labels), count))
    print('')

    print('%d features:' % dataset.n_features)
    for idx, name in enumerate(dataset.feature_names):
        length = dataset.feature_lengths[idx]
        print('  %s (dim=%d)' % (name, length))
    print('')

    print('%d motions:' % len(dataset.X))
    for idx, curr in enumerate(dataset.X):
        label_indexes = np.where(dataset.y[idx, :] == 1)[0]
        labels = ', '.join([dataset.unique_labels[label_idx] for label_idx in label_indexes])
        print('  motion %.3d: %s, tags %s' % (idx, curr.shape, labels))


# Export all variations of a dataset: vicon + mmm, with and without normalization.
def export_all(args):
    # Load MMM dataset WITHOUT normalization
    args.motion_type = 'mmm-nlopt'
    args.disable_normalization = True
    args.disable_smoothing = True
    print('Loading MMM data without normalization ...')
    mmm = data.load_dataset_from_args(args)
    print('done, %d motions and %d features loaded' % (mmm.n_samples, mmm.n_features))
    print('')

    # Load MMM dataset WITH normalization
    args.motion_type = 'mmm-nlopt'
    args.disable_normalization = False
    args.disable_smoothing = True
    print('Loading MMM data with normalization ...')
    normalized_mmm = data.load_dataset_from_args(args)
    normalized_mmm.feature_names = ['normalized_' + name for name in normalized_mmm.feature_names]
    print('done, %d motions and %d features loaded' % (normalized_mmm.n_samples, normalized_mmm.n_features))
    print('')

    # Load Vicon dataset
    args.motion_type = 'vicon'
    args.disable_normalization = True
    args.disable_smoothing = True
    print('Loading vicon data without normalization ...')
    vicon = data.load_dataset_from_args(args)
    print('done, %d motions and %d features loaded' % (vicon.n_samples, vicon.n_features))
    print('')

    # Manually perform normalization on Vicon using the root_rot and root_pos information
    print('Performing normalization on vicon dataset ...')
    normalized_vicon = vicon.copy()
    root_pos_indexes = mmm.indexes_for_feature('root_pos')
    root_rot_indexes = mmm.indexes_for_feature('root_rot')
    for sample_idx in xrange(normalized_vicon.n_samples):
        X_curr = normalized_vicon.X[sample_idx]
        n_frames = X_curr.shape[0]
        marker_pos_indexes = normalized_vicon.indexes_for_feature('marker_pos')
        assert len(marker_pos_indexes) % 3 == 0
        n_markers = len(marker_pos_indexes) / 3

        root_pos0 = mmm.X[sample_idx][0, root_pos_indexes]  # do not use the normalized dataset here!
        root_rot = mmm.X[sample_idx][:, root_rot_indexes]   # do not use the normalized dataset here!
        assert root_rot.shape == (n_frames, 3)

        # Normalize marker positions
        root_pose0_inv = np.linalg.inv(pose_matrix(root_pos0, (0., 0., root_rot[0][2])))
        for marker_idx in xrange(n_markers):
            start_idx = marker_pos_indexes[0] + marker_idx * 3
            end_idx = start_idx + 3
            marker_pos = X_curr[:, start_idx:end_idx]
            n_samples = marker_pos.shape[0]
            marker_pos_plus_one = np.hstack((marker_pos, np.ones((n_samples, 1))))
            assert marker_pos_plus_one.shape == (n_samples, 4)
            normalized_marker_pos = np.dot(root_pose0_inv, marker_pos_plus_one.T).T[:, 0:3]
            assert normalized_marker_pos.shape == marker_pos.shape
            X_curr[:, start_idx:end_idx] = normalized_marker_pos

        # Normalize velocities and accelerations
        marker_vel_indexes = normalized_vicon.indexes_for_feature('marker_vel')
        marker_acc_indexes = normalized_vicon.indexes_for_feature('marker_acc')
        for idx in xrange(n_frames):
            root_rot_inv = np.linalg.inv(rotation_matrix(root_rot[idx][0], root_rot[idx][1], root_rot[idx][2]))
            for marker_idx in xrange(n_markers):
                vel_start_idx = marker_vel_indexes[0] + marker_idx * 3
                vel_end_idx = vel_start_idx + 3
                vel = X_curr[idx, vel_start_idx:vel_end_idx]
                X_curr[idx, vel_start_idx:vel_end_idx] = np.dot(root_rot_inv, vel)

                acc_start_idx = marker_acc_indexes[0] + marker_idx * 3
                acc_end_idx = acc_start_idx + 3
                acc = X_curr[idx, acc_start_idx:acc_end_idx]
                X_curr[idx, acc_start_idx:acc_end_idx] = np.dot(root_rot_inv, acc)
        # No need to normalize marker_vel_norm and marker_acc_norm
    normalized_vicon.feature_names = ['normalized_' + name for name in normalized_vicon.feature_names]
    print('done, %d motions and %d features loaded' % (normalized_vicon.n_samples, normalized_vicon.n_features))
    print('')

    # Merge all datasets into one
    print('merging datasets ...')
    final_dataset = mmm.copy()
    final_dataset.merge_with_dataset(normalized_mmm)
    final_dataset.merge_with_dataset(vicon)
    final_dataset.merge_with_dataset(normalized_vicon)
    assert final_dataset.feature_names == mmm.feature_names + normalized_mmm.feature_names + vicon.feature_names + normalized_vicon.feature_names
    assert final_dataset.n_features == mmm.n_features + normalized_mmm.n_features + vicon.n_features + normalized_vicon.n_features
    print('done, %d motions and %d features' % (final_dataset.n_samples, final_dataset.n_features))
    print('')

    if args.output is not None:
        print('saving dataset ...')
        with open(args.output, 'wb') as f:
            pickle.dump(final_dataset, f)
        print('done')



def merge(args):
    # Merges two datasets into one
    dataset1 = data.load_dataset_from_args(args.dataset)
    dataset2 = data.load_dataset_from_args(args.other_dataset)
    dataset1.merge_with_dataset(dataset2)


def export(dataset, args):
    fmt = args.format
    output = args.output

    transformers = data.transformers_from_args(args)
    if len(transformers) > 0:
        print('applying transformers %s ...' % transformers)
        dataset = dataset.dataset_from_transformers(transformers)

    print('exporting dataset ...')
    if fmt == 'matlab' or fmt == 'matlab-stacked':
        matlab_data = {}
        if fmt == 'matlab-stacked':
            X_pad = pad_sequences(dataset.X)
            matlab_data['X'] = X_pad
            matlab_data['T'] = X_pad.shape[1]
        else:
            matlab_data['X'] = dataset.X
        matlab_data['y'] = dataset.y
        matlab_data['unique_labels'] = dataset.unique_labels
        matlab_data['feature_names'] = dataset.feature_names
        matlab_data['feature_lengths'] = dataset.feature_lengths
        scipy.io.savemat(output, matlab_data)
    elif fmt == 'pickle':
        with open(output, 'wb') as f:
            pickle.dump(dataset, f)
    print('done!')


def main(args):
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    args.disable_cache = True  # we do not want to load stale data here!
    path = args.dataset
    if not os.path.exists(path):
        exit('dataset at path "%s" does not exist' % path)

    # check and export-all does not need the loaded dataset
    if args.action == 'check':
        check(args)
        return
    elif args.action == 'export-all':
        export_all(args)
        return

    print('loading data set "%s" ...' % path)
    dataset = data.load_dataset_from_args(args)

    action_func = None
    if args.action == 'report':
        action_func = report
    elif args.action == 'export':
        action_func = export
    elif args.action == 'plot':
        action_func = plot
    assert action_func is not None
    action_func(dataset, args)


if __name__ == '__main__':
    main(get_parser().parse_args())
