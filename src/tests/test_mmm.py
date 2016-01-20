import os
from unittest import TestCase
import numpy as np


import toolkit.dataset.mmm as mmm


TESTDATA_PATH = os.path.join(os.path.dirname(__file__), 'fixtures')


class TestMotion(TestCase):
    def setUp(self):
        path = os.path.join(TESTDATA_PATH, 'test_mmm.xml')
        self.assertTrue(os.path.exists(path))
        self.motion = mmm.parse_motions(path)[0]
        self.assertIsNotNone(self.motion)

    def test_features(self):
        names = ['joint_pos', 'joint_vel', 'joint_vel_norm', 'joint_acc', 'joint_acc_norm',
                 'root_pos', 'root_vel', 'root_vel_norm', 'root_acc', 'root_acc_norm',
                 'com_pos', 'com_vel', 'com_vel_norm', 'com_acc', 'com_acc_norm',
                 'left_hand_pos', 'left_hand_vel', 'left_hand_vel_norm', 'left_hand_acc', 'left_hand_acc_norm',
                 'right_hand_pos', 'right_hand_vel', 'right_hand_vel_norm', 'right_hand_acc', 'right_hand_acc_norm',
                 'left_foot_pos', 'left_foot_vel', 'left_foot_vel_norm', 'left_foot_acc', 'left_foot_acc_norm',
                 'right_foot_pos', 'right_foot_vel', 'right_foot_vel_norm', 'right_foot_acc', 'right_foot_acc_norm',
                 'root_rot', 'root_rot_norm',
                 'angular_momentum', 'angular_momentum_norm']
        self.assertSetEqual(set(names), set(mmm.FEATURE_NAMES))
        ref_lengths = [40, 40, 1, 40, 1, 3, 3, 1, 3, 1, 3, 3, 1, 3, 1, 3, 3, 1, 3, 1, 3, 3, 1, 3, 1, 3, 3, 1, 3, 1, 3,
                       3, 1, 3, 1, 3, 1, 3, 1]
        self.assertTrue(len(names), len(ref_lengths))

        features, lengths = self.motion.features(names, normalize=True)
        self.assertEqual(features.shape, (265, 196))
        self.assertListEqual(lengths, ref_lengths)
        self.assertTrue(len(names), len(lengths))

        # Load reference
        features_ref = np.load(os.path.join(TESTDATA_PATH, 'test_mmm_features_reference.npy'))
        self.assertEqual(features_ref.shape, features.shape)

        # Iterate over all features one by one to give meaningful failure reason
        start_idx = 0
        for idx, name in enumerate(names):
            length = lengths[idx]
            end_idx = start_idx + length
            is_equal = np.allclose(features[:, start_idx:end_idx], features_ref[:, start_idx:end_idx])
            self.assertTrue(is_equal, 'features for %s are not equal' % name)
            start_idx += length
