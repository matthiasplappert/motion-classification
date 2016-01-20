import os
from unittest import TestCase

from toolkit.dataset.mmm import parse_motions


TESTDATA_PATH = os.path.join(os.path.dirname(__file__), 'fixtures')
TESTDATA_VALID_FILE = os.path.join(TESTDATA_PATH, 'test_parser_valid.xml')
TESTDATA_INVALID_FILES = ['/dev/null',
                          '/does/not/exist',
                          os.path.join(TESTDATA_PATH, 'test_parser_invalid1.xml'),
                          os.path.join(TESTDATA_PATH, 'test_parser_invalid2.xml'),
                          os.path.join(TESTDATA_PATH, 'test_parser_invalid3.xml')]


class TestMotion(TestCase):
    def setUp(self):
        self.motions = parse_motions(TESTDATA_VALID_FILE)
        self.assertGreaterEqual(len(self.motions), 1)

    def test_motions(self):
        self.assertEqual(len(self.motions), 1)
        motion = self.motions[0]
        self.assertEqual(len(motion.frames), 2)

    def test_motion(self):
        motion = self.motions[0]
        self.assertEqual(len(motion.joint_names), 3)
        self.assertListEqual(motion.joint_names, ['a', 'b', 'c'])
        self.assertEqual(len(motion.frames), 2)

    def test_complete_frame(self):
        motion = self.motions[0]
        len_joints = len(motion.joint_names)
        frame = motion.frames[1]

        self.assertEqual(frame.timestep, 0.01)
        self.assertListEqual(frame.root_pos, [1.23, 4.56, 7.89])
        self.assertListEqual(frame.root_rot, [-1.23, -4.56, -7.89])
        self.assertListEqual(frame.joint_pos, [1.1, 2.2, 3.3])
        self.assertListEqual(frame.joint_vel, [4.4, 5.5, 6.6])
        self.assertListEqual(frame.joint_acc, [7.7, 8.8, 9.9])

    def test_optional_values_in_frame(self):
        motion = self.motions[0]
        frame = motion.frames[0]

        self.assertEqual(frame.timestep, 0)
        self.assertIsNone(frame.root_pos)
        self.assertIsNone(frame.root_rot)
        self.assertListEqual(frame.joint_pos, [42, 43, 44])
        self.assertIsNone(frame.joint_vel)
        self.assertIsNone(frame.joint_acc)

    def test_invalid_files(self):
        for f in TESTDATA_INVALID_FILES:
            self.assertRaises(Exception, parse_motions, f)