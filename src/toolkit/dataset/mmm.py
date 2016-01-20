import logging
import os
from collections import namedtuple
import xml.etree.ElementTree as ET
import timeit

import numpy as np

import pysimox
import pymmm


SUPPORTED_JOINTS = ['BPx_joint', 'BPy_joint', 'BPz_joint', 'BTx_joint', 'BTy_joint', 'BTz_joint', 'BUNx_joint',
                    'BUNy_joint', 'BUNz_joint', 'LAx_joint', 'LAy_joint', 'LAz_joint', 'LEx_joint', 'LEz_joint',
                    'LHx_joint', 'LHy_joint', 'LHz_joint', 'LKx_joint', 'LSx_joint', 'LSy_joint', 'LSz_joint',
                    'LWx_joint', 'LWy_joint', 'RAx_joint', 'RAy_joint', 'RAz_joint', 'REx_joint', 'REz_joint',
                    'RHx_joint', 'RHy_joint', 'RHz_joint', 'RKx_joint', 'RSx_joint', 'RSy_joint', 'RSz_joint',
                    'RWx_joint', 'RWy_joint', 'BLNx_joint', 'BLNy_joint', 'BLNz_joint']
FEATURE_NAMES = ['joint_pos', 'joint_vel', 'joint_vel_norm', 'joint_acc', 'joint_acc_norm',
                 'root_pos', 'root_vel', 'root_vel_norm', 'root_acc', 'root_acc_norm',
                 'com_pos', 'com_vel', 'com_vel_norm', 'com_acc', 'com_acc_norm',
                 'left_hand_pos', 'left_hand_vel', 'left_hand_vel_norm', 'left_hand_acc', 'left_hand_acc_norm',
                 'right_hand_pos', 'right_hand_vel', 'right_hand_vel_norm', 'right_hand_acc', 'right_hand_acc_norm',
                 'left_foot_pos', 'left_foot_vel', 'left_foot_vel_norm', 'left_foot_acc', 'left_foot_acc_norm',
                 'right_foot_pos', 'right_foot_vel', 'right_foot_vel_norm', 'right_foot_acc', 'right_foot_acc_norm',
                 'root_rot', 'root_rot_norm',
                 'angular_momentum', 'angular_momentum_norm']


Frame = namedtuple('Frame', 'timestep, root_pos, root_rot, joint_pos, joint_vel, joint_acc, angular_momentum, com')


# Internal constants
_POS_TO_SEGMENT_MAP = {'left_hand_pos': 'LeftHandSegment_joint',
                       'right_hand_pos': 'RightHandSegment_joint',
                       'left_foot_pos': 'LeftFootHeight_joint',
                       'right_foot_pos': 'RightFootHeight_joint',
                       'left_hand_vel': 'LeftHandSegment_joint',
                       'right_hand_vel': 'RightHandSegment_joint',
                       'left_foot_vel': 'LeftFootHeight_joint',
                       'right_foot_vel': 'RightFootHeight_joint',
                       'left_hand_acc': 'LeftHandSegment_joint',
                       'right_hand_acc': 'RightHandSegment_joint',
                       'left_foot_acc': 'LeftFootHeight_joint',
                       'right_foot_acc': 'RightFootHeight_joint'}
_SEGMENT_POS_NAMES = ['left_hand_pos', 'right_hand_pos', 'left_foot_pos', 'right_foot_pos']
_SEGMENT_VEL_NAMES = ['left_hand_vel', 'right_hand_vel', 'left_foot_vel', 'right_foot_vel']
_SEGMENT_ACC_NAMES = ['left_hand_acc', 'right_hand_acc', 'left_foot_acc', 'right_foot_acc']
_POS_NAMES = [_name for _name in FEATURE_NAMES if _name.endswith('_pos')]
_VEL_NAMES = [_name for _name in FEATURE_NAMES if _name.endswith('_vel')]
_ACC_NAMES = [_name for _name in FEATURE_NAMES if _name.endswith('_acc')]


def rotation_matrix(roll, pitch, yaw):
    r11 = np.cos(yaw) * np.cos(pitch)
    r12 = np.cos(yaw) * np.sin(pitch) * np.sin(roll) - np.sin(yaw) * np.cos(roll)
    r13 = np.cos(yaw) * np.sin(pitch) * np.cos(roll) + np.sin(yaw) * np.sin(roll)
    r21 = np.sin(yaw) * np.cos(pitch)
    r22 = np.sin(yaw) * np.sin(pitch) * np.sin(roll) + np.cos(yaw) * np.cos(roll)
    r23 = np.sin(yaw) * np.sin(pitch) * np.cos(roll) - np.cos(yaw) * np.sin(roll)
    r31 = -np.sin(pitch)
    r32 = np.cos(pitch) * np.sin(roll)
    r33 = np.cos(pitch) * np.cos(roll)
    matrix = np.array([[r11, r12, r13],
                       [r21, r22, r23],
                       [r31, r32, r33]])
    return matrix


def pose_matrix(root_pos, root_rot):
    mat = np.eye(4)
    mat[0:3, 0:3] = rotation_matrix(root_rot[0], root_rot[1], root_rot[2])
    mat[0:3, 3] = root_pos
    return mat.astype('float32')  # this cast is VERY IMPORTANT b/c otherwise pysimox and pymmm need float32!


class Motion(object):
    def __init__(self, frames, joint_names, model_path, subject_height, subject_mass):
        self.joint_names = joint_names
        self.frames = frames
        self.n_frames = len(frames)
        self.model_path = model_path
        self.subject_height = subject_height  # in meters
        self.subject_mass = subject_mass      # in kilogram
        self.dt = frames[1].timestep - frames[0].timestep  # assuming equidistant frames
        assert self.dt > 0.
        self.robot = None
        self.robot_node_set = None

    def features(self, names, normalize=True):
        if not self.robot:
            model_reader = pymmm.ModelReaderXML()
            mmm_model = model_reader.loadModel(self.model_path)
            model_processor_winter = pymmm.ModelProcessorWinter()
            model_processor_winter.setup(self.subject_height, self.subject_mass)
            processed_model = model_processor_winter.convertModel(mmm_model)
            self.robot = pymmm.buildModel(processed_model, False)
            self.robot_node_set = pysimox.RobotNodeSet.createRobotNodeSet(self.robot, 'RobotNodeSet', self.joint_names,
                                                                          '', '', True)

        start_time = timeit.default_timer()
        if len(names) == 0:
            raise ValueError('must at least specify one feature name')
        for name in names:
            if name not in FEATURE_NAMES:
                raise ValueError('unknown feature %s' % name)

        features = {}

        # Calculate pos, acc and vel without normalization. We calculate those regardless of the requested features
        # since resolving dependencies between them is not worth the effort.
        self._calculate_positions(features)
        self._calculate_velocities(features)
        self._calculate_accelerations(features)

        # Calculate other normalized features.
        if 'root_rot' in names or 'root_rot_norm' in names:
            self._calculate_and_normalize_root_rot(features, normalize)
        if 'angular_momentum' in names or 'angular_momentum_norm' in names:
            self._calculate_and_normalize_angular_momentum(features, normalize)

        # Normalize pos, vel and acc.
        if normalize:
            self._normalize_positions(features)
            self._normalize_velocities(features)
            self._normalize_accelerations(features)

        # Calculate norm features
        self._calculate_norm_features(features)

        # Assemble features into feature matrix.
        X = np.zeros((self.n_frames, 0))
        lengths = []
        for name in names:
            feature = np.array(features[name])
            assert feature.shape[0] == self.n_frames
            X = np.hstack((X, feature))
            lengths.append(feature.shape[1])
        n_features = np.sum(lengths)
        assert X.shape == (self.n_frames, n_features)
        logging.info('loading features took %fs' % (timeit.default_timer() - start_time))
        return X, lengths

    def _configure_robot_with_frame(self, frame):
        pose = pose_matrix(frame.root_pos, frame.root_rot)
        self.robot.setGlobalPose(pose, True)
        self.robot_node_set.setJointValues(frame.joint_pos)

    def _calculate_positions(self, features):
        for name in _POS_NAMES:
            features[name] = []
        features['root_pose'] = []
        for segment_name in set(_POS_TO_SEGMENT_MAP.values()):
            features[segment_name] = []
        for frame in self.frames:
            self._configure_robot_with_frame(frame)
            features['root_pose'].append(self.robot.getGlobalPose())
            for name in _POS_NAMES:
                pos = None
                if name == 'joint_pos':
                    pos = np.array(frame.joint_pos)
                elif name == 'root_pos':
                    pos = np.array(frame.root_pos)
                elif name == 'com_pos':
                    pos = self.robot.getCoMGlobal().reshape(3,)
                elif name in _SEGMENT_POS_NAMES:
                    segment_name = _POS_TO_SEGMENT_MAP[name]
                    node = self.robot.getRobotNode(segment_name)
                    segment_pose = node.getGlobalPose()
                    pos = segment_pose[0:3, 3].reshape(3,)
                    features[segment_name].append(segment_pose)
                assert pos is not None
                features[name].append(pos)

    def _calculate_velocities(self, features):
        vel_to_pos_map = {}
        for name in _VEL_NAMES:
            vel_to_pos_map[name] = name[:-3] + 'pos'
            features[name] = []
        self._calculate_vel_or_acc(features, _VEL_NAMES, vel_to_pos_map)

    def _calculate_accelerations(self, features):
        acc_to_vel_map = {}
        for name in _ACC_NAMES:
            acc_to_vel_map[name] = name[:-3] + 'vel'
            features[name] = []
        self._calculate_vel_or_acc(features, _ACC_NAMES, acc_to_vel_map)

    def _calculate_vel_or_acc(self, features, names, antiderivative_map):
        for idx, frame in enumerate(self.frames):
            prev_idx = idx - 1 if idx > 0 else 0
            next_idx = idx + 1 if idx < self.n_frames - 1 else self.n_frames - 1
            for name in names:
                antiderivative_name = antiderivative_map[name]
                prev_value = features[antiderivative_name][prev_idx]
                next_value = features[antiderivative_name][next_idx]
                derivative = (next_value - prev_value) / (2. * self.dt)
                features[name].append(derivative)

    def _normalize_positions(self, features):
        #root_rot0 = self.frames[0].root_rot
        root_rot0 = (0., 0., self.frames[0].root_rot[2])
        root_pose0_inv = np.linalg.inv(pose_matrix(self.frames[0].root_pos, root_rot0))
        for name in _POS_NAMES:
            if name == 'joint_pos':
                continue
            pos = np.hstack((np.array(features[name]), np.ones((self.n_frames, 1))))
            features[name] = np.dot(root_pose0_inv, pos.T).T[:, 0:3]

    def _normalize_velocities(self, features):
        self._normalize_vel_or_acc(features, _VEL_NAMES)

    def _normalize_accelerations(self, features):
        self._normalize_vel_or_acc(features, _ACC_NAMES)

    def _normalize_vel_or_acc(self, features, names):
        # TODO: implement different normalization types to segment
        for idx, frame in enumerate(self.frames):
            root_rot_inv = np.linalg.inv(features['root_pose'][idx][0:3, 0:3])
            for name in names:
                if name == 'joint_vel' or name == 'joint_acc':
                    continue
                value = features[name][idx]
                if name in _SEGMENT_VEL_NAMES or name in _SEGMENT_ACC_NAMES:
                    segment_name = _POS_TO_SEGMENT_MAP[name]
                    node_rot_inv = np.linalg.inv(features[segment_name][idx][0:3, 0:3])
                    value = np.dot(node_rot_inv, value)
                else:
                    value = np.dot(root_rot_inv, value)
                features[name][idx] = value

    def _calculate_and_normalize_root_rot(self, features, normalize):
        offset = np.zeros(3)
        frame0 = self.frames[0]
        feature = []
        for idx, frame in enumerate(self.frames):
            rot = np.array(frame.root_rot)
            if normalize:
                prev_idx = idx - 1 if idx > 0 else 0
                prev_rot = np.array(self.frames[prev_idx].root_rot)

                # Detect overflow (that is a jump from +pi to -pi) and underflow (jump from -pi to +pi). If such
                # a jump happens, we normalize by adding +2pi (-2pi respectively) as a correction factor to all
                # rotations following the jump.
                threshold = 2. * np.pi - 0.1
                delta_root_rot = prev_rot - rot
                overflow_indexes = np.where(delta_root_rot > threshold)[0]
                if len(overflow_indexes) > 0:
                    offset[overflow_indexes] += 2. * np.pi
                underflow_indexes = np.where(delta_root_rot < -threshold)[0]
                if len(underflow_indexes) > 0:
                    offset[underflow_indexes] -= 2. * np.pi
                rot = (rot + offset) - np.array(frame0.root_rot)
            feature.append(rot)
        features['root_rot'] = feature

    def _calculate_and_normalize_angular_momentum(self, features, normalize):
        n_frames = len(self.frames)

        # Pre-calculate some dynamic properties
        inverse_kinematics = pysimox.DifferentialIK(self.robot_node_set)
        segments_with_mass = [node.getName() for node in self.robot.getRobotNodes() if node.getMass() > 0.]
        frame_segment_data = []
        for idx, frame in enumerate(self.frames):
            self._configure_robot_with_frame(frame)
            segment_map = {}
            for segment in segments_with_mass:
                node = self.robot.getRobotNode(segment)
                data = {'com': node.getCoMGlobal().reshape(3,),
                        'jac': inverse_kinematics.getJacobianMatrix(node)}
                segment_map[segment] = data
            frame_segment_data.append(segment_map)

        # Calculate angular momentum
        feature = []
        for idx, frame in enumerate(self.frames):
            self._configure_robot_with_frame(frame)
            prev_idx = idx - 1 if idx > 0 else 0
            next_idx = idx + 1 if idx < n_frames-1 else n_frames-1
            angular_momentum = np.zeros(3)
            for segment in segments_with_mass:
                node = self.robot.getRobotNode(segment)

                # Calculate linear velocity
                prev_segment_com = frame_segment_data[prev_idx][segment]['com']
                next_segment_com = frame_segment_data[next_idx][segment]['com']
                segment_com_vel = next_segment_com - prev_segment_com
                com_vel = features['com_pos'][next_idx] - features['com_pos'][prev_idx]
                linear_vel = (segment_com_vel - com_vel) / (2. * self.dt)

                # Calculate angular velocity
                joint_vel = features['joint_vel'][idx]
                angular_vel = np.dot(frame_segment_data[idx][segment]['jac'], joint_vel)[3:6]

                # Calculate angular momentum for this segment
                delta_com = frame_segment_data[idx][segment]['com'] - features['com_pos'][idx]
                if normalize:
                    root_rot_inv = np.linalg.inv(self.robot.getGlobalPose()[0:3, 0:3])
                    delta_com = np.dot(root_rot_inv, delta_com)
                    linear_vel = np.dot(root_rot_inv, linear_vel)
                inertia_tensor = node.getInertiaMatrix() * 1000000.  # m^2 -> mm^2
                segment_am = node.getMass() * np.cross(delta_com, linear_vel)
                segment_am += np.dot(inertia_tensor, angular_vel)
                assert segment_am.shape == angular_momentum.shape
                angular_momentum += segment_am
            angular_momentum /= 1000000.  # mm^2 -> m^2
            # TODO: normalization of the velocities/positions is not necessary. However, subject-specifc
            # values like the height should be normalized eventually.
            # Also see http://jeb.biologists.org/content/211/4/467.full.pdf
            feature.append(angular_momentum)
        features['angular_momentum'] = feature

    def _calculate_norm_features(self, features):
        for name in FEATURE_NAMES:
            if not name.endswith('_norm'):
                continue
            non_norm_name = name[:-5]
            if non_norm_name not in features:
                continue
            norms = np.linalg.norm(features[non_norm_name], axis=1).reshape((self.n_frames, 1))
            features[name] = norms


def parse_motions(path):
    xml_tree = ET.parse(path)
    xml_root = xml_tree.getroot()
    xml_motions = xml_root.findall('Motion')
    motions = []

    # TODO: currently we only read the first motion, which is usually the movement. Some files also contain other
    # motions, which are usually objects and/or obstacles in the scene. We should be somehow able to handle this better
    # in case the human motion is not the first motion in the file
    if len(xml_motions) > 1:
        logging.warn('more than one <Motion> tag in file "%s", only parsing the first one', path)
    motions.append(_parse_motion(xml_motions[0], path))

    return motions


def _parse_motion(xml_motion, path):
    # Extract model information
    xml_model_file = xml_motion.find('Model/File')
    if xml_model_file is None:
        raise RuntimeError('model file not found')
    model_path = str(os.path.abspath(os.path.join(os.path.dirname(path), xml_model_file.text)))
    if not os.path.exists(model_path):
        raise RuntimeError('model "%s" does not exist' % model_path)

    # Extract subject information
    xml_height = xml_motion.find('ModelProcessorConfig/Height')
    if xml_height is None:
        raise RuntimeError('subject height not found')
    xml_mass = xml_motion.find('ModelProcessorConfig/Mass')
    if xml_mass is None:
        raise RuntimeError('subject mass not found')
    subject_height, subject_mass = float(xml_height.text), float(xml_mass.text)

    xml_joint_order = xml_motion.find('JointOrder')
    if xml_joint_order is None:
        raise RuntimeError('<JointOrder> not found')

    joint_names = []
    joint_indexes = []
    for idx, xml_joint in enumerate(xml_joint_order.findall('Joint')):
        name = xml_joint.get('name')
        if name is None:
            raise RuntimeError('<Joint> has no name')
        elif name not in SUPPORTED_JOINTS:
            logging.warn('joint %s is unsupported - skipping', name)
            continue
        joint_indexes.append(idx)
        joint_names.append(name)

    frames = []
    xml_frames = xml_motion.find('MotionFrames')
    if xml_frames is None:
        raise RuntimeError('<MotionFrames> not found')
    for xml_frame in xml_frames.findall('MotionFrame'):
        frames.append(_parse_frame(xml_frame, joint_indexes))

    return Motion(frames, joint_names, model_path, subject_height, subject_mass)


def _parse_frame(xml_frame, joint_indexes):
    n_joints = len(joint_indexes)
    xml_timestep = xml_frame.find('Timestep')
    if xml_timestep is None:
        raise RuntimeError('<Timestep> not found')
    timestep = float(xml_timestep.text)

    xml_joint_pos = xml_frame.find('JointPosition')
    if xml_joint_pos is None:
        raise RuntimeError('<JointPosition> not found')
    joint_pos = _parse_list(xml_joint_pos, n_joints, joint_indexes)

    # Optional attributes
    joint_vel = joint_acc = root_pos = root_rot = angular_momentum = com = None

    xml_joint_vel = xml_frame.find('JointVelocity')
    if xml_joint_vel is not None:
        joint_vel = _parse_list(xml_joint_vel, n_joints, joint_indexes)

    xml_joint_acc = xml_frame.find('JointAcceleration')
    if xml_joint_acc is not None:
        joint_acc = _parse_list(xml_joint_acc, n_joints, joint_indexes)

    xml_root_pos = xml_frame.find('RootPosition')
    if xml_root_pos is not None:
        root_pos = _parse_list(xml_root_pos, 3)

    xml_root_rot = xml_frame.find('RootRotation')
    if xml_root_rot is not None:
        root_rot = _parse_list(xml_root_rot, 3)

    xml_angular_momentum = xml_frame.find('AngularMomentum')
    if xml_angular_momentum is not None:
        angular_momentum = _parse_list(xml_angular_momentum, 3)

    xml_com = xml_frame.find('CoM')
    if xml_com is not None:
        com = _parse_list(xml_com, 3)

    return Frame(timestep=timestep, root_pos=root_pos, root_rot=root_rot,
                 joint_pos=joint_pos, joint_acc=joint_acc, joint_vel=joint_vel,
                 angular_momentum=angular_momentum, com=com)


def _parse_list(xml_elem, length, indexes=None):
    if indexes is None:
        indexes = range(length)
    elems = [float(x) for idx, x in enumerate(xml_elem.text.rstrip().split(' ')) if idx in indexes]
    if len(elems) != length:
        raise RuntimeError('invalid number of elements')
    return elems
