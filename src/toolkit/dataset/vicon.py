import c3d
import numpy as np


FEATURE_NAMES = ['marker_pos', 'marker_vel', 'marker_vel_norm', 'marker_acc', 'marker_acc_norm']
SUPPORTED_MARKER_NAMES = ['RPSI', 'LPSI', 'L3', 'STRN', 'T10', 'C7', 'CLAV', 'RBAK', 'LBAK', 'LSHO', 'LUPA', 'LAEL',
                          'LAOL', 'LWTS', 'LWPS', 'LFRA', 'LIFD', 'LHPS', 'LHTS', 'RSHO', 'RUPA', 'RAEL', 'RAOL',
                          'RWTS', 'RWPS', 'RFRA', 'RIFD', 'RHTS', 'RHPS', 'RBHD', 'LFHD', 'RFHD', 'LBHD', 'LHIP',
                          'RHIP', 'RASI', 'LASI', 'LKNE', 'LTHI', 'LANK', 'LTIP', 'LTOE', 'LMT1', 'LMT5', 'LHEE',
                          'RKNE', 'RTHI', 'RANK', 'RTIP', 'RTOE', 'RMT1', 'RMT5', 'RHEE']


class Motion(object):
    def __init__(self, frames):
        self.frames = frames

    def features(self, names, normalize=True):
        for name in names:
            if name not in FEATURE_NAMES:
                raise ValueError('unknown feature %s' % name)

        n_frames = len(self.frames)

        # The marker positions are given directly by the frames
        marker_pos = np.array(self.frames)
        assert marker_pos.shape[0] == n_frames
        assert marker_pos.shape[1] % 3 == 0

        marker_vel, marker_acc = None, None
        if 'marker_vel' in names or 'marker_vel_norm' in names or 'marker_acc' in names or 'marker_acc_norm' in names:
            # Always compute the velocities since they are needed for the acceleration
            marker_vel = []
            dt = 0.01
            for idx in xrange(n_frames):
                prev_idx = idx - 1 if idx > 0 else 0
                next_idx = idx + 1 if idx < n_frames - 1 else n_frames - 1
                prev_pos = marker_pos[prev_idx]
                next_pos = marker_pos[next_idx]
                vel = (next_pos - prev_pos) / (2. * dt)
                marker_vel.append(vel)

            if 'marker_acc' in names or 'marker_acc_norm' in names:
                marker_acc = []
                for idx in xrange(n_frames):
                    prev_idx = idx - 1 if idx > 0 else 0
                    next_idx = idx + 1 if idx < n_frames - 1 else n_frames - 1
                    prev_vel = marker_vel[prev_idx]
                    next_vel = marker_vel[next_idx]
                    acc = (next_vel - prev_vel) / (2. * dt)
                    marker_acc.append(acc)

        # Assemble features
        X = np.zeros((n_frames, 0))
        lengths = []
        for name in names:
            data = None
            if name == 'marker_pos':
                data = marker_pos
            elif name == 'marker_vel':
                data = np.array(marker_vel)
            elif name == 'marker_vel_norm':
                data = np.linalg.norm(marker_vel, axis=1).reshape((n_frames, 1))
            elif name == 'marker_acc':
                data = np.array(marker_acc)
            elif name == 'marker_acc_norm':
                data = np.linalg.norm(marker_acc, axis=1).reshape((n_frames, 1))
            assert data is not None
            assert data.shape[0] == n_frames
            lengths.append(data.shape[1])
            X = np.hstack((X, data))
        return X, lengths


def parse_motions(path):
    reader = c3d.Reader(open(path, 'rb'))
    frames = []

    # Extract marker names (some markers are named <Prefix>:<MarkerName>) and filter those to only include
    # the ones that we support across all motions
    marker_names = [marker.rstrip().split(':')[-1] for marker in reader.point_labels]
    supported_indexes = [idx for idx, name in enumerate(marker_names) if name in SUPPORTED_MARKER_NAMES]

    for idx, points, _ in reader.read_frames():
        xyz_coordinates = points[supported_indexes, 0:3]
        assert xyz_coordinates.shape == (len(SUPPORTED_MARKER_NAMES), 3)
        frame = xyz_coordinates.flatten()
        frames.append(frame)
    return [Motion(frames)]
