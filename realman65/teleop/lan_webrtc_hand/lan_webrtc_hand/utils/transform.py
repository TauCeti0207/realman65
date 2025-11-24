import numpy as np
from scipy.spatial.transform import Rotation as R

class Transform:
    def __init__(self):
        pass
        self.TRANSFORM_TO_WORLD = np.ascontiguousarray(np.eye(4))
        self.TRANSFORM_TO_WORLD[:3, :3] = R.from_euler('xyz', [-90, 0, -90], degrees=True).as_matrix()
        self.WORLD_TO_TRANSFORM = np.ascontiguousarray(np.linalg.inv(self.TRANSFORM_TO_WORLD))

    def pose2mat(self, pos, quat):
        homo_pose_mat = np.eye(4)
        homo_pose_mat[:3, :3] = R.from_quat(quat).as_matrix()
        homo_pose_mat[:3, 3] = pos
        return homo_pose_mat

    def mat2pose(self, homo_pose_mat):
        pos = homo_pose_mat[:3, 3]
        quat = self.mat2quat(homo_pose_mat[:3, :3])
        return pos, quat

    def mat2quat(self, rmat):
        # https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/utils/transform_utils.py
        """
        Converts given rotation matrix to quaternion.

        Args:
            rmat (np.array): 3x3 rotation matrix

        Returns:
            np.array: (x,y,z,w) float quaternion angles
        """
        M = rmat

        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = np.array(
            [
                [m00 - m11 - m22, np.float32(0.0), np.float32(0.0), np.float32(0.0)],
                [m01 + m10, m11 - m00 - m22, np.float32(0.0), np.float32(0.0)],
                [m02 + m20, m12 + m21, m22 - m00 - m11, np.float32(0.0)],
                [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
            ]
        )
        K /= 3.0
        # quaternion is Eigen vector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        inds = np.array([3, 0, 1, 2])
        q1 = V[inds, np.argmax(w)]
        if q1[0] < 0.0:
            np.negative(q1, q1)
        inds = np.array([1, 2, 3, 0])
        return q1[inds]

    def convert_left_to_right_coordinates(self, left_pos, left_quat):

        x = left_pos[0]
        y = -left_pos[1]  # flip y from left to right
        z = left_pos[2]
        qx = -left_quat[0]  # flip rotation from left to right
        qy = left_quat[1]
        qz = -left_quat[2]  # flip rotation from left to right
        qw = left_quat[3]

        transform = self.pose2mat(np.array([x, y, z]), np.array([qx, qy, qz, qw]))

        transform = np.ascontiguousarray(transform)

        transform = self.TRANSFORM_TO_WORLD @ transform

        right_pos, right_quat = self.mat2pose(transform)

        return right_pos, right_quat