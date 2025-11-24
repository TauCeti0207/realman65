#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import threading
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
import tf2_ros
from scipy.spatial.transform import Rotation as R

from realman65.my_robot.realman_65_interface import Realman65Interface
from realman65.sensor.Quest_sensor import QuestSensor
from realman65.utils.data_handler import matrix_to_xyz_rpy, debug_print


ADJ_MAT = np.array([
    [0, 0, -1, 0],
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
], dtype=float)
# ADJ_MAT = np.array([
#     [0, 0, 1, 0],
#     [0, -1, 0, 0],
#     [1, 0, 0, 0],
#     [0, 0, 0, 1],
# ], dtype=float)


class VRArmTeleop(Node):
    def __init__(self):
        super().__init__('rm65_vr_teleop')
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self.pose_lock = threading.Lock()
        self.target_pose = None
        self.pos_offset = None
        self.rot_offset = None
        self.retarget_done = False
        self.running = True

        debug_print("teleop", "Initializing RM65 interface...", "INFO")
        self.rm_interface = Realman65Interface(auto_setup=False)
        self.rm_interface.set_up()

        self.quest_sensor = QuestSensor("quest_vr")
        self.quest_sensor.set_up()

        self.vr_thread = threading.Thread(target=self._vr_worker, daemon=True)
        self.arm_thread = threading.Thread(target=self._arm_worker, daemon=True)
        self.vr_thread.start()
        # self.arm_thread.start()

    def destroy_node(self):
        self.running = False
        self.vr_thread.join(timeout=2.0)
        self.arm_thread.join(timeout=2.0)
        debug_print("teleop", "Teleop stopped.", "INFO")
        self.rm_interface.reset()
        super().destroy_node()

    def _publish_tf(self, parent, child, pose6):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = parent
        t.child_frame_id = child
        t.transform.translation.x = float(pose6[0])
        t.transform.translation.y = float(pose6[1])
        t.transform.translation.z = float(pose6[2])
        q = R.from_euler('xyz', pose6[3:]).as_quat()
        t.transform.rotation.x = float(q[0])
        t.transform.rotation.y = float(q[1])
        t.transform.rotation.z = float(q[2])
        t.transform.rotation.w = float(q[3])
        self.tf_broadcaster.sendTransform(t)

    @staticmethod
    def _euler_to_rot(euler_xyz):
        return R.from_euler('xyz', euler_xyz, degrees=False)

    def _retarget_once(self, vr_pose, robot_pose):
        vr_pos = np.array(vr_pose[:3])
        vr_rot = self._euler_to_rot(vr_pose[3:])
        rb_pos = np.array(robot_pose[:3])
        rb_rot = self._euler_to_rot(robot_pose[3:])
        self.pos_offset = rb_pos - vr_pos
        self.rot_offset = rb_rot * vr_rot.inv()
        self.retarget_done = True
        debug_print("teleop", f"Retarget: pos_offset={self.pos_offset}", "INFO")

    def _apply_retarget(self, vr_pose):
        vr_pos = np.array(vr_pose[:3])
        vr_rot = self._euler_to_rot(vr_pose[3:])
        mapped_pos = vr_pos + self.pos_offset
        mapped_rot = (self.rot_offset * vr_rot).as_euler('xyz')
        return np.concatenate([mapped_pos, mapped_rot])

    def _read_left_vr_pose(self):
        transformations, _ = self.quest_sensor.sensor.get_transformations_and_buttons()
        if not transformations or 'l' not in transformations:
            return None
        tf_raw = np.asarray(transformations['l'])
        if tf_raw.shape != (4, 4):
            return None
        return matrix_to_xyz_rpy(ADJ_MAT @ tf_raw)

    def _vr_worker(self, hz=60.0):
        period = 1.0 / hz
        while self.running:
            try:
                left_vr_pose = self._read_left_vr_pose()
            except Exception as exc:
                debug_print("teleop", f"read VR failed: {exc}", "WARNING")
                time.sleep(0.1)
                continue

            if left_vr_pose is None:
                time.sleep(period)
                continue

            self._publish_tf("world", "quest_vr_left", left_vr_pose)

            with self.pose_lock:
                if not self.retarget_done:
                    robot_pose_map = self.rm_interface.get_end_effector_pose()
                    if robot_pose_map and "left_arm" in robot_pose_map:
                        self._retarget_once(left_vr_pose, robot_pose_map["left_arm"])
                        self._publish_tf("world", "rm_left_end_effector", robot_pose_map["left_arm"])
                if self.retarget_done:
                    self.target_pose = self._apply_retarget(left_vr_pose)
                    self._publish_tf("world", "rm_left_target", self.target_pose)

            time.sleep(period)

    def _arm_worker(self, hz=30.0):
        period = 1.0 / hz
        while self.running:
            pose_to_send = None
            with self.pose_lock:
                if self.target_pose is not None and self.retarget_done:
                    pose_to_send = self.target_pose.copy()

            if pose_to_send is not None:
                try:
                    self.rm_interface.set_end_effector_pose('left_arm', pose_to_send.tolist())
                except Exception as exc:
                    debug_print("teleop", f"send pose failed: {exc}", "WARNING")

            time.sleep(period)


def main():
    rclpy.init()
    node = VRArmTeleop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nShutdown requested.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()