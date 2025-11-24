#!/usr/bin/env python3
import os
import json
import threading
import numpy as np
import rclpy
from rclpy.node import Node
import tf2_ros
import tf_transformations
from sensor_msgs.msg import JointState
from ament_index_python.packages import get_package_share_directory


class HandTeleopNode(Node):
    def __init__(self):
        super().__init__('hand_teleop_interface')

        self.declare_parameter("device", "brainco2")
        self.device = self.get_parameter('device').value

        self.joint_limits = self._get_joint_limits(self.device)
        self.lower_limit = [l for l, _ in self.joint_limits]
        self.upper_limit = [u for _, u in self.joint_limits]

        self.finger_list, self.parent_list = self._get_finger_mappings()

        self.init_status = False
        self.offset = []
        self.scale = []
        self.current_angles = [0.0] * 12
        self.retargeted_angles = [0.0] * 12
        self.open_angles = []
        self.close_angles = []

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.finger_cmd_pub = self.create_publisher(JointState, '/device/finger', 2)

        self.hz = 100.0

        self.lock = threading.Lock()

        self.offset_thread = threading.Thread(target=self.offset_loop, daemon=True)
        self.offset_thread.start()

        self.recv_thread = threading.Thread(target=self.recv_loop, daemon=True)
        self.recv_thread.start()

        self.send_thread = threading.Thread(target=self.send_loop, daemon=True)
        self.send_thread.start()

    def _get_joint_limits(self, device):
        if device == "brainco1":
            return [
                (0.0, 1.5708), (0.0, 0.8727), (0.0, 1.309), (0.0, 1.309),
                (0.0, 1.309), (0.0, 1.309), (0.0, 1.5708), (0.0, 0.8727),
                (0.0, 1.309), (0.0, 1.309), (0.0, 1.309), (0.0, 1.309)
            ]
        else:
            return [
                (0.0, 1.0297), (0.0, 1.5707), (0.0, 1.4137), (0.0, 1.4137),
                (0.0, 1.4137), (0.0, 1.4137), (0.0, 1.0297), (0.0, 1.5707),
                (0.0, 1.4137), (0.0, 1.4137), (0.0, 1.4137), (0.0, 1.4137)
            ]

    def _get_finger_mappings(self):
        hands = ["left_", "right_"]
        fingers = ["Hand_Thumb3", "Hand_Thumb1", "Hand_Index2",
                   "Hand_Middle2", "Hand_Ring2", "Hand_Pinky2"]
        parents = ["Hand_Thumb2", "Hand_Thumb0", "Hand_Index1",
                   "Hand_Middle1", "Hand_Ring1", "Hand_Pinky1"]

        finger_list, parent_list = [], []
        for hand in hands:
            for f, p in zip(fingers, parents):
                finger_list.append(hand + f)
                parent_list.append(hand + p)
        return finger_list, parent_list

    def offset_reader(self, file_path="offset.json"):
        pkg_path = get_package_share_directory("lan_webrtc_hand")
        offset_path = os.path.join(pkg_path, "quest", file_path)

        try:
            with open(offset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.init_status = True
            return data[0], data[1]
        except Exception as e:
            self.get_logger().error(f"Failed to read offset file: {e}", once=True)
            return [0.0] * 12, [0.0] * 12

    def offset_loop(self):
        rate = self.create_rate(self.hz)
        while not self.init_status:
            close_angles, open_angles = self.offset_reader()
            if self.init_status:
                self.get_logger().info("Offset successfully read!")
                break
            else:
                self.get_logger().info("Offset not ready, retrying...", once=True)
            rate.sleep()

        self.get_logger().info(f"Offset data: {close_angles}, {open_angles}")

        offset = -np.array(close_angles)
        scale = []

        for i in range(len(self.joint_limits)):
            scale.append(
                (self.joint_limits[i][0] - self.joint_limits[i][1]) /
                (open_angles[i] - close_angles[i] if open_angles[i] != close_angles[i] else 1.0)
            )

        with self.lock:
            self.offset = offset
            self.scale = scale
            self.open_angles = open_angles
            self.close_angles = close_angles

    def recv_loop(self):
        rate = self.create_rate(self.hz)
        while rclpy.ok():
            angles = self._get_finger_angles()
            with self.lock:
                self.current_angles = angles
            rate.sleep()

    def _get_finger_angles(self):
        angles = []
        for i, (finger, parent) in enumerate(zip(self.finger_list, self.parent_list)):
            try:
                trans = self.tf_buffer.lookup_transform(parent, finger, rclpy.time.Time())
                quat = [trans.transform.rotation.x,
                        trans.transform.rotation.y,
                        trans.transform.rotation.z,
                        trans.transform.rotation.w]
                euler = tf_transformations.euler_from_quaternion(quat)
                rot = euler[0] if i in (1, 7) else euler[2]
                angles.append(rot)
            except Exception as e:
                self.get_logger().warn(f"TF lookup failed {i}, {finger}->{parent}: {str(e)}", once=True)
                return self.current_angles
        return angles

    def send_loop(self):
        rate = self.create_rate(self.hz)
        while rclpy.ok():
            if self.init_status:
                with self.lock:
                    self.retarget()
                    self.publish_mros_msg()
            rate.sleep()

    def retarget(self):
        if len(self.current_angles) != 12 or np.all(np.array(self.current_angles) == 0.0):
            self.retargeted_angles = [0.0] * 12
            return
        retarget_angles = (np.array(self.current_angles) + self.offset) * self.scale + self.upper_limit
        self.retargeted_angles = [
            np.clip(retarget_angles[i], self.lower_limit[i], self.upper_limit[i])
            for i in range(12)
        ]

    def angles_to_controller(self, angles):
        angle_range = np.array(self.upper_limit) - np.array(self.lower_limit)
        normalized = (np.asarray(angles) - self.lower_limit) / angle_range
        return np.clip(normalized * 100.0, 0.0, 100.0)

    def publish_mros_msg(self):
        try:
            if len(self.retargeted_angles) != 12:
                self.get_logger().warn("Published finger len mismatch")
                cmd = np.zeros(12)
            else:
                l_pose, r_pose = self.retargeted_angles[:6], self.retargeted_angles[6:]
                if self.device == "brainco1":
                    hand_angles = np.concatenate([l_pose, r_pose], dtype=float).tolist()
                    cmd = self.angles_to_controller(hand_angles)
                else:
                    cmd = np.concatenate([l_pose, r_pose], dtype=float)

            msg = JointState()
            msg.name = [
                "left_thumb", "left_thumb_aux", "left_index", "left_middle", "left_ring", "left_pinky",
                "right_thumb", "right_thumb_aux", "right_index", "right_middle", "right_ring", "right_pinky"
            ]
            msg.position = cmd.tolist()
            msg.velocity = [2.5367, 2.6180, 2.2689, 2.2689, 2.2689, 2.2689] * 2
            msg.effort = []
            self.finger_cmd_pub.publish(msg)

        except Exception as e:
            self.get_logger().warn(f"Finger Publish Error: {str(e)}")


def main():
    rclpy.init()
    node = HandTeleopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Node interrupted")
    except Exception as e:
        node.get_logger().error(f"Runtime error: {str(e)}")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
