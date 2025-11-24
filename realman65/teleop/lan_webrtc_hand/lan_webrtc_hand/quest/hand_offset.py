#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import tf2_ros
import tf_transformations
import threading
import json
from ament_index_python.packages import get_package_share_directory
import os


class HandOffset(Node):
    def __init__(self):
        super().__init__('hand_offset_reader')

        self.declare_parameter("device", "brainco2")
        self.device = self.get_parameter('device').value

        self.joint_limits = self._get_joint_limits(self.device)

        # self.joint_limits = [(0.0, 1.0297), (0.0, 1.5707), (0.0, 1.4137), (0.0, 1.4137), (0.0, 1.4137), (0.0, 1.4137),
        #                      (0.0, 1.0297), (0.0, 1.5707), (0.0, 1.4137), (0.0, 1.4137), (0.0, 1.4137), (0.0, 1.4137)]

        self.hand_names = ["left_", "right_"]
        finger_list = ["Hand_Thumb3", "Hand_Thumb1", "Hand_Index2", "Hand_Middle2", "Hand_Ring2", "Hand_Pinky2"]
        parent_list = ["Hand_Thumb2", "Hand_Thumb0", "Hand_Index1", "Hand_Middle1", "Hand_Ring1", "Hand_Pinky1"]

        self.finger_list = []
        self.parent_list = []
        for hand in self.hand_names:
            for finger, parent in zip(finger_list, parent_list):
                self.finger_list.append(hand + finger)
                self.parent_list.append(hand + parent)

        pkg_path = get_package_share_directory("lan_webrtc_hand")
        self.offset_path = os.path.join(pkg_path, "quest/offset.json")
        os.makedirs(os.path.dirname(self.offset_path), exist_ok=True)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.init_rate = self.create_rate(1 / 3)

        self.thread_init = threading.Thread(target=self.initialize)
        self.thread_init.start()
        
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
        
    def _get_finger_angles(self):
        angles = []
        for i, (finger, parent) in enumerate(zip(self.finger_list, self.parent_list)):
            try:
                trans = self.tf_buffer.lookup_transform(
                    parent, finger, rclpy.time.Time())
                trans_quat = [
                    trans.transform.rotation.x,
                    trans.transform.rotation.y,
                    trans.transform.rotation.z,
                    trans.transform.rotation.w
                ]
                trans_euler = tf_transformations.euler_from_quaternion(trans_quat)

                if i == 1 or i == 7:
                    rot = trans_euler[0]
                else:
                    rot = trans_euler[2]
                angles.append(rot)
            except Exception as e:
                self.get_logger().warn(
                    f"Cannot get transformation {i}, {finger}, {parent}: {str(e)}")
                return []
        return angles

    def initialize(self):
        close_angles = []
        open_angles = []

        while len(close_angles) != 12 or len(open_angles) != 12:
            self.get_logger().info("Close your hand for 3 seconds...")
            self.init_rate.sleep()
            close_angles = self._get_finger_angles()

            self.get_logger().info("Open your hand for 3 seconds...")
            self.init_rate.sleep()
            open_angles = self._get_finger_angles()

        offset = [close_angles, open_angles]
        with open(self.offset_path, 'w', encoding='utf-8') as f:
            json.dump(offset, f, indent=4)

        self.get_logger().info("Initialization complete! Saving offsets and shutting down...")

        def shutdown():
            self.destroy_node()
            rclpy.shutdown()

        self.executor.create_task(shutdown)


def main():
    rclpy.init()
    node = HandOffset()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Node interrupted")
    except Exception as e:
        node.get_logger().error(f"Node runtime error: {str(e)}")
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
