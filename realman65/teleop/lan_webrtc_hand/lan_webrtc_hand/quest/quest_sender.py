#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unity 双目接收端的 Python 发送程序：
- 发布两路视频（左/右目），名称可配置（默认 left/right）
- 可选创建 DataChannel 订阅者，接收 Unity 的手柄数据并打印

要求：先启动 signaling_server.py，然后运行本程序；Unity 端运行 WebRTCStreamer.cs（作为视频订阅 + 数据发布）
/usr/bin/python3 unity_dual_sender.py --server ws://127.0.0.1:8000 --room quest3_room
"""
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
import tf_transformations
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Float32MultiArray, Bool
import argparse
import asyncio
import logging
import signal
import threading
import time
from typing import Optional
import json
import cv2
import numpy as np
import copy

from lan_webrtc_hand.server.video_publisher import create_room_publisher_with_track, CallbackVideoTrack
from lan_webrtc_hand.server.data_subscriber import DataSubscriber
from lan_webrtc_hand.utils.transform import Transform
from lan_webrtc_hand.utils.camera import CameraCapture
from lan_webrtc_hand.utils.headset import HeadsetData
from arm_teleoppy.teleop_srv.teleop_client import TeleopClient

import tf2_ros

from scipy.spatial.transform import Rotation as R

from sensor_msgs.msg import Joy

received_msg = None


async def run(server: str, room: str, camera: int, width: int, height: int, fps: int,
              left_name: str, right_name: str, subscribe_data: bool, verbose: bool):
    if verbose:
        logging.getLogger("webrtc_client").setLevel(logging.DEBUG)
        logging.getLogger("video_publisher").setLevel(logging.DEBUG)
        logging.getLogger("data_subscriber").setLevel(logging.DEBUG)

    cap = CameraCapture(camera, width, height, fps)
    cap.start()

    left_track = CallbackVideoTrack(
        cap.get_left, width=width, height=height, fps=fps)
    right_track = CallbackVideoTrack(
        cap.get_right, width=width, height=height, fps=fps)

    # 启动两个发布者（名称与 Unity 默认匹配）
    left_client = await create_room_publisher_with_track(
        server_url=server, room_id=room, display_name=left_name, track=left_track
    )
    right_client = await create_room_publisher_with_track(
        server_url=server, room_id=room, display_name=right_name, track=right_track
    )

    # 可选：数据订阅者，接收 Unity 的手柄数据
    data_sub: Optional[DataSubscriber] = None
    if subscribe_data:
        def on_msg(pub_id: str, msg: str):
            global received_msg
            decoded_msg = msg.decode('utf-8')  # 先解码为字符串
            received_msg = json.loads(decoded_msg)  # 解析为 JSON 对象

        data_sub = DataSubscriber(
            server_url=server,
            room_id=room,
            name="PyDataSub",
            data_channel_label="data",
            data_channel_options={"ordered": True, "protocol": ""},
            on_message_callback=on_msg,
        )
        await data_sub.connect()

    stop_event = asyncio.Event()

    def _on_sigint(*_):
        try:
            stop_event.set()
        except Exception:
            pass

    loop = asyncio.get_event_loop()
    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(s, _on_sigint)
        except NotImplementedError:
            pass

    await stop_event.wait()

    # 清理
    if data_sub:
        await data_sub.close()
    await left_client.disconnect()
    await right_client.disconnect()
    await asyncio.sleep(0.2)
    cap.stop()


class TFManager(Node):
    def __init__(self):
        super().__init__("vr_unity_node")

        self.declare_parameter('control_type', "upper_body")
        self.control_type = self.get_parameter('control_type').value

        self.declare_parameter('device', "brainco2")
        self.device = self.get_parameter('device').value

        self.broadcaster = TransformBroadcaster(self)
        self.lock = threading.Lock()

        self.prev_l_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        self.prev_r_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

        self.last_A_status = self.last_B_status = False
        self.last_X_status = self.last_Y_status = False
        self.last_l_button_thumbstick = False

        self._is_pub = False
        self._is_run = False
        self.l_pause = False
        self.r_pause = False

        self.l_fingers_pos = np.array([{'x': 0.0, 'y': 0.0, 'z': 0.0} for _ in range(20)])
        self.l_fingers_quat = np.array([{'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0} for _ in range(20)])
        self.r_fingers_pos = np.array([{'x': 0.0, 'y': 0.0, 'z': 0.0} for _ in range(20)])
        self.r_fingers_quat = np.array([{'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0} for _ in range(20)])

        self.twist_mode = 1

        self.prev_headset_data = HeadsetData()
        self.curr_headset_data = HeadsetData()

        self.T_map_l_hand = np.eye(4)
        self.T_root_l_hand = np.eye(4)
        self.T_map_r_hand = np.eye(4)
        self.T_root_r_hand = np.eye(4)

        self.wrpy = np.zeros(3)
        self.vxyz = np.zeros(3)
        self.twist = np.zeros(3)

        self.trig_pub = self.create_publisher(Float32MultiArray, '/device/trigger', 2)
        self.twist_pub = self.create_publisher(TwistStamped, '/desire/twist', 2)
        self.button_pub = self.create_publisher(Joy, '/device/buttons', 2)
        self.tele_cli = TeleopClient(self, '/srv/teleop')

        #Tiga
        self.save_frame2buffer_flag = False
        # 定义 Publisher，发布 Bool 类型的消息，话题名可以自定义
        self.flag_publisher = self.create_publisher(Bool, "save_frame2buffer_flag", 10)
        # 创建一个定时器，每隔 0.1 秒发布一次
        self.timer = self.create_timer(0.1, self.publish_flag)

        self.transform = Transform()

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.hz = 60
        self.update_thread = threading.Thread(target=self.run, daemon=True)
        self.publish_thread = threading.Thread(target=self.pub, daemon=True)
        self.update_thread.start()
        self.publish_thread.start()

    def publish_flag(self):
        msg = Bool()
        msg.data = self.save_frame2buffer_flag
        self.flag_publisher.publish(msg)

    def run(self):
        global received_msg
        rate = self.create_rate(self.hz)
        while rclpy.ok():
            if received_msg is not None:
                self.update_controller_state()
            rate.sleep()

    def pub(self):
        rate = self.create_rate(self.hz)
        while rclpy.ok():
            if self._is_pub:
                self.publish_headset_data()
                self.publish_controller_data()
                self.publish_buttons()
                if self.device != "claw":
                    self.publish_finger_data()
            rate.sleep()

    def update_controller_state(self):
        global received_msg
        data = received_msg.copy()

        curr_headset_data = HeadsetData()

        h_matrix = np.reshape(data["eyePose"], (4, 4))
        l_matrix = np.reshape(data["l"], (4, 4))
        r_matrix = np.reshape(data["r"], (4, 4))

        h_pose = self._matrix_to_pose(h_matrix)
        l_pose = self._matrix_to_pose(l_matrix)
        r_pose = self._matrix_to_pose(r_matrix)

        l_thumbstick_x, l_thumbstick_y = data["leftJS"]
        l_index_trigger = data["leftTrig"]
        l_hand_trigger = data["leftGrip"]
        button_X, button_Y = data["X"], data["Y"]
        l_button_thumbstick = data["LThU"]

        r_thumbstick_x, r_thumbstick_y = data["rightJS"]
        r_index_trigger = data["rightTrig"]
        r_hand_trigger = data["rightGrip"]
        button_A, button_B = data["A"], data["B"]
        r_button_thumbstick = data["RThU"]

        curr_headset_data.timestamp = data["timestamp"]
        curr_headset_data.poses['head'] = h_pose
        curr_headset_data.poses['left'] = l_pose
        curr_headset_data.poses['right'] = r_pose

        curr_headset_data.controllers['left'] = {
            'thumbstick_x': l_thumbstick_x,
            'thumbstick_y': l_thumbstick_y,
            'index_trigger': l_index_trigger,
            'hand_trigger': l_hand_trigger,
            'button_one': button_X,
            'button_two': button_Y,
            'button_thumbstick': l_button_thumbstick,
        }

        curr_headset_data.controllers['right'] = {
            'thumbstick_x': r_thumbstick_x,
            'thumbstick_y': r_thumbstick_y,
            'index_trigger': r_index_trigger,
            'hand_trigger': r_hand_trigger,
            'button_one': button_A,
            'button_two': button_B,
            'button_thumbstick': r_button_thumbstick,
        }

        self.l_fingers_pos = data['LeftFingerPositions']
        self.l_fingers_quat = data['LeftFingerRotations']
        self.r_fingers_pos = data['RightFingerPositions']
        self.r_fingers_quat = data['RightFingerRotations']

        self._handle_buttons(button_A, button_X, button_Y)

        with self.lock:
            self.curr_headset_data = curr_headset_data

    def _handle_buttons(self, button_A, button_X, button_Y):
        if button_A and not self.last_A_status:
            self.last_A_status = True
            if self.tele_cli:
                self.tele_cli.reset()
                print("call Reset")
                self._is_pub, self._is_run = True, False

        if button_X and not self.last_X_status:
            self.last_X_status = True
            if self.tele_cli:
                self.tele_cli.run()
                print("call Run")
                self._is_run = True

        if button_Y and not self.last_Y_status:
            self.last_Y_status = True
            if self.tele_cli:
                self.tele_cli.pause()
                print("call Pause")
                self._is_run = False

        if not button_X:
            self.last_X_status = False
        if not button_Y:
            self.last_Y_status = False
        if not button_A:
            self.last_A_status = False

    def _diff_vel(self):
        prev_h = self.prev_headset_data.poses['head']
        curr_h = self.curr_headset_data.poses['head']

        prev_h_pos, curr_h_pos = prev_h[:3], curr_h[:3]
        prev_h_quat, curr_h_quat = prev_h[3:], curr_h[3:]

        dt = float(self.curr_headset_data.timestamp - self.prev_headset_data.timestamp) * 0.001
        if dt <= 0:
            dt = 0.001

        alpha = 0.99
        rpy_prev = R.from_quat(prev_h_quat).as_euler("xyz")
        rpy_curr = R.from_quat(curr_h_quat).as_euler("xyz")

        self.wrpy = np.clip(self.wrpy, -4.0 * np.pi, 4.0 * np.pi)
        self.vxyz = np.clip(self.vxyz, -10.0, 10.0)

        rm_inv = R.from_euler('z', rpy_curr[2]).as_matrix().T
        self.vxyz = rm_inv.dot(self.vxyz)

        self.wrpy = np.where(np.abs(self.wrpy) < 0.1, 0, self.wrpy)
        self.vxyz = np.where(np.abs(self.vxyz) < 0.1, 0, self.vxyz)

        self.wrpy = alpha * ((rpy_curr - rpy_prev) / dt) + (1 - alpha) * self.wrpy
        self.vxyz = alpha * ((curr_h_pos - prev_h_pos) / dt) + (1 - alpha) * self.vxyz

        self.prev_headset_data = copy.deepcopy(self.curr_headset_data)

    def _twist_mode_switch(self, l_button_thumbstick):
        if l_button_thumbstick and l_button_thumbstick is not self.last_l_button_thumbstick:
            self.twist_mode = 1 - self.twist_mode
            print(f"Current twist mode: {self.twist_mode}", flush=True)

    def _create_transform(self, parent_frame: str, child_frame: str, position, rotation):
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = parent_frame
        transform.child_frame_id = child_frame

        transform.transform.translation.x = float(position[0])
        transform.transform.translation.y = float(position[1])
        transform.transform.translation.z = float(position[2])

        transform.transform.rotation.x = float(rotation[0])
        transform.transform.rotation.y = float(rotation[1])
        transform.transform.rotation.z = float(rotation[2])
        transform.transform.rotation.w = float(rotation[3])

        return transform

    def _pose_to_matrix(self, pos, quat):
        T = np.eye(4)
        T[:3, :3] = R.from_quat(quat).as_matrix()
        T[:3, 3] = pos
        return T

    def _matrix_to_pose(self, T):
        pos = T[:3, 3]
        quat = R.from_matrix(T[:3, :3]).as_quat()
        return np.hstack([pos, quat])

    def publish_headset_data(self):

        h_pose = self.curr_headset_data.poses['head']
        l_pose = self.curr_headset_data.poses['left']
        r_pose = self.curr_headset_data.poses['right']

        root_pos = np.array([0.0, 0.0, 1.0]) + h_pose[:3]
        h_pos = np.array([0.0, 0.0, 0.6])

        if self.control_type == "whole_body":

            if self.twist_mode == 1:
                h_rpy = R.from_quat(h_pose[3:]).as_euler('xyz')
                root_quat = R.from_euler('xyz', [0, 0, h_rpy[2]]).as_quat()
                h_quat = R.from_euler('xyz', [h_rpy[0], h_rpy[1], 0]).as_quat()

            else:
                root_quat = np.array([0, 0, 0, 1])
                h_quat = h_pose[3:]

            root_pose = np.hstack([root_pos, root_quat])
            h_pose = np.hstack([h_pos, h_quat])

        else:
            h_pose = np.hstack([h_pos, h_pose[3:]])

            root_quat = np.array([0.0, 0.0, 0.0, 1.0])
            root_pose = np.hstack([root_pos, root_quat])

        root_matrix = self._pose_to_matrix(root_pos, root_quat)

        # determine whether l/r hand is out of sight
        if np.any(l_pose[:3] != 0):
            if self.l_pause:
                self.l_pause = False

            self.T_map_l_hand = self._pose_to_matrix(l_pose[:3], l_pose[3:])
            self.T_root_l_hand = np.linalg.inv(root_matrix) @ self.T_map_l_hand
        else:
            self.l_pause = True

            if self.control_type == "whole_body":
                T_map_hand_new = root_matrix @ self.T_root_l_hand
                l_pose = self._matrix_to_pose(T_map_hand_new)
            else:
                l_pose = self.prev_l_pose.copy()

        if np.any(r_pose[:3] != 0):
            if self.r_pause:
                self.r_pause = False

            self.T_map_r_hand = self._pose_to_matrix(r_pose[:3], r_pose[3:])
            self.T_root_r_hand = np.linalg.inv(root_matrix) @ self.T_map_r_hand
        else:
            if self.lock:
                r_pose = self.prev_r_pose.copy()
            self.r_pause = True

            if self.control_type == "whole_body":
                T_map_hand_new = root_matrix @ self.T_root_r_hand
                r_pose = self._matrix_to_pose(T_map_hand_new)
            else:
                r_pose = self.prev_r_pose.copy()

        tf_l = self._create_transform(
            parent_frame="map",
            child_frame="l_hand",
            position=l_pose[:3] + np.array([0.0, 0.0, 1.0]),
            rotation=l_pose[3:]
        )

        tf_r = self._create_transform(
            parent_frame="map",
            child_frame="r_hand",
            position=r_pose[:3] + np.array([0.0, 0.0, 1.0]),
            rotation=r_pose[3:]
        )

        t_torso = self._create_transform(
            parent_frame="map",
            child_frame="root",
            position=root_pose[:3],
            rotation=root_pose[3:]
        )

        t_head = self._create_transform(
            parent_frame="root",
            child_frame="head",
            position=h_pose[:3],
            rotation=h_pose[3:]
        )

        if self._is_pub:
            self.broadcaster.sendTransform([tf_l, tf_r, t_torso, t_head])

        if self.lock:
            self.prev_l_pose = l_pose.copy()
            self.prev_r_pose = r_pose.copy()

    def publish_buttons(self):
        joy_msg = Joy()

        joy_msg.axes = [
            self.curr_headset_data.controllers['left']['thumbstick_x'],
            self.curr_headset_data.controllers['left']['thumbstick_y'],
            self.curr_headset_data.controllers['left']['index_trigger'],
            self.curr_headset_data.controllers['left']['hand_trigger'],

            self.curr_headset_data.controllers['right']['thumbstick_x'],
            self.curr_headset_data.controllers['right']['thumbstick_y'],
            self.curr_headset_data.controllers['right']['index_trigger'],
            self.curr_headset_data.controllers['right']['hand_trigger'],
        ]

        joy_msg.buttons = [
            self.curr_headset_data.controllers['left']['button_one'],
            self.curr_headset_data.controllers['left']['button_two'],
            self.curr_headset_data.controllers['left']['button_thumbstick'],

            self.curr_headset_data.controllers['right']['button_one'],
            self.curr_headset_data.controllers['right']['button_two'],
            self.curr_headset_data.controllers['right']['button_thumbstick'],
        ]

        self.button_pub.publish(joy_msg)

    def publish_controller_data(self):
        l_controller = self.curr_headset_data.controllers['left']
        r_controller = self.curr_headset_data.controllers['right']
        l_button_thumbstick = l_controller['button_thumbstick']

        if self.control_type == "whole_body":

            self._twist_mode_switch(l_button_thumbstick)
            self.last_l_button_thumbstick = l_button_thumbstick

            if self.twist_mode == 1:
                self._diff_vel()
                self.twist = [self.vxyz[0] * 1.3, self.vxyz[1] * 1.3, self.wrpy[2] * 1.3]
                if not self._is_run:
                    self.twist = np.zeros(3)
            else:
                self.twist = [l_controller["thumbstick_y"], -l_controller["thumbstick_x"], -r_controller["thumbstick_x"]]

        if self.control_type == "whole_body" and self._is_pub:
            twist_msg = TwistStamped()
            twist_msg.header.stamp = self.get_clock().now().to_msg()
            twist_msg.twist.linear.x, twist_msg.twist.linear.y = self.twist[:2]
            twist_msg.twist.angular.z = self.twist[2]
            self.twist_pub.publish(twist_msg)

        if self.device == "claw" and self._is_pub:
            trig_msg = Float32MultiArray(data=[l_controller['index_trigger'], r_controller['index_trigger']])
            self.trig_pub.publish(trig_msg)

    def publish_finger_data(self):
        left_transforms = self.publish_left_fingers_tf()
        right_transforms = self.publish_right_fingers_tf()
        if self._is_pub:
            self.broadcaster.sendTransform(left_transforms + right_transforms)

    def publish_left_fingers_tf(self):
        return self._publish_fingers(self.l_fingers_pos, self.l_fingers_quat, "left")

    def publish_right_fingers_tf(self):
        return self._publish_fingers(self.r_fingers_pos, self.r_fingers_quat, "right")

    def _publish_fingers(self, fingers_pos, fingers_quat, prefix):
        transforms = []
        finger_list = [
            "Hand_Thumb0", "Hand_Thumb1", "Hand_Thumb2", "Hand_Thumb3",
            "Hand_Index1", "Hand_Index2", "Hand_Index3",
            "Hand_Middle1", "Hand_Middle2", "Hand_Middle3",
            "Hand_Ring1", "Hand_Ring2", "Hand_Ring3",
            "Hand_Pinky0", "Hand_Pinky1", "Hand_Pinky2", "Hand_Pinky3",
            "Hand_WristRoot", "Hand_ForearmStub", "Hand_Start"
        ]

        for i, name in enumerate(finger_list):

            pos = np.array([fingers_pos[i]['x'], fingers_pos[i]['y'], fingers_pos[i]['z']], dtype=float)
            quat = np.array([fingers_quat[i]['x'], fingers_quat[i]['y'], fingers_quat[i]['z'], fingers_quat[i]['w']], dtype=float)

            transform = self._create_transform("map", f"{prefix}_{name}", pos + np.array([0.0, 0.0, 1.0]), quat)
            transforms.append(transform)

        return transforms


def main():
    rclpy.init()

    node = TFManager()

    node.declare_parameter('server', 'ws://127.0.0.1:8000')
    node.declare_parameter('room', '10.192.1.3')
    node.declare_parameter('camera', 10)
    node.declare_parameter('width', 1920)
    node.declare_parameter('height', 1080)
    node.declare_parameter('fps', 30)
    node.declare_parameter('left_name', 'left')
    node.declare_parameter('right_name', 'right')
    node.declare_parameter('no_data_sub', False)
    node.declare_parameter('verbose', False)

    server = node.get_parameter('server').value
    room = node.get_parameter('room').value
    camera = node.get_parameter('camera').value
    width = node.get_parameter('width').value
    height = node.get_parameter('height').value
    fps = node.get_parameter('fps').value
    left_name = node.get_parameter('left_name').value
    right_name = node.get_parameter('right_name').value
    subscribe_data = not node.get_parameter('no_data_sub').value
    verbose = node.get_parameter('verbose').value

    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")

    ros2_thread = threading.Thread(target=rclpy.spin, args=(node,))
    ros2_thread.start()

    try:
        asyncio.run(run(server=server,
                        room=room,
                        camera=camera,
                        width=width,
                        height=height,
                        fps=fps,
                        left_name=left_name,
                        right_name=right_name,
                        subscribe_data=subscribe_data,
                        verbose=verbose))
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
