#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""使用 TF 中的 Quest VR 手柄位姿遥控 RM65 机械臂。"""

import threading
import time
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.time import Time
import tf2_ros

from realman65.my_robot.realman_65_interface import Realman65Interface
from realman65.utils.data_handler import debug_print

from std_msgs.msg import String, Bool   # Bool 已不再需要时可移除
import json


ADJ_MAT = np.array([
    [-1, 0,  0, 0],
    [0, -1,  0, 0],
    [0, 0,  1, 0],
    [0, 0,  0, 1],
], dtype=np.float64)

# ADJ_MAT = np.array([
#     [1, 0,  0, 0],
#     [0, 1,  0, 0],
#     [0, 0,  1, 0],
#     [0, 0,  0, 1],
# ], dtype=np.float64)

# ==================== 全局共享状态 ====================

pose_lock = threading.Lock()
target_pose: Optional[np.ndarray] = None   # 目标末端位姿（6×1）
pos_offset: Optional[np.ndarray] = None    # retarget 位置偏移
rot_offset: Optional[R] = None             # retarget 姿态偏移


running = True
retarget_done = False




controller_lock = threading.Lock()
pending_gripper_cmd: Optional[int] = None
last_button_state = {"button_one": False, "index_trigger": False}



# ==================== ROS2 TF 客户端节点 ====================

class TFTeleopClient(rclpy.node.Node):
    """TF + 控制器 JSON 订阅节点。"""
    def __init__(self, node_name: str = "rm65_vr_tf_client") -> None:
        super().__init__(node_name)
        self.tf_buffer = tf2_ros.Buffer()
        # 不启用内部线程，由外部 executor 统一 spin
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self, spin_thread=False)
        self.create_subscription(String, "quest/controller_state", self._on_controller_state, 10)
        
    
    def _on_controller_state(self, msg: String) -> None:
        global pending_gripper_cmd, last_button_state
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warning("Invalid controller JSON payload")
            return

        left = payload.get("left", {})
        button_one_pressed = bool(left.get("button_one", False))
        index_val = float(left.get("index_trigger", 0.0))
        index_pressed = index_val > 0.8  # 阈值视情况调整

        with controller_lock:
            if button_one_pressed and not last_button_state["button_one"]:
                pending_gripper_cmd = 1  # 夹紧
            if index_pressed and not last_button_state["index_trigger"]:
                pending_gripper_cmd = 0  # 松开
            last_button_state["button_one"] = button_one_pressed
            last_button_state["index_trigger"] = index_pressed
            # debug_print("pending_gripper_cmd",pending_gripper_cmd,"INFO")
            


def euler_to_rot(euler_xyz: np.ndarray) -> R:
    return R.from_euler("xyz", euler_xyz, degrees=False)


def retarget_once(vr_pose: np.ndarray, robot_pose: np.ndarray) -> None:
    """根据当前 VR/机械臂姿态求一次偏移。"""
    global pos_offset, rot_offset, retarget_done
    vr_pos = vr_pose[:3]
    vr_rot = euler_to_rot(vr_pose[3:])

    rb_pos = robot_pose[:3]
    rb_rot = euler_to_rot(robot_pose[3:])

    pos_offset = rb_pos - vr_pos
    rot_offset = vr_rot.inv()  * rb_rot
    retarget_done = True
    debug_print("teleop", f"Retarget OK, pos_offset={pos_offset}", "INFO")


def apply_retarget(vr_pose: np.ndarray) -> np.ndarray:
    """把当前 VR 位姿映射为机械臂目标位姿。"""
    mapped_pos = vr_pose[:3] + pos_offset
    mapped_rot = (euler_to_rot(vr_pose[3:]) * rot_offset).as_euler("xyz")
    return np.concatenate([mapped_pos, mapped_rot])



def lookup_hand_pose(tf_client, parent_frame: str, child_frame: str):
    transform = tf_client.tf_buffer.lookup_transform(parent_frame, child_frame, Time())

    pos = transform.transform.translation
    quat = transform.transform.rotation
    rot = R.from_quat([quat.x, quat.y, quat.z, quat.w]).as_matrix()

    pose_mat = np.eye(4)
    pose_mat[:3, :3] = rot
    pose_mat[:3, 3] = [pos.x, pos.y, pos.z]

    pose_mat = ADJ_MAT @ pose_mat

    adj_pos = pose_mat[:3, 3]
    adj_rpy = R.from_matrix(pose_mat[:3, :3]).as_euler("xyz")

    return np.concatenate([adj_pos, adj_rpy])


# ==================== 线程函数 ====================

def vr_worker(tf_client: TFTeleopClient,
              rm_interface: Realman65Interface,
              hz: float = 60.0,
              parent_frame: str = "map",
              hand_frame: str = "l_hand") -> None:
    """读取 TF、执行 retarget，更新全局 target_pose。"""
    global target_pose, retarget_done
    period = 1.0 / hz

    while running:
        left_vr_pose = lookup_hand_pose(tf_client, parent_frame, hand_frame)
        if left_vr_pose is None:
            time.sleep(period)
            continue
        # debug_print("left_vr_pose",left_vr_pose,"INFO")
        with pose_lock:
            if not retarget_done:
                robot_pose_map = rm_interface.get_end_effector_pose()
                if robot_pose_map and "left_arm" in robot_pose_map:
                    robot_pose = np.asarray(robot_pose_map["left_arm"], dtype=np.float64)
                    retarget_once(left_vr_pose, robot_pose)
                else:
                    time.sleep(period)
                    continue

            if retarget_done:
                target_pose = apply_retarget(left_vr_pose)
                # debug_print("target_pose", target_pose, "INFO")

        time.sleep(period)


def arm_worker(rm_interface: Realman65Interface, hz: float = 30.0) -> None:
    """按照 target_pose 控制机械臂。"""
    global running
    global pending_gripper_cmd, target_pose
    period = 1.0 / hz
    while running:
        pose_to_send = None
        with pose_lock:
            if retarget_done and target_pose is not None:
                pose_to_send = target_pose.copy()

        if pose_to_send is not None:
            try:
                print(f"[DEBUG] 发送机械臂位姿：{pose_to_send}")
                # rm_interface.update(np.asarray(pose_to_send, dtype=np.float64))
                # time.sleep(0.00001)
            except Exception as exc:
                debug_print("teleop", f"send pose failed: {exc}", "WARNING")
        
        cmd = None
        with controller_lock:
            if pending_gripper_cmd is not None:
                cmd = pending_gripper_cmd
                pending_gripper_cmd = None

        if cmd is not None:
            debug_print("cmd",cmd,"INFO")
            try:
                rm_interface.set_gripper("left_arm", cmd)
            except Exception as exc:
                debug_print("teleop", f"gripper command failed: {exc}", "WARNING")
        
        time.sleep(period)


# ==================== 主入口 ====================

def main() -> None:
    global running

    rclpy.init()
    tf_client = TFTeleopClient()

    # 单线程 executor 驱动 TF 回调
    executor = SingleThreadedExecutor()
    executor.add_node(tf_client)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    debug_print("teleop", "Initializing RM65 interface...", "INFO")
    rm_interface = Realman65Interface(auto_setup=False)
    rm_interface.set_up()
    rm_interface.reset()  # 如需上电后先复位，可解除注释
    time.sleep(1)

    vr_thread = threading.Thread(
        target=vr_worker, args=(tf_client, rm_interface), daemon=True)
    arm_thread = threading.Thread(
        target=arm_worker, args=(rm_interface,), daemon=True)

    vr_thread.start()
    arm_thread.start()

    debug_print("teleop", "Teleop threads started. Press Ctrl+C to stop.", "INFO")

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        rm_interface.reset()
        debug_print("teleop", "User requested shutdown.", "INFO")
    finally:
        running = False
        vr_thread.join(timeout=2.0)
        arm_thread.join(timeout=2.0)

        executor.shutdown()
        tf_client.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=2.0)

        debug_print("teleop", "Teleop stopped.", "INFO")


if __name__ == "__main__":
    main()