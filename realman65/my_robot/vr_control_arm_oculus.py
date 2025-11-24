#!/usr/bin/env python
# -*- coding: utf-8 -*-
import threading
import time
import numpy as np
from scipy.spatial.transform import Rotation as R

from realman65.my_robot.realman_65_interface import Realman65Interface
from realman65.sensor.Quest_sensor import QuestSensor
from realman65.utils.data_handler import debug_print, matrix_to_xyz_rpy
from realman65.third_party import Realman_IK

import sys

import time

import numpy as np

from realman65.utils.data_handler import matrix_to_xyz_rpy, compute_local_delta_pose, debug_print, euler_to_matrix, compute_rotate_matrix

from scipy.spatial.transform import Rotation as R
from typing import Callable, Optional

from oculus_reader import OculusReader

# 固定 Quest → 机械臂的左乘矩阵
ADJ_MAT = np.array([
    [0, 0, -1, 0],
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
], dtype=float)

# ADJ_MAT = np.array([
#     [1, 0, 0, 0],  # 机械臂 x <- VR x
#     [0, 0, 1, 0],  # 机械臂 y <- VR z
#     [0, 1, 0, 0],  # 机械臂 z <- VR y
#     [0, 0, 0, 1],
# ], dtype=float)

# -------------------- 全局状态 --------------------
pose_lock = threading.Lock()
target_pose = None
pos_offset = None
rot_offset = None

running = True
retarget_done = False

# -------------------- 工具函数 --------------------
def euler_to_rot(euler_xyz):
    return R.from_euler('xyz', euler_xyz, degrees=False)

def retarget_once(vr_pose, robot_pose):
    global pos_offset, rot_offset, retarget_done
    vr_pos = np.array(vr_pose[:3], dtype=float)
    vr_rot = euler_to_rot(vr_pose[3:])

    rb_pos = np.array(robot_pose[:3], dtype=float)
    rb_rot = euler_to_rot(robot_pose[3:])

    pos_offset = rb_pos - vr_pos
    rot_offset = rb_rot * vr_rot.inv()     # rot_offset * VR = Robot
    retarget_done = True
    debug_print("teleop", f"Retarget: pos_offset={pos_offset}", "INFO")

def apply_retarget(vr_pose):
    vr_pos = np.array(vr_pose[:3], dtype=float)
    vr_rot = euler_to_rot(vr_pose[3:])
    mapped_pos = vr_pos + pos_offset
    mapped_rot = (rot_offset * vr_rot).as_euler('xyz')
    return np.concatenate([mapped_pos, mapped_rot])

def read_left_vr_pose(quest_sensor):
    """读取原始左手柄齐次矩阵 -> 左乘 ADJ_MAT -> 转成 xyz+rpy"""
    transformations, _ = quest_sensor.sensor.get_transformations_and_buttons()
    if not transformations or 'l' not in transformations:
        return None
    tf_raw = np.asarray(transformations['l'], dtype=float)
    # debug_print("queset_vr left",tf_raw,"INFO")
    if tf_raw.shape != (4, 4):
        return None
    tf_adj = ADJ_MAT @ tf_raw
    return matrix_to_xyz_rpy(tf_adj)

# -------------------- VR 线程 --------------------
def vr_worker(quest_sensor, rm_interface, hz=60.0):
    global target_pose
    period = 1.0 / hz
    while running:
        try:
            left_vr_pose = read_left_vr_pose(quest_sensor)
        except Exception as exc:
            debug_print("teleop", f"read VR failed: {exc}", "WARNING")
            time.sleep(0.1)
            continue

        if left_vr_pose is None:
            time.sleep(period)
            continue

        with pose_lock:
            if not retarget_done:
                robot_pose_map = rm_interface.get_end_effector_pose()
                debug_print("robot_pose_map",robot_pose_map,"INFO")
                if robot_pose_map and "left_arm" in robot_pose_map:
                    retarget_once(left_vr_pose, robot_pose_map["left_arm"])
            if retarget_done:
                target_pose = apply_retarget(left_vr_pose)
                debug_print("target_pose",target_pose,"INFO")
                

        time.sleep(period)

# -------------------- 机械臂线程 --------------------
def arm_worker(rm_interface, hz=30.0):
    global running
    period = 1.0 / hz
    while running:
        pose_to_send = None
        with pose_lock:
            if target_pose is not None and retarget_done:
                pose_to_send = target_pose.copy().tolist()
        if pose_to_send is not None:
            try:
                # temp_pose = np.array([-0.43842, 0.014762, 0.2402192, -1.46355, -1.15518, 1.3611],dtype=float).astype(np.float32)
                rm_interface.set_end_effector_pose('left_arm', pose_to_send)
                # rm_interface.set_end_effector_pose('left_arm',temp_pose )
            except Exception as exc:
                debug_print("teleop", f"send pose failed: {exc}", "WARNING")
        time.sleep(period)

# -------------------- 主入口 --------------------
def main():
    global running
    debug_print("teleop", "Initializing RM65 interface...", "INFO")
    rm_interface = Realman65Interface(auto_setup=False)
    rm_interface.set_up()
    rm_interface.reset()
    

    quest_sensor = QuestSensor("quest_vr")
    quest_sensor.set_up()  # 只初始化 OculusReader

    vr_thread = threading.Thread(target=vr_worker, args=(quest_sensor, rm_interface), daemon=True)
    arm_thread = threading.Thread(target=arm_worker, args=(rm_interface,), daemon=True)
    vr_thread.start()
    arm_thread.start()

    debug_print("teleop", "Teleop threads started. Press Ctrl+C to stop.", "INFO")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        debug_print("teleop", "User requested shutdown.", "INFO")
        rm_interface.reset()
    finally:
        running = False
        vr_thread.join(timeout=2.0)
        arm_thread.join(timeout=2.0)
        debug_print("teleop", "Teleop stopped.", "INFO")

if __name__ == "__main__":
    main()