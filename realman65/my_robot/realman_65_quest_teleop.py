#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
                        RM65 QuestVR 遥操系统
================================================================================
作者: Kris_Young
创建时间: 2025/10
版本: v1.0

功能描述:
    支持双臂RM65机械臂的QuestVR遥操作，包括位姿控制和夹爪控制

系统架构:
    - 继承 my_robot/base_robot.py 基础机器人框架
    - 组合 controller/Realman_controller.py 机械臂控制器  
    - 组合 sensor/Quest_sensor.py VR传感器

主要特性:
    - 通过配置字典灵活控制设备启用状态
    - 支持按需组合使用（单臂/双臂、有无摄像头等）
    - 高频实时控制，支持50Hz遥操频率
    - 完整的数据采集和日志记录

配置示例:
    - 仅右臂遥操: DEVICE_CONFIG['arms']['left_arm'] = False
    - 禁用摄像头: DEVICE_CONFIG['cameras'] = {'head': False, 'left_wrist': False, 'right_wrist': False}
    - 禁用夹爪: DEVICE_CONFIG['gripper'] = False
    - 修改IP地址: RM65_CONFIG['left_arm_ip'] = "192.168.1.20"
    - 更换摄像头: CAMERA_CONFIG['head'] = "123456789"
================================================================================
"""

import os
import sys
sys.path.append("./")

from realman65.utils.data_handler import debug_print, matrix_to_xyz_rpy, apply_local_delta_pose
from realman65.sensor.Realsense_sensor import RealsenseSensor
from pathlib import Path
import json
from datetime import datetime
import csv
import time
import numpy as np
from Robotic_Arm.rm_robot_interface import rm_thread_mode_e
from sensor.Quest_sensor import QuestSensor
from controller.Realman_controller import RealmanController
from my_robot.base_robot import Robot


# 获取当前脚本所在目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 构造third_party/oculus_reader的绝对路径（根据你的目录结构）
oculus_reader_path = os.path.join(current_dir, "../third_party/oculus_reader")
# 将路径添加到Python搜索路径
sys.path.append(oculus_reader_path)

# 基础机器人类 - 提供统一的机器人控制框架

# RM65控制器 - 直接IP控制，无需ROS，支持高频透传
# 摄像头传感器 - 采集视觉数据
# QuestVR传感器 - 获取VR手柄位姿数据
# 数据处理工具 - 坐标变换和姿态计算

# RM65线程模式枚举

# ================================ 配置参数 ================================
# 遥操系统配置 - 统一管理所有硬件参数，提高可配置性和可维护性
# 使用方法: 根据实际硬件环境修改对应配置项

# 机械臂网络配置 - 调用 controller/Realman_controller.py -> set_up(rm_ip)
RM65_CONFIG = {
    'left_arm_ip': "192.168.2.19",    # 左臂机械臂IP地址，需与实际设备IP匹配
    'right_arm_ip': "192.168.1.18",   # 右臂机械臂IP地址，需与实际设备IP匹配
    'thread_mode': rm_thread_mode_e.RM_TRIPLE_MODE_E,  # 三线程模式，提升响应性能
    'control_freq': 0.014285,             # 控制周期70Hz
}

# 摄像头设备配置 - 调用 sensor/Realsense_sensor.py -> set_up(serial_number)
# 获取方法: 运行 python tools/realsense_serial.py 查看连接的设备序列号
CAMERA_CONFIG = {
    'head': '111',          # 头部摄像头序列号 - 请替换为实际序列号
    'left_wrist': '111',    # 左腕摄像头序列号 - 请替换为实际序列号
    'right_wrist': '111',   # 右腕摄像头序列号 - 请替换为实际序列号
    'is_depth': False,      # 是否启用深度数据采集
}

# RM65初始关节角度配置（度制）- 安全起始位置
# 调用 controller/Realman_controller.py -> reset(start_angles)
JOINT_CONFIG = {
    'left_start_position': [0, 47, 102, 7, -62, -160],  # 左臂6自由度安全位置
    # 右臂6自由度安全位置(与左臂YZ面镜像的角度)
    'right_start_position': [-78, 56, 86, 11, -61, -40],
}

# 设备启用控制配置 - 灵活控制各设备模块的启用状态
# 支持按需配置，如仅右臂遥操、禁用摄像头等场景
DEVICE_CONFIG = {
    'arms': {
        'left_arm': True,       # 是否启用左臂控制
        'right_arm': False,      # 是否启用右臂控制
    },
    'cameras': {
        'head': False,           # 是否启用头部摄像头
        'left_wrist': False,     # 是否启用左腕摄像头
        'right_wrist': False,    # 是否启用右腕摄像头
    },
    'gripper': False,            # 是否启用夹爪控制
    'quest_vr': True,           # 是否启用QuestVR传感器
}

# 数据采集配置 - 调用 my_robot/base_robot.py -> set_collect_type()
COLLECT_CONFIG = {
    "arm": ["joint", "qpos", "gripper"],        # 机械臂数据类型
    "image": ["color"],                          # 图像数据类型
    "teleop": ["end_pose", "extra", "raw_pose"],  # 遥操数据类型
}
# ========================================================================


class RM65QuestRobot(Robot):
    def __init__(self, start_episode=0):
        """
        初始化RM65 QuestVR遥操机器人
        根据DEVICE_CONFIG配置灵活创建启用的设备组件
        调用: my_robot/base_robot.py -> Robot.__init__()
        """
        super().__init__(start_episode)

        # 根据配置动态创建控制器 - controller/Realman_controller.py
        self.controllers = {"arm": {}}
        self.last_teleop_log_path = None
        if DEVICE_CONFIG['arms']['left_arm']:
            self.controllers["arm"]["left_arm"] = RealmanController("left_arm")
        if DEVICE_CONFIG['arms']['right_arm']:
            self.controllers["arm"]["right_arm"] = RealmanController(
                "right_arm")

        # 根据配置动态创建传感器
        self.sensors = {}

        # 摄像头传感器 - sensor/Realsense_sensor.py
        if any(DEVICE_CONFIG['cameras'].values()):
            self.sensors["image"] = {}
            if DEVICE_CONFIG['cameras']['head']:
                self.sensors["image"]["cam_head"] = RealsenseSensor("cam_head")
            if DEVICE_CONFIG['cameras']['left_wrist']:
                self.sensors["image"]["cam_left_wrist"] = RealsenseSensor(
                    "cam_left_wrist")
            if DEVICE_CONFIG['cameras']['right_wrist']:
                self.sensors["image"]["cam_right_wrist"] = RealsenseSensor(
                    "cam_right_wrist")

        # QuestVR传感器 - sensor/Quest_sensor.py
        if DEVICE_CONFIG['quest_vr']:
            if "teleop" not in self.sensors:
                self.sensors["teleop"] = {}
            self.sensors["teleop"]["quest_vr"] = QuestSensor("quest_vr")

    def set_up(self):
        """
        根据配置初始化所有启用的组件
        调用链路: 根据配置动态调用各组件的set_up()方法
        """
        # 设置RM65机械臂控制器 - 直接TCP连接，启用高频模式
        # 调用 controller/Realman_controller.py -> set_up(rm_ip, thread_mode, dT)
        if DEVICE_CONFIG['arms']['left_arm']:
            self.controllers["arm"]["left_arm"].set_up(
                rm_ip=RM65_CONFIG['left_arm_ip'],
                thread_mode=RM65_CONFIG['thread_mode'],
                dT=RM65_CONFIG['control_freq']
            )
            debug_print(
                "robot", f"Left arm initialized at {RM65_CONFIG['left_arm_ip']}", "INFO")

        if DEVICE_CONFIG['arms']['right_arm']:
            self.controllers["arm"]["right_arm"].set_up(
                rm_ip=RM65_CONFIG['right_arm_ip'],
                thread_mode=RM65_CONFIG['thread_mode'],
                dT=RM65_CONFIG['control_freq']
            )
            debug_print(
                "robot", f"Right arm initialized at {RM65_CONFIG['right_arm_ip']}", "INFO")

        # 设置摄像头传感器 - 调用 sensor/Realsense_sensor.py -> set_up(serial_number, is_depth)
        if "image" in self.sensors:
            if "cam_head" in self.sensors["image"]:
                self.sensors["image"]["cam_head"].set_up(
                    CAMERA_CONFIG['head'],
                    is_depth=CAMERA_CONFIG['is_depth']
                )
                debug_print(
                    "robot", f"Head camera initialized: {CAMERA_CONFIG['head']}", "INFO")

            if "cam_left_wrist" in self.sensors["image"]:
                self.sensors["image"]["cam_left_wrist"].set_up(
                    CAMERA_CONFIG['left_wrist'],
                    is_depth=CAMERA_CONFIG['is_depth']
                )
                debug_print(
                    "robot", f"Left wrist camera initialized: {CAMERA_CONFIG['left_wrist']}", "INFO")

            if "cam_right_wrist" in self.sensors["image"]:
                self.sensors["image"]["cam_right_wrist"].set_up(
                    CAMERA_CONFIG['right_wrist'],
                    is_depth=CAMERA_CONFIG['is_depth']
                )
                debug_print(
                    "robot", f"Right wrist camera initialized: {CAMERA_CONFIG['right_wrist']}", "INFO")

        # 设置QuestVR传感器 - 调用 sensor/Quest_sensor.py -> set_up()
        if DEVICE_CONFIG['quest_vr'] and "teleop" in self.sensors:
            self.sensors["teleop"]["quest_vr"].set_up()
            debug_print("robot", "QuestVR sensor initialized", "INFO")

        # 配置数据采集类型 - 调用 my_robot/base_robot.py -> set_collect_type()
        self.set_collect_type(COLLECT_CONFIG)

        debug_print(
            "robot", "All configured devices initialized successfully", "INFO")

    def reset(self):
        """
        根据配置复位启用的机械臂到安全位置
        调用 controller/Realman_controller.py -> reset(start_angles)
        """
        if DEVICE_CONFIG['arms']['left_arm']:
            self.controllers["arm"]["left_arm"].reset(
                JOINT_CONFIG['left_start_position'])
            debug_print("robot", "Left arm reset to safe position", "INFO")

        if DEVICE_CONFIG['arms']['right_arm']:
            self.controllers["arm"]["right_arm"].reset(
                JOINT_CONFIG['right_start_position'])
            debug_print("robot", "Right arm reset to safe position", "INFO")

    def start_teleop(self):
        """
        启动QuestVR遥操功能
        根据配置动态检查设备状态并启动遥操控制
        """
        if not DEVICE_CONFIG['quest_vr']:
            debug_print(
                "teleop", "QuestVR is disabled in config, cannot start teleop", "ERROR")
            return

        time.sleep(3)

        # 等待数据稳定 - 调用 my_robot/base_robot.py -> get()
        debug_print(
            "teleop", "Waiting for stable data from enabled devices...", "INFO")
        while True:
            data = self.get()

            # 检查QuestVR数据
            quest_data = None
            if DEVICE_CONFIG['quest_vr'] and "quest_vr" in data[1]:
                quest_data = data[1]["quest_vr"].get("end_pose")
                if quest_data is None:
                    debug_print(
                        "teleop", "QuestVR end_pose missing while waiting", "INFO")

            # 检查机械臂数据
            left_qpos = None
            right_qpos = None
            if DEVICE_CONFIG['arms']['left_arm'] and "left_arm" in data[0]:
                left_qpos = data[0]["left_arm"]["qpos"]
            if DEVICE_CONFIG['arms']['right_arm'] and "right_arm" in data[0]:
                right_qpos = data[0]["right_arm"]["qpos"]

            # 检查必要的数据是否就绪
            data_ready = True
            missing_items = []
            if DEVICE_CONFIG['quest_vr'] and quest_data is None:
                data_ready = False
                missing_items.append("quest_vr")
            if DEVICE_CONFIG['arms']['left_arm'] and left_qpos is None:
                data_ready = False
                missing_items.append("left_qpos")
            if DEVICE_CONFIG['arms']['right_arm'] and right_qpos is None:
                data_ready = False
                missing_items.append("right_qpos")

            if data_ready:
                break
            debug_print(
                "teleop", f"Waiting for devices: {missing_items}", "INFO")
            time.sleep(0.1)

        # 记录基准位姿 - 仅记录启用的机械臂的基准位姿
        left_base_pose = None
        right_base_pose = None
        if DEVICE_CONFIG['arms']['left_arm']:
            left_base_pose = data[0]["left_arm"]["qpos"]
        if DEVICE_CONFIG['arms']['right_arm']:
            right_base_pose = data[0]["right_arm"]["qpos"]

        debug_print("teleop", "Data stable, starting teleoperation...", "INFO")

        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / \
            f"quest_rm65_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        log_file_handle = None
        log_writer = None

        header = [
            "timestamp",
            "quest_left_x", "quest_left_y", "quest_left_z",
            "quest_left_roll", "quest_left_pitch", "quest_left_yaw",
            "quest_right_x", "quest_right_y", "quest_right_z",
            "quest_right_roll", "quest_right_pitch", "quest_right_yaw",
            "arm_left_x", "arm_left_y", "arm_left_z",
            "arm_left_roll", "arm_left_pitch", "arm_left_yaw",
            "arm_right_x", "arm_right_y", "arm_right_z",
            "arm_right_roll", "arm_right_pitch", "arm_right_yaw",
        ]

        def pose_to_list(pose):
            if pose is None:
                return [np.nan] * 6
            pose_array = np.asarray(pose, dtype=np.float64).flatten()
            if pose_array.size < 6:
                padded = np.full(6, np.nan)
                padded[:pose_array.size] = pose_array
                pose_array = padded
            return pose_array[:6].tolist()

        def quest_pose_lists(raw_pose):
            if raw_pose is None:
                nan_block = [np.nan] * 6
                return nan_block, nan_block
            pose_array = np.asarray(raw_pose, dtype=np.float64).flatten()
            if pose_array.size < 12:
                padded = np.full(12, np.nan)
                padded[:pose_array.size] = pose_array
                pose_array = padded
            return pose_array[:6].tolist(), pose_array[6:12].tolist()

        try:
            log_file_handle = log_path.open("w", newline="")
            log_writer = csv.writer(log_file_handle)
            log_writer.writerow(header)
            self.last_teleop_log_path = log_path
            debug_print("teleop", f"Logging teleop data to {log_path}", "INFO")
        except Exception as exc:
            debug_print(
                "teleop", f"Failed to create log file {log_path}: {exc}", "WARNING")
            log_file_handle = None
            log_writer = None
            self.last_teleop_log_path = None

        # 遥操主循环 - 根据配置动态构建控制指令
        log_counter = 0
        try:
            while True:
                try:
                    data = self.get()
                    log_counter += 1

                    # 获取QuestVR姿态数据
                    quest_pose = None
                    if DEVICE_CONFIG['quest_vr'] and "quest_vr" in data[1]:
                        quest_pose = data[1]["quest_vr"].get("end_pose")
                        if log_counter % 10 == 0:
                            debug_print(
                                "teleop", f"Quest pose raw: {quest_pose}", "INFO")

                    if quest_pose is not None:
                        # QuestVR数据解析：12维数组分离为左右手6维增量姿态
                        left_delta = quest_pose[:6]    # 左手增量姿态（相对于上一次位置的变化）
                        right_delta = quest_pose[6:]   # 右手增量姿态（相对于上一次位置的变化）

                        # 坐标变换：将增量叠加到基准位姿，得到绝对目标位姿
                        # 调用 utils/data_handler.py -> apply_local_delta_pose()
                        left_target = None
                        right_target = None
                        if DEVICE_CONFIG['arms']['left_arm'] and left_base_pose is not None:
                            left_target = apply_local_delta_pose(
                                left_base_pose, left_delta)
                        if DEVICE_CONFIG['arms']['right_arm'] and right_base_pose is not None:
                            right_target = apply_local_delta_pose(
                                right_base_pose, right_delta)

                        # 解析QuestVR按钮数据用于夹爪控制 - 调用 sensor/Quest_sensor.py -> get_state()["extra"]
                        # 数据格式参考: my_robot/oculus_data_format.md
                        left_gripper = 0.0   # 默认关闭状态(0.0)
                        right_gripper = 0.0  # 默认关闭状态(0.0)

                        # if DEVICE_CONFIG['gripper'] and "quest_vr" in data[1]:
                        if "quest_vr" in data[1]:
                            buttons = data[1]["quest_vr"].get("extra")
                            print(buttons)

                        # 根据配置动态构建控制指令 - 使用teleop_qpos触发高频透传控制
                        # 调用链路（左右臂相同）：
                        # my_robot/base_robot.py -> move()
                        # -> controller/controller.py -> move()
                        # -> controller/arm_controller.py -> move_controller()
                        # -> controller/Realman_controller.py -> set_position_teleop()
                        # -> third_party/Realman_IK/rm_ik.py -> solve() (逆解)
                        # -> Robotic_Arm SDK -> Movej_CANFD() (硬件透传控制)
                        move_data = {"arm": {}}

                        if DEVICE_CONFIG['arms']['left_arm'] and left_target is not None:
                            move_data["arm"]["left_arm"] = {
                                "teleop_qpos": left_target,    # 触发set_position_teleop() -> 逆解 -> Movej_CANFD()透传
                            }
                            if DEVICE_CONFIG['gripper']:
                                # 夹爪控制，范围0.0-1.0
                                move_data["arm"]["left_arm"]["gripper"] = left_gripper

                        if DEVICE_CONFIG['arms']['right_arm'] and right_target is not None:
                            move_data["arm"]["right_arm"] = {
                                "teleop_qpos": right_target,   # 触发set_position_teleop() -> 逆解 -> Movej_CANFD()透传
                            }
                            if DEVICE_CONFIG['gripper']:
                                # 夹爪控制，范围0.0-1.0
                                move_data["arm"]["right_arm"]["gripper"] = right_gripper

                        # 发送控制指令到机械臂控制器
                        if move_data["arm"]:  # 仅在有启用的机械臂时发送指令
                            if log_counter % 10 == 0:
                                debug_print(
                                    "teleop", f"Move command: {move_data}", "INFO")
                            self.move(move_data)

                    if log_writer is not None:
                        quest_raw = None
                        if DEVICE_CONFIG['quest_vr'] and "quest_vr" in data[1]:
                            quest_raw = data[1]["quest_vr"].get("raw_pose")

                        left_raw_vals, right_raw_vals = quest_pose_lists(
                            quest_raw)

                        left_arm_vals = pose_to_list(
                            data[0].get("left_arm", {}).get("qpos")
                            if DEVICE_CONFIG['arms']['left_arm'] else None
                        )
                        right_arm_vals = pose_to_list(
                            data[0].get("right_arm", {}).get("qpos")
                            if DEVICE_CONFIG['arms']['right_arm'] else None
                        )

                        row = [time.time()] + left_raw_vals + \
                            right_raw_vals + left_arm_vals + right_arm_vals
                        log_writer.writerow(row)
                        if log_counter % 10 == 0 and log_file_handle is not None:
                            log_file_handle.flush()

                    time.sleep(RM65_CONFIG['control_freq'])  # 使用配置的控制频率

                except KeyboardInterrupt:
                    debug_print(
                        "teleop", "User interrupted, stopping teleop", "INFO")
                    break
                except ValueError as e:
                    debug_print(
                        "teleop", f"Data processing error: {e}", "WARNING")
                    time.sleep(0.1)
                except RuntimeError as e:
                    debug_print("teleop", f"Control error: {e}", "ERROR")
                    time.sleep(0.1)
                except Exception as e:
                    debug_print("teleop", f"Unexpected error: {e}", "ERROR")
                    time.sleep(0.1)
        finally:
            if log_file_handle is not None:
                log_file_handle.flush()
                log_file_handle.close()

        self.reset()

    def test_get_arm_state(self, arm_name="left_arm", retries=3, retry_delay=0.2):
        """
        测试函数：获取机械臂当前状态

        """
        if "arm" not in self.controllers or arm_name not in self.controllers["arm"]:
            raise ValueError(
                f"Arm '{arm_name}' is not enabled in DEVICE_CONFIG.")

        arm = self.controllers["arm"][arm_name]

        if getattr(arm, "controller", None) is None:
            debug_print("test", f"{arm_name} 未初始化，自动执行 set_up()", "WARNING")
            self.set_up()
            arm = self.controllers["arm"][arm_name]

        if getattr(arm, "controller", None) is None:
            raise RuntimeError(
                f"Arm '{arm_name}' controller is not connected. Please check the network/IP settings.")

        succ, full_state = -1, None
        for attempt in range(1, retries + 1):
            succ, full_state = arm.controller.rm_get_current_arm_state()
            if succ == 0 and isinstance(full_state, dict):
                break
            debug_print(
                "test",
                f"{arm_name} 获取原始状态失败 (succ={succ})，第 {attempt} 次重试",
                "WARNING",
            )
            time.sleep(retry_delay)

        if succ != 0 or not isinstance(full_state, dict):
            raise RuntimeError(
                f"Failed to read arm state after {retries} attempts (succ={succ}).")

        # 使用封装后的接口获取状态，会额外包含夹爪信息
        arm_state = arm.get_state()

        # 打印关节/位姿的主要字段，方便人工核查
        joint = np.array(full_state.get("joint", []), dtype=np.float64)
        pose = np.array(full_state.get("pose", []), dtype=np.float64)
        gripper = arm_state.get("gripper")

        if joint.size:
            debug_print(
                "test", f"{arm_name} joint(deg): {np.round(joint, 3)}", "INFO")
        if pose.size:
            debug_print(
                "test", f"{arm_name} pose: {np.round(pose, 4)}", "INFO")
        debug_print("test", f"{arm_name} gripper state: {gripper}", "INFO")

        # 计算一次差值，确保封装接口与底层数据一致
        if "joint" in arm_state and joint.size:
            wrapper_joint = np.array(arm_state["joint"], dtype=np.float64)
            joint_diff = np.max(np.abs(wrapper_joint - joint))
            debug_print(
                "test", f"{arm_name} joint diff(wrapper-raw): {joint_diff:.6f}", "INFO")

        return {
            "raw_state": full_state,
            "wrapper_state": arm_state,
        }

    # def test():
    #     # 实例化RoboticArm类
    #     arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    #     # 创建机械臂连接，打印连接id
    #     handle = arm.rm_create_robot_arm("192.168.2.19", 8080)
    #     print(handle.id)

    #     # 关节阻塞运动到[0, 20, 70, 0, 90, 0]
    #     print(arm.rm_movej([0, 0, 0, 0, 0, -60], 10, 0, 0, 1))

    #     arm.rm_delete_robot_arm()


if __name__ == "__main__":
    import os
    os.environ["INFO_LEVEL"] = "INFO"

    robot = RM65QuestRobot()

    try:
        robot.set_up()      # 初始化系统
        robot.reset()       # 复位到安全位置
        robot.start_teleop() # 启动遥操
        # robot.test_get_arm_state("left_arm")  # 测试获取左臂状态

    except Exception as e:
        debug_print("main", f"System error: {e}", "ERROR")
        robot.reset()       # 复位到安全位置
    except KeyboardInterrupt:
        debug_print("main", "User interrupted, stopping system", "INFO")
        robot.reset()       # 复位到安全位置
    finally:
        debug_print("main", "System shutdown", "INFO")
        robot.reset()       # 复位到安全位置
