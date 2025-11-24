'''
Descripttion: 
Author: tauceti0207
version: 
Date: 2025-11-02 10:16:52
LastEditors: tauceti0207
LastEditTime: 2025-11-03 15:58:13
'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from re import T
import sys
import os


from realman65.utils.data_handler import debug_print
from pathlib import Path
import csv
import copy
import time
from typing import Dict, Iterable, Optional, Sequence
import numpy as np
from Robotic_Arm.rm_robot_interface import rm_thread_mode_e
from realman65.controller.Realman_controller import RealmanController
import json
from scipy.spatial.transform import Rotation as R

from realman65.my_robot.pink_ik import ArmIKSolver
import threading

DEFAULT_RM65_CONFIG = {
    'left_arm_ip': "192.168.2.19",
    'right_arm_ip': "192.168.1.18",
    'thread_mode': rm_thread_mode_e.RM_TRIPLE_MODE_E,
    'control_freq': 0.014285,
}

DEFAULT_JOINT_CONFIG = {
    # 'left_start_position': [-28, 17, 83, -3, 67, -179],
    # 'left_start_position': [-1.48, 33.7, 79, 0, 60, -158], # 最开始初始
    # 'left_start_position': [-35, 36, 86, 1.9, 50, -158], # temp 靠近杯子
    'left_start_position': [-95, 10, 58, 5, 65, -169], # 有障碍物 棕色杯子
    # 'left_start_position': [-5, 10, 58, 5, 65, -169], # 有障碍物 瑞星杯子
    'right_start_position': [-95, 10, 58, 5, 65, -169],
}

DEFAULT_DEVICE_CONFIG = {
    'arms': {
        'left_arm': False,
        'right_arm': True,
    },
    'gripper': True,            # 是否启用夹爪控制
    'quest_vr': False,           # 是否启用QuestVR传感器
}

DEFAULT_COLLECT_CONFIG = {
    "arm": ["joint", "qpos", "gripper"],
    "teleop": ["end_pose", "extra", "raw_pose"],
}


class Realman65Interface:
    """简单封装RM65机械臂控制接口。"""

    def __init__(self,
                 device_config: Optional[Dict] = None,
                 rm_config: Optional[Dict] = None,
                 joint_config: Optional[Dict] = None,
                 auto_setup: bool = False) -> None:
        self.device_config = copy.deepcopy(DEFAULT_DEVICE_CONFIG)
        if device_config:
            self._merge_dict(self.device_config, device_config)

        self.rm_config = copy.deepcopy(DEFAULT_RM65_CONFIG)
        if rm_config:
            self.rm_config.update(rm_config)

        self.joint_config = copy.deepcopy(DEFAULT_JOINT_CONFIG)
        if joint_config:
            self._merge_dict(self.joint_config, joint_config)

        self.controllers: Dict[str, RealmanController] = {}
        self._initialize_arms()
        
        self.ik_solver =  ArmIKSolver(
        # "/home/shui/sam2_pointclouds/rm_description/urdf/rm_65.urdf",
        # ee_frame="ee_link",
        "/home/shui/cloudfusion/DA_D03_description/urdf/rm_65_without_gripper.urdf",
        ee_frame="ee_link",
        visualize=False)
        self.init_ik = False
        
        self.target_joint_angles = None
        
        if auto_setup:
            self.set_up()

        self.control_thread = threading.Thread(target=self.start_control)
        self.control_thread.start()

    def start_control(self):
        if not self.device_config['arms']['left_arm'] and not self.device_config['arms']['right_arm']:
            return
        if self.device_config['arms']['left_arm']:
            arm_name = 'left_arm'
        elif self.device_config['arms']['right_arm']:
            arm_name = 'right_arm'
        self.send_update_joint_angles(arm_name)

    def set_up(self) -> None:
        """连接并初始化已启用的机械臂。"""
        for arm_name, controller in self.controllers.items():
            ip_key = f"{arm_name}_ip"
            if ip_key not in self.rm_config:
                raise KeyError(f"{ip_key} 未在 RM65_CONFIG 中配置")
            controller.set_up(
                rm_ip=self.rm_config[ip_key],
                thread_mode=self.rm_config['thread_mode'],
                dT=self.rm_config['control_freq'],
            )
            debug_print(
                "robot", f"{arm_name} connected to {self.rm_config[ip_key]}", "INFO")

    def reset(self) -> None:
        """将已启用的机械臂复位到预设安全姿态。"""
        for arm_name, controller in self.controllers.items():
            joint_key = f"{arm_name.replace('_arm', '')}_start_position"
            start_pose = self.joint_config.get(joint_key)
            if start_pose is None:
                debug_print("robot", f"{arm_name} 缺少复位角度配置，跳过", "WARNING")
                continue
            controller.reset(start_pose)
            debug_print(
                "robot", f"{arm_name} reset to preset joint angles", "INFO")

    def get_end_effector_pose(self,
                              arm_names: Optional[Iterable[str]] = None,
                              retries: int = 3,
                              retry_delay: float = 0.2) -> Dict[str, Optional[Sequence[float]]]:
        """读取末端位姿 (xyz + rpy)。"""
        return self._collect_state_field("pose", arm_names, retries, retry_delay)

    def low_pass_filter(self,target_joint_angles,arm_name):
        current_angles = np.radians(self.get_joint_angles().get(arm_name))
        if target_joint_angles is None:
            return None
        send_angles = 0.6*np.array(target_joint_angles) + 0.4*np.array(current_angles)
        return send_angles
        
    def get_joint_angles(self,
                         arm_names: Optional[Iterable[str]] = None,
                         retries: int = 3,
                         retry_delay: float = 0.2) -> Dict[str, Optional[Sequence[float]]]:
        """读取关节角 (单位为度)。"""
        return self._collect_state_field("joint", arm_names, retries, retry_delay)

    def set_end_effector_pose(self,
                              arm_name: str,
                              target_pose: Sequence[float],
                              *,
                              use_ik: bool = True) -> None:
        """设置末端位姿，默认走IK解算。"""
        controller = self._ensure_arm_ready(arm_name)
        pose_np = np.asarray(target_pose, dtype=float).flatten()  # 转为ndarray并展平

        
        if use_ik:
            # IK解算出角度
            # controller.set_position_teleop(pose)
            
            # 使用我们自己的IK
            target_pos = pose_np[:3]  # [x, y, z]
            target_rpy = pose_np[3:]  # [r, p, y]（弧度）
            rot = R.from_euler('xyz', target_rpy, degrees=False)  # 从rpy创建旋转对象
            target_quat = rot.as_quat("xyzw")  # 转换为四元数 [x, y, z, w]
            joint_target = self.ik_solver.move_to_pose_and_get_joints(target_pos, target_quat)

            while True:
                joint_rad  = self.low_pass_filter(joint_target, arm_name)
            
                if joint_rad is None:
                    return
                joint_deg = np.rad2deg(joint_rad)
                # debug_print("robot", 
                # f"{arm_name} 自定义IK解算: 目标位姿={target_pose}, 关节指令（度）={joint_deg}", 
                # "INFO")
                
                success = controller.controller.rm_movej_canfd(joint_deg.tolist(), False)
                if success != 0:
                    raise RuntimeError(f"发送关节角度失败，返回码: {success}")
         
        debug_print("robot", f"{arm_name} 控制指令执行完成", "INFO")

    def update(self,target_pose: Sequence[float],*,use_ik: bool = True):
        pose_np = np.asarray(target_pose, dtype=float).flatten()  # 转为ndarray并展平
        if use_ik:
            target_pos = pose_np[:3]  # [x, y, z]
            target_rpy = pose_np[3:]  # [r, p, y]（弧度）
            rot = R.from_euler('xyz', target_rpy, degrees=False)  # 从rpy创建旋转对象
            target_quat = rot.as_quat("xyzw")  # 转换为四元数 [x, y, z, w]
            joint_target = self.ik_solver.move_to_pose_and_get_joints(target_pos, target_quat)
            if joint_target is None:
                return
            self.target_joint_angles = joint_target
            self.init_ik = True

    def send_update_joint_angles(self,arm_name):
        try:
            controller = self._ensure_arm_ready(arm_name)
            while True:
                if not self.init_ik:
                    time.sleep(0.1)
                    continue
                joint_rad  = self.low_pass_filter(self.target_joint_angles, arm_name)
                if joint_rad is None:
                    continue
                joint_deg = np.rad2deg(joint_rad)
                success = controller.controller.rm_movej_canfd(joint_deg.tolist(), False)
                if success != 0:
                    raise RuntimeError(f"发送关节角度失败，返回码: {success}")
                time.sleep(0.005)
        except Exception as e:
            print(f"发送关节角度线程异常: {e}")

    def set_joint_angles(self,
                         arm_name: str,
                         joint_angles: Sequence[float],
                         *,
                         blocking: bool = False) -> None:
        """设置关节角，可选阻塞模式。"""
        controller = self._ensure_arm_ready(arm_name)
        joint_list = np.asarray(joint_angles, dtype=float).flatten().tolist()
        if (len(joint_list) == 6):
            joint_list.append(0.0)
        if blocking:
            result = controller.controller.rm_movej(joint_list, 5, 0, 0, 1)
            if result != 0:
                raise RuntimeError(
                    f"{arm_name} rm_movej failed with code {result}")
        else:
            controller.set_joint(joint_list)
        debug_print(
            "robot", f"{arm_name} joint command -> {joint_list}", "INFO")

        # -------------------- 新增：夹爪控制接口 --------------------
    def set_gripper(self,
                    arm_name: str,
                    value: int,
                    speed: int = 1000,
                    force: int = 1000,
                    blocking: bool = False,
                    timeout: int = 0) -> None:
        """
        控制夹爪松紧（复用底层 rm_set_gripper_pick/rm_set_gripper_release）
        Args:
            arm_name: 机械臂名称（如 "left_arm"）
            value: 控制指令（1=夹紧，0=松开）
            speed: 夹爪运动速度（1~1000，默认1000）
            force: 夹爪夹紧力阈值（50~1000，仅夹紧时生效，默认200）
            blocking: 是否阻塞等待（True=等待到位，False=非阻塞，默认False）
            timeout: 超时时间（阻塞模式=等待秒数，非阻塞模式=0立即返回，默认0）
        Raises:
            ValueError: value不是0/1，或夹爪未启用
            RuntimeError: 控制指令执行失败
        """
        # 检查夹爪是否启用
        if not self.device_config.get('gripper', False):
            raise ValueError(f"夹爪未在 DEVICE_CONFIG 中启用，请先设置 'gripper': True")

        # 校验控制指令
        if value not in (0, 1):
            raise ValueError(
                f"set_gripper value 必须是 0（松开）或 1（夹紧），当前输入：{value}")

        controller = self._ensure_arm_ready(arm_name)
        debug_print(
            "robot", f"{arm_name} 夹爪控制：{'夹紧' if value == 1 else '松开'}force={force}）", "INFO")

        # 调用底层SDK方法
        if value == 1:
            # 夹紧：复用 rm_set_gripper_pick（力控夹取）
            result = controller.controller.rm_set_gripper_pick(
                speed, force, blocking, timeout)
        else:
            # 松开：复用 rm_set_gripper_release（运动到最大开口）
            result = controller.controller.rm_set_gripper_release(
                speed, blocking, timeout)

        # 校验执行结果
        if result != 0:
            error_msg = {
                0: "成功",
                1: "参数错误或机械臂状态异常",
                -1: "数据发送失败",
                -2: "数据接收失败",
                -3: "返回值解析失败",
                -4: "超时"
            }.get(result, f"未知错误（状态码：{result}）")
            raise RuntimeError(f"{arm_name} 夹爪控制失败：{error_msg}")

    # -------------------- 新增：夹爪状态获取接口 --------------------
    def get_gripper_state(self,
                          arm_names: Optional[Iterable[str]] = None,
                          retries: int = 3,
                          retry_delay: float = 0.2) -> Dict[str, Optional[int]]:
        """
        获取夹爪状态（简化返回：1=夹紧，0=松开，None=获取失败）
        Args:
            arm_names: 要查询的机械臂名称列表（None=查询所有启用的机械臂）
            retries: 失败重试次数（默认3）
            retry_delay: 重试间隔（默认0.2秒）
        Returns:
            字典：key=机械臂名称，value=状态（1=夹紧，0=松开，None=失败）
        """
        # 检查夹爪是否启用
        if not self.device_config.get('gripper', False):
            debug_print("robot", "夹爪未启用，返回空状态", "WARNING")
            return {}

        results: Dict[str, Optional[int]] = {}
        for arm_name in self._iter_arms(arm_names):
            controller = self._ensure_arm_ready(arm_name)
            state = self._fetch_gripper_state(controller, retries, retry_delay)
            results[arm_name] = state
            # debug_print("robot", f"{arm_name} 夹爪状态：{'夹紧' if state == 1 else '松开' if state == 0 else '未知'}", "INFO")
        return results

    # -------------------- 新增：内部辅助方法 --------------------

    def _fetch_gripper_state(self,
                             controller: RealmanController,
                             retries: int,
                             retry_delay: float) -> Optional[int]:
        """底层获取夹爪状态，映射为 1=夹紧，0=松开（基于实际结构体字段修正）"""
        for attempt in range(1, retries + 1):
            try:
                # 复用底层 rm_get_gripper_state 接口
                succ, gripper_dict = controller.controller.rm_get_gripper_state()
                if succ != 0:
                    debug_print(
                        "robot", f"{controller.name} 夹爪状态获取失败（状态码：{succ}），第{attempt}次重试", "WARNING")
                    time.sleep(retry_delay)
                    continue

                # 核心映射逻辑：基于 mode 字段（优先）+ actpos 字段（辅助）
                mode = gripper_dict.get("mode", 0)  # 工作状态（1-6）
                actpos = gripper_dict.get("actpos", 500)  # 开口度（0=最小，1000=最大）

                # 根据结构体定义和实际返回数据判断：
                # mode含义：1=张开到最大空闲，2=闭合到最小空闲，3=停止空闲，4=正在闭合，5=正在张开，6=力控停止（已夹紧）
                if mode in (2, 4, 6):
                    # mode=2（已夹紧空闲）、4（正在夹紧）、6（力控夹紧停止）→ 均判定为夹紧状态（1）
                    return 1
                elif mode in (1, 5):
                    # mode=1（已松开空闲）、5（正在松开）→ 均判定为松开状态（0）
                    return 0
                elif mode == 3:
                    # mode=3（停止空闲）：通过开口度辅助判断（actpos≤100=夹紧，>100=松开）
                    return 1 if actpos <= 100 else 0
                else:
                    # 未知mode：默认按开口度判断（接近最小=夹紧，接近最大=松开）
                    debug_print(
                        "robot", f"{controller.name} 未知夹爪mode：{mode}，按开口度判断", "WARNING")
                    return 1 if actpos <= 100 else 0

            except Exception as exc:
                debug_print(
                    "robot", f"{controller.name} 夹爪状态解析异常：{exc}，第{attempt}次重试", "WARNING")
                time.sleep(retry_delay)

        debug_print(
            "robot", f"{controller.name} 夹爪状态获取失败（已重试{retries}次）", "ERROR")
        return None

    # -------------------- Internal helpers --------------------
    def _initialize_arms(self) -> None:
        for flag, enabled in self.device_config.get("arms", {}).items():
            if enabled:
                arm_name = flag if flag.endswith("_arm") else f"{flag}_arm"
                self.controllers[arm_name] = RealmanController(arm_name)

    def _ensure_arm_ready(self, arm_name: str) -> RealmanController:
        if arm_name not in self.controllers:
            raise ValueError(f"{arm_name} 未启用，请检查 DEVICE_CONFIG")
        controller = self.controllers[arm_name]
        if getattr(controller, "controller", None) is None:
            debug_print("robot", f"{arm_name} 未连接，尝试自动 set_up()", "WARNING")
            self.set_up()
        if getattr(controller, "controller", None) is None:
            raise RuntimeError(f"{arm_name} controller 未连接，请检查网络或电源")
        return controller

    def _collect_state_field(self,
                             field: str,
                             arm_names: Optional[Iterable[str]],
                             retries: int,
                             retry_delay: float) -> Dict[str, Optional[Sequence[float]]]:
        results: Dict[str, Optional[Sequence[float]]] = {}
        for name in self._iter_arms(arm_names):
            controller = self._ensure_arm_ready(name)
            results[name] = self._fetch_state_field(
                controller, field, retries, retry_delay)
        return results

    def _iter_arms(self, arm_names: Optional[Iterable[str]]) -> Iterable[str]:
        if arm_names is None:
            return list(self.controllers.keys())
        return [name for name in arm_names if name in self.controllers]

    def _fetch_state_field(self,
                           controller: RealmanController,
                           field: str,
                           retries: int,
                           retry_delay: float) -> Optional[Sequence[float]]:
        succ, full_state = -1, None
        for attempt in range(1, retries + 1):
            try:
                succ, full_state = controller.controller.rm_get_current_arm_state()
                if succ == 0 and isinstance(full_state, dict):
                    break
            except Exception as exc:
                debug_print(
                    "robot", f"{controller.name} 读取状态异常：{exc}", "WARNING")
            debug_print("robot",
                        f"{controller.name} 获取状态失败 (succ={succ})，第 {attempt} 次重试",
                        "WARNING")
            time.sleep(retry_delay)

        value = None
        if succ == 0 and isinstance(full_state, dict):
            value = full_state.get(field)

        if value is None:
            try:
                wrapper = controller.get_state() or {}
            except Exception as exc:
                debug_print(
                    "robot", f"{controller.name} get_state 失败：{exc}", "WARNING")
                wrapper = {}
            value = wrapper.get(field)

        if value is None:
            return None

        try:
            return np.asarray(value, dtype=float).flatten().tolist()
        except Exception:
            return list(value)

    @staticmethod
    def _merge_dict(dst: Dict, src: Dict) -> None:
        for key, value in src.items():
            if isinstance(value, dict) and isinstance(dst.get(key), dict):
                Realman65Interface._merge_dict(dst[key], value)
            else:
                dst[key] = value





if __name__ == "__main__":
    # 环境变量设置（日志级别）
    print("pass")
    # test_gripper()