#!/usr/bin/env python3
"""
最简线程版：VR控制双机械臂（单类实现）
- 封装连接、VR接收、位姿缓存、左右臂跟随等逻辑到一个类中
- 只保留核心流程，便于快速对接与测试
"""

from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformListener, Buffer
from rclpy.node import Node
import rclpy
import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Optional, Tuple, Dict, Any, List
import threading
import socket
import json
import time
import sys
import os

sys.path.append(os.path.dirname(__file__))


# ROS2 相关导入

from Robotic_Arm.rm_robot_interface import *  # noqa: F401,F403


# 绕z轴顺时针转90度的旋转矩阵
global_transform_Rz_cw90 = np.array([
    [0,  1, 0],
    [-1, 0, 0],
    [0,  0, 1]
])

# 绕z轴逆时针转90度的旋转矩阵
global_transform_Rz_ccw90 = np.array([
    [0,  -1, 0],
    [1, 0, 0],
    [0,  0, 1]
])

# 绕z轴转180度的旋转矩阵
global_transform_Rz180 = np.array([
    [-1,  0, 0],
    [0, -1, 0],
    [0,  0, 1]
])


# 绕x轴顺时针转90度的旋转矩阵（Rx_cw90）
global_transform_Rx_cw90 = np.array([
    [1, 0, 0],
    [0, 0, 1],   # cos(-90°)=0, -sin(-90°)=1
    [0, -1, 0]   # sin(-90°)=-1, cos(-90°)=0
])

# 绕x轴逆时针转90度的旋转矩阵（Rx_ccw90）
global_transform_Rx_ccw90 = np.array([
    [1, 0, 0],
    [0, 0, -1],  # cos90°=0, -sin90°=-1
    [0, 1, 0]    # sin90°=1, cos90°=0
])


# 绕x轴转180度的旋转矩阵（Rx180）
global_transform_Rx180 = np.array([
    [1, 0, 0],
    [0, -1, 0],  # cos180°=-1, -sin180°=0
    [0, 0, -1]   # sin180°=0, cos180°=-1
])


# 绕y轴顺时针转90度的旋转矩阵（Ry_cw90）
global_transform_Ry_cw90 = np.array([
    [0, 0, -1],  # cos(-90°)=0, sin(-90°)=-1
    [0, 1, 0],
    [1, 0, 0]    # -sin(-90°)=1, cos(-90°)=0
])

# 绕y轴逆时针转90度的旋转矩阵（Ry_ccw90）
global_transform_Ry_ccw90 = np.array([
    [0, 0, 1],   # cos90°=0, sin90°=1
    [0, 1, 0],
    [-1, 0, 0]   # -sin90°=-1, cos90°=0
])

# 绕y轴转180度的旋转矩阵（Ry180）
global_transform_Ry180 = np.array([
    [-1, 0, 0],  # cos180°=-1, sin180°=0
    [0, 1, 0],
    [0, 0, -1]   # -sin180°=0, cos180°=-1
])


# 绕y轴逆时针转90度再转180度的旋转矩阵（Ry_ccw90_plus_180）
global_transform_Ry_ccw90_plus_180 = global_transform_Ry180 @ global_transform_Ry_ccw90


class SimpleVRDoubleArmThreaded:
    def __init__(
        self,
        left_ip: str = "192.168.1.18",
        right_ip: str = "192.168.2.19",
        port: int = 8080,
        level: int = 3,
        use_ros2_tf: bool = True,  # 是否使用ROS2 TF
    ) -> None:
        # 连接配置
        self.left_ip = left_ip
        self.right_ip = right_ip
        self.port = port
        self.level = level

        self.left_pos_offset = None  # 左机械臂位置偏移
        self.right_pos_offset = None  # 右机械臂位置偏移
        self.left_quat_offset = None  # 左机械臂旋转偏移
        self.right_quat_offset = None  # 右机械臂旋转偏移

        self.initialized = False
        self.vr_received = False
        self.robot_received = False

        # 运行状态
        self.running = False
        self._lock = threading.Lock()

        # 机器人句柄
        self.left_robot: Optional[RoboticArm] = None
        self.right_robot: Optional[RoboticArm] = None

        # 机器人自身位姿
        self.left_robot_pos: Optional[List[float, float, float]] = None
        self.left_robot_quat: Optional[List[float, float, float, float]] = None
        self.right_robot_pos: Optional[List[float, float, float]] = None
        self.right_robot_quat: Optional[List[float,
                                             float, float, float]] = None

        # VR 位姿缓存（position: (x,y,z); rotation: (qx,qy,qz,qw)）
        self.vr_left_pos: Optional[List[float, float, float]] = None
        self.vr_left_quat: Optional[List[float, float, float, float]] = None
        self.vr_right_pos: Optional[List[float, float, float]] = None
        self.vr_right_quat: Optional[List[float, float, float, float]] = None

        # ROS2 TF相关
        self.ros_node: Optional['Node'] = None
        self.tf_buffer: Optional['Buffer'] = None
        self.tf_listener: Optional['TransformListener'] = None

        # 线程
        self.vr_thread: Optional[threading.Thread] = None
        self.left_control_thread: Optional[threading.Thread] = None
        self.right_control_thread: Optional[threading.Thread] = None
        self.update_state_thread: Optional[threading.Thread] = None

    # -------------------- 基础工具方法（并入类） --------------------
    def _connect_robot(self, ip: str, port: int, level: int, mode: Optional[int] = None) -> Optional[RoboticArm]:
        if RoboticArm is None:
            print("[连接] 未找到 RoboticArm 类，请确认 SDK 路径")
            return None
        robot = RoboticArm(rm_thread_mode_e(mode)) if (
            mode is not None and 'rm_thread_mode_e' in globals()
        ) else RoboticArm()
        handle = robot.rm_create_robot_arm(ip, port, level)
        if getattr(handle, 'id', -1) == -1:
            print(f"[连接] 连接 {ip}:{port} 失败")
            return None
        print(f"[连接] 连接 {ip}:{port} 成功，句柄: {handle.id}")
        return robot

    def _disconnect_robot(self, robot: Optional[RoboticArm]) -> None:
        if robot is None:
            return
        try:
            ret = robot.rm_delete_robot_arm()
            print("[断开] 成功" if ret == 0 else "[断开] 失败")
        except Exception:
            pass

    def _init_ros2(self) -> None:
        """初始化ROS2节点和TF监听器"""

        try:
            import rclpy
            from tf2_ros import TransformListener, Buffer

            if not rclpy.ok():
                rclpy.init()

            self.ros_node = rclpy.create_node('vr_tf_listener')
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self.ros_node)
            print("[信息] ROS2 TF监听器初始化成功")
        except Exception as e:
            print(f"[错误] 初始化ROS2失败: {e}")
            raise

    # -------------------- 线程循环 --------------------

    def _vr_receiver_loop(self) -> None:
        """VR 数据接收线程"""
        print("[信息] VR 接收线程启动")

        # ROS2 TF模式
        self._tf_receiver_loop()

        print("[信息] VR 接收线程结束")

    def _tf_receiver_loop(self) -> None:
        """TF数据接收循环"""

        import rclpy

        while self.running:
            try:
                # 处理ROS2回调
                rclpy.spin_once(self.ros_node, timeout_sec=0.01)

                # 获取左手TF
                try:
                    from rclpy.time import Time
                    left_transform = self.tf_buffer.lookup_transform(
                        'map', 'l_hand', Time())
                    left_pos = (
                        left_transform.transform.translation.x,
                        left_transform.transform.translation.y,
                        left_transform.transform.translation.z
                    )
                    left_rot = (
                        left_transform.transform.rotation.x,
                        left_transform.transform.rotation.y,
                        left_transform.transform.rotation.z,
                        left_transform.transform.rotation.w
                    )
                    rotated_left_pos = global_transform_Rz180 @ np.array(
                        left_pos)
                    rotated_left_rot = R.from_matrix(
                        global_transform_Ry_ccw90 @ R.from_quat(left_rot).as_matrix()).as_quat()

                    # rotated_left_pos = left_pos
                    # rotated_left_rot = left_rot

                    self.vr_received = True
                    with self._lock:
                        self.vr_left_pos = rotated_left_pos
                        self.vr_left_quat = rotated_left_rot
                        # print(f"[调试] 左手TF:{self.vr_left_pos}, {self.vr_left_quat}")
                except Exception:
                    print("[警告] 无法获取左手TF")  # TF不可用时忽略

                # 获取右手TF
                try:
                    from rclpy.time import Time
                    right_transform = self.tf_buffer.lookup_transform(
                        'map', 'r_hand', Time())
                    right_pos = (
                        right_transform.transform.translation.x,
                        right_transform.transform.translation.y,
                        right_transform.transform.translation.z
                    )
                    right_rot = (
                        right_transform.transform.rotation.x,
                        right_transform.transform.rotation.y,
                        right_transform.transform.rotation.z,
                        right_transform.transform.rotation.w
                    )

                    rotated_right_pos = global_transform_Rz180 @ np.array(
                        right_pos)
                    rotated_right_rot = R.from_matrix(
                        global_transform_Ry_ccw90 @ R.from_quat(right_rot).as_matrix()).as_quat()

                    # rotated_right_pos = right_pos
                    # rotated_right_rot = right_rot

                    self.vr_received = True
                    with self._lock:
                        self.vr_right_pos = rotated_right_pos
                        self.vr_right_quat = rotated_right_rot
                        # print(f"[调试] 右手TF:{self.vr_right_pos}, {self.vr_right_quat}")
                except Exception:
                    print("[警告] 无法获取右手TF")  # TF不可用时忽略

                time.sleep(0.02)

            except Exception as e:
                print(f"[错误] TF 接收失败: {e}")
                time.sleep(0.02)

    def retarget(self):
        # self.left_robot_pos: Optional[List[float, float, float]] = None
        # self.left_robot_quat: Optional[List[float,float,float,float]] = None
        # self.right_robot_pos: Optional[List[float, float, float]] = None
        # self.right_robot_quat: Optional[List[float,float,float,float]] = None

        # # VR 位姿缓存（position: (x,y,z); rotation: (qx,qy,qz,qw)）
        # self.vr_left_pos: Optional[List[float, float, float]] = None
        # self.vr_left_quat: Optional[List[float,float,float,float]] = None
        # self.vr_right_pos: Optional[List[float, float, float]] = None
        # self.vr_right_quat: Optional[List[float,float,float,float]] = None

        if not self.vr_received or not self.robot_received:
            print("[警告] 无法获取VR或机器人位姿")
            return

        l_pos_offset = np.zeros(3)
        r_pos_offset = np.zeros(3)
        l_quat_offset = np.array([0, 0, 0, 1])
        r_quat_offset = np.array([0, 0, 0, 1])

        print(f"robot_left:{self.left_robot_pos},vr_left: {self.vr_left_pos}")
        l_pos_offset = np.array(self.left_robot_pos) - \
            np.array(self.vr_left_pos)
        r_pos_offset = np.array(self.right_robot_pos) - \
            np.array(self.vr_right_pos)
        l_quat_offset = (R.from_quat(self.vr_left_quat).inv()
                         * R.from_quat(self.left_robot_quat)).as_quat()
        r_quat_offset = (R.from_quat(self.vr_right_quat).inv()
                         * R.from_quat(self.right_robot_quat)).as_quat()

        print(f"l_pos_offset: {l_pos_offset}")
        self.left_pos_offset = l_pos_offset
        self.right_pos_offset = r_pos_offset
        self.left_quat_offset = l_quat_offset
        self.right_quat_offset = r_quat_offset
        self.initialized = True

    def arm_update_state(self) -> None:
        # 获取状态
        # 获取机械臂完整状态（返回值是元组：(状态码, 状态字典)）
        left_state_tuple = self.left_robot.rm_get_current_arm_state()
        left_status_code = left_state_tuple[0]  # 状态码（0通常表示成功）
        left_state_dict = left_state_tuple[1]   # 核心状态字典（包含joint、pose、err）
        print("left_state_dict:", left_state_dict)
        right_state_tuple = self.right_robot.rm_get_current_arm_state()
        right_status_code = right_state_tuple[0]  # 状态码（0通常表示成功）
        right_state_dict = right_state_tuple[1]   # 核心状态字典（包含joint、pose、err）
        print("right_state_dict:", right_state_dict)
        # 提取pose列表，分离“位置坐标”和“欧拉角”
        left_pose_list = left_state_dict["pose"]
        # 提取pose列表，分离“位置坐标”和“欧拉角”
        right_pose_list = right_state_dict["pose"]

        print("left_robot机械臂当前状态:")
        print(
            f"  - 位置坐标 (x,y,z): ({left_pose_list[0]:.6f}, {left_pose_list[1]:.6f}, {left_pose_list[2]:.6f})")
        print(
            f"  - 欧拉角 (roll,pitch,yaw): ({left_pose_list[3]:.3f}, {left_pose_list[4]:.3f}, {left_pose_list[5]:.3f})")

        print("right_robot机械臂当前状态:")
        print(
            f"  - 位置坐标 (x,y,z): ({right_pose_list[0]:.6f}, {right_pose_list[1]:.6f}, {right_pose_list[2]:.6f})")
        print(
            f"  - 欧拉角 (roll,pitch,yaw): ({right_pose_list[3]:.3f}, {right_pose_list[4]:.3f}, {right_pose_list[5]:.3f})")

        left_robot_euler_list = [
            # 打包成列表：[rx, ry, rz]
            left_pose_list[3], left_pose_list[4], left_pose_list[5]]
        self.left_robot_pos = left_pose_list[0:3]
        self.left_robot_quat = self.left_robot.rm_algo_euler2quaternion(
            left_robot_euler_list)

        right_robot_euler_list = [
            # 打包成列表：[rx, ry, rz]
            right_pose_list[3], right_pose_list[4], right_pose_list[5]]
        self.right_robot_pos = right_pose_list[0:3]
        # 将欧拉角转四元数
        self.right_robot_quat = self.right_robot.rm_algo_euler2quaternion(
            right_robot_euler_list)

        self.robot_received = True
        time.sleep(0.01)  # 状态更新频率50Hz，与控制频率匹配

    def left_arm_control(self) -> None:
        print("[调试] 左机械臂控制线程运行中")
        try:
            i = 0
            while self.running:
                if not self.initialized:
                    self.retarget()
                    time.sleep(0.1)
                    print("Left init success")

                else:

                    target_pos = (np.array(self.vr_left_pos) +
                                  np.array(self.left_pos_offset)).tolist()
                    target_quat = ((R.from_quat(
                        self.vr_left_quat) * R.from_quat(self.left_quat_offset)).as_quat()).tolist()
                    target_euler = self.left_robot.rm_algo_quaternion2euler(
                        target_quat)
                    pose = target_pos + target_quat
                    try:
                        if (pose != None):
                            # self.left_robot.rm_movej_p(pose=pose, v=20, r=0, connect=0, block=0)
                            print(f"pose: {pose}")
                            # print(f"currnt robot pose:{self.left_robot_pos},{self.left_robot_quat}")
                            # print(self.left_robot.rm_movep_canfd(pose, False, 1, 60))

                            self.left_robot.rm_movep_canfd(pose, False, 1, 60)

                            # self.left_robot.rm_movej_p(pose=pose, v=10, r=0, connect=0, block=0)

                            # self.left_robot.rm_movej([0, 0, 0, 0, 10, 0], 20, 0, 0, 1)
                            # self.left_robot.rm_movej([0, 0, 0, 0, 30, 0], 20, 0, 0, 1)
                            # self.left_robot.rm_movej([0, 0, 0, 0, 50, 0], 20, 0, 0, 1)
                            # self.left_robot.rm_movej([0, 0, 0, 0, 70, 0], 20, 0, 0, 1)
                            # self.left_robot.rm_movej([0, 0, 0, 0, 90, 0], 20, 0, 0, 1)

                            # [-5.8979997634887695, 50.257999420166016, 65.19000244140625, -173.90499877929688, -63.75600051879883, -2.0280001163482666]
                            # time.sleep(1)
                            # self.left_robot.rm_movej_canfd( [-5.8979997634887695, 50.257999420166016, 65.19000244140625, -173.90499877929688, -63.75600051879883, -2.0280001163482666], True, 0, 1, 50)
                            # time.sleep(0.1)
                            # self.left_robot.rm_movej_canfd( [-5.8979997634887695, 50.257999420166016, 65.19000244140625, -173.90499877929688, -43.75600051879883, -2.0280001163482666], True, 0, 1, 50)
                            # time.sleep(0.1)
                            # self.left_robot.rm_movej_canfd( [-5.8979997634887695, 50.257999420166016, 65.19000244140625, -173.90499877929688, -23.75600051879883, -2.0280001163482666], True, 0, 1, 50)
                            # angle_init = -23.75600051879883
                            # angle_final = -63.75600051879883
                            # self.left_robot.rm_movej_canfd( [-5.8979997634887695, 50.257999420166016, 65.19000244140625, -173.90499877929688, angle_init, -2.0280001163482666], False, 0, 1, 50)
                            # time.sleep(1)
                            # for i in range(100):
                            #     angle_curr = angle_init + (angle_final - angle_init)*i/100
                            #     self.left_robot.rm_movej_canfd( [-5.8979997634887695, 50.257999420166016, 65.19000244140625, -173.90499877929688, angle_curr, -2.0280001163482666], True, 0, 1, 50)
                            #     time.sleep(0.01)

                    except Exception as e:
                        print(f"[控制-左机械臂] 异常: {e}")
                time.sleep(0.001)

        except KeyboardInterrupt:
            print("[调试] 左机械臂控制线程异常退出")

    def right_arm_control(self) -> None:
        print("[调试] 右机械臂控制线程运行中")
        try:
            while self.running:
                if not self.initialized:
                    self.retarget()
                    time.sleep(0.1)
                    print("Right init success")
                else:

                    target_pos = (np.array(self.vr_right_pos) +
                                  np.array(self.right_pos_offset)).tolist()
                    target_quat = ((R.from_quat(
                        self.vr_right_quat) * R.from_quat(self.right_quat_offset)).as_quat()).tolist()
                    target_euler = self.right_robot.rm_algo_quaternion2euler(
                        target_quat)
                    pose = target_pos + target_quat
                    try:
                        if (pose != None):
                            print(f"pose: {pose}")
                            # print(f"currnt robot pose:{self.right_robot_pos},{self.right_robot_quat}")
                            # self.right_robot.rm_movej_p(pose=pose, v=5, r=0, connect=0, block=0)
                            # self.right_robot.rm_movep_canfd(pose, False, 1, 60)

                            # self.right_robot.rm_movej_p(
                            #     pose=pose, v=10, r=0, connect=0, block=0)
                            self.right_robot.rm_movej(
                                [0, 0, 0, 0, 0, 0], 10, 0, 0, 1)

                    except Exception as e:
                        print(f"[控制-右机械臂] 异常: {e}")
                time.sleep(0.001)
        except KeyboardInterrupt:
            print("[调试] 右机械臂控制线程异常退出")

    def init_robot(self) -> None:
        # 初始化机械臂
        pose = [-0.388573, 0.024712, 0.153489, -3.046, -0.015, 3.027]
        # self.left_robot.rm_movep_canfd(pose, False, 1, 60)
        # self.right_robot.rm_movep_canfd(pose, False, 1, 60)
        # time.sleep(1)

    # -------------------- 生命周期 --------------------
    def start(self) -> bool:
        print("[启动] 连接左右机械臂...")

        self._init_ros2()

        self.left_robot = self._connect_robot(
            self.left_ip, self.port, self.level, mode=2)
        self.right_robot = self._connect_robot(
            self.right_ip, self.port, self.level)
        if self.left_robot is None or self.right_robot is None:
            print("[启动] 机械臂连接失败")
            return False
        if self.left_robot is not None and self.right_robot is not None:
            print("[启动] 初始化机械臂...")
            self.init_robot()

        self.running = True

        # 启动线程：VR接收 + 左臂控制 + 右臂控制
        self.vr_thread = threading.Thread(
            target=self._vr_receiver_loop, name="vr_receiver", daemon=True)
        self.left_control_thread = threading.Thread(
            target=self.left_arm_control, name="left_arm_control", daemon=True)
        self.right_control_thread = threading.Thread(
            target=self.right_arm_control, name="right_arm_control", daemon=True)
        self.update_state_thread = threading.Thread(
            target=self.arm_update_state, name="left_arm_update_state", daemon=True)

        self.vr_thread.start()
        # self.left_control_thread.start()
        self.right_control_thread.start()
        self.update_state_thread.start()
        print("[启动] 线程已启动")
        return True

    def run_until_stop(self) -> None:
        print("[运行] 按 Ctrl+C 退出")

        # 定义60秒后自动终止的函数
        def auto_shutdown():
            print("\n[定时终止] 60秒已到，开始自动退出...")
            # 触发机械臂缓停
            print("机械臂运动轨迹缓停")
            if self.left_robot:
                print("left_robot 缓停:", self.left_robot.rm_set_arm_slow_stop())
            if self.right_robot:
                print("right_robot 缓停:", self.right_robot.rm_set_arm_slow_stop())
            # 终止主循环
            self.running = False

        # 启动60秒定时器（60秒后执行auto_shutdown）
        self._auto_timer = threading.Timer(60.0, auto_shutdown)
        self._auto_timer.start()  # 启动定时器

        try:
            while self.running:
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("[运行] 用户中断")

            print("机械臂运动轨迹缓停")
            print("left_robot 缓停:", self.left_robot.rm_set_arm_slow_stop())
            print("right_robot 缓停:", self.right_robot.rm_set_arm_slow_stop())
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        print("[关闭] 停止线程并释放资源...")
        self.running = False

        # 等待线程结束
        for t in [self.left_control_thread, self.right_control_thread, self.update_state_thread, self.vr_thread]:
            if t is not None and t.is_alive():
                t.join(timeout=1.0)

        # 断开机械臂
        if self.left_robot:
            self._disconnect_robot(self.left_robot)
            self.left_robot = None
        if self.right_robot:
            self._disconnect_robot(self.right_robot)
            self.right_robot = None

        print("[关闭] 完成")


def main():
    ctrl = SimpleVRDoubleArmThreaded(
        left_ip="192.168.1.18",
        right_ip="192.168.2.19",
        port=8080,
        level=3,
    )
    if ctrl.start():
        ctrl.run_until_stop()
    else:
        print("[启动] 失败")


if __name__ == "__main__":
    main()
