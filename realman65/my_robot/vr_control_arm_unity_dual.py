#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import threading
import time
from typing import Optional, Callable

import numpy as np
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.time import Time
import tf2_ros

from realman65.my_robot.realman_65_interface_dual import Realman65Interface
from realman65.utils.data_handler import debug_print

from std_msgs.msg import String, Bool   
import json

from termcolor import cprint


ADJ_MAT = np.array([
    [1, 0,  0, 0],
    [0, 1,  0, 0],
    [0, 0,  1, 0],
    [0, 0,  0, 1],
], dtype=np.float64)



# ==================== ROS2 TF 客户端节点 ====================

class TFTeleopClient(rclpy.node.Node):
    """TF + 控制器 JSON 订阅节点（支持回调注入）。"""
    def __init__(self, on_controller_state: Optional[Callable[[dict], None]] = None,
                 node_name: str = "rm65_vr_tf_client") -> None:
        super().__init__(node_name)
        self.tf_buffer = tf2_ros.Buffer()
        # 不启用内部线程，由外部 executor 统一 spin
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self, spin_thread=False)
        self._on_controller_state_cb = on_controller_state
        self.create_subscription(String, "quest/controller_state", self._on_controller_state, 10)

    def _on_controller_state(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warning("Invalid controller JSON payload")
            return
        if callable(self._on_controller_state_cb):
            self._on_controller_state_cb(payload)


def euler_to_rot(euler_xyz: np.ndarray) -> R:
    return R.from_euler("xyz", euler_xyz, degrees=False)


def retarget_once(side: str, vr_pose: np.ndarray, robot_pose: np.ndarray) -> None:
    """根据当前 VR/机械臂姿态求一次偏移（按左右臂分开）。"""
    global pos_offset_left, rot_offset_left, left_retarget_done
    global pos_offset_right, rot_offset_right, right_retarget_done

    vr_pos = vr_pose[:3]
    vr_rot = euler_to_rot(vr_pose[3:])

    rb_pos = robot_pose[:3]
    rb_rot = euler_to_rot(robot_pose[3:])

    if side == "left":
        pos_offset_left = rb_pos - vr_pos
        rot_offset_left = vr_rot.inv() * rb_rot
        left_retarget_done = True
        debug_print("teleop", f"Left retarget OK, pos_offset={pos_offset_left}", "INFO")
    elif side == "right":
        pos_offset_right = rb_pos - vr_pos
        rot_offset_right = vr_rot.inv() * rb_rot
        right_retarget_done = True
        debug_print("teleop", f"Right retarget OK, pos_offset={pos_offset_right}", "INFO")


def apply_retarget(side: str, vr_pose: np.ndarray) -> np.ndarray:
    """把当前 VR 位姿映射为机械臂目标位姿（左右臂分开）。"""
    if side == "left":
        if pos_offset_left is None or rot_offset_left is None:
            return vr_pose.copy()
        mapped_pos = vr_pose[:3] + pos_offset_left
        mapped_rot = (euler_to_rot(vr_pose[3:]) * rot_offset_left).as_euler("xyz")
    else:
        if pos_offset_right is None or rot_offset_right is None:
            return vr_pose.copy()
        mapped_pos = vr_pose[:3] + pos_offset_right
        mapped_rot = (euler_to_rot(vr_pose[3:]) * rot_offset_right).as_euler("xyz")
    return np.concatenate([mapped_pos, mapped_rot])


def lookup_hand_pose(tf_client, parent_frame: str, child_frame: str):
    try:
        transform = tf_client.tf_buffer.lookup_transform(parent_frame, child_frame, Time())
    except Exception as exc:
        try:
            debug_print("teleop", f"TF lookup failed for {child_frame}: {exc}", "WARNING")
        except Exception:
            pass
        return None

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

class DualArmTeleop:
    """双臂 VR 遥操作器（封装 ROS/TF、重定标、控制、自动退出）。"""

    def __init__(self,
                 rm_interface: Realman65Interface,
                 parent_frame: str = "map",
                 hand_frame_left: str = "l_hand",
                 hand_frame_right: str = "r_hand",
                 hz_vr: float = 60.0,
                 hz_arm: float = 30.0,
                 enable_auto_shutdown: bool = False,
                 auto_shutdown_s: Optional[float] = None) -> None:
        self.rm_interface = rm_interface
        self.parent_frame = parent_frame
        self.hand_frame_left = hand_frame_left
        self.hand_frame_right = hand_frame_right
        self.hz_vr = float(hz_vr)
        self.hz_arm = float(hz_arm)
        self.enable_auto_shutdown = bool(enable_auto_shutdown)
        self.auto_shutdown_s = auto_shutdown_s

        # 运行控制
        self._running = threading.Event()
        self._running.set()
        # 遥操开关：左手扳机开启，右手扳机停止并复位
        self._teleop_enabled: bool = False
        self._start_time: Optional[float] = None

        # 线程与 ROS 执行器
        self.executor: Optional[SingleThreadedExecutor] = None
        self.spin_thread: Optional[threading.Thread] = None
        self.vr_thread: Optional[threading.Thread] = None
        self.arm_thread: Optional[threading.Thread] = None

        # ROS2 节点
        self.tf_client: Optional[TFTeleopClient] = None

        # 同步与状态
        self.pose_lock = threading.Lock()
        self.controller_lock = threading.Lock()

        # 目标与偏移（不使用全局变量）
        self.left_target_pose: Optional[np.ndarray] = None
        self.right_target_pose: Optional[np.ndarray] = None
        self.pos_offset_left: Optional[np.ndarray] = None
        self.rot_offset_left: Optional[R] = None
        self.pos_offset_right: Optional[np.ndarray] = None
        self.rot_offset_right: Optional[R] = None
        self.left_retarget_done = False
        self.right_retarget_done = False

        # 夹爪状态
        self.left_pending_gripper_cmd: Optional[int] = None
        self.right_pending_gripper_cmd: Optional[int] = None
        self.last_button_state = {
            "left_button_one": False, "left_index_trigger": False, "left_hand_trigger": False,
            "right_button_one": False, "right_index_trigger": False, "right_hand_trigger": False
        }

    # -------------------- Public API --------------------
    def start(self) -> None:
        """启动 ROS、控制线程与自动关闭计时。"""
        # 启动 ROS
        rclpy.init()
        self.tf_client = TFTeleopClient(on_controller_state=self._on_controller_state)
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.tf_client)
        self.spin_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.spin_thread.start()

        # 启动机器人控制线程（接口外部准备）
        self.rm_interface.start_control()

        # 启动 VR/控制线程
        self.vr_thread = threading.Thread(target=self._vr_worker, daemon=True)
        self.arm_thread = threading.Thread(target=self._arm_worker, daemon=True)
        self.vr_thread.start()
        self.arm_thread.start()

        # 仅在启用自动退出时开始计时
        if self.enable_auto_shutdown and self.auto_shutdown_s is not None:
            self._start_time = time.time()
        else:
            self._start_time = None
        debug_print("teleop", "Teleop started.", "INFO")

    def stop(self) -> None:
        """优雅停止所有线程并清理 ROS（幂等，可重复调用）。"""
        # 发出停止信号（即使已清除也继续做清理）
        self._running.clear()

        # 尝试停止控制线程/会话（如果接口提供），优先减少并发
        try:
            self.rm_interface.stop_control()
        except Exception:
            debug_print("teleop", "Failed to stop control thread.", "WARNING")

        # 等待线程结束
        if self.vr_thread:
            self.vr_thread.join(timeout=2.0)
        if self.arm_thread:
            self.arm_thread.join(timeout=2.0)

        # 关闭 ROS
        if self.executor:
            try:
                self.executor.shutdown()
            except Exception:
                debug_print("teleop", "Failed to shutdown executor.", "WARNING")
        if self.tf_client:
            try:
                self.tf_client.destroy_node()
            except Exception:
                pass
        try:
            rclpy.shutdown()
        except Exception:
            pass
        if self.spin_thread:
            self.spin_thread.join(timeout=2.0)

        debug_print("teleop", "Teleop stopped.", "INFO")

    def run(self) -> None:
        """阻塞运行，直到 CTRL+C 或自动退出。"""
        self.start()
        try:
            while self._running.is_set():
                time.sleep(0.2)
                if self.enable_auto_shutdown and self.auto_shutdown_s is not None and self._start_time is not None:
                    if time.time() - self._start_time >= self.auto_shutdown_s:
                        debug_print("teleop", f"Auto shutdown after {self.auto_shutdown_s}s.", "INFO")
                        break
        except KeyboardInterrupt:
            debug_print("teleop", "User requested shutdown.", "INFO")
        finally:
            # 先停止线程与 ROS，再做复位，避免并发调用 SDK
            self.stop()
            try:
                self.rm_interface.reset()
            except Exception:
                pass

    # -------------------- Internal helpers --------------------
    def _on_controller_state(self, payload: dict) -> None:
        """控制器状态订阅回调：更新夹爪命令（边沿触发）。"""
        left = payload.get("left", {})
        right = payload.get("right", {})
        left_index_val = float(left.get("index_trigger", 0.0))
        left_index_pressed = left_index_val > 0.5  # 阈值可按需调整
        left_hand_val = float(left.get("hand_trigger", 0.0))
        left_hand_pressed = left_hand_val > 0.8
        right_index_val = float(right.get("index_trigger", 0.0))
        right_index_pressed = right_index_val > 0.5
        right_hand_val = float(right.get("hand_trigger", 0.0))
        right_hand_pressed = right_hand_val > 0.8

        with self.controller_lock:
            # 左侧 index_trigger 控制夹爪：按下=夹紧，松开=松开（边沿触发）
            if left_index_pressed != self.last_button_state["left_index_trigger"]:
                self.left_pending_gripper_cmd = 1 if left_index_pressed else 0
            self.last_button_state["left_index_trigger"] = left_index_pressed

            # 左手扳机：开启遥操（边沿触发）
            if left_hand_pressed and not self.last_button_state["left_hand_trigger"]:
                self._enable_teleop()
            self.last_button_state["left_hand_trigger"] = left_hand_pressed

            # 右侧 index_trigger 控制夹爪：按下=夹紧，松开=松开（边沿触发）
            if right_index_pressed != self.last_button_state["right_index_trigger"]:
                self.right_pending_gripper_cmd = 1 if right_index_pressed else 0
            self.last_button_state["right_index_trigger"] = right_index_pressed
            # 右手扳机：停止遥操并复位（边沿触发，后台执行）
            if right_hand_pressed and not self.last_button_state["right_hand_trigger"]:
                threading.Thread(target=self._disable_teleop_and_reset, daemon=True).start()
            self.last_button_state["right_hand_trigger"] = right_hand_pressed

    def _perform_stop_and_reset(self) -> None:
        """在独立线程中执行停止控制与复位，避免在 ROS 回调内阻塞或死锁。"""
        try:
            debug_print("teleop", "Emergency stop requested: stopping workers", "WARNING")
            # 停止循环并统一走 stop() 清理（包含 stop_control）
            self._running.clear()
            try:
                self.stop()
            except Exception as exc:
                debug_print("teleop", f"teleop stop failed: {exc}", "WARNING")
            # 机械臂复位
            try:
                self.rm_interface.reset()
            except Exception as exc:
                debug_print("teleop", f"reset failed: {exc}", "WARNING")
            debug_print("teleop", "Emergency stop sequence completed", "INFO")
        except Exception as exc:
            debug_print("teleop", f"Emergency stop unexpected error: {exc}", "WARNING")

    @staticmethod
    def _euler_to_rot(euler_xyz: np.ndarray) -> R:
        return R.from_euler("xyz", euler_xyz, degrees=False)

    def _retarget_once(self, side: str, vr_pose: np.ndarray, robot_pose: np.ndarray) -> None:
        """根据当前 VR/机械臂姿态求一次偏移（左右臂分开）。"""
        vr_pos = vr_pose[:3]
        vr_rot = self._euler_to_rot(vr_pose[3:])

        rb_pos = robot_pose[:3]
        rb_rot = self._euler_to_rot(robot_pose[3:])

        if side == "left":
            self.pos_offset_left = rb_pos - vr_pos
            self.rot_offset_left = vr_rot.inv() * rb_rot
            self.left_retarget_done = True
            debug_print("teleop", f"Left retarget OK, pos_offset={self.pos_offset_left}", "INFO")
        elif side == "right":
            self.pos_offset_right = rb_pos - vr_pos
            self.rot_offset_right = vr_rot.inv() * rb_rot
            self.right_retarget_done = True
            debug_print("teleop", f"Right retarget OK, pos_offset={self.pos_offset_right}", "INFO")

    def _apply_retarget(self, side: str, vr_pose: np.ndarray) -> np.ndarray:
        """把当前 VR 位姿映射为机械臂目标位姿（左右臂分开）。"""
        if side == "left":
            if self.pos_offset_left is None or self.rot_offset_left is None:
                return vr_pose.copy()
            mapped_pos = vr_pose[:3] + self.pos_offset_left
            mapped_rot = (self._euler_to_rot(vr_pose[3:]) * self.rot_offset_left).as_euler("xyz")
        else:
            if self.pos_offset_right is None or self.rot_offset_right is None:
                return vr_pose.copy()
            mapped_pos = vr_pose[:3] + self.pos_offset_right
            mapped_rot = (self._euler_to_rot(vr_pose[3:]) * self.rot_offset_right).as_euler("xyz")
        return np.concatenate([mapped_pos, mapped_rot])

    def _lookup_hand_pose(self, child_frame: str) -> Optional[np.ndarray]:
        if not self.tf_client:
            return None
        try:
            transform = self.tf_client.tf_buffer.lookup_transform(self.parent_frame, child_frame, Time())
        except Exception as exc:
            debug_print("teleop", f"TF lookup failed for {child_frame}: {exc}", "WARNING")
            return None

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

    # -------------------- Workers --------------------
    def _vr_worker(self) -> None:
        period = 1.0 / self.hz_vr
        while self._running.is_set():
            
            # 未开启遥操时不下发控制与夹爪命令
            if not self._teleop_enabled:
                time.sleep(period)
                continue
            
            left_vr_pose = self._lookup_hand_pose(self.hand_frame_left)
            right_vr_pose = self._lookup_hand_pose(self.hand_frame_right)
            if left_vr_pose is None and right_vr_pose is None:
                time.sleep(period)
                continue
            with self.pose_lock:
                robot_pose_map = self.rm_interface.get_end_effector_pose()
                if left_vr_pose is not None:
                    if not self.left_retarget_done and robot_pose_map and "left_arm" in robot_pose_map:
                        robot_pose_left = np.asarray(robot_pose_map["left_arm"], dtype=np.float64)
                        self._retarget_once("left", left_vr_pose, robot_pose_left)
                    if self.left_retarget_done:
                        self.left_target_pose = self._apply_retarget("left", left_vr_pose)
                if right_vr_pose is not None:
                    if not self.right_retarget_done and robot_pose_map and "right_arm" in robot_pose_map:
                        robot_pose_right = np.asarray(robot_pose_map["right_arm"], dtype=np.float64)
                        robot_pose_right[1] += 0.5  # 右臂 y 轴偏移
                        self._retarget_once("right", right_vr_pose, robot_pose_right)
                    if self.right_retarget_done:
                        self.right_target_pose = self._apply_retarget("right", right_vr_pose)
            time.sleep(period)

    def _arm_worker(self) -> None:
        period = 1.0 / self.hz_arm
        while self._running.is_set():
            # 未开启遥操时不下发控制与夹爪命令
            if not self._teleop_enabled:
                time.sleep(period)
                continue
            left_pose_to_send = None
            right_pose_to_send = None
            with self.pose_lock:
                if self.left_retarget_done and self.left_target_pose is not None:
                    left_pose_to_send = self.left_target_pose.copy()
                if self.right_retarget_done and self.right_target_pose is not None:
                    right_pose_to_send = self.right_target_pose.copy()

            try:
                # 任意一侧有目标就下发（另一侧传 None）
                if left_pose_to_send is not None or right_pose_to_send is not None:
                    cprint(f"left_pose_to_send: {left_pose_to_send}", "green")
                    cprint(f"right_pose_to_send: {right_pose_to_send}", "red")
                    self.rm_interface.update(left_pose_to_send, right_pose_to_send)
            except Exception as exc:
                debug_print("teleop", f"send dual pose failed: {exc}", "WARNING")

            # 夹爪命令下发
            left_cmd = None
            right_cmd = None
            with self.controller_lock:
                if self.left_pending_gripper_cmd is not None:
                    left_cmd = self.left_pending_gripper_cmd
                    self.left_pending_gripper_cmd = None
                if self.right_pending_gripper_cmd is not None:
                    right_cmd = self.right_pending_gripper_cmd
                    self.right_pending_gripper_cmd = None

            if left_cmd is not None:
                debug_print("cmd", {"left": left_cmd}, "INFO")
                try:
                    self.rm_interface.set_gripper("left_arm", left_cmd)
                except Exception as exc:
                    debug_print("teleop", f"left gripper command failed: {exc}", "WARNING")
            if right_cmd is not None:
                debug_print("cmd", {"right": right_cmd}, "INFO")
                try:
                    self.rm_interface.set_gripper("right_arm", right_cmd)
                except Exception as exc:
                    debug_print("teleop", f"right gripper command failed: {exc}", "WARNING")

            # 自动退出计时（如果启用）
            if self.enable_auto_shutdown and self.auto_shutdown_s is not None and self._start_time is not None:
                if time.time() - self._start_time >= self.auto_shutdown_s:
                    self._running.clear()

            time.sleep(period)


    # -------------------- Teleop 开关与复位 --------------------
    def _enable_teleop(self) -> None:
        """开启遥操作：启动底层控制会话并允许VR/臂线程工作。"""
        # 重新启动底层控制线程（幂等）
        try:
            self.rm_interface.start_control()
        except Exception as exc:
            debug_print("teleop", f"start_control failed: {exc}", "WARNING")
        self._teleop_enabled = True
        debug_print("teleop", "Teleop enabled by left hand trigger.", "INFO")

    def _disable_teleop_and_reset(self) -> None:
        """停止遥操并复位机械臂（后台执行，避免阻塞主循环）。"""
        # 先关闭遥操，防止继续下发控制
        self._teleop_enabled = False
        # 清理待发夹爪命令，避免误发
        with self.controller_lock:
            self.left_pending_gripper_cmd = None
            self.right_pending_gripper_cmd = None

        def _do_stop_and_reset():
            try:
                # 停止底层控制线程，避免与复位并发
                try:
                    self.rm_interface.stop_control()
                except Exception as exc:
                    debug_print("teleop", f"stop_control failed: {exc}", "WARNING")
                time.sleep(0.1)
                # 执行复位
                debug_print("teleop", "Resetting robot (via right hand trigger)...", "INFO")
                try:
                    self.rm_interface.reset()
                except Exception as exc:
                    debug_print("teleop", f"reset failed: {exc}", "WARNING")
                # 复位后清空重定标与目标，确保下一次开启遥操重新配准
                with self.pose_lock:
                    self.left_target_pose = None
                    self.right_target_pose = None
                    self.pos_offset_left = None
                    self.rot_offset_left = None
                    self.pos_offset_right = None
                    self.rot_offset_right = None
                    self.left_retarget_done = False
                    self.right_retarget_done = False
                debug_print("teleop", "Robot reset completed and retarget state cleared.", "INFO")
            except Exception as exc:
                debug_print("teleop", f"disable+reset failed unexpectedly: {exc}", "WARNING")

        threading.Thread(target=_do_stop_and_reset, daemon=True).start()



# ==================== 主入口 ====================

def main() -> None:
    # 初始化接口并连接
    debug_print("teleop", "Initializing RM65 interface...", "INFO")
    rm_interface = Realman65Interface(auto_setup=False)
    rm_interface.set_up()
    rm_interface.reset()

    # 构建并运行遥操作器（默认不启用自动退出）
    teleop = DualArmTeleop(
        rm_interface=rm_interface,
        parent_frame="map",
        hand_frame_left="l_hand",
        hand_frame_right="r_hand",
        hz_vr=60.0,
        hz_arm=30.0,
        enable_auto_shutdown=False,
        auto_shutdown_s=10.0,
    )
    teleop.run()


if __name__ == "__main__":
    main()