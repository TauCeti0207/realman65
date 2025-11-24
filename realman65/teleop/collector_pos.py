#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS2 rclpy：订阅 PoseArray 与 controller_msgs/JointState（仅 finger q[0]）
并根据一个 Bool 开关话题（True=开始录制，False=停止录制）分段写入 CSV。
每次停止录制后 idx+1，等待下次 True 开新文件继续记。

CSV（每个 PoseArray 写一行，包含前两个 pose 的位姿）：
  type, stamp_sec, stamp_nanosec, frame_id,
  p0_idx, p0_px, p0_py, p0_pz, p0_ox, p0_oy, p0_oz, p0_ow,
  p1_idx, p1_px, p1_py, p1_pz, p1_ox, p1_oy, p1_oz, p1_ow,
  finger_name0, finger_q0, time
"""

import os
import csv
import math
import queue
from datetime import datetime
from threading import Thread, Event, Lock
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy
)
from geometry_msgs.msg import PoseArray
from std_msgs.msg import Bool  # <— 新增：flag 话题

# 友好提示：若未 source 工作区，清晰报错
try:
    from controller_msgs.msg import JointState
except Exception as _e:
    JointState = None
    _IMPORT_ERR = _e


# ------------------ 异步 CSV 写入器（单文件） ------------------
class CSVAsyncWriter:
    def __init__(self, filepath: str, header, flush_every_n: int = 10000,
                 max_queue: int = 200000):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.file = open(filepath, 'w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(header)

        self.q = queue.Queue(maxsize=max_queue)
        self.flush_every_n = max(1, int(flush_every_n))
        self._since_flush = 0
        self.stop_evt = Event()
        self.dropped = 0

        self.t = Thread(target=self._run, daemon=True)
        self.t.start()

    def put_row(self, row):
        try:
            self.q.put_nowait(row)
        except queue.Full:
            self.dropped += 1

    def _run(self):
        while not self.stop_evt.is_set() or not self.q.empty():
            try:
                row = self.q.get(timeout=0.1)
            except queue.Empty:
                row = None
            if row is not None:
                self.writer.writerow(row)
                self._since_flush += 1
                if self._since_flush >= self.flush_every_n:
                    self.file.flush()
                    self._since_flush = 0

        if self._since_flush > 0:
            self.file.flush()
        self.file.close()

    def stop(self):
        self.stop_evt.set()
        self.t.join()


# ------------------ 主节点 ------------------
class PoseAndFingerLogger(Node):
    def __init__(self,
                 pose_topic='/dad03/state/pose',        # 若话题不同，改这里
                 finger_topic='/dad03/state/finger',     # 若话题不同，改这里
                 joint_topic='/dad03/state/joint', 
                 flag_topic='/save_frame2buffer_flag',   # Bool：True=开始, False=停止
                 out_dir='data_logs',
                 filename_prefix='pose_finger'):
        super().__init__('pose_and_finger_q0_logger')

        if JointState is None:
            raise RuntimeError(
                "无法导入 controller_msgs/msg/JointState。\n"
                f"原始错误：{_IMPORT_ERR}\n"
                "请先执行：source /opt/ros/$ROS_DISTRO/setup.bash && "
                "source ~/teleop/colcon_ws/install/setup.bash"
            )

        # 传感器流 QoS；如需更稳可改 RELIABLE
        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10000,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE
        )

        self.out_dir = out_dir
        self.filename_prefix = filename_prefix

        # 当前录制状态/文件
        self.recording = False
        self.file_idx = 0
        self.csv_writer = None
        self.csv_path = None

        # 缓存：最新 finger 值（sec, nsec, fid, name0, q0）
        self._finger_lock = Lock()
        self._latest_finger = None

        self._joint_lock = Lock()
        self._joint_names = ["neck_pitch_joint","neck_yaw_joint", 
                             "left_shoulder_pitch_joint", "left_shoulder_roll_joint","left_shoulder_yaw_joint","left_elbow_joint"
                             ,"left_wrist_yaw_joint", "left_wrist_pitch_joint", "left_hand_yaw_joint"
                             ,"right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint"
                             ,"right_wrist_yaw_joint", "right_wrist_pitch_joint", "right_hand_yaw_joint"]
        self._latest_joint = None

        # 订阅
        self.pose_sub = self.create_subscription(PoseArray, pose_topic, self.pose_cb, qos)
        self.finger_sub = self.create_subscription(JointState, finger_topic, self.finger_cb, qos)
        self.joint_sub = self.create_subscription(JointState, joint_topic, self.joint_cb, qos)
        self.flag_sub = self.create_subscription(Bool, flag_topic, self.flag_cb, qos)

        self.get_logger().info(f'PoseArray 订阅：{pose_topic}')
        self.get_logger().info(f'Finger(q[0]) 订阅：{finger_topic}')
        self.get_logger().info(f'Flag 订阅：{flag_topic}')
        self.get_logger().info(f'输出目录：{self.out_dir}')

        # 提示下 finger 回调的名字（避免 IDE 警告未使用）
        # _ = finger_cb  # noqa: F841

    # ---------- 开始/停止录制 ----------
    def start_recording(self):
        if self.recording:
            self.get_logger().warn('已在录制中，忽略 start。')
            return
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        # 文件名：prefix_idx_timestamp.csv，便于检索
        self.csv_path = os.path.join(
            self.out_dir, f'{self.filename_prefix}_{self.file_idx:03d}_{ts}.csv'
        )
        header = [
            'type',
            'stamp_sec', 'stamp_nanosec', 'frame_id',
            # pose[0]
            'p0_idx',
            'p0_px', 'p0_py', 'p0_pz',
            'p0_ox', 'p0_oy', 'p0_oz', 'p0_ow',
            # pose[1]
            'p1_idx',
            'p1_px', 'p1_py', 'p1_pz',
            'p1_ox', 'p1_oy', 'p1_oz', 'p1_ow',
            # finger（来自最新一次 JointState）
            'finger_name0', 'finger_q0'
            , "neck_pitch_joint"
            , "neck_yaw_joint"
            , "left_shoulder_pitch_joint"
            , "left_shoulder_roll_joint"
            , "left_shoulder_yaw_joint"
            , "left_elbow_joint"
            , "left_wrist_yaw_joint"
            , "left_wrist_pitch_joint"
            , "left_hand_yaw_joint"
            , "right_shoulder_pitch_joint"
            , "right_shoulder_roll_joint"
            , "right_shoulder_yaw_joint"
            , "right_elbow_joint"
            , "right_wrist_yaw_joint"
            , "right_wrist_pitch_joint"
            , "right_hand_yaw_joint"
            , 'time'  # = time.time()
        ]
        self.csv_writer = CSVAsyncWriter(
            filepath=self.csv_path, header=header,
            flush_every_n=10000, max_queue=200000
        )
        self.recording = True
        self.get_logger().info(f'开始录制 -> {self.csv_path}')

    def stop_recording(self):
        if not self.recording:
            self.get_logger().warn('当前未在录制，忽略 stop。')
            return
        try:
            if self.csv_writer:
                self.get_logger().info('正在停止写线程并关闭文件…')
                self.csv_writer.stop()
                self.get_logger().info(
                    f'写线程已停。若队列曾满而丢弃：dropped={self.csv_writer.dropped}'
                )
        finally:
            self.csv_writer = None
            self.recording = False
            self.get_logger().info(f'已保存：{self.csv_path}')
            self.csv_path = None
            # 文件序号 +1，等待下次 True 开新文件
            self.file_idx += 1
            self.get_logger().info(f'文件索引已递增至 {self.file_idx}')

    # ---------- Flag 回调 ----------
    def flag_cb(self, msg: Bool):
        flag = bool(msg.data)
        if flag and not self.recording:
            self.start_recording()
        elif (not flag) and self.recording:
            self.stop_recording()
        else:
            # 状态未变化：忽略
            pass

    # ---------- PoseArray 回调：仅在录制中才写入 ----------
    def pose_cb(self, msg: PoseArray):
        if not self.recording or self.csv_writer is None:
            return  # 不在录制中，直接丢弃
        sec, nsec = msg.header.stamp.sec, msg.header.stamp.nanosec
        fid = msg.header.frame_id

        # 最新 finger
        with self._finger_lock:
            f = self._latest_finger
        if f is None:
            fname, fq0 = '', float('nan')
        else:
            _, _, _, fname, fq0 = f

        with self._joint_lock:
            joint_names, joints_values = self._latest_joint
            

        # 取前两个 pose，不足则用占位
        def pack_pose(p, idx):
            if p is None:
                return (-1, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan)
            return (
                idx,
                p.position.x, p.position.y, p.position.z,
                p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w
            )

        p0 = msg.poses[0] if len(msg.poses) > 0 else None
        p1 = msg.poses[1] if len(msg.poses) > 1 else None
        p0_pack = pack_pose(p0, 0)
        p1_pack = pack_pose(p1, 1)

        self.csv_writer.put_row([
            'pose',           # 固定
            sec, nsec, fid,
            *p0_pack,
            *p1_pack,
            fname, fq0,
            joints_values[0],
            joints_values[1],
            joints_values[2],
            joints_values[3],
            joints_values[4],
            joints_values[5],
            joints_values[6],
            joints_values[7],
            joints_values[8],
            joints_values[9],
            joints_values[10],
            joints_values[11],
            joints_values[12],
            joints_values[13],
            joints_values[14],
            joints_values[15],
            time.time()
        ])

    # ---------- JointState 回调：仅缓存 finger q[0] ----------
    def finger_cb(self, msg: JointState):
        sec, nsec = msg.header.stamp.sec, msg.header.stamp.nanosec
        fid = msg.header.frame_id
        name0 = msg.name[0] if len(msg.name) > 0 else ''
        q0 = float('nan') if len(msg.q) == 0 else float(msg.q[0])
        with self._finger_lock:
            self._latest_finger = (sec, nsec, fid, name0, q0)

    def joint_cb(self, msg: JointState):
        sec, nsec = msg.header.stamp.sec, msg.header.stamp.nanosec
        fid = msg.header.frame_id
        names = msg.name if len(msg.name) > 0 else ''
        values = float('nan') if len(msg.q) == 0 else msg.q
        # print(values)
        with self._joint_lock:
            self._latest_joint = (names, values)

    def destroy_node(self):
        try:
            # 若正在录制，确保关闭文件
            if getattr(self, 'recording', False):
                self.stop_recording()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PoseAndFingerLogger(
        pose_topic='/dad03/state/pose',        # 若话题不同，改这里
        finger_topic='/dad03/state/finger',    # 若话题不同，改这里
        joint_topic='/dad03/state/joint',
        flag_topic='/save_frame2buffer_flag',  # Bool flag 话题
        out_dir='data_logs',
        filename_prefix='pose_finger'
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
