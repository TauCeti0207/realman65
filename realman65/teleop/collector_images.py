#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
import cv2
import time
import os

from tiga_save_buffer2disk import persist_shortbuffer_pairs


class DualCamRecorder(Node):
    def __init__(self):
        super().__init__('dual_cam_recorder_simple')

        # ============ 固定参数 ============
        self.flag_topic = '/save_frame2buffer_flag'
        self.head_camera_id = 0
        self.chest_camera_id = 1
        self.video_width = 1280
        self.video_height = 720
        self.video_fps = 30
        self.output_base_dir = "/home/guest/teleop/video_server/tiga_data/pair_output"
        self.edge_margin = 5
        self.max_time_diff = 0.05
        self.start_index = 0

        # ============ 状态与缓存 ============
        self.save_frame2buffer_flag = False
        self.episode_idx = 0

        self.head_frame_buffer = []
        self.head_frame_timestep = []
        self.chest_frame_buffer = []
        self.chest_frame_timestep = []

        # ============ 初始化相机 ============
        self.get_logger().info(f'打开摄像头：头部={self.head_camera_id}, 胸部={self.chest_camera_id}')
        self.head_camera = cv2.VideoCapture(self.head_camera_id)
        self.chest_camera = cv2.VideoCapture(self.chest_camera_id)

        if not self.head_camera.isOpened():
            raise RuntimeError(f'无法打开头部摄像头 {self.head_camera_id}')
        if not self.chest_camera.isOpened():
            raise RuntimeError(f'无法打开胸部摄像头 {self.chest_camera_id}')

        for cam in [self.head_camera, self.chest_camera]:
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.video_width)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.video_height)
            cam.set(cv2.CAP_PROP_FPS, self.video_fps)

        # ============ 订阅Flag ============
        self.flag_sub = self.create_subscription(
            Bool,
            self.flag_topic,
            self.flag_cb,
            10
        )

        # ============ 定时器：按FPS采集 ============
        self.timer = self.create_timer(1.0 / self.video_fps, self.timer_cb)

        self.get_logger().info('双相机录制节点启动完成 ✅')

    # ----------------------------------------------------
    # 订阅回调：只修改标志
    # ----------------------------------------------------
    def flag_cb(self, msg: Bool):
        self.save_frame2buffer_flag = msg.data
        self.get_logger().info(f'更新采集标志: {self.save_frame2buffer_flag}')

    # ----------------------------------------------------
    # 定时器回调：采集 / 保存
    # ----------------------------------------------------
    def timer_cb(self):
        ret_head, head_frame = self.head_camera.read()
        ret_chest, chest_frame = self.chest_camera.read()


        if not ret_head or not ret_chest:
            self.get_logger().warn('读取相机帧失败')
            return

        if self.save_frame2buffer_flag:
            # 正在收集数据
            ts = time.time()
            self.head_frame_buffer.append(head_frame)
            self.head_frame_timestep.append(ts)
            
            ts = time.time()
            self.chest_frame_buffer.append(chest_frame)
            self.chest_frame_timestep.append(ts)
        else:
            # 不再收集，但如果缓存不为空就保存
            if len(self.head_frame_buffer) > 0:
                out_dir = f"{self.output_base_dir}{self.episode_idx}"
                os.makedirs(out_dir, exist_ok=True)
                self.get_logger().info(f"Flag=False，保存缓存到 {out_dir}")

                persist_shortbuffer_pairs(
                    self.head_frame_buffer, self.head_frame_timestep,
                    self.chest_frame_buffer, self.chest_frame_timestep,
                    out_dir=out_dir,
                    edge_margin=self.edge_margin,
                    max_time_diff=self.max_time_diff,
                    start_index=self.start_index,
                )

                self.episode_idx += 1
                # 不需要手动清空，persist_shortbuffer_pairs 内部已经清理了

    # ----------------------------------------------------
    # 节点退出时释放资源
    # ----------------------------------------------------
    def destroy_node(self):
        if self.head_camera.isOpened():
            self.head_camera.release()
        if self.chest_camera.isOpened():
            self.chest_camera.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DualCamRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
