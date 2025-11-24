#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unity 双目接收端的 Python 发送程序：
- 发布两路视频（左/右目），名称可配置（默认 left/right）
- 可选创建 DataChannel 订阅者，接收 Unity 的手柄数据并打印

要求：先启动 signaling_server.py，然后运行本程序；Unity 端运行 WebRTCStreamer.cs（作为视频订阅 + 数据发布）
/usr/bin/python3 unity_dual_sender.py --server ws://127.0.0.1:8000 --room quest3_room
"""

import logging
import threading
import time
from typing import Optional
import cv2
import numpy as np


logger = logging.getLogger("unity_dual_sender")


class CameraCapture:
    """采集摄像头帧，并提供左右目分割的帧"""

    def __init__(self, camera_index: int, width: int, height: int, fps: int):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()
        self._latest: Optional[np.ndarray] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> bool:
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap or not self.cap.isOpened():
            logger.warning("摄像头打开失败，使用黑帧代替")
            self._running = True
            self._thread = threading.Thread(
                target=self._black_loop, daemon=True)
            self._thread.start()
            return True

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    def _loop(self):
        interval = 1.0 / max(1, self.fps)
        while self._running:
            ok, frame = self.cap.read() if self.cap else (False, None)
            if not ok or frame is None:
                frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            with self._lock:
                self._latest = frame
            time.sleep(interval)

    def _black_loop(self):
        interval = 1.0 / max(1, self.fps)
        black = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        while self._running:
            with self._lock:
                self._latest = black.copy()
            time.sleep(interval)

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass

    def get_left(self) -> np.ndarray:
        with self._lock:
            f = None if self._latest is None else self._latest.copy()
        if f is None:
            # return np.zeros((self.height, self.width // 2, 3), dtype=np.uint8)
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # h, w = f.shape[:2]
        # mid = w // 2
        # return f[:, :mid]
        return f

    def get_right(self) -> np.ndarray:
        with self._lock:
            f = None if self._latest is None else self._latest.copy()
        if f is None:
            # return np.zeros((self.height, self.width // 2, 3), dtype=np.uint8)
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # h, w = f.shape[:2]
        # mid = w // 2
        # return f[:, mid:]
        return f
