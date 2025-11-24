#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: webrtc_client.py
功能: 通用WebRTC客户端组件
职责: 处理WebRTC连接、ICE协商、媒体/数据通道管理 
"""

import asyncio
import logging
import time
from typing import Optional, Callable, Any, List
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceServer, RTCConfiguration, MediaStreamTrack, RTCRtpSender
from av import VideoFrame, CodecContext
from fractions import Fraction
import numpy as np

logger = logging.getLogger("webrtc_client")


class WebRTCPeer:
    """通用WebRTC对等连接管理"""

    def __init__(self,
                 peer_id: str,
                 peer_name: str = None,
                 ice_servers: List[str] = None,
                 on_connection_state_change: Callable = None,
                 on_ice_candidate: Callable = None):
        """
        初始化WebRTC对等连接

        Args:
            peer_id: 对等连接的唯一标识
            peer_name: 对等连接的显示名称
            ice_servers: ICE服务器列表，默认使用Google STUN服务器
            on_connection_state_change: 连接状态改变回调 callback(peer_id, state)
            on_ice_candidate: ICE候选回调 callback(peer_id, candidate)
        """
        self.peer_id = peer_id
        self.peer_name = peer_name or peer_id
        self.pc = None
        self.state = "new"
        self.created_at = time.time()

        # 回调函数
        self.on_connection_state_change = on_connection_state_change
        self.on_ice_candidate = on_ice_candidate

        # ICE服务器配置
        if ice_servers is None:
            ice_servers = [
                "stun:stun.l.google.com:19302",
                "stun:stun1.l.google.com:19302"
            ]

        self.ice_servers = [RTCIceServer(urls=url) for url in ice_servers]

        # 媒体轨道和数据通道
        self.tracks = []
        self.data_channels = {}

        logger.info(f"初始化WebRTC对等连接: {self.peer_name} ({self.peer_id})")

    async def create_connection(self):
        """创建WebRTC连接"""
        try:
            # 如果已存在连接则直接返回
            if self.pc is not None:
                return True
            # 创建RTCPeerConnection
            config = RTCConfiguration(iceServers=self.ice_servers)
            self.pc = RTCPeerConnection(config)

            # 连接状态监控
            @self.pc.on("connectionstatechange")
            async def on_connectionstatechange():
                old_state = self.state
                self.state = self.pc.connectionState

                if old_state != self.state:
                    logger.info(
                        f"📡 {self.peer_name} P2P连接状态: {old_state} → {self.state}")

                    # 调用外部回调
                    if self.on_connection_state_change:
                        try:
                            if asyncio.iscoroutinefunction(self.on_connection_state_change):
                                await self.on_connection_state_change(self.peer_id, self.state)
                            else:
                                self.on_connection_state_change(
                                    self.peer_id, self.state)
                        except Exception as e:
                            logger.error(f"连接状态回调出错: {e}")

            # ICE候选处理
            @self.pc.on("icecandidate")
            def on_icecandidate_internal(candidate):
                if candidate and self.on_ice_candidate:
                    try:
                        # 创建异步任务来处理ICE候选
                        asyncio.create_task(
                            self._handle_ice_candidate_callback(candidate))
                    except Exception as e:
                        logger.error(f"ICE候选回调出错: {e}")

            # 数据通道接收处理（用于接收端）
            @self.pc.on("datachannel")
            def on_datachannel(channel):
                logger.info(f"📨 接收到数据通道: {channel.label}")
                self.data_channels[channel.label] = channel

                # 设置数据通道事件（可以被外部重写）
                self._setup_data_channel_events(channel)

            return True

        except Exception as e:
            logger.error(f"创建WebRTC连接失败: {e}")
            return False

    async def _handle_ice_candidate_callback(self, candidate):
        """处理ICE候选回调"""
        try:
            if asyncio.iscoroutinefunction(self.on_ice_candidate):
                await self.on_ice_candidate(self.peer_id, candidate)
            else:
                self.on_ice_candidate(self.peer_id, candidate)
        except Exception as e:
            logger.error(f"ICE候选回调处理出错: {e}")

    def force_codec(self, pc, sender, forced_codec):
        kind = forced_codec.split("/")[0]
        codecs = RTCRtpSender.getCapabilities(kind).codecs
        transceiver = next(t for t in pc.getTransceivers() if t.sender == sender)
        transceiver.setCodecPreferences(
            [codec for codec in codecs if codec.mimeType == forced_codec]
        )

    def add_track(self, track):
        """添加媒体轨道"""
        if not self.pc:
            raise RuntimeError("WebRTC连接尚未创建")

        # self.pc.addTrack(track)

        self.video_sender = self.pc.addTrack(track)
        self.force_codec(self.pc, self.video_sender, 'video/H264')

        self.tracks.append(track)
        logger.info(f"✅ 已添加媒体轨道给 {self.peer_name}: {track.kind}")

    def create_data_channel(self,
                            label: str,
                            on_open: Callable = None,
                            on_close: Callable = None,
                            on_message: Callable = None,
                            ordered: bool = True,
                            max_packet_life_time: Optional[int] = None,
                            max_retransmits: Optional[int] = None,
                            protocol: Optional[str] = None,
                            negotiated: Optional[bool] = None,
                            channel_id: Optional[int] = None):
        """创建数据通道（支持WebRTC数据通道属性配置）"""
        if not self.pc:
            raise RuntimeError("WebRTC连接尚未创建")

        normalized_protocol = protocol if isinstance(protocol, str) else ""
        normalized_negotiated = bool(
            negotiated) if negotiated is not None else False

        channel = self.pc.createDataChannel(
            label,
            ordered=ordered,
            maxPacketLifeTime=max_packet_life_time,
            maxRetransmits=max_retransmits,
            protocol=normalized_protocol,
            negotiated=normalized_negotiated,
            id=channel_id,
        )
        self.data_channels[label] = channel

        # 设置事件处理器
        if on_open:
            channel.on("open", lambda: self._safe_callback(on_open, channel))

        if on_close:
            channel.on("close", lambda: self._safe_callback(on_close, channel))

        if on_message:
            channel.on("message", lambda msg: self._safe_callback(
                on_message, channel, msg))

        logger.info(f"✅ 已创建数据通道给 {self.peer_name}: {label}")
        return channel

    def _setup_data_channel_events(self, channel):
        """设置数据通道事件（可被子类重写）"""
        @channel.on("open")
        def on_open():
            logger.info(f"📨 数据通道已打开: {channel.label} ({self.peer_name})")

        @channel.on("close")
        def on_close():
            logger.info(f"📨 数据通道已关闭: {channel.label} ({self.peer_name})")

        @channel.on("message")
        def on_message(message):
            logger.info(
                f"📨 收到数据: {channel.label} ({self.peer_name}): {message}")

    def _safe_callback(self, callback, *args):
        """安全执行回调函数"""
        try:
            if asyncio.iscoroutinefunction(callback):
                asyncio.create_task(callback(*args))
            else:
                callback(*args)
        except Exception as e:
            logger.error(f"回调函数执行出错: {e}")

    async def create_offer(self):
        """创建offer"""
        if not self.pc:
            raise RuntimeError("WebRTC连接尚未创建")

        try:
            # 仅在稳定态创建本地 offer，避免重复/竞态
            if getattr(self.pc, "signalingState", "") != "stable":
                logger.debug(
                    f"跳过创建offer，当前signalingState={self.pc.signalingState}")
                return None
            offer = await self.pc.createOffer()
            await self.pc.setLocalDescription(offer)
            logger.info(f"✅ 创建offer成功: {self.peer_name}")
            return self.pc.localDescription.sdp
        except Exception as e:
            logger.error(f"创建offer失败: {e}")
            return None

    async def create_answer(self):
        """创建answer"""
        if not self.pc:
            raise RuntimeError("WebRTC连接尚未创建")

        try:
            # 仅在已接收远端 offer 的状态创建 answer
            if getattr(self.pc, "signalingState", "") != "have-remote-offer":
                logger.debug(
                    f"跳过创建answer，当前signalingState={self.pc.signalingState}")
                return None
            answer = await self.pc.createAnswer()
            await self.pc.setLocalDescription(answer)
            logger.info(f"✅ 创建answer成功: {self.peer_name}")
            return self.pc.localDescription.sdp
        except Exception as e:
            logger.error(f"创建answer失败: {e}")
            return None

    async def handle_offer(self, sdp: str):
        """处理offer"""
        if not self.pc:
            raise RuntimeError("WebRTC连接尚未创建")

        try:
            # 仅在稳定态处理远端 offer，避免重复/竞态
            if getattr(self.pc, "signalingState", "") != "stable":
                logger.debug(
                    f"忽略处理offer，当前signalingState={self.pc.signalingState}")
                return False
            offer = RTCSessionDescription(sdp=sdp, type="offer")
            await self.pc.setRemoteDescription(offer)
            logger.info(f"✅ Offer处理完成: {self.peer_name}")
            return True
        except Exception as e:
            logger.error(f"处理offer失败: {e}")
            return False

    async def handle_answer(self, sdp: str):
        """处理answer"""
        if not self.pc:
            raise RuntimeError("WebRTC连接尚未创建")

        try:
            # 仅在本地已有 offer 的状态处理远端 answer
            if getattr(self.pc, "signalingState", "") != "have-local-offer":
                logger.debug(
                    f"忽略处理answer，当前signalingState={self.pc.signalingState}")
                return True
            answer = RTCSessionDescription(sdp=sdp, type="answer")
            await self.pc.setRemoteDescription(answer)
            logger.info(f"✅ Answer处理完成: {self.peer_name}")
            return True
        except Exception as e:
            logger.error(f"处理answer失败: {e}")
            return False

    async def handle_ice_candidate(self, candidate):
        """处理ICE候选"""
        if not self.pc:
            return False

        try:
            await self.pc.addIceCandidate(candidate)
            return True
        except Exception as e:
            logger.error(f"添加ICE候选失败: {e}")
            return False

    def get_data_channel(self, label: str):
        """获取数据通道"""
        return self.data_channels.get(label)

    def send_data(self, label: str, data: str):
        """通过数据通道发送数据"""
        channel = self.data_channels.get(label)
        if channel and channel.readyState == "open":
            channel.send(data)
            return True
        else:
            logger.warning(
                f"数据通道不可用: {label} (状态: {channel.readyState if channel else 'None'})")
            return False

    def is_connected(self):
        """检查是否已连接"""
        return self.state == "connected"

    def is_failed(self):
        """检查连接是否失败"""
        return self.state in ["failed", "closed", "disconnected"]

    def get_stats(self):
        """获取连接统计"""
        uptime = time.time() - self.created_at
        return {
            "peer_id": self.peer_id,
            "peer_name": self.peer_name,
            "state": self.state,
            "uptime": uptime,
            "tracks_count": len(self.tracks),
            "data_channels_count": len(self.data_channels),
            "data_channels": list(self.data_channels.keys())
        }

    async def close(self):
        """关闭连接"""
        logger.info(f"🔒 关闭WebRTC连接: {self.peer_name}")

        # 关闭数据通道
        for channel in self.data_channels.values():
            try:
                if hasattr(channel, 'close'):
                    channel.close()
            except Exception as e:
                logger.debug(f"关闭数据通道时的警告: {e}")

        # 关闭PC连接
        if self.pc:
            try:
                await self.pc.close()
            except Exception as e:
                logger.debug(f"关闭P2P连接时的警告: {e}")

        # 清理状态
        self.data_channels.clear()
        self.tracks.clear()
        self.state = "closed"


class WebRTCManager:
    """WebRTC连接管理器"""

    def __init__(self):
        self.peers = {}

    def create_peer(self, peer_id: str, peer_name: str = None, **kwargs) -> WebRTCPeer:
        """创建WebRTC对等连接"""
        if peer_id in self.peers:
            logger.warning(f"对等连接已存在: {peer_id}")
            return self.peers[peer_id]

        peer = WebRTCPeer(peer_id, peer_name, **kwargs)
        self.peers[peer_id] = peer
        return peer

    def get_peer(self, peer_id: str) -> Optional[WebRTCPeer]:
        """获取WebRTC对等连接"""
        return self.peers.get(peer_id)

    async def remove_peer(self, peer_id: str):
        """移除WebRTC对等连接"""
        peer = self.peers.pop(peer_id, None)
        if peer:
            await peer.close()

    async def close_all(self):
        """关闭所有连接"""
        tasks = []
        for peer_id in list(self.peers.keys()):
            tasks.append(self.remove_peer(peer_id))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def get_stats(self):
        """获取所有连接统计"""
        return {
            peer_id: peer.get_stats()
            for peer_id, peer in self.peers.items()
        }

    def get_connected_peers(self):
        """获取已连接的对等连接列表"""
        return [
            peer for peer in self.peers.values()
            if peer.is_connected()
        ]

    def get_failed_peers(self):
        """获取失败的对等连接列表"""
        return [
            peer for peer in self.peers.values()
            if peer.is_failed()
        ]


class IVideoTrack(MediaStreamTrack):
    """统一的视频轨接口。业务层应引用该基类，而不是直接引用 aiortc 类型。"""
    kind = "video"


class VideoCaptureTrack(IVideoTrack):
    """从OpenCV摄像头采集的视频轨。"""

    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480, fps: int = 30):
        super().__init__()
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.is_running = False
        self._pts = 0
        self._time_base = Fraction(1, fps)

    async def start(self):
        import cv2
        self.cap = cv2.VideoCapture(self.camera_index)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            ok, _ = self.cap.read()
            if ok:
                self.is_running = True
                return True
        # 失败则进入虚拟模式
        self.cap = None
        self.is_running = True
        return True

    async def stop(self):
        self.is_running = False
        try:
            if self.cap:
                self.cap.release()
        finally:
            await super().stop()

    def _generate_virtual_frame(self):
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        return frame

    async def recv(self):
        if not self.is_running:
            raise Exception("Track is stopped")
        import cv2
        if self.cap is not None:
            ok, frame = self.cap.read()
            if not ok:
                frame = self._generate_virtual_frame()
        else:
            frame = self._generate_virtual_frame()
        video_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        video_frame.pts = self._pts
        video_frame.time_base = self._time_base
        self._pts += 1
        return video_frame


class CallbackVideoTrack(IVideoTrack):
    """从回调函数获取帧的视频轨。回调返回 ndarray 或 VideoFrame。"""

    def __init__(self, frame_provider, width: int, height: int, fps: int = 30):
        super().__init__()
        self.frame_provider = frame_provider
        self.width = width
        self.height = height
        self.fps = fps
        self._pts = 0
        self._time_base = Fraction(1, fps)
        self._start = time.time()

    async def recv(self):
        # 简单的匀速控制
        pts_time = self._pts * self._time_base
        wait = pts_time - (time.time() - self._start)
        if wait > 0:
            await asyncio.sleep(wait)

        frame = self.frame_provider()
        if asyncio.iscoroutine(frame):
            frame = await frame

        if isinstance(frame, VideoFrame):
            video_frame = frame
        elif hasattr(frame, 'shape'):
            video_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        else:
            black = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            video_frame = VideoFrame.from_ndarray(black, format="bgr24")

        video_frame.pts = self._pts
        video_frame.time_base = self._time_base
        self._pts += 1

        return video_frame


