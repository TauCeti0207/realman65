#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: video_publisher.py
功能: 底层视频发布组件（SDK） - 使用通用WebRTC组件
"""

import asyncio
import logging
import uuid
from typing import Dict

from .signal_client import SignalClient
from .webrtc_client import WebRTCManager, IVideoTrack, VideoCaptureTrack as SDKVideoCaptureTrack, CallbackVideoTrack as SDKCallbackVideoTrack

logger = logging.getLogger("video_publisher")

VideoCaptureTrack = SDKVideoCaptureTrack
CallbackVideoTrack = SDKCallbackVideoTrack


class RoomClient:
    """房间客户端 - 使用通用WebRTC组件"""

    def __init__(self, server_url: str, room_id: str, display_name: str, client_type: str = "video_publisher"):
        self.server_url = server_url
        self.room_id = room_id
        self.display_name = display_name
        self.client_type = client_type
        self.client_id = f"{client_type[:5]}_{uuid.uuid4().hex[:8]}"

        # 房间信息
        self.joined_room = False

        # 使用通用信令组件
        self.signal_client = SignalClient(
            server_url=server_url,
            client_id=self.client_id,
            room_id=room_id,
            display_name=display_name,
            client_type=client_type
        )

        # 使用通用WebRTC管理器
        self.webrtc_manager = WebRTCManager()

        # 视频轨道
        self.video_track = None

        # 状态
        self.is_shutting_down = False

        # 设置信令消息处理器
        self._setup_signal_handlers()

        logger.info(f"初始化房间客户端: {self.client_id}")

    def _setup_signal_handlers(self):
        """设置信令消息处理器"""
        self.signal_client.set_message_handler(
            'registered', self._handle_registered)
        self.signal_client.set_message_handler(
            'room_joined', self._handle_room_joined)
        self.signal_client.set_message_handler(
            'client_joined', self._handle_client_joined)
        self.signal_client.set_message_handler(
            'client_left', self._handle_client_left)
        self.signal_client.set_message_handler(
            'video_request', self._handle_video_request)
        self.signal_client.set_message_handler('answer', self._handle_answer)
        self.signal_client.set_message_handler(
            'ice_candidate', self._handle_ice_candidate)

    def set_video_track(self, track: IVideoTrack):
        """设置视频轨道"""
        self.video_track = track
        logger.info("视频轨道已设置")

    async def connect_and_join_room(self):
        """连接并加入房间"""
        # 启动信令客户端连接（带重连）
        signaling_task = asyncio.create_task(self.signal_client.start())

        # 启动WebRTC连接监控任务
        monitor_task = asyncio.create_task(self._monitor_webrtc_connections())

        try:
            await asyncio.gather(signaling_task, monitor_task)
        except Exception as e:
            if not self.is_shutting_down:
                logger.error(f"连接错误: {e}")

    async def _monitor_webrtc_connections(self):
        """监控WebRTC连接状态"""
        while not self.is_shutting_down:
            try:
                await asyncio.sleep(10)  # 每10秒检查一次

                # 清理失败的连接
                failed_peers = self.webrtc_manager.get_failed_peers()
                for peer in failed_peers:
                    logger.info(f"清理失败的WebRTC连接: {peer.peer_name}")
                    await self.webrtc_manager.remove_peer(peer.peer_id)

                # 报告连接状态
                connected_peers = self.webrtc_manager.get_connected_peers()
                if connected_peers:
                    logger.info(f"📡 WebRTC连接状态: {len(connected_peers)} 个连接正常")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"监控WebRTC连接时出错: {e}")
                await asyncio.sleep(5)

    # 信令消息处理器
    async def _handle_registered(self, data: dict):
        """处理注册确认"""
        logger.info(f"✅ 注册成功: {data.get('client_id')}")

    async def _handle_room_joined(self, data: dict):
        """处理加入房间成功"""
        self.joined_room = True
        logger.info(f"✅ 成功加入房间: {data.get('room_id')}")
        logger.info(f"房间人数: {data.get('client_count')}")
        logger.info(f"我的显示名: {self.display_name}")

        # 通知视频可用
        await self.signal_client.send_message({
            'type': 'video_available',
            'publisher_name': self.display_name
        })

        logger.info("📺 视频发布可用，等待订阅者连接...")

    async def _handle_client_joined(self, data: dict):
        """处理客户端加入"""
        client = data.get('client', {})
        client_type = client.get('client_type')
        display_name = client.get('display_name', 'Unknown')
        logger.info(f"👋 新成员加入: {display_name} ({client_type})")

    async def _handle_client_left(self, data: dict):
        """处理客户端离开"""
        client_id = data.get('client_id')
        display_name = data.get('display_name', 'Unknown')
        logger.info(f"👋 {display_name} 离开房间")

        # 清理对应的WebRTC连接
        await self.webrtc_manager.remove_peer(client_id)

    async def _handle_video_request(self, data: dict):
        """处理视频订阅请求"""
        subscriber_id = data.get('subscriber_id')
        subscriber_name = data.get('subscriber_name', 'Unknown')
        logger.info(f"📺 {subscriber_name} 请求订阅我的视频")

        # 创建WebRTC连接
        peer = self.webrtc_manager.create_peer(
            peer_id=subscriber_id,
            peer_name=subscriber_name,
            on_ice_candidate=self._on_ice_candidate
        )

        # 创建WebRTC连接
        success = await peer.create_connection()
        if success:
            # 添加视频轨道
            if self.video_track:
                peer.add_track(self.video_track)

            logger.info(f"🔗 为 {subscriber_name} 创建WebRTC连接")

            # 创建并发送offer
            sdp = await peer.create_offer()
            if sdp:
                await self.signal_client.send_message({
                    'type': 'offer',
                    'target_client_id': subscriber_id,
                    'sdp': sdp
                })
                logger.info(f"📤 已向 {subscriber_name} 发送视频offer")

    async def _on_ice_candidate(self, peer_id: str, candidate):
        """ICE候选回调"""
        await self.signal_client.send_message({
            'type': 'ice_candidate',
            'target_client_id': peer_id,
            'candidate': candidate
        })

    async def _handle_answer(self, data: dict):
        """处理接收到的answer"""
        from_client_id = data.get('from_client_id')
        sdp = data.get('sdp')

        logger.info(f"📥 收到来自 {from_client_id} 的answer")

        peer = self.webrtc_manager.get_peer(from_client_id)
        if peer and sdp:
            await peer.handle_answer(sdp)

    async def _handle_ice_candidate(self, data: dict):
        """处理ICE候选"""
        from_client_id = data.get('from_client_id')
        candidate = data.get('candidate')

        peer = self.webrtc_manager.get_peer(from_client_id)
        if peer and candidate:
            await peer.handle_ice_candidate(candidate)

    def get_connection_stats(self):
        """获取连接统计"""
        signaling_stats = self.signal_client.get_stats()
        webrtc_stats = self.webrtc_manager.get_stats()
        connected_count = len(self.webrtc_manager.get_connected_peers())

        return {
            "signaling_connected": signaling_stats["is_connected"],
            "room_joined": self.joined_room,
            "total_webrtc_connections": len(webrtc_stats),
            "active_webrtc_connections": connected_count,
            "reconnect_attempts": signaling_stats["reconnect_attempts"],
            "total_disconnections": signaling_stats["total_disconnections"],
            "connection_uptime": signaling_stats["connection_uptime"]
        }

    async def disconnect(self):
        """断开连接"""
        logger.info("🔌 开始断开房间连接...")
        self.is_shutting_down = True

        # 关闭所有WebRTC连接
        await self.webrtc_manager.close_all()

        # 停止信令客户端
        await self.signal_client.stop()

        logger.info("✅ 房间连接断开完成")


async def create_room_publisher_with_track(server_url: str, room_id: str, display_name: str, track: IVideoTrack) -> RoomClient:
    """便捷工厂：用外部给定的 MediaStreamTrack 创建并启动房间发布客户端。

    用法：
        track = CallbackVideoTrack(my_provider, width, height, fps)
        room = await create_room_publisher_with_track(server, room, name, track)
    """
    client = RoomClient(
        server_url=server_url,
        room_id=room_id,
        display_name=display_name,
        client_type="video_publisher",
    )
    client.set_video_track(track)
    asyncio.create_task(client.connect_and_join_room())
    return client


class VideoPublisher:
    """视频发布客户端（可选的高层封装）。

    注：保留该类供上层使用，但移除了命令行运行与测试逻辑。
    """

    def __init__(self, server_url: str, room_id: str, display_name: str, camera_index: int = 0):
        self.server_url = server_url
        self.room_id = room_id
        self.display_name = display_name
        self.camera_index = camera_index

        self.video_track = None
        self.room_client = None
        self.is_running = False
        self.is_shutting_down = False

    async def initialize(self) -> bool:
        try:
            self.video_track = VideoCaptureTrack(self.camera_index)
            success = await self.video_track.start()
            if not success:
                return False

            self.room_client = RoomClient(
                server_url=self.server_url,
                room_id=self.room_id,
                display_name=self.display_name,
                client_type="video_publisher"
            )
            self.room_client.set_video_track(self.video_track)
            return True
        except Exception:
            return False

    async def start(self) -> bool:
        if not await self.initialize():
            return False
        self.is_running = True
        try:
            await self.room_client.connect_and_join_room()
        finally:
            self.is_running = False
        return True

    async def stop(self):
        if self.is_shutting_down:
            return
        self.is_shutting_down = True
        if self.video_track:
            try:
                await asyncio.wait_for(self.video_track.stop(), timeout=3.0)
            except Exception:
                pass
        if self.room_client:
            try:
                await asyncio.wait_for(self.room_client.disconnect(), timeout=10.0)
            except Exception:
                pass
