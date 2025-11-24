#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: data_subscriber.py
功能: 数据订阅端 - 使用通用WebRTC组件
职责: 订阅数据发布者的传感器数据，保持原有业务逻辑不变
特性: 使用通用信令组件和WebRTC组件替换原有WebSocket连接
依赖: aiortc>=1.7.0, websockets>=10.0
使用: python3 data_subscriber.py --server ws://localhost:8000 --room room123 --name "数据订阅者"
"""

import asyncio
import logging
import json
import uuid
import argparse
import signal
import sys

# 导入通用组件
from .signal_client import SignalClient
from .webrtc_client import WebRTCManager

logger = logging.getLogger("data_subscriber")


class DataSubscriber:
    def __init__(self, server_url, room_id, name, target_publishers=None, verbose=False,
                 data_channel_label: str = "data", data_channel_options: dict = None,
                 ice_servers=None, on_message_callback=None):
        self.server_url = server_url
        self.room_id = room_id
        self.name = name
        self.client_id = f"datasub_{uuid.uuid4().hex[:8]}"
        self.verbose = verbose
        self.target_publishers = set(target_publishers or [])
        self.should_exit = False
        self.data_channel_label = data_channel_label
        self.data_channel_options = data_channel_options or {}
        self.ice_servers = ice_servers

        self.on_message_callback = on_message_callback

        # 使用通用信令组件
        self.signal_client = SignalClient(
            server_url=server_url,
            client_id=self.client_id,
            room_id=room_id,
            display_name=name,
            client_type="data_subscriber"
        )

        # 使用通用WebRTC管理器
        self.webrtc_manager = WebRTCManager()

        self.pc_by_pubid = {}

        # 设置信令消息处理器
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """设置信令消息处理器"""
        self.signal_client.set_message_handler(
            'registered', self._handle_registered)
        self.signal_client.set_message_handler(
            'room_joined', self._handle_room_joined)
        self.signal_client.set_message_handler(
            'data_available', self._handle_data_available)
        self.signal_client.set_message_handler(
            'client_joined', self._handle_client_joined)
        self.signal_client.set_message_handler('offer', self._handle_offer)
        self.signal_client.set_message_handler(
            'ice_candidate', self._handle_ice_candidate)

    async def connect(self):
        """启动连接"""
        logger.info("已注册并加入房间，等待数据发布者...")
        # 启动信令客户端
        await self.signal_client.start()

    async def _handle_registered(self, data: dict):
        """处理注册确认"""
        if self.verbose:
            logger.debug(f"注册成功: {data}")

    async def _handle_room_joined(self, data):

        pubs = data.get("publishers", [])
        data_pubs = data.get("data_publishers", [])  # 新增：获取数据发布者
        logger.info(f"当前房间有发布者: {pubs}")
        logger.info(f"当前房间有数据发布者: {data_pubs}")  # 新增：显示数据发布者

        # 处理视频发布者（原有逻辑）
        for pub in pubs:
            if pub.get("client_type") == "video_publisher":
                pubid = pub.get("client_id")
                pubname = pub.get("display_name")
                if (not self.target_publishers) or \
                        (pubid in self.target_publishers or pubname in self.target_publishers):
                    await self._subscribe_video(pubid)

        # 新增：处理数据发布者
        for pub in data_pubs:
            if pub.get("client_type") == "data_publisher":
                pubid = pub.get("client_id")
                pubname = pub.get("display_name")
                if (not self.target_publishers) or \
                        (pubid in self.target_publishers or pubname in self.target_publishers):
                    await self._subscribe_data(pubid)

    async def _handle_data_available(self, data):
        pubid = data["publisher_id"]
        pubname = data.get("publisher_name", pubid)
        if (not self.target_publishers) or \
           (pubid in self.target_publishers or pubname in self.target_publishers):
            await self._subscribe_data(pubid)

    async def _handle_client_joined(self, data: dict):
        """新客户端加入：若是数据发布者则按策略订阅"""
        client = data.get('client', {})
        client_id = client.get('client_id')
        client_type = client.get('client_type')
        display_name = client.get('display_name', client_id)
        if client_type != 'data_publisher':
            return
        # 仅当未限制目标或命中目标时订阅
        if (not self.target_publishers) or \
           (client_id in self.target_publishers or display_name in self.target_publishers):
            await self._subscribe_data(client_id)

    async def _subscribe_video(self, publisher_id):
        """订阅视频（原有逻辑保持不变）"""
        if publisher_id in self.pc_by_pubid:
            logger.info(f"已订阅视频 {publisher_id}，跳过")
            return
        logger.info(f"订阅视频发布者: {publisher_id}")
        await self._send_message({
            "type": "subscribe_video",
            "publisher_id": publisher_id
        })

    async def _subscribe_data(self, publisher_id):
        if publisher_id in self.pc_by_pubid:
            logger.info(f"已订阅 {publisher_id}，跳过")
            return
        logger.info(f"订阅数据发布者: {publisher_id}")
        await self._send_message({
            "type": "subscribe_data",
            "publisher_id": publisher_id
        })

    async def _on_connection_state_change(self, peer_id: str, state: str):
        """WebRTC连接状态变化回调"""
        if self.verbose:
            logger.debug(f"连接状态变化: {peer_id} -> {state}")

    async def _on_ice_candidate(self, peer_id: str, candidate):
        """ICE候选回调"""
        await self._send_message({
            "type": "ice_candidate",
            "target_client_id": peer_id,
            "candidate": candidate
        })

    def _on_data_channel_open(self, channel):
        """数据通道打开回调"""
        # 从数据通道标签中提取发布者ID（如果有的话）
        label = channel.label
        logger.info(f"数据通道已建立: {label}")

    def _on_data_channel_message(self, channel, msg):
        """数据通道消息回调"""

        peer_id = None
        for pub_id, peer in self.webrtc_manager.peers.items():
            if peer.get_data_channel(channel.label) == channel:
                peer_id = pub_id
                break

        if peer_id:
            # logger.info(f"收到数据({peer_id}): {msg}")
            if callable(self.on_message_callback):
                try:
                    self.on_message_callback(peer_id, msg)
                except Exception as e:
                    logger.debug(f"上层on_message回调异常: {e}")
        else:
            pass
        #     logger.info(f"收到数据: {msg}")

    async def _handle_offer(self, data):
        pubid = data["from_client_id"]
        sdp = data["sdp"]
        logger.info(f"收到offer: {pubid}")

        # 创建WebRTC连接
        peer = self.webrtc_manager.create_peer(
            peer_id=pubid,
            peer_name=f"publisher_{pubid}",
            on_connection_state_change=self._on_connection_state_change,
            on_ice_candidate=self._on_ice_candidate,
            ice_servers=self.ice_servers
        )

        # 创建WebRTC连接
        success = await peer.create_connection()
        if not success:
            logger.error(f"创建WebRTC连接失败: {pubid}")
            return

        opts = self.data_channel_options
        if opts.get('negotiated'):
            if self.data_channel_label not in peer.data_channels:
                chan = peer.create_data_channel(
                    self.data_channel_label,
                    on_open=None,
                    on_close=None,
                    on_message=None,
                    ordered=opts.get('ordered', True),
                    max_packet_life_time=opts.get('maxPacketLifeTime'),
                    max_retransmits=opts.get('maxRetransmits'),
                    protocol=opts.get('protocol'),
                    negotiated=True,
                    channel_id=opts.get('id'),
                )
                # 绑定消息回调

                @chan.on("message")
                def _on_msg(msg):
                    # logger.info(f"收到数据({pubid}): {msg}")
                    if callable(self.on_message_callback):
                        try:
                            self.on_message_callback(pubid, msg)
                        except Exception as e:
                            logger.debug(f"上层on_message回调异常: {e}")

        # 保存连接引用以保持兼容性
        self.pc_by_pubid[pubid] = peer.pc

        def setup_data_channel(channel):
            logger.info(
                f"数据通道已建立: {channel.label} ({pubid}), negotiated={getattr(channel, 'negotiated', None)}, id={getattr(channel, 'id', None)}")
            if channel.label != self.data_channel_label:
                logger.info(f"忽略非目标数据通道: {channel.label}")
                return

            # 设置消息处理（保持原有格式）
            @channel.on("message")
            def on_message(msg):
                # logger.info(f"收到数据({pubid}): {msg}")
                if callable(self.on_message_callback):
                    try:
                        self.on_message_callback(pubid, msg)
                    except Exception as e:
                        logger.debug(f"上层on_message回调异常: {e}")

        # 重写数据通道设置方法
        peer._setup_data_channel_events = setup_data_channel

        # 处理offer（防重复/仅在stable）
        success = await peer.handle_offer(sdp)
        if not success:
            logger.error(f"处理offer失败: {pubid}")
            return

        try:
            answer_sdp = await peer.create_answer()
            if answer_sdp:
                await self._send_message({
                    "type": "answer",
                    "target_client_id": pubid,
                    "sdp": answer_sdp
                })
            else:
                logger.debug("跳过发送answer：未生成SDP（状态不匹配或重复）")
        except Exception as e:
            logger.debug(f"创建/发送answer异常(可能重复): {e}")

    async def _handle_ice_candidate(self, data):
        pubid = data.get("from_client_id")
        candidate = data.get("candidate")
        if self.verbose:
            logger.debug(f"收到ICE候选: {data}")

        if pubid and candidate:
            peer = self.webrtc_manager.get_peer(pubid)
            if peer:
                await peer.handle_ice_candidate(candidate)

    async def _send_message(self, msg):
        """发送消息（通过信令组件）"""
        await self.signal_client.send_message(msg)

    async def close(self):
        self.should_exit = True

        # 关闭所有WebRTC连接
        await self.webrtc_manager.close_all()

        # 清理兼容性字典
        self.pc_by_pubid.clear()

        # 停止信令客户端
        await self.signal_client.stop()


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="ws://localhost:8000")
    parser.add_argument("--room", default="room123")
    parser.add_argument("--name", default="数据订阅者")
    parser.add_argument("--subscribe", help="指定订阅发布者名称，多个用,分隔")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    loglevel = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=loglevel, format='%(asctime)s | %(levelname)s | %(message)s')
    target_publishers = [x.strip() for x in args.subscribe.split(
        ",")] if args.subscribe else []
    ds = DataSubscriber(args.server, args.room, args.name,
                        target_publishers, args.verbose)

    def signal_handler():
        logger.info("收到中断, 退出...")
        asyncio.create_task(ds.close())

    loop = asyncio.get_running_loop()
    for sig in [signal.SIGINT, signal.SIGTERM]:
        loop.add_signal_handler(sig, signal_handler)
    try:
        await ds.connect()
    finally:
        await ds.close()


if __name__ == "__main__":
    asyncio.run(main())
