#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: signal_client.py
功能: 通用WebRTC信令客户端组件
职责: 处理WebSocket连接、消息收发、自动重连、房间管理
特性: 支持自动注册、加入房间、消息处理器、断线重连
"""

import asyncio
import json
import logging
import time
import random
import websockets

logger = logging.getLogger("signal_client")


class SignalClient:
    """通用信令客户端 - 支持自动重连和消息处理器"""
    
    def __init__(self, server_url: str, client_id: str, room_id: str, 
                 display_name: str, client_type: str = "subscriber"):
        self.server_url = server_url
        self.client_id = client_id
        self.room_id = room_id
        self.display_name = display_name
        self.client_type = client_type
        
        # 连接状态
        self.websocket = None
        self.is_connected = False
        self.is_shutting_down = False
        
        # 重连参数
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = -1  # -1表示无限重连
        self.base_reconnect_delay = 1.0
        self.max_reconnect_delay = 30.0
        
        # 消息处理器
        self.message_handlers = {}
        
        # 连接统计
        self.connection_established_time = None
        self.total_disconnections = 0
    
    def set_message_handler(self, msg_type: str, handler):
        """设置消息处理器"""
        self.message_handlers[msg_type] = handler
    
    async def start(self):
        """启动信令客户端（带自动重连）"""
        while not self.is_shutting_down:
            try:
                await self._connect_and_run()
            except Exception as e:
                if not self.is_shutting_down:
                    self.reconnect_attempts += 1
                    self.total_disconnections += 1
                    
                    delay = min(
                        self.base_reconnect_delay * (2 ** min(self.reconnect_attempts - 1, 6)) + random.uniform(0, 1),
                        self.max_reconnect_delay
                    )
                    
                    logger.warning(f"📡 信令服务器连接失败: {e}")
                    logger.info(f"⏳ {delay:.1f}秒后重连... (尝试 {self.reconnect_attempts})")
                    
                    if self.max_reconnect_attempts > 0 and self.reconnect_attempts >= self.max_reconnect_attempts:
                        logger.error("达到最大重连次数，停止重连")
                        break
                    
                    await asyncio.sleep(delay)
    
    async def _connect_and_run(self):
        """连接并运行"""
        logger.info(f"🔗 连接到信令服务器: {self.server_url} (尝试 {self.reconnect_attempts + 1})")
        
        # 建立WebSocket连接
        self.websocket = await websockets.connect(
            f"{self.server_url}/ws/{self.client_id}",
            ping_interval=20,
            ping_timeout=10
        )
        
        self.is_connected = True
        self.reconnect_attempts = 0
        self.connection_established_time = time.time()
        logger.info("✅ 信令服务器连接成功")
        
        # 注册和加入房间
        await self._register_and_join_room()
        
        # 开始消息处理
        await self._message_loop()
    
    async def _register_and_join_room(self):
        """注册并加入房间"""
        # 注册客户端
        await self.send_message({
            'type': 'register',
            'client_type': self.client_type,
            'client_name': self.display_name
        })
        logger.info(f"📝 已发送注册消息 (类型: {self.client_type})")
        
        # 等待一下再加入房间
        await asyncio.sleep(0.1)
        
        # 加入房间
        await self.send_message({
            'type': 'join_room',
            'room_id': self.room_id,
            'display_name': self.display_name
        })
        logger.info(f"🚪 已发送加入房间消息: {self.room_id}")
    
    async def _message_loop(self):
        """消息处理循环"""
        try:
            async for message in self.websocket:
                if self.is_shutting_down:
                    break
                await self._handle_message(message)
        except websockets.exceptions.ConnectionClosed:
            if not self.is_shutting_down:
                logger.warning("📡 信令服务器连接断开 (P2P连接继续工作)")
            self.is_connected = False
        except Exception as e:
            if not self.is_shutting_down:
                logger.error(f"信令消息循环错误: {e}")
            self.is_connected = False
        finally:
            self.is_connected = False
    
    async def _handle_message(self, message: str):
        """处理服务器消息"""
        try:
            data = json.loads(message)
            msg_type = data.get('type')
            
            logger.debug(f"📨 收到信令消息: {msg_type}")
            
            # 调用对应的处理器
            handler = self.message_handlers.get(msg_type)
            if handler:
                await handler(data)
            else:
                logger.debug(f"未处理的消息类型: {msg_type}")
            
        except json.JSONDecodeError:
            logger.error("无效的JSON消息")
        except Exception as e:
            logger.error(f"处理信令消息错误: {e}")
    
    async def send_message(self, message: dict):
        """发送消息"""
        if self.is_connected and self.websocket and not self.websocket.closed:
            try:
                await self.websocket.send(json.dumps(message))
                return True
            except Exception as e:
                logger.error(f"发送信令消息失败: {e}")
                self.is_connected = False
                return False
        else:
            logger.warning("📡 信令服务器未连接，消息发送失败")
            return False
    
    def get_stats(self):
        """获取统计信息"""
        uptime = time.time() - self.connection_established_time if self.connection_established_time else 0
        
        return {
            "is_connected": self.is_connected,
            "reconnect_attempts": self.reconnect_attempts,
            "total_disconnections": self.total_disconnections,
            "connection_uptime": uptime
        }
    
    async def stop(self):
        """停止信令客户端"""
        logger.info("🔌 停止信令客户端...")
        self.is_shutting_down = True
        
        if self.websocket:
            try:
                await self.websocket.close()
            except Exception as e:
                logger.debug(f"关闭信令连接时的警告: {e}")
        
        self.is_connected = False