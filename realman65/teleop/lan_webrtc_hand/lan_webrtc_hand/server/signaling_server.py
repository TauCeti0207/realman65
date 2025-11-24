
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: signaling_server.py
功能: 视频流信令服务器 - 修复版
特性: 视频发布订阅、房间管理、WebRTC信令转发 
使用: python3 signaling_server.py --port 8000
"""

import argparse
import asyncio
import json
import logging
import time
from typing import Dict, Optional, List
from enum import Enum
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pathlib import Path
import uvicorn

import rclpy
from rclpy.node import Node

logger = logging.getLogger("signaling_server")


class ClientType(str, Enum):
    """客户端类型"""
    PUBLISHER = "video_publisher"  # 视频发布者
    SUBSCRIBER = "subscriber"      # 视频订阅者
    DATA_PUBLISHER = "data_publisher"   # 数据发布者
    DATA_SUBSCRIBER = "data_subscriber"  # 数据订阅者


class Client:
    """客户端信息"""

    def __init__(self, client_id: str, client_type: ClientType, websocket: WebSocket, display_name: str = None):
        self.client_id = client_id
        self.client_type = client_type
        self.websocket = websocket
        self.display_name = display_name or client_id
        self.room_id: Optional[str] = None
        self.connected_at = time.time()
        self.video_available = False  # 标记发布者是否已发布视频
        self.is_disconnected = False  # 标记连接是否已断开

    def to_dict(self):
        return {
            "client_id": self.client_id,
            "client_type": self.client_type,
            "display_name": self.display_name,
            "room_id": self.room_id,
            "connected_at": self.connected_at,
            "video_available": self.video_available
        }


class Room:
    """房间"""

    def __init__(self, room_id: str):
        self.room_id = room_id
        self.clients: Dict[str, Client] = {}
        self.publishers: Dict[str, Client] = {}  # 视频发布者
        self.subscribers: Dict[str, Client] = {}  # 视频订阅者
        self.data_publishers: Dict[str, Client] = {}
        self.data_subscribers: Dict[str, Client] = {}
        self.created_at = time.time()

    def add_client(self, client: Client):
        """添加客户端"""
        self.clients[client.client_id] = client
        client.room_id = self.room_id

        if client.client_type == ClientType.PUBLISHER:
            self.publishers[client.client_id] = client
            print(f"📹 发布者 {client.display_name} 加入房间 {self.room_id}")
        elif client.client_type == ClientType.SUBSCRIBER:
            self.subscribers[client.client_id] = client
            print(f"👁️ 订阅者 {client.display_name} 加入房间 {self.room_id}")
        elif client.client_type == ClientType.DATA_PUBLISHER:
            self.data_publishers[client.client_id] = client
            print(f"📤 数据发布者 {client.display_name} 加入房间 {self.room_id}")
        elif client.client_type == ClientType.DATA_SUBSCRIBER:
            self.data_subscribers[client.client_id] = client
            print(f"📥 数据订阅者 {client.display_name} 加入房间 {self.room_id}")

    def remove_client(self, client_id: str) -> Optional[Client]:
        """移除客户端"""
        client = self.clients.pop(client_id, None)
        if client:
            self.publishers.pop(client_id, None)
            self.subscribers.pop(client_id, None)
            self.data_publishers.pop(client_id, None)
            self.data_subscribers.pop(client_id, None)
            client.room_id = None
            print(f"👋 {client.display_name} 离开房间 {self.room_id}")
        return client

    def get_other_clients(self, exclude_client_id: str) -> List[Client]:
        """获取其他客户端"""
        return [client for client_id, client in self.clients.items()
                if client_id != exclude_client_id]

    def get_available_publishers(self) -> List[Client]:
        """获取可用的视频发布者"""
        return [pub for pub in self.publishers.values() if pub.video_available]

    def get_available_data_publishers(self) -> List[Client]:
        """获取可用的数据发布者"""
        return list(self.data_publishers.values())

    def is_empty(self) -> bool:
        """房间是否为空"""
        return len(self.clients) == 0

    def to_dict(self):
        return {
            "room_id": self.room_id,
            "client_count": len(self.clients),
            "publisher_count": len(self.publishers),
            "subscriber_count": len(self.subscribers),
            "available_publishers": len(self.get_available_publishers()),
            "clients": [client.to_dict() for client in self.clients.values()],
            "created_at": self.created_at
        }


class VideoSignalingServer:
    """视频流信令服务器"""

    def __init__(self):
        self.clients: Dict[str, Client] = {}  # 所有客户端
        self.rooms: Dict[str, Room] = {}      # 所有房间
        self.cleanup_task: Optional[asyncio.Task] = None

    async def start(self):
        """启动后台任务"""
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        print("🚀 视频信令服务器启动")

    async def stop(self):
        """停止服务器"""
        if self.cleanup_task:
            self.cleanup_task.cancel()
        print("🛑 视频信令服务器停止")

    async def _cleanup_loop(self):
        """清理循环"""
        while True:
            try:
                await asyncio.sleep(30)  # 30秒清理一次
                await self._cleanup_empty_rooms()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"清理任务错误: {e}")

    async def _cleanup_empty_rooms(self):
        """清理空房间"""
        current_time = time.time()
        empty_rooms = []

        for room_id, room in self.rooms.items():
            if room.is_empty() and (current_time - room.created_at) > 60:  # 1分钟后清理
                empty_rooms.append(room_id)

        for room_id in empty_rooms:
            del self.rooms[room_id]
            print(f"🗑️ 清理空房间: {room_id}")

    async def handle_connection(self, websocket: WebSocket, client_id: str):
        """处理WebSocket连接"""
        await websocket.accept()
        print(f"🔗 新连接: {client_id}")

        client = None
        try:
            # 等待注册消息
            message = await websocket.receive_text()
            data = json.loads(message)

            if data.get('type') != 'register':
                await websocket.close(code=4000, reason="Must register first")
                return

            # 创建客户端
            raw_type = data.get('client_type', ClientType.SUBSCRIBER)
            try:
                client_type = ClientType(raw_type)
            except ValueError:
                await websocket.close(code=4001, reason=f"未知客户端类型: {raw_type}")
                return

            display_name = data.get('client_name', client_id)
            client = Client(client_id, client_type, websocket, display_name)

            self.clients[client_id] = client
            print(f"✅ 注册客户端: {client.display_name} ({client.client_type})")

            # 发送注册确认
            await self._send_message(websocket, {
                'type': 'registered',
                'client_id': client_id,
                'client_type': client_type,
                'display_name': display_name
            })

            # 消息处理循环
            async for message in websocket.iter_text():
                await self._handle_message(client, message)

        except WebSocketDisconnect:
            print(f"🔌 连接断开: {client_id}")
        except Exception as e:
            logger.error(f"连接处理错误: {e}")
        finally:
            # 标记客户端为已断开
            if client:
                client.is_disconnected = True
            await self._handle_disconnect(client_id)

    def _is_websocket_connected(self, websocket: WebSocket) -> bool:
        """检查WebSocket连接是否仍然有效"""
        try:
            # 检查WebSocket的连接状态
            if hasattr(websocket, 'client_state'):
                from starlette.websockets import WebSocketState
                return websocket.client_state == WebSocketState.CONNECTED
            # 后备检查方法
            return not websocket.closed if hasattr(websocket, 'closed') else True
        except:
            return False

    async def _send_message(self, websocket: WebSocket, message: dict):
        """发送消息"""
        try:
            # 检查连接状态
            if not self._is_websocket_connected(websocket):
                logger.debug("WebSocket连接已关闭，跳过消息发送")
                return False

            await websocket.send_text(json.dumps(message))
            return True
        except Exception as e:
            logger.debug(f"发送消息失败: {e}")
            return False

    async def _send_message_to_client(self, client: Client, message: dict):
        """向客户端发送消息"""
        if client.is_disconnected:
            logger.debug(f"客户端 {client.client_id} 已断开，跳过消息发送")
            return False

        return await self._send_message(client.websocket, message)

    async def _broadcast_to_room(self, room_id: str, message: dict, exclude_client_id: str = None):
        """房间广播"""
        room = self.rooms.get(room_id)
        if not room:
            return

        failed_clients = []
        for client_id, client in room.clients.items():
            if exclude_client_id and client_id == exclude_client_id:
                continue

            success = await self._send_message_to_client(client, message)
            if not success:
                failed_clients.append(client_id)

        # 清理失败的连接
        for client_id in failed_clients:
            await self._handle_disconnect(client_id)

    async def _handle_message(self, client: Client, message: str):
        """处理客户端消息"""
        try:
            data = json.loads(message)
            msg_type = data.get('type')

            print(f"📨 [{client.display_name}] {msg_type}")
            if msg_type == 'subscribe_data':
                await self._handle_subscribe_data(client, data)
            elif msg_type == 'data_available':
                await self._handle_data_available(client, data)
            elif msg_type == 'join_room':
                await self._handle_join_room(client, data)
            elif msg_type == 'leave_room':
                await self._handle_leave_room(client)
            elif msg_type == 'video_available':
                await self._handle_video_available(client, data)
            elif msg_type == 'subscribe_video':
                await self._handle_subscribe_video(client, data)
            elif msg_type == 'offer':
                await self._handle_offer(client, data)
            elif msg_type == 'answer':
                await self._handle_answer(client, data)
            elif msg_type == 'ice_candidate':
                await self._handle_ice_candidate(client, data)
            else:
                print(f"⚠️ 未知消息: {msg_type}")

        except json.JSONDecodeError:
            logger.error("无效JSON消息")
        except Exception as e:
            logger.error(f"消息处理错误: {e}")

    async def _handle_subscribe_data(self, client: Client, data: dict):
        """订阅数据请求"""
        publisher_id = data.get('publisher_id')
        if not publisher_id or not client.room_id:
            return
        publisher = self.clients.get(publisher_id)
        if not publisher or publisher.room_id != client.room_id:
            await self._send_message_to_client(client, {
                'type': 'error',
                'error': 'Data publisher not found'
            })
            return
        # 告知数据发布者有订阅请求
        await self._send_message_to_client(publisher, {
            'type': 'data_request',
            'subscriber_id': client.client_id,
            'subscriber_name': client.display_name
        })
        print(f"📥 {client.display_name} 请求订阅 {publisher.display_name} 的数据通道")

    async def _handle_data_available(self, client: Client, data: dict):
        """通知数据通道可以被订阅"""
        if not client.room_id or client.client_type != ClientType.DATA_PUBLISHER:
            return
        publisher_name = data.get('publisher_name', client.display_name)
        room = self.rooms.get(client.room_id)
        if room:
            for subscriber in room.data_subscribers.values():
                if subscriber.client_id != client.client_id:
                    await self._send_message_to_client(subscriber, {
                        'type': 'data_available',
                        'publisher_id': client.client_id,
                        'publisher_name': publisher_name
                    })
            print(f"📤 {publisher_name} 的数据通道在房间 {client.room_id} 可用")

    async def _handle_join_room(self, client: Client, data: dict):
        """处理加入房间"""
        room_id = data.get('room_id')
        display_name = data.get('display_name')

        if not room_id:
            await self._send_message_to_client(client, {
                'type': 'error',
                'error': 'Missing room_id'
            })
            return

        # 更新显示名
        if display_name:
            client.display_name = display_name

        # 如果已在其他房间，先离开
        if client.room_id:
            await self._handle_leave_room(client)

        # 创建或获取房间
        if room_id not in self.rooms:
            self.rooms[room_id] = Room(room_id)
            print(f"🏗️ 创建房间: {room_id}")

        room = self.rooms[room_id]
        room.add_client(client)

        # 发送加入确认
        await self._send_message_to_client(client, {
            'type': 'room_joined',
            'room_id': room_id,
            'client_count': len(room.clients),
            'publishers': [p.to_dict() for p in room.publishers.values()],
            'data_publishers': [p.to_dict() for p in room.data_publishers.values()],
            'other_clients': [c.to_dict() for c in room.get_other_clients(client.client_id)]
        })

        # 通知房间其他成员
        await self._broadcast_to_room(room_id, {
            'type': 'client_joined',
            'client': client.to_dict()
        }, exclude_client_id=client.client_id)

    async def _handle_leave_room(self, client: Client):
        """处理离开房间"""
        if not client.room_id:
            return

        room_id = client.room_id
        room = self.rooms.get(room_id)

        if room:
            room.remove_client(client.client_id)

            # 通知房间其他成员
            await self._broadcast_to_room(room_id, {
                'type': 'client_left',
                'client_id': client.client_id,
                'display_name': client.display_name
            })

            # 如果房间空了，删除房间
            if room.is_empty():
                del self.rooms[room_id]
                print(f"🗑️ 删除空房间: {room_id}")

        # 只向未断开的客户端发送离开确认
        if not client.is_disconnected:
            await self._send_message_to_client(client, {
                'type': 'room_left',
                'room_id': room_id
            })

    async def _handle_video_available(self, client: Client, data: dict):
        """处理视频可用通知"""
        if not client.room_id or client.client_type != ClientType.PUBLISHER:
            return

        # 标记发布者视频可用
        client.video_available = True

        publisher_name = data.get('publisher_name', client.display_name)

        # 通知房间内的订阅者
        room = self.rooms.get(client.room_id)
        if room:
            # 只通知订阅者
            for subscriber in room.subscribers.values():
                if subscriber.client_id != client.client_id:
                    await self._send_message_to_client(subscriber, {
                        'type': 'video_available',
                        'publisher_id': client.client_id,
                        'publisher_name': publisher_name
                    })

            print(f"📺 {publisher_name} 的视频在房间 {client.room_id} 可用")

    async def _handle_subscribe_video(self, client: Client, data: dict):
        """处理视频订阅请求"""
        publisher_id = data.get('publisher_id')

        if not publisher_id or not client.room_id:
            return

        publisher = self.clients.get(publisher_id)
        if not publisher or publisher.room_id != client.room_id:
            await self._send_message_to_client(client, {
                'type': 'error',
                'error': 'Publisher not found'
            })
            return

        # 通知发布者有订阅请求
        await self._send_message_to_client(publisher, {
            'type': 'video_request',
            'subscriber_id': client.client_id,
            'subscriber_name': client.display_name
        })

        print(f"📺 {client.display_name} 请求订阅 {publisher.display_name} 的视频")

    async def _handle_offer(self, client: Client, data: dict):
        """处理WebRTC Offer"""
        target_client_id = data.get('target_client_id')
        sdp = data.get('sdp')

        if not target_client_id or not sdp:
            return

        target_client = self.clients.get(target_client_id)
        if not target_client or target_client.room_id != client.room_id:
            return

        # 转发offer
        success = await self._send_message_to_client(target_client, {
            'type': 'offer',
            'from_client_id': client.client_id,
            'sdp': sdp
        })

        if success:
            print(
                f"🤝 转发offer: {client.display_name} -> {target_client.display_name}")

    async def _handle_answer(self, client: Client, data: dict):
        """处理WebRTC Answer"""
        target_client_id = data.get('target_client_id')
        sdp = data.get('sdp')

        if not target_client_id or not sdp:
            return

        target_client = self.clients.get(target_client_id)
        if not target_client or target_client.room_id != client.room_id:
            return

        # 转发answer
        success = await self._send_message_to_client(target_client, {
            'type': 'answer',
            'from_client_id': client.client_id,
            'sdp': sdp
        })

        if success:
            print(
                f"🤝 转发answer: {client.display_name} -> {target_client.display_name}")

    async def _handle_ice_candidate(self, client: Client, data: dict):
        """处理ICE候选"""
        target_client_id = data.get('target_client_id')
        candidate = data.get('candidate')

        if not target_client_id or not candidate:
            return

        target_client = self.clients.get(target_client_id)
        if not target_client or target_client.room_id != client.room_id:
            return

        # 转发ICE候选
        await self._send_message_to_client(target_client, {
            'type': 'ice_candidate',
            'from_client_id': client.client_id,
            'candidate': candidate
        })

    async def _handle_disconnect(self, client_id: str):
        """处理客户端断开"""
        client = self.clients.pop(client_id, None)
        if not client:
            return

        print(f"🔌 客户端断开: {client.display_name}")

        # 标记为已断开
        client.is_disconnected = True

        # 从房间移除（不发送离开确认给断开的客户端）
        if client.room_id:
            await self._handle_leave_room(client)

    def get_stats(self):
        """获取统计信息"""
        return {
            "total_clients": len(self.clients),
            "total_rooms": len(self.rooms),
            "publishers": len([c for c in self.clients.values() if c.client_type == ClientType.PUBLISHER]),
            "subscribers": len([c for c in self.clients.values() if c.client_type == ClientType.SUBSCRIBER]),
            "rooms": [room.to_dict() for room in self.rooms.values()]
        }


server = VideoSignalingServer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期"""
    await server.start()
    yield
    await server.stop()


# 创建FastAPI应用
app = FastAPI(
    title="视频流信令服务器",
    version="1.0.0",
    lifespan=lifespan
)

# 添加CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 挂载前端Web UI（/ui）
_BASE_DIR = Path(__file__).resolve().parent
_UI_DIR = _BASE_DIR / "webui"
if _UI_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(_UI_DIR), html=True), name="ui")

@app.get("/ui/")
async def ui_root():
    if _UI_DIR.exists():
        return RedirectResponse(url="/ui/index.html")
    return {"message": "UI not found. Place web files under 'webui' directory."}


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket端点"""
    await server.handle_connection(websocket, client_id)


@app.get("/")
async def root():
    """根路径"""
    return {"message": "视频流信令服务器", "version": "1.0.0"}


@app.get("/stats")
async def get_stats():
    """获取统计信息"""
    return server.get_stats()


@app.get("/rooms")
async def get_rooms():
    """获取所有房间"""
    return {
        "rooms": [room.to_dict() for room in server.rooms.values()]
    }


@app.get("/rooms/{room_id}")
async def get_room_info(room_id: str):
    """获取房间信息"""
    room = server.rooms.get(room_id)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")
    return room.to_dict()


class MainNode(Node):
    def __init__(self):
        super().__init__('server')

        self.declare_parameter('host', '0.0.0.0')
        self.declare_parameter('port', 8000)
        self.declare_parameter('verbose', False)

    async def run_server(self):
        host = self.get_parameter('host').value
        port = self.get_parameter('port').value
        verbose = self.get_parameter('verbose').value

        log_level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(level=log_level)

        self.get_logger().info(f"🚀 启动视频流信令服务器")
        self.get_logger().info(f"🌐 地址: http://{host}:{port}")
        self.get_logger().info(f"🔌 WebSocket: ws://{host}:{port}/ws/{{client_id}}")
        self.get_logger().info(f"📊 统计: http://{host}:{port}/stats")

        config = uvicorn.Config(
            app,
            host=host,
            port=port,
            log_level="info" if verbose else "warning"
        )
        server_instance = uvicorn.Server(config)
        await server_instance.serve()

def main(args=None):
    rclpy.init(args=args)
    node = MainNode()
    
    loop = asyncio.get_event_loop()
    
    try:
        loop.run_until_complete(node.run_server())
    except KeyboardInterrupt:
        node.get_logger().info("\n👋 服务器已停止")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
