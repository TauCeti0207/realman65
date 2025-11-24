
#!/usr/bin/env python3
import socket
import threading
import time
import json
import struct
import cv2
import numpy as np
from datetime import datetime
import os
from headset_utils import HeadsetData, HeadsetFeedback, convert_left_to_right_coordinates
import queue
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
import tf_transformations
import numpy as np
import threading
import multiprocessing as mp
from geometry_msgs.msg import TransformStamped
import time
import cv2
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
from tf2_ros import TransformBroadcaster
# from arm_teleoppy.teleop_srv.teleop_client import TeleopClient
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Float32MultiArray
from tiga_save_buffer2disk import persist_shortbuffer_pairs
from std_msgs.msg import Bool

class UDPCameraServer(Node):
    def __init__(self): 
        super().__init__("vr_unity_node")

        self.broadcaster = TransformBroadcaster(self)
        self.lock = threading.Lock()

        self.server_ip = "0.0.0.0"  
        self.client_ip = "127.0.0.1"  # quest ip

        # 端口配置
        self.data_receive_port = 8003     # 接收头显数据
        self.status_send_port = 8004      # 发送状态数据
        self.left_video_port = 8001       # 发送左眼视频
        self.right_video_port = 8002      # 发送右眼视频

        # 摄像头配置
        # self.camera_mode = "tiga"  # "single", "dual", "split"
        # self.left_camera_id = 10     # 左眼摄像头ID
        # self.right_camera_id = 3     # 右眼摄像头ID
        # self.head_camera_id = 10
        # self.chest_camera_id = 4

        # self.video_width = 640
        # self.video_height = 480
        # self.video_fps = 30
        # self.jpeg_quality = 80

        # 摄像头对象
        # self.left_camera = None
        # self.right_camera = None
        self.single_camera = None

        # 状态变量
        self.running = False
        self.client_address = None
        self.last_headset_data = None
        self.camera_initialized = False

        # 模拟机器人状态
        self.head_out_of_sync = False
        self.left_out_of_sync = False
        self.right_out_of_sync = False
        self.robot_info = "机器人状态正常 - 摄像头已启动"

        # 模拟手臂位置
        self.left_arm_position = [0.5, 1.2, 0.3]
        self.right_arm_position = [-0.5, 1.2, 0.3]
        self.left_arm_rotation = [0.0, 0.0, 0.0, 1.0]
        self.right_arm_rotation = [0.0, 0.0, 0.0, 1.0]

        # Socket对象
        self.data_socket = None
        self.status_socket = None
        self.left_video_socket = None
        self.right_video_socket = None

        # 线程对象
        self.threads = []
        self.headset_data = HeadsetData()

        # 帧统计
        self.left_frame_count = 0
        self.right_frame_count = 0
        self.last_fps_time = time.time()
        self.actual_fps = 0

        self.l_matrix = np.eye(4)
        self.r_matrix = np.eye(4)
        self.h_matrix = np.eye(4)

        self.l_quat_offset = np.eye(4)
        self.l_pos_offset = np.zeros(3)
        self.r_quat_offset = np.eye(4)
        self.r_pos_offset = np.zeros(3)
        self.h_quat_offset = np.eye(4)
        self.h_pos_offset = np.zeros(3)

        # 左手控制器状态
        self.l_thumbstick_x = 0.0
        self.l_thumbstick_y = 0.0
        self.l_index_trigger = 0.0
        self.l_hand_trigger = 0.0
        self.l_button_one = False
        self.l_button_two = False
        self.l_button_thumbstick = False

        # 右手控制器状态
        self.r_thumbstick_x = 0.0
        self.r_thumbstick_y = 0.0
        self.r_index_trigger = 0.0
        self.r_hand_trigger = 0.0
        self.button_A = False
        self.button_B = False
        self.r_button_thumbstick = False

        self.last_A_status = False
        self.last_B_status = False
        self.last_X_status = False
        self.last_Y_status = False

        self.cv_image1 = None
        self.cv_image2 = None
        self.running = True
        self.pos_scale = 1
        self.publish_status = False

        self.print_sec = 5

        self.publish_hz = 60
        self.dt = 1.0 / self.publish_hz

        # self.cli = TeleopClient(self, '/srv/teleop')

        self.trig_pub = self.create_publisher(
            Float32MultiArray,
            '/device/trigger',
            10)

        self.twist_pub = self.create_publisher(
            TwistStamped,
            '/desire/twist',
            10
        )

        # 添加机械臂控制发布器
        self.arm_pose_pub = self.create_publisher(
            TransformStamped,
            '/robot_arm/target_pose',
            10
        )
        
        self.arm_joint_pub = self.create_publisher(
            Float32MultiArray,
            '/robot_arm/joint_positions',
            10
        )
        
        self.gripper_pub = self.create_publisher(
            Float32MultiArray,
            '/robot_arm/gripper_control',
            10
        )

        self.save_frame2buffer_flag = False

        self.head_frame_buffer = []
        self.chest_frame_buffer = []
        self.head_frame_timestep = []
        self.chest_frame_timestep = []
        self.episode_idx = 0

        # 定义 Publisher，发布 Bool 类型的消息，话题名可以自定义
        self.flag_publisher = self.create_publisher(Bool, "save_frame2buffer_flag", 10)
        # 创建一个定时器，每隔 0.1 秒发布一次
        self.timer = self.create_timer(0.1, self.publish_flag)

    def publish_flag(self):
        msg = Bool()
        msg.data = self.save_frame2buffer_flag
        self.flag_publisher.publish(msg)
        # self.get_logger().info(f"Publishing flag: {msg.data}")

    def initialize_cameras(self):
        """初始化摄像头"""
        # try:
        #     print("正在初始化摄像头...")
        #     if self.camera_mode == "single":
        #         self.single_camera = cv2.VideoCapture(self.head_camera_id)
        #         if not self.single_camera.isOpened():
        #             raise Exception(f"无法打开摄像头 {self.head_camera_id}")
        #         # ret, frame = self.single_camera.read()
        #         # print("read:",ret)
        #         self.single_camera.set(
        #             cv2.CAP_PROP_FRAME_WIDTH, self.video_width)
        #         self.single_camera.set(
        #             cv2.CAP_PROP_FRAME_HEIGHT, self.video_height)
        #         self.single_camera.set(cv2.CAP_PROP_FPS, self.video_fps)
        #     elif self.camera_mode == "dual":
        #         self.left_camera = cv2.VideoCapture(self.left_camera_id)
        #         self.right_camera = cv2.VideoCapture(self.right_camera_id)
        #         if not self.left_camera.isOpened():
        #             raise Exception(f"无法打开左眼摄像头 {self.left_camera_id}")
        #         if not self.right_camera.isOpened():
        #             raise Exception(f"无法打开右眼摄像头 {self.right_camera_id}")
        #         for camera in [self.left_camera, self.right_camera]:
        #             camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.video_width)
        #             camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.video_height)
        #             camera.set(cv2.CAP_PROP_FPS, self.video_fps)

        #     elif self.camera_mode == "split":
        #         self.single_camera = cv2.VideoCapture(self.left_camera_id)
        #         if not self.single_camera.isOpened():
        #             raise Exception(f"无法打开摄像头 {self.left_camera_id}")
        #         self.single_camera.set(
        #             cv2.CAP_PROP_FRAME_WIDTH, self.video_width * 2)
        #         self.single_camera.set(
        #             cv2.CAP_PROP_FRAME_HEIGHT, self.video_height)
        #         self.single_camera.set(cv2.CAP_PROP_FPS, self.video_fps)
        #     elif self.camera_mode == "tiga":
        #         self.head_camera = cv2.VideoCapture(self.head_camera_id)
        #         self.chest_camera = cv2.VideoCapture(self.chest_camera_id)
        #         if not self.head_camera.isOpened():
        #             raise Exception(f"无法打开头部摄像头 {self.head_camera_id}")
        #         if not self.chest_camera.isOpened():
        #             raise Exception(f"无法打开胸部摄像头 {self.chest_camera_id}")
        #         for camera in [self.head_camera, self.chest_camera]:
        #             camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.video_width)
        #             camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.video_height)
        #             camera.set(cv2.CAP_PROP_FPS, self.video_fps)


        #     self.camera_initialized = True

        # except Exception as e:
        #     print(f"摄像头初始化失败: {e}")
        #     self.list_available_cameras()
        #     return False

        # 不使用摄像头，直接返回True
        self.camera_initialized = True
        return True

    def list_available_cameras(self):
        available_cameras = []

        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    available_cameras.append(i)
                    print(
                        f"  摄像头 {i}: 可用 (分辨率: {frame.shape[1]}x{frame.shape[0]})")
                cap.release()

        if not available_cameras:
            print("  未检测到可用摄像头")
        else:
            print(f"\n建议使用: camera_id = {available_cameras[0]}")

    def capture_left_frame(self):
        """捕获左眼画面"""
        # if self.camera_mode == "single":
        #     ret, frame = self.single_camera.read()
        #     # print("read :",ret)
        #     if ret:
        #         frame = cv2.resize(
        #             frame, (self.video_width, self.video_height))
        #         cv2.putText(frame, "LEFT", (10, 30),
        #                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        #         return frame

        # elif self.camera_mode == "dual":
        #     # 双摄像头模式：使用左眼摄像头
        #     ret, frame = self.left_camera.read()
        #     if ret:
        #         frame = cv2.resize(
        #             frame, (self.video_width, self.video_height))
        #         return frame

        # elif self.camera_mode == "split":
        #     ret, frame = self.single_camera.read()
        #     if ret:
        #         h, w = frame.shape[:2]
        #         left_frame = frame[:, :w//2]  # 取左半部分
        #         left_frame = cv2.resize(
        #             left_frame, (self.video_width, self.video_height))
        #         return left_frame
            
        
        # elif self.camera_mode == "tiga":
        #     ret, frame = self.head_camera.read()
        #     if ret:
        #         return frame

        return None

    def capture_right_frame(self):
        """捕获右眼画面"""
        # if self.camera_mode == "single":
        #     ret, frame = self.single_camera.read()
        #     if ret:
        #         frame = cv2.resize(
        #             frame, (self.video_width, self.video_height))
        #         cv2.putText(frame, "RIGHT", (10, 30),
        #                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        #         return frame

        # elif self.camera_mode == "dual":
        #     ret, frame = self.right_camera.read()
        #     if ret:
        #         frame = cv2.resize(
        #             frame, (self.video_width, self.video_height))
        #         return frame

        # elif self.camera_mode == "split":
        #     ret, frame = self.single_camera.read()
        #     if ret:
        #         h, w = frame.shape[:2]
        #         right_frame = frame[:, w//2:]  # 取右半部分
        #         right_frame = cv2.resize(
        #             right_frame, (self.video_width, self.video_height))
        #         return right_frame
        
        # elif self.camera_mode == "tiga":
        #     ret, frame = self.chest_camera.read()
        #     if ret:
        #         return frame

        return None

    def start_server(self):
        try:
            # 初始化摄像头
            if not self.initialize_cameras():
                return

            # 创建Socket
            self.data_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.status_socket = socket.socket(
                socket.AF_INET, socket.SOCK_DGRAM)
            self.left_video_socket = socket.socket(
                socket.AF_INET, socket.SOCK_DGRAM)
            self.right_video_socket = socket.socket(
                socket.AF_INET, socket.SOCK_DGRAM)

            # 绑定端口
            self.data_socket.bind((self.server_ip, self.data_receive_port))
            print(f"数据接收服务启动 - 端口: {self.data_receive_port}")

            self.running = True

            self.threads = [
                threading.Thread(target=self.receive_data_thread, daemon=True),
                threading.Thread(target=self.send_status_thread, daemon=True),
                threading.Thread(
                    target=self.send_left_video_thread, daemon=True),
                threading.Thread(
                    target=self.send_right_video_thread, daemon=True),
                threading.Thread(
                    target=self.console_interface_thread, daemon=True),
                threading.Thread(target=self.fps_monitor_thread, daemon=True),
                threading.Thread(target=self.pub, daemon=True)
            ]

            for thread in self.threads:
                thread.start()

            print("UDP摄像头服务器启动成功!")
            print("端口配置:")
            print(f"  数据接收: {self.data_receive_port}")
            print(f"  状态发送: {self.status_send_port}")
            print(f"  左眼视频: {self.left_video_port}")
            print(f"  右眼视频: {self.right_video_port}")
            # print(f"摄像头模式: {self.camera_mode}")
            print("\n控制命令:")
            print("  'head' - 切换头部同步状态")
            print("  'left' - 切换左臂同步状态")
            print("  'right' - 切换右臂同步状态")
            print("  'info <消息>' - 设置机器人信息")
            print("  'mode <single/dual/split>' - 切换摄像头模式")
            print("  'quality <1-100>' - 设置JPEG质量")
            print("  'fps' - 显示帧率信息")
            print("  'quit' - 退出服务器")

        except Exception as e:
            print(f"服务器启动失败: {e}")
            self.stop_server()

    def pub(self):
        rate = self.create_rate(self.publish_hz)
        while True:
            if self.publish_status == True:
                self.publish()
            rate.sleep()

    def receive_data_thread(self):
        """接收头显数据线程"""
        print("数据接收线程启动")

        cnt = 0
        t0 = time.time()
        while self.running:
            try:
                data, addr = self.data_socket.recvfrom(8192)

                # 记录客户端地址
                if self.client_address is None:
                    self.client_address = addr
                    self.client_ip = addr[0]
                    print(f"客户端连接: {addr}")

                # 解析JSON数据
                json_str = data.decode('utf-8')
                headset_data = json.loads(json_str)
                print(headset_data)
                print('=================================================\n')
                self.last_headset_data = headset_data

                # self.print_headsets_data(headset_data)

                cnt += 1
                if time.time() - t0 >= 5:
                    print(f'recv rate: {cnt / 5.0}')
                    t0 = time.time()
                    cnt = 0

            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"数据接收错误: {e}")

    def send_status_thread(self):
        """发送状态数据线程"""
        print("状态发送线程启动")

        while self.running:
            try:
                if self.client_address is not None:
                    # 构造状态数据
                    status_data = {
                        "headOutOfSync": self.head_out_of_sync,
                        "leftOutOfSync": self.left_out_of_sync,
                        "rightOutOfSync": self.right_out_of_sync,
                        "info": f"{self.robot_info} | FPS: {self.actual_fps:.1f}",
                        "leftArmPosition": self.left_arm_position,
                        "leftArmRotation": self.left_arm_rotation,
                        "rightArmPosition": self.right_arm_position,
                        "rightArmRotation": self.right_arm_rotation,
                        "timestamp": time.time()
                    }

                    # 发送状态数据
                    json_str = json.dumps(status_data)
                    data = json_str.encode('utf-8')

                    self.status_socket.sendto(
                        data, (self.client_ip, self.status_send_port))

                time.sleep(0.1)  # 10Hz发送频率

            except Exception as e:
                if self.running:
                    print(f"状态发送错误: {e}")

    def send_left_video_thread(self):
        """发送左眼视频线程"""
        print("左眼视频发送线程启动")

        while self.running:
            try:
                # if self.client_address is not None and self.camera_initialized: 
                #     frame = self.capture_left_frame()
                    
                #     if self.save_frame2buffer_flag:
                #         self.head_frame_buffer.append(frame)
                #         self.head_frame_timestep.append(time.time())
                #     else:
                #         if len(self.head_frame_buffer) != 0:
                #             print("saving 2 disk")
                #             persist_shortbuffer_pairs(
                #                 self.head_frame_buffer, self.head_frame_timestep,
                #                 self.chest_frame_buffer, self.chest_frame_timestep,
                #                 out_dir=f"/home/guest/teleop/video_server/tiga_data/pair_output{self.episode_idx}",
                #                 edge_margin=5,        # 去掉短 buffer 首尾各 5 帧
                #                 max_time_diff=0.05,   # 可选，允许最大时间差 50ms，不需要可设为 None
                #                 start_index=0,
                #             )
                #             self.episode_idx = self.episode_idx + 1


                #     if frame is not None:
                #         # 编码为JPEG
                #         _, buffer = cv2.imencode(
                #             '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                #         frame_data = buffer.tobytes()

                #         # 发送视频数据
                #         self.left_video_socket.sendto(
                #             frame_data, (self.client_ip, self.left_video_port))
                #         self.left_frame_count += 1

                # time.sleep(1.0 / self.video_fps)
                time.sleep(0.1)  # 简单的延时，避免占用过多CPU

            except Exception as e:
                if self.running:
                    print(f"左眼视频发送错误: {e}")

    def send_right_video_thread(self):
        """发送右眼视频线程"""
        print("右眼视频发送线程启动")

        while self.running:
            try:
                # if self.client_address is not None and self.camera_initialized:
                #     # 捕获右眼画面
                #     frame = self.capture_right_frame()
                #     if self.save_frame2buffer_flag:
                #         self.chest_frame_buffer.append(frame)
                #         self.chest_frame_timestep.append(time.time())

                #     if frame is not None:
                #         # 编码为JPEG
                #         _, buffer = cv2.imencode(
                #             '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                #         frame_data = buffer.tobytes()

                #         # 发送视频数据
                #         self.right_video_socket.sendto(
                #             frame_data, (self.client_ip, self.right_video_port))

                #         self.right_frame_count += 1

                # time.sleep(1.0 / self.video_fps)
                time.sleep(0.1)  # 简单的延时，避免占用过多CPU

            except Exception as e:
                if self.running:
                    print(f"右眼视频发送错误: {e}")

    def fps_monitor_thread(self):
        """帧率监控线程"""
        while self.running:
            try:
                time.sleep(5.0)  # 每5秒更新一次
                current_time = time.time()
                elapsed = current_time - self.last_fps_time

                if elapsed > 0:
                    total_frames = self.left_frame_count + self.right_frame_count
                    self.actual_fps = total_frames / elapsed / 2  # 除以2因为有两路视频

                    # 重置计数器
                    self.left_frame_count = 0
                    self.right_frame_count = 0
                    self.last_fps_time = current_time

            except Exception as e:
                if self.running:
                    print(f"帧率监控错误: {e}")

    def console_interface_thread(self):
        """控制台交互线程"""
        while self.running:
            try:
                command = input().strip().lower()

                if command == 'quit':
                    print("正在关闭服务器...")
                    self.stop_server()
                    break
                elif command == 'head':
                    self.head_out_of_sync = not self.head_out_of_sync
                    print(
                        f"头部同步状态: {'失同步' if self.head_out_of_sync else '同步'}")
                elif command == 'left':
                    self.left_out_of_sync = not self.left_out_of_sync
                    print(
                        f"左臂同步状态: {'失同步' if self.left_out_of_sync else '同步'}")
                elif command == 'right':
                    self.right_out_of_sync = not self.right_out_of_sync
                    print(
                        f"右臂同步状态: {'失同步' if self.right_out_of_sync else '同步'}")
                elif command.startswith('info '):
                    self.robot_info = command[5:]
                    print(f"机器人信息已更新: {self.robot_info}")
                elif command.startswith('mode '):
                    new_mode = command[5:].strip()
                    if new_mode in ['single', 'dual', 'split']:
                        self.change_camera_mode(new_mode)
                    else:
                        print("无效模式，支持: single, dual, split")
                elif command.startswith('quality '):
                    try:
                        quality = int(command[8:].strip())
                        if 1 <= quality <= 100:
                            self.jpeg_quality = quality
                            print(f"JPEG质量已设置为: {quality}")
                        else:
                            print("质量值应在1-100之间")
                    except ValueError:
                        print("无效的质量值")
                elif command == 'fps':
                    print(f"当前FPS: {self.actual_fps:.1f}")
                elif command == 'status':
                    self.print_server_status()
                elif command == 'cameras':
                    self.list_available_cameras()

            except EOFError:
                break
            except Exception as e:
                print(f"控制台输入错误: {e}")

    def change_camera_mode(self, new_mode): 
 
        if self.single_camera:
            self.single_camera.release()
            self.single_camera = None
        if self.left_camera:
            self.left_camera.release()
            self.left_camera = None
        if self.right_camera:
            self.right_camera.release()
            self.right_camera = None
 
        self.camera_mode = new_mode
        self.camera_initialized = False

        if self.initialize_cameras():
            print(f"成功切换到 {new_mode} 模式")
        else:
            print(f"切换到 {new_mode} 模式失败")

    def print_headset_data(self, message):
        # if hasattr(self, '_last_print_time'):
        #     if time.time() - self._last_print_time < 2.0:  # 每2秒最多打印一次
        #         return

        # self._last_print_time = time.time()

        # print(f"\n=== 头显数据 ({datetime.now().strftime('%H:%M:%S')}) ===")
        # print(
        #     f"头部位置: ({data['HPosition']['x']:.2f}, {data['HPosition']['y']:.2f}, {data['HPosition']['z']:.2f})")
        # print(
        #     f"左手位置: ({data['LPosition']['x']:.2f}, {data['LPosition']['y']:.2f}, {data['LPosition']['z']:.2f})")
        # print(
        #     f"右手位置: ({data['RPosition']['x']:.2f}, {data['RPosition']['y']:.2f}, {data['RPosition']['z']:.2f})")
        # print(f"{data}")
        # global global_headset_data
        # print(type(message))
        data = message
        headset_data = HeadsetData()

        headset_data.h_pos[0] = data['HPosition']['x']
        headset_data.h_pos[1] = data['HPosition']['y']
        headset_data.h_pos[2] = data['HPosition']['z']
        headset_data.h_quat[0] = data['HRotation']['x']
        headset_data.h_quat[1] = data['HRotation']['y']
        headset_data.h_quat[2] = data['HRotation']['z']
        headset_data.h_quat[3] = data['HRotation']['w']
        headset_data.l_pos[0] = data['LPosition']['x']
        headset_data.l_pos[1] = data['LPosition']['y']
        headset_data.l_pos[2] = data['LPosition']['z']
        headset_data.l_quat[0] = data['LRotation']['x']
        headset_data.l_quat[1] = data['LRotation']['y']
        headset_data.l_quat[2] = data['LRotation']['z']
        headset_data.l_quat[3] = data['LRotation']['w']
        headset_data.l_thumbstick_x = data['LThumbstick']['x']
        headset_data.l_thumbstick_y = data['LThumbstick']['y']
        headset_data.l_index_trigger = data['LIndexTrigger']
        headset_data.l_hand_trigger = data['LHandTrigger']
        headset_data.l_button_one = data['LButtonOne']
        headset_data.l_button_two = data['LButtonTwo']
        headset_data.l_button_thumbstick = data['LButtonThumbstick']
        headset_data.r_pos[0] = data['RPosition']['x']
        headset_data.r_pos[1] = data['RPosition']['y']
        headset_data.r_pos[2] = data['RPosition']['z']
        headset_data.r_quat[0] = data['RRotation']['x']
        headset_data.r_quat[1] = data['RRotation']['y']
        headset_data.r_quat[2] = data['RRotation']['z']
        headset_data.r_quat[3] = data['RRotation']['w']
        headset_data.r_thumbstick_x = data['RThumbstick']['x']
        headset_data.r_thumbstick_y = data['RThumbstick']['y']
        headset_data.r_index_trigger = data['RIndexTrigger']
        headset_data.r_hand_trigger = data['RHandTrigger']
        headset_data.r_button_one = data['RButtonOne']
        headset_data.r_button_two = data['RButtonTwo']
        headset_data.r_button_thumbstick = data['RButtonThumbstick']
        headset_data.h_pos, headset_data.h_quat = convert_left_to_right_coordinates(headset_data.h_pos, headset_data.h_quat)
        headset_data.l_pos, headset_data.l_quat = convert_left_to_right_coordinates(headset_data.l_pos, headset_data.l_quat)
        headset_data.r_pos, headset_data.r_quat = convert_left_to_right_coordinates(headset_data.r_pos, headset_data.r_quat)
        # global_headset_data = headset_data
        self.headset_data = headset_data
        self.update_controller_state(headset_data)
        # print(headset_data.h_pos)

    def print_server_status(self):
        """打印服务器状态"""
        print("\n=== 服务器状态 ===")
        print(f"运行状态: {'运行中' if self.running else '已停止'}")
        print(f"客户端地址: {self.client_address}")
        print(f"摄像头模式: {self.camera_mode}")
        print(f"摄像头状态: {'已初始化' if self.camera_initialized else '未初始化'}")
        print(f"JPEG质量: {self.jpeg_quality}")
        print(f"当前FPS: {self.actual_fps:.1f}")
        print(f"头部同步: {'失同步' if self.head_out_of_sync else '同步'}")
        print(f"左臂同步: {'失同步' if self.left_out_of_sync else '同步'}")
        print(f"右臂同步: {'失同步' if self.right_out_of_sync else '同步'}") 
    def stop_server(self):
        self.running = False

        if self.single_camera:
            self.single_camera.release()
        if self.left_camera:
            self.left_camera.release()
        if self.right_camera:
            self.right_camera.release()

        if self.data_socket:
            self.data_socket.close()
        if self.status_socket:
            self.status_socket.close()
        if self.left_video_socket:
            self.left_video_socket.close()
        if self.right_video_socket:
            self.right_video_socket.close()

        print("服务器已停止")

    def update_controller_state(self, data):
        # 头部位置和姿态
        h_pos = np.copy(data.h_pos) if hasattr(data, 'h_pos') else np.zeros(3)
        h_quat = np.copy(data.h_quat) if hasattr(data, 'h_quat') else np.array([0, 0, 0, 1])
        h_mat = np.eye(4)
        h_mat = tf_transformations.quaternion_matrix(h_quat)
        # h_mat = tf_transformations.euler_matrix(0.3,0,0, axes='sxyz')
        h_mat[:3, 3] = h_pos
        self.h_matrix = h_mat
        print("___________________________________________",self.h_matrix )

        # 左手控制器
        l_pos = np.copy(data.l_pos) if hasattr(data, 'l_pos') else np.zeros(3)
        l_quat = np.copy(data.l_quat) if hasattr(data, 'l_quat') else np.array([0, 0, 0, 1])
        l_mat = np.eye(4)
        l_mat = tf_transformations.quaternion_matrix(l_quat)
        l_mat[:3, 3] = l_pos
        self.l_matrix = l_mat

        r_pos = np.copy(data.r_pos) if hasattr(data, 'r_pos') else np.zeros(3)
        r_quat = np.copy(data.r_quat) if hasattr(data, 'r_quat') else np.array([0, 0, 0, 1])
        r_mat = np.eye(4)
        r_mat = tf_transformations.quaternion_matrix(r_quat)
        r_mat[:3, 3] = r_pos
        self.r_matrix = r_mat

        self.l_thumbstick_x = data.l_thumbstick_x if hasattr(data, 'l_thumbstick_x') else 0.0
        self.l_thumbstick_y = data.l_thumbstick_y if hasattr(data, 'l_thumbstick_y') else 0.0
        self.l_index_trigger = data.l_index_trigger if hasattr(data, 'l_index_trigger') else 0.0
        self.l_hand_trigger = data.l_hand_trigger if hasattr(data, 'l_hand_trigger') else 0.0
        self.l_button_one = data.l_button_one if hasattr(data, 'l_button_one') else False
        self.l_button_two = data.l_button_two if hasattr(data, 'l_button_two') else False
        self.l_button_thumbstick = data.l_button_thumbstick if hasattr(data, 'l_button_thumbstick') else False

        # 右手控制器
        self.r_pos = np.copy(data.r_pos) if hasattr(data, 'r_pos') else np.zeros(3)
        self.r_quat = np.copy(data.r_quat) if hasattr(data, 'r_quat') else np.array([0, 0, 0, 1])
        self.r_thumbstick_x = data.r_thumbstick_x if hasattr(data, 'r_thumbstick_x') else 0.0
        self.r_thumbstick_y = data.r_thumbstick_y if hasattr(data, 'r_thumbstick_y') else 0.0
        self.r_index_trigger = data.r_index_trigger if hasattr(data, 'r_index_trigger') else 0.0
        self.r_hand_trigger = data.r_hand_trigger if hasattr(data, 'r_hand_trigger') else 0.0
        self.button_A = data.r_button_one if hasattr(data, 'r_button_one') else False
        self.button_B = data.r_button_two if hasattr(data, 'r_button_two') else False
        self.r_button_thumbstick = data.r_button_thumbstick if hasattr(data, 'r_button_thumbstick') else False

        if self.button_A == True and self.last_A_status == False:
            # self.cli.reset()
            self.last_A_status = True
            self.publish_status = True
            print("call A")
            self.save_frame2buffer_flag = False
            self.head_frame_buffer.clear()
            self.head_frame_timestep.clear()
            self.chest_frame_buffer.clear()
            self.chest_frame_timestep.clear()


        if self.l_button_one == True and self.last_X_status == False:
            # self.cli.run()
            self.last_X_status = True
            print("call X")
            self.save_frame2buffer_flag = True     

        if self.l_button_two == True and self.last_Y_status == False:
            self.last_Y_status = True
            # self.cli.pause()
            print("call Y")

        if self.l_hand_trigger == True and self.last_l_hand_trigger_status == False:
            self.last_l_hand_trigger_status = True
            print("call l_hand_trigger Pause or Continue")
            if self.save_frame2buffer_flag:
                print("pause and save2disk")
            else:
                print("continue and save2buffer")
            self.save_frame2buffer_flag = not self.save_frame2buffer_flag

        if self.l_button_one == False:
            self.last_X_status = False
        if self.l_button_two == False:
            self.last_Y_status = False
        if self.button_A == False:
            self.last_A_status = False

        if self.l_hand_trigger == False:
            self.last_l_hand_trigger_status = False

    def publish(self):
        l_mat = self.l_matrix.copy()
        r_mat = self.r_matrix.copy()
        h_mat = self.h_matrix.copy()

        # Left transform
        tf_l = TransformStamped()
        tf_l.header.stamp = self.get_clock().now().to_msg()
        
        tf_l.header.frame_id = "map"
        tf_l.child_frame_id = "l_hand"

        target_l_pos = self.pos_scale * l_mat[:3, 3] + self.l_pos_offset
        target_l_pose = tf_transformations.quaternion_from_matrix(l_mat @ self.l_quat_offset)

        tf_l.transform.translation.x = target_l_pos[0]
        tf_l.transform.translation.y = target_l_pos[1]
        tf_l.transform.translation.z = target_l_pos[2]
        tf_l.transform.rotation.x = target_l_pose[0]
        tf_l.transform.rotation.y = target_l_pose[1]
        tf_l.transform.rotation.z = target_l_pose[2]
        tf_l.transform.rotation.w = target_l_pose[3]

        # Right transform
        tf_r = TransformStamped()
        tf_r.header.stamp = self.get_clock().now().to_msg()
        tf_r.header.frame_id = "map"
        tf_r.child_frame_id = "r_hand"

        target_r_pos = self.pos_scale * r_mat[:3, 3] + self.r_pos_offset
        target_r_pose = tf_transformations.quaternion_from_matrix(r_mat @ self.r_quat_offset)

        tf_r.transform.translation.x = target_r_pos[0]
        tf_r.transform.translation.y = target_r_pos[1]
        tf_r.transform.translation.z = target_r_pos[2]
        tf_r.transform.rotation.x = target_r_pose[0]
        tf_r.transform.rotation.y = target_r_pose[1]
        tf_r.transform.rotation.z = target_r_pose[2]
        tf_r.transform.rotation.w = target_r_pose[3]

        t_torso = TransformStamped()
        t_head = TransformStamped()
        target_h_pos = h_mat[:3, 3] + self.h_pos_offset
        target_h_pose = tf_transformations.quaternion_from_matrix(h_mat @ self.h_quat_offset)

        t_torso.header.stamp = self.get_clock().now().to_msg()
        t_torso.header.frame_id = "map"
        t_torso.child_frame_id = "root"
        t_torso.transform.translation.x = target_h_pos[0]
        t_torso.transform.translation.y = target_h_pos[1]
        t_torso.transform.translation.z = target_h_pos[2] + 1.0
        t_torso.transform.rotation.x = 0.0
        t_torso.transform.rotation.y = 0.0
        t_torso.transform.rotation.z = 0.0
        t_torso.transform.rotation.w = 1.0

        t_head.header.stamp = self.get_clock().now().to_msg()
        t_head.header.frame_id = "root"
        t_head.child_frame_id = "head"
        t_head.transform.translation.x = 0.0
        t_head.transform.translation.y = 0.0
        t_head.transform.translation.z = 0.6
        t_head.transform.rotation.x = target_h_pose[0]
        t_head.transform.rotation.y = target_h_pose[1]
        t_head.transform.rotation.z = target_h_pose[2]
        t_head.transform.rotation.w = target_h_pose[3]

        
        self.broadcaster.sendTransform([tf_l, tf_r, t_torso, t_head])

        twist_msg = TwistStamped()
        twist_msg.header.stamp = self.get_clock().now().to_msg()
        # print("self.l_thumbstick_y",self.l_thumbstick_y)
        twist_msg.twist.linear.x = float(self.l_thumbstick_y)
        twist_msg.twist.linear.y = float(self.l_thumbstick_x)
        twist_msg.twist.linear.z = 0.0
        twist_msg.twist.angular.x = 0.0
        twist_msg.twist.angular.y = 0.0
        twist_msg.twist.angular.z = float(self.r_thumbstick_x)
        self.twist_pub.publish(twist_msg)

        trig_msg = Float32MultiArray()
        # trig_msg.data = [0.0, 0.0]
        trig_msg.data.append(self.l_index_trigger)
        trig_msg.data.append(self.r_index_trigger)
        self.trig_pub.publish(trig_msg)

        # 发布机械臂控制数据
        # 使用右手控制器位置和姿态作为机械臂目标位置
        arm_pose_msg = TransformStamped()
        arm_pose_msg.header.stamp = self.get_clock().now().to_msg()
        arm_pose_msg.header.frame_id = "map"
        arm_pose_msg.child_frame_id = "robot_arm_target"
        
        # 使用右手控制器的位置和姿态
        target_arm_pos = self.pos_scale * self.r_matrix[:3, 3] + self.r_pos_offset
        target_arm_quat = tf_transformations.quaternion_from_matrix(self.r_matrix @ self.r_quat_offset)
        
        arm_pose_msg.transform.translation.x = target_arm_pos[0]
        arm_pose_msg.transform.translation.y = target_arm_pos[1]
        arm_pose_msg.transform.translation.z = target_arm_pos[2]
        arm_pose_msg.transform.rotation.x = target_arm_quat[0]
        arm_pose_msg.transform.rotation.y = target_arm_quat[1]
        arm_pose_msg.transform.rotation.z = target_arm_quat[2]
        arm_pose_msg.transform.rotation.w = target_arm_quat[3]
        
        self.arm_pose_pub.publish(arm_pose_msg)
        
        # 发布夹爪控制（使用右手扳机键控制）
        gripper_msg = Float32MultiArray()
        gripper_msg.data = [self.r_index_trigger]  # 0.0-1.0 控制夹爪开合
        self.gripper_pub.publish(gripper_msg)
        
        # 可选：发布关节位置（如果需要关节空间控制）
        # joint_msg = Float32MultiArray()
        # joint_msg.data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # 6个关节角度
        # self.arm_joint_pub.publish(joint_msg)


def main(args=None):
    rclpy.init(args=args)
    server = UDPCameraServer()
    server.start_server()
    
    ros2_thread = threading.Thread(target=rclpy.spin, args=(server,))
    ros2_thread.start()
    try:
        for thread in server.threads:
            if thread.is_alive():
                thread.join()

    except KeyboardInterrupt:
        server.stop_server()
        server.destroy_node()
        rclpy.shutdown()
    except Exception as e:
        server.stop_server()
        server.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
