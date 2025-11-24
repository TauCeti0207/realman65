import sys
sys.path.append("./")

from geometry_msgs.msg import Pose, Point, Quaternion
from franka_msgs.msg import *

from controller.arm_controller import ArmController
from utils.ros_publisher import ROSPublisher, start_publishing
from utils.ros_subscriber import ROSSubscriber
from utils.data_handler import debug_print

import threading
import rospy
from scipy.spatial.transform import Rotation as R
import numpy as np

class RealmanRosController(ArmController):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.controller_type = "user_controller"
        self.controller = None

    def set_up(self, arm_name):
        subscriber = ROSSubscriber(f"/{arm_name}/rm_driver/Arm_Current_State", Arm_Current_State)
        self.pub_thread = {}

        # 初始化发布获取状态消息的节点
        state_publisher = ROSPublisher(f"/{arm_name}/rm_driver/GetArmState_Cmd", GetArmState_Command, continuous=True)
        state_msg = GetArmState_Command()
        state_msg.command = ''
        state_publisher.update_msg(state_msg)
        self.pub_thread["state"] = threading.Thread(target=start_publishing, args=(state_publisher,))
        self.pub_thread["state"].start()

        # 初始化发布关节角的节点
        joint_publisher = ROSPublisher(f"/{arm_name}/rm_driver/MoveJ_Cmd", MoveJ, continuous=False)
        self.pub_thread["joint"] = threading.Thread(target=start_publishing, args=(joint_publisher,))
        self.pub_thread["joint"].start()

        # 初始化发布末端位姿的节点
        eef_publisher = ROSPublisher(f"/{arm_name}/rm_driver/MoveP_Fd_Cmd", CartePos , continuous=False)
        self.pub_thread["eef"] = threading.Thread(target=start_publishing, args=(eef_publisher,))
        self.pub_thread["eef"].start()

        

        self.controller = {
            "subscriber": subscriber,
            "state_publisher": state_publisher,
            "joint_publisher": joint_publisher,
            "eef_publisher": eef_publisher,
            "gripper_publisher": gripper_publisher,
        }