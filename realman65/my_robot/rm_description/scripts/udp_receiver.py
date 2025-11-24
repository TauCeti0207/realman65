'''
Descripttion: 
Author: tauceti0207
version: 
Date: 2025-10-27 16:30:35
LastEditors: tauceti0207
LastEditTime: 2025-10-27 17:03:54
'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Zhang Jingwei
# Description: Receive joint states over UDP and publish to ROS 2 topic

import json
import socket
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class UDPJointStateReceiver(Node):
    def __init__(self):
        super().__init__('udp_joint_state_receiver')
        self.publisher = self.create_publisher(JointState, 'joint_states', 10)

        # Create a UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Bind to the address and port where data is sent
        self.sock.bind(('127.0.0.1', 12345))

        # Start a thread to receive UDP messages
        self.timer = self.create_timer(0.01, self.receive_udp_data)  # 10Hz

    def receive_udp_data(self):
        """Receive UDP data and publish it as ROS2 JointState message"""
        data, _ = self.sock.recvfrom(1024)  # Buffer size of 1024 bytes
        joint_states = json.loads(data.decode())

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = joint_states["name"]
        msg.position = joint_states["position"]
        msg.velocity = joint_states["velocity"]
        msg.effort = joint_states["effort"]

        # Publish the message to ROS 2 topic
        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    receiver = UDPJointStateReceiver()

    try:
        rclpy.spin(receiver)
    except KeyboardInterrupt:
        pass
    finally:
        receiver.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
