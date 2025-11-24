#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Zhang Jingwei
# Description: Load local URDF, visualize with MeshCat, and solve IK using Pink

import numpy as np
import pinocchio as pin
from pinocchio.romeo_wrapper import RomeoWrapper
from pinocchio.robot_wrapper import RobotWrapper
import qpsolvers
import pink
from pink import solve_ik
from pink.tasks import FrameTask, PostureTask
from pink.visualization import start_meshcat_visualizer
from scipy.spatial.transform import Rotation as R
import time
from sensor_msgs.msg import JointState
import time
import threading
import socket
import json


class ArmIKSolver:
    def __init__(self, urdf_path, ee_frame="ee_link", visualize=True):
        """
        Initialize the arm IK solver.
        Args:
            urdf_path (str): local URDF file path
            ee_frame (str): end-effector frame name
            visualize (bool): whether to open MeshCat viewer
        """
        # === Step 1: Load URDF ===
        self.robot = RobotWrapper.BuildFromURDF(urdf_path)
        self.model = self.robot.model
        self.data = self.robot.data

        # === Step 2: Initialize Pink Configuration ===
        q0 = pin.randomConfiguration(self.model)
        self.configuration = pink.Configuration(self.model, self.data, q0)

        # === Step 3: Visualization ===
        if visualize:
            self.viz = start_meshcat_visualizer(
                self.robot)  # ✅ 传入 RobotWrapper
            self.viz.display(q0)

        # === Step 4: Define IK tasks ===
        self.ee_task = FrameTask(
            ee_frame,
            position_cost=2.0,
            orientation_cost=1.0,
        )
        self.posture_task = PostureTask(cost=1e-3)
        self.posture_task.set_target(self.configuration.q)

        self.tasks = [self.ee_task, self.posture_task]
        self.ee_task.set_target_from_configuration(self.configuration)

        # === Step 5: Select solver ===
        self.solver = "daqp" if "daqp" in qpsolvers.available_solvers else qpsolvers.available_solvers[
            0]
        print(f"[Pink IK] Using solver: {self.solver}")

        self.dt = 1/100
        # self.rate = RateLimiter(frequency=1/self.dt, warn=False)

    def set_target(self, pos, quat):
        """Set EE target pose"""
        R = pin.Quaternion(quat[0], quat[1], quat[2],
                           quat[3]).toRotationMatrix()
        target = self.ee_task.transform_target_to_world
        target.translation = np.array(pos)
        target.rotation = R
        self.ee_task.set_target(target)

    def step(self):
        """Perform one IK iteration"""
        dq = solve_ik(
            self.configuration,
            self.tasks,
            self.dt,
            solver=self.solver,
            damping=0.05
        )
        self.configuration.integrate_inplace(dq, self.dt)
        if hasattr(self, "viz"):
            self.viz.display(self.configuration.q)

    def run_demo(self):
        """Circular trajectory demo"""
        t = 0.0
        t_start = time.time()
        target_position = np.array([-0.415195, 0.014819, 0.209343])
        target_quat = R.from_euler("xyz", [-2.966, -0.976, 2.99]).as_quat()
        while True:
            radius = 0.2 
            center = np.array([-0.415195, 0.014819, 0.209343])
            omega = 2 * np.pi / 3.0
            pos = center + \
                np.array([0.0, radius * np.cos(omega * t),
                         radius * np.sin(omega * t)])
            quat = R.from_euler("xyz", [-2.966, -0.976, 2.99]).as_quat()
            self.set_target(pos, quat)
            self.step()
            elapse_time = time.time() - t_start
            time.sleep(max(0.0, self.dt - elapse_time))
            t_start = time.time()
            t += self.dt
    
    def move_to_pose_and_get_joints(self, target_pos, target_quat):
        self.set_target(target_pos, target_quat)               
        self.step()
        return self.get_q()
        

    def get_q(self):
        return self.configuration.q

    def publish_joint_states(self, udp_ip, udp_port):
        """Publish joint states over UDP"""
        # Create a UDP socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        while True:
            joint_angles = self.get_q()  # Get joint angles
            joint_states = {
                "name": [f"joint_{i}" for i in range(len(joint_angles))],
                "position": joint_angles.tolist(),
                # Placeholder for velocity
                "velocity": [0.0] * len(joint_angles),
                "effort": [0.0] * len(joint_angles)    # Placeholder for effort
            }

            # Convert joint states to JSON
            joint_states_json = json.dumps(joint_states)

            # Send the JSON data over UDP
            sock.sendto(joint_states_json.encode(), (udp_ip, udp_port))
            time.sleep(0.01)  # Publish at 10Hz


if __name__ == "__main__":
    solver = ArmIKSolver(
        "../urdf/rm_65.urdf",
        ee_frame="ee_link",
        visualize=False
    )

    # run_thread = threading.Thread(target=solver.run_demo)
    # run_thread.start()
    
    radius = 0.2 
    center = np.array([-0.415195, 0.014819, 0.209343])
    omega = 2 * np.pi / 3.0
    pos = center + \
    np.array([0.0, radius * np.cos(omega * 0),
                radius * np.sin(omega * 0)])
    quat = R.from_euler("xyz", [-2.966, -0.976, 2.99]).as_quat()
    result = solver.move_to_pose_and_get_joints(pos, quat)
    print(result)

    # solver_thread = threading.Thread(
    #     target=solver.publish_joint_states, args=(udp_ip, udp_port))
    # solver_thread.start()
