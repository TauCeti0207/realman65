#!/usr/bin/env python3
# -*- coding: utf-8 -*-


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
from typing import Dict, Iterable, Optional, Union

class DualArmIKSolver:
    def __init__(self, urdf_path,
                 left_ee="left_ee_link",
                 right_ee="right_ee_link",
                 visualize=True):

        # === Step 1: Load dual-arm URDF ===
        pkg_dir = "/home/shui/cloudfusion/DA_D03_description/urdf" 

        self.robot = RobotWrapper.BuildFromURDF(urdf_path, package_dirs=[pkg_dir])
        # self.robot = RobotWrapper.BuildFromURDF(urdf_path)
        self.model = self.robot.model
        self.data = self.robot.data

        # === Step 2: Initialize Pink configuration ===
        # 默认从 home 姿态开始，你可按需要修改为你的双臂初始姿态
        q0 = np.radians([-1.48, 33.7, 79, 0, 60, -158, -1.48, 33.7, 79, 0, 60, -158])
        self.configuration = pink.Configuration(self.model, self.data, q0)

        # === Step 3: Visualization ===
        if visualize:
            self.viz = start_meshcat_visualizer(self.robot)
            self.viz.display(q0)

        self.left_ee = left_ee
        self.right_ee = right_ee

        # === Step 4: Define IK tasks ===
        self.left_task = FrameTask(
            self.left_ee,
            position_cost=2.0,
            orientation_cost=1.0,
        )
        self.right_task = FrameTask(
            self.right_ee,
            position_cost=2.0,
            orientation_cost=1.0,
        )

        # 姿态保持任务（代价小）
        self.posture_task = PostureTask(cost=1e-3)
        self.posture_task.set_target(self.configuration.q)

        # 所有任务
        self.tasks = [self.left_task, self.right_task, self.posture_task]

        # 初始目标 = 当前位姿
        self.left_task.set_target_from_configuration(self.configuration)
        self.right_task.set_target_from_configuration(self.configuration)

        # === Step 5: Select QP solver ===
        self.solver = (
            "daqp" if "daqp" in qpsolvers.available_solvers
            else qpsolvers.available_solvers[0]
        )
        print(f"[Pink Dual-Arm IK] Using solver: {self.solver}")

        self.dt = 1 / 100.0
        self.max_iter = 3000


    # ============================================================
    #               ========  Target setting  ========
    # ============================================================

    def _quat_xyzw_to_R(self, quat):
        """Convert xyzw quaternion to rotation matrix"""
        return pin.Quaternion(quat[3], quat[0], quat[1], quat[2]).toRotationMatrix()


    def set_left_target(self, pos, quat_xyzw):
        Rmat = self._quat_xyzw_to_R(quat_xyzw)
        target = self.left_task.transform_target_to_world
        target.translation = np.array(pos)
        target.rotation = Rmat
        self.left_task.set_target(target)


    def set_right_target(self, pos, quat_xyzw):
        Rmat = self._quat_xyzw_to_R(quat_xyzw)
        target = self.right_task.transform_target_to_world
        target.translation = np.array(pos)
        target.rotation = Rmat
        self.right_task.set_target(target)

    def step(self):
        dq = solve_ik(
            self.configuration,
            self.tasks,
            self.dt,
            solver=self.solver,
            damping=0.05,
        )
        self.configuration.integrate_inplace(dq, self.dt)

        if hasattr(self, "viz"):
            self.viz.display(self.configuration.q)

    def move_dual_arm(self,
                      left_pos, left_quat,
                      right_pos, right_quat,
                      pos_threshold=1e-4,
                      ori_threshold=1e-3,
                      debug_print=False):

        # Set target
        self.set_left_target(left_pos, left_quat)
        self.set_right_target(right_pos, right_quat)

        it = 0
        while it < self.max_iter:
            it += 1
            self.step()

            errL = self.left_task.compute_error(self.configuration)
            errR = self.right_task.compute_error(self.configuration)

            pos_err_L = np.linalg.norm(errL[:3])
            ori_err_L = np.linalg.norm(errL[3:])

            pos_err_R = np.linalg.norm(errR[:3])
            ori_err_R = np.linalg.norm(errR[3:])

            if (pos_err_L < pos_threshold and ori_err_L < ori_threshold and
                pos_err_R < pos_threshold and ori_err_R < ori_threshold):

                if debug_print:
                    print(f"✅ Dual-arm IK converged in {it} iterations.")
                    print(f"Left  pos error = {pos_err_L:.6f}, ori = {ori_err_L:.6f}")
                    print(f"Right pos error = {pos_err_R:.6f}, ori = {ori_err_R:.6f}")
                return self.get_q()

        print("❌ Dual-arm IK failed to converge.")
        print(f"Left  pos error = {pos_err_L:.6f}, ori = {ori_err_L:.6f}")
        print(f"Right pos error = {pos_err_R:.6f}, ori = {ori_err_R:.6f}")
        return self.get_q()

    def get_q(self):
        return self.configuration.q
    
    def run_demo(self):
        """Circular trajectory demo"""
        t = 0.0
        t_start = time.time()
        while True:
            radius = 0.2 
            center = np.array([-0.25, 0.191, 0.229])
            omega = 2 * np.pi / 3.0
            left_pos = center + \
                np.array([0.0, radius * np.cos(omega * t),
                         radius * np.sin(omega * t)])
            left_quat = R.from_euler('zyx',[ 3.05727925, -0.02256491,  0.66418082]).as_quat("xyzw")
            right_pos = center + np.array([0.0, radius * np.cos(omega * t)+0.5,radius * np.sin(omega * t)])
            right_quat = left_quat.copy()
            
            self.set_left_target(left_pos, left_quat)
            self.set_right_target(right_pos, right_quat)
            self.step()
            elapse_time = time.time() - t_start
            time.sleep(max(0.0, self.dt - elapse_time))
            t_start = time.time()
            t += self.dt
            # error_vector = self.ee_task.compute_error(self.configuration)
            # print(np.linalg.norm(error_vector[3:]))

if __name__ == "__main__":
    solver = DualArmIKSolver("/home/shui/cloudfusion/DA_D03_description/urdf/rm_65_dual_without_gripper.urdf","left_ee_link","right_ee_link",visualize=True)


    udp_ip = "127.0.0.1"
    udp_port = 12345

    run_thread = threading.Thread(target=solver.run_demo)
    run_thread.start()