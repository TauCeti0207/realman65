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
from typing import Dict, Iterable, Optional, Union

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
        pkg_dir = "/home/shui/cloudfusion/DA_D03_description/urdf"
        self.robot = RobotWrapper.BuildFromURDF(urdf_path,package_dirs=[pkg_dir])
        self.model = self.robot.model
        self.data = self.robot.data

        # === Step 2: Initialize Pink Configuration ===
        # q0 = pin.randomConfiguration(self.model)
        q0 = np.radians([-1.48, 33.7, 79, 0, 60, -158])
        self.configuration = pink.Configuration(self.model, self.data, q0)

        # === Step 3: Visualization ===
        if visualize:
            self.viz = start_meshcat_visualizer(
                self.robot)  # ✅ 传入 RobotWrapper
            self.viz.display(q0)
        self.ee_frame = ee_frame

        # === Step 4: Define IK tasks ===
        self.ee_task = FrameTask(
            self.ee_frame,
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
        # self.fk = FKServer(urdf_path,free_flyer=False)
        # self.rate = RateLimiter(frequency=1/self.dt, warn=False)
        self.calculate_threshold = 1e-2
        self.max_iter = 1000

    def set_target(self, pos, quat_xyzw):
        """
        Set EE target pose.
        Args:
            pos (np.array): 目标位置 [x, y, z]
            quat_xyzw (np.array): 目标四元数, 格式为 [x, y, z, w]
        """
        # pin.Quaternion 构造函数需要 (w, x, y, z)
        # 我们从 quat_xyzw 中正确地提取它们
        R = pin.Quaternion(quat_xyzw[3],  # w
                           quat_xyzw[0],  # x
                           quat_xyzw[1],  # y
                           quat_xyzw[2]   # z
                           ).toRotationMatrix()
        
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
        # target_position = np.array([-0.415195, 0.014819, 0.209343])
        # target_quat = R.from_euler("xyz", [3., -0.027, 0.663]).as_quat("wxyz")
        while True:
            time0 = time.time()
            radius = 0.2 
            center = np.array([-0.25, 0.191, 0.229])
            omega = 2 * np.pi / 3.0
            pos = center + \
                np.array([0.0, radius * np.cos(omega * t),
                         radius * np.sin(omega * t)])
            quat = R.from_euler('xyz',[ 3.05727925, -0.02256491,  0.66418082]).as_quat("xyzw")
            time1 = time.time()
            self.set_target(pos, quat)
            time2 = time.time()
            self.step()
            time3 = time.time()
            elapse_time = time.time() - t_start
            time.sleep(max(0.0, self.dt - elapse_time))
            print(f"time0:{time0},time1:{time1-time0},time2:{time2-time1},time3:{time3-time2}")
            t_start = time.time()
            t += self.dt
            # error_vector = self.ee_task.compute_error(self.configuration)
            # print(np.linalg.norm(error_vector[3:]))
    
    def move_to_pose_and_get_joints(self, target_pos, target_quat, 
                                    pos_threshold=1e-4,  # 位置阈值 (0.1mm)
                                    ori_threshold=1e-3,  # 姿态阈值 (约 0.05 度)
                                    debug_print=False):   # 增加一个调试开关
        
        # 1. 设置目标
        self.set_target(target_pos, target_quat)
        iter = 0

        while True:
            iter += 1

            # 2. 执行一步 IK
            self.step() 

            # 3. === 关键：从 Task 对象获取内部误差 ===
            # self.configuration 在 self.step() 中被更新
            # compute_error 返回 6D 误差向量 [pos_err_3d, ori_err_3d]
            error_vector = self.ee_task.compute_error(self.configuration)
            
            # 分别计算位置和姿态误差的范数
            pos_err_norm = np.linalg.norm(error_vector[:3]) # 位置误差 (米)
            ori_err_norm = np.linalg.norm(error_vector[3:]) # 姿态误差 (类弧度)
            
            # (可选) 打印当前误差，用于调试
            # if debug_print and iter % 100 == 0:
            #     print(f"Iter {iter} | Pos Err: {pos_err_norm:.6f} | Ori Err: {ori_err_norm:.6f}")

            # 4. 使用分离的阈值进行判断
            if pos_err_norm < pos_threshold and ori_err_norm < ori_threshold:
                if debug_print:
                    print(f"✅ Converged in {iter} iterations.")
                    print(f"Final Position Error: {pos_err_norm:.6f} m")
                    print(f"Final Orientation Error: {ori_err_norm:.6f}")
                return self.get_q()
            
            # 5. 检查是否超时
            if iter > self.max_iter:
                print(f"❌ Failed to converge after {self.max_iter} iterations.")
                print(f"Current Position Error: {pos_err_norm:.6f} m (Target: {pos_threshold})")
                print(f"Current Orientation Error: {ori_err_norm:.6f} (Target: {ori_threshold})")
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
            time.sleep(0.01)  # Publish at 100Hz

class FKServer:
    def __init__(
        self,
        urdf_path: str,
        package_dirs: Optional[Iterable[str]] = None,
        free_flyer: bool = False,
    ):
        # 解析 package://
        if package_dirs is None:
            package_dirs = []
        model = pin.buildModelFromUrdf(
            urdf_path,
        )
        data = model.createData()

        self.model = model
        self.data = data
        self.free_flyer = free_flyer

        # 预存关节名顺序（跳过 universe / root_joint）
        # model.names 与 model.idx_qs 可用来理解顺序
        self.actuated_joint_names = [
            jn for jn in model.names if jn not in ("universe",)
        ]

        # 默认姿势
        self.q_neutral = pin.neutral(model)

    # ------------------------ 输入适配 ------------------------ #
    def _q_from_vector_or_dict(
        self,
        q_input: Union[np.ndarray, Dict[str, float]],
    ) -> np.ndarray:
        """
        支持两种输入：
          - 向量：长度 == model.nq
          - 字典：{joint_name: angle}（单位：弧度）。缺失关节自动用 neutral。
            若 free_flyer=True，字典里一般无需设置 base 的 7 个自由度（使用 neutral）。
        """
        q = self.q_neutral.copy()

        if isinstance(q_input, np.ndarray):
            assert q_input.shape[0] == self.model.nq, \
                f"q size mismatch: got {q_input.shape[0]}, expect {self.model.nq}"
            q[:] = q_input
            return q

        if isinstance(q_input, dict):
            # 遍历所有关节，根据 joint 配置在 q 向量中的段落写入
            for jid, jn in enumerate(self.model.names):
                if jn == "universe":
                    continue
                jidx = self.model.getJointId(jn)
                idx_q = self.model.idx_qs[jidx]     # 在 q 中的起始索引
                nq_j = self.model.nqs[jidx]         # 该关节的 nq 维度
                if jn in q_input:
                    val = q_input[jn]
                    if nq_j == 1:
                        q[idx_q] = float(val)
                    else:
                        # 多自由度关节（比如 free-flyer 的 7D [x,y,z,qx,qy,qz,qw]）
                        v = np.asarray(val, dtype=float).reshape(-1)
                        assert v.size == nq_j, f"{jn} expects {nq_j} params, got {v.size}"
                        q[idx_q:idx_q+nq_j] = v
            return q

        raise TypeError(
            "q_input must be a np.ndarray of shape (nq,) or Dict[str, float]")

    # ------------------------ FK 核心 ------------------------ #
    def forward_kinematics(
        self,
        q_input: Union[np.ndarray, Dict[str, float]],
        ee_frame: str,
    ) -> Dict[str, np.ndarray]:
        """
        计算末端（frame 名称为 ee_frame）的位姿。
        返回：
            {
              "position": [3,],
              "rotation": [3,3],
              "quaternion_xyzw": [4,],  # Pinocchio 内部顺序
              "quaternion_wxyz": [4,],  # 常见顺序
              "T_ee_world": [4,4],
            }
        """
        # 准备 q
        q = self._q_from_vector_or_dict(q_input)

        # 前向计算
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        # 找到 frame
        fid = self.model.getFrameId(ee_frame)
        assert fid != len(self.model.frames), f"Frame '{ee_frame}' not found!"
        placement: pin.SE3 = self.data.oMf[fid]  # world 到 frame 的位姿

        R = placement.rotation.copy()    # (3,3)
        p = placement.translation.copy()  # (3,)

        # 四元数
        q_xyzw = pin.Quaternion(R).normalized().coeffs()  # [x, y, z, w]
        q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])

        # 4x4 齐次
        T = np.eye(4)
        T[:3, :3] = R
        T[:3,  3] = p

        return {
            "position": p,
            "rotation": R,
            "quaternion_xyzw": np.array(q_xyzw),
            "quaternion_wxyz": q_wxyz,
            "T_ee_world": T,
        }



if __name__ == "__main__":
    solver = ArmIKSolver(
        "/home/shui/cloudfusion/DA_D03_description/urdf/rm_65_without_gripper.urdf",
        ee_frame="ee_link",
        visualize=True
    )
    udp_ip = "127.0.0.1"
    udp_port = 12345
    # fk = FKServer("./rm_description/urdf/rm_65.urdf")
    # q_input = np.radians([-64,47,54,-9,75,-208])
    # pose = fk.forward_kinematics(q_input, "ee_link")
    # rotation = pose['rotation']
    # print(R.from_matrix(rotation).as_euler("xyz", degrees=False))
    run_thread = threading.Thread(target=solver.run_demo)
    run_thread.start()
    
    # pos = np.array([-0.151, 0.371, 0.229])
    # quat = R.from_euler("xyz", np.array([2.975, 0.044,  2.55])).as_quat("xyzw")
    # result = solver.move_to_pose_and_get_joints(pos, quat)
    # print(result)

    # solver_thread = threading.Thread(
    #     target=solver.publish_joint_states, args=(udp_ip, udp_port))
    # solver_thread.start()
