#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FK Solver with Pinocchio: load URDF, compute end-effector pose from joint angles.

Requirements:
    pip install pinocchio robot_descriptions  # (robot_descriptions 可选)
"""

from typing import Dict, Iterable, Optional, Union
import os
import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation as R


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

    # ------------------------ 小工具 ------------------------ #
    def joint_order_hint(self) -> None:
        """打印模型关节顺序（便于你按向量方式喂入 q）"""
        print(f"[model.nq={self.model.nq}, nv={self.model.nv}]")
        for jn in self.model.names:
            if jn == "universe":
                continue
            jidx = self.model.getJointId(jn)
            idx_q = self.model.idx_qs[jidx]
            nq_j = self.model.nqs[jidx]
            print(f"{jn:>24s} : q[{idx_q}:{idx_q+nq_j}] (nq={nq_j})")


# ========================== 示例 ========================== #
if __name__ == "__main__":
    # 例1：固定基座机器人（如 Franka、UR5 等）
    # urdf = "/path/to/your_robot.urdf"
    # package_paths = ["/path/to/ros_ws/install/share", "/opt/ros/.../share"]

    # 例2：如果你使用 robot_descriptions 的资源（可选）
    # from robot_descriptions.loaders.pinocchio import load_robot_description
    # model = load_robot_description("franka_panda")
    # -> 这种方式已直接得到 model/data，可另写一个构造器接入；这里先用通用 URDF 方式。

    # urdf = os.environ.get("URDF_PATH", "")  # 通过环境变量传入，或直接写死路径
    # if not urdf:
    #     print("请设置环境变量 URDF_PATH 指向你的 URDF 文件；下面演示使用占位路径。")
    urdf = "../urdf/rm_65.urdf"

    fk = FKServer(
        urdf_path=urdf,
        free_flyer=False,  # 大多数固定基座臂为 False
    )

    fk.joint_order_hint()

    # --- 方式A：按字典输入（推荐，按关节名更直观），单位：弧度
    # q_dict = {    }

    # --- 方式B：按向量输入（长度必须等于 model.nq）
    # q_vec = fk.q_neutral.copy()
    # q_vec[...] = q_vec[...] + 0.0

    # 末端 frame 名称（URDF 中通常是末端的 visual/collision/工具坐标，常叫 "tool0", "ee_link", "tcp" 等）
    q_vec = np.radians(
        [9.196999549865723, 35.672000885009766, 21.599000930786133,
            174.76400756835938, -66.4229965209961, -347.98699951171875]
    )
    ee_name = "ee_link"  # 替换为你的末端 frame 名

    pose = fk.forward_kinematics(q_vec, ee_name)
    print("\n--- FK Result ---")
    print("position (m):\n", pose["position"])
    print(f"target pos:  {-0.455978, -0.060171, 0.473091}")

    print("quaternion (wxyz):\n", pose["quaternion_xyzw"])
    print(
        f"target quat:  {R.from_euler('xyz', [-2.966, -0.976, 2.99], degrees=False).as_quat()}")
