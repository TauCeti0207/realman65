import pinocchio as pin
import numpy as np
import time
import matplotlib.pyplot as plt
import qpsolvers
import pink
from pink import solve_ik
from pink.tasks import FrameTask, PostureTask
from pink.visualization import start_meshcat_visualizer
from pinocchio.robot_wrapper import RobotWrapper
from scipy.spatial.transform import Rotation as R
import threading
import os

# =============================
#   1. 导纳控制器类（核心：力→位置转换）
# =============================
class AdmittanceController:
    def __init__(self, M, D, K, dt, init_x=np.zeros(6)):
        """
        导纳控制器初始化
        :param M: 惯性矩阵（6维：平移3+旋转3）
        :param D: 阻尼矩阵
        :param K: 刚度矩阵
        :param dt: 仿真时间步
        :param init_x: 初始任务空间位姿（6维：位置3+旋转向量3）
        """
        self.M = M
        self.D = D
        self.K = K
        self.dt = dt
        self.M_inv = np.linalg.inv(M)  # 预计算逆矩阵提升效率
        self.x = init_x  # 当前任务空间位姿（6维）
        self.v = np.zeros(6)  # 当前任务空间速度（6维）

    def update(self, x_ref, F_ext):
        """
        导纳控制更新：输入参考位姿和外部力，输出期望位姿
        :param x_ref: 参考轨迹位姿（6维）
        :param F_ext: 外部力/力矩（6维）
        :return: 期望任务空间位姿（6维）
        """
        spring_force = self.K.dot(self.x - x_ref)  # 弹簧力（位置偏差）
        damper_force = self.D.dot(self.v)          # 阻尼力（速度）
        acc = self.M_inv.dot(F_ext - spring_force - damper_force)  # 加速度

        # 欧拉积分更新速度和位姿
        self.v = self.v + self.dt * acc
        self.x = self.x + self.dt * self.v

        return self.x.copy()

# =============================
#   2. 双臂IK求解器类（核心：位置→关节角度转换，基于pink库）
# =============================
class DualArmIKSolver:
    def __init__(self, urdf_path, left_ee="left_ee_link", right_ee="right_ee_link", visualize=True):
        """
        初始化双臂IK求解器（RM65双臂）
        :param urdf_path: URDF文件路径
        :param left_ee: 左臂末端连杆名
        :param right_ee: 右臂末端连杆名
        :param visualize: 是否可视化
        """
        # 1. 加载双臂机器人模型
        pkg_dir = "/home/shui/cloudfusion/DA_D03_description/urdf"
        self.robot = RobotWrapper.BuildFromURDF(urdf_path, package_dirs=[pkg_dir])
        self.model = self.robot.model
        self.data = self.robot.data

        # 2. 初始关节角度（RM65双臂home位姿，可根据实际调整）
        q0 = np.radians([0, 44, 114, 3, -66, 179, 0, 44, 114, 3, -66, 179])
        self.configuration = pink.Configuration(self.model, self.data, q0)

        # 3. 可视化初始化
        self.visualize = visualize
        if self.visualize:
            self.viz = start_meshcat_visualizer(self.robot)
            self.viz.display(q0)
            time.sleep(1)  # 等待可视化加载

        # 4. 末端连杆ID
        self.left_ee = left_ee
        self.right_ee = right_ee
        self.left_frame_id = self.model.getFrameId(left_ee)
        self.right_frame_id = self.model.getFrameId(right_ee)

        # 5. 定义IK任务（帧任务：位置+姿态，姿态保持任务）
        # 左臂任务：位置代价2.0，姿态代价1.0
        self.left_task = FrameTask(
            left_ee,
            position_cost=2.0,
            orientation_cost=1.0,
        )
        # 右臂任务：同左臂
        self.right_task = FrameTask(
            right_ee,
            position_cost=2.0,
            orientation_cost=1.0,
        )
        # 姿态保持任务（防止关节偏离初始位姿过远）
        self.posture_task = PostureTask(cost=1e-3)
        self.posture_task.set_target(self.configuration.q)

        # 任务列表
        self.tasks = [self.left_task, self.right_task, self.posture_task]

        # 6. 初始目标位姿：当前末端位姿
        self.left_task.set_target_from_configuration(self.configuration)
        self.right_task.set_target_from_configuration(self.configuration)

        # 7. 选择QP求解器（优先daqp，否则用第一个可用的）
        self.solver = "daqp" if "daqp" in qpsolvers.available_solvers else qpsolvers.available_solvers[0]
        print(f"[IK Solver] Using QP solver: {self.solver}")

        # 8. 仿真参数
        self.dt = 1 / 100.0  # 100Hz控制频率
        self.max_iter = 100  # IK最大迭代次数
        self.viz_freq = 10  # 可视化更新频率：每10步更新一次

    def _quat_xyzw_to_R(self, quat):
        """将xyzw四元数转换为旋转矩阵"""
        return pin.Quaternion(quat[3], quat[0], quat[1], quat[2]).toRotationMatrix()

    def _rot_vec_to_quat_xyzw(self, rot_vec):
        """将旋转向量（3维）转换为xyzw四元数"""
        rot_mat = pin.exp(rot_vec)  # 旋转向量→旋转矩阵（Pinocchio）
        quat_wxyz = pin.Quaternion(rot_mat).coeffs()  # wxyz格式
        quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]  # 转为xyzw
        return np.array(quat_xyzw)

    def set_left_target(self, pos, rot_vec):
        """
        设置左臂目标位姿（位置+旋转向量）
        :param pos: 位置（3维）
        :param rot_vec: 旋转向量（3维）
        """
        # 旋转向量→旋转矩阵→目标变换
        rot_mat = pin.exp(rot_vec)
        target = self.left_task.transform_target_to_world
        target.translation = np.array(pos)
        target.rotation = rot_mat
        self.left_task.set_target(target)

    def set_right_target(self, pos, rot_vec):
        """设置右臂目标位姿（位置+旋转向量）"""
        rot_mat = pin.exp(rot_vec)
        target = self.right_task.transform_target_to_world
        target.translation = np.array(pos)
        target.rotation = rot_mat
        self.right_task.set_target(target)

    def get_ee_pose(self, frame_id):
        """
        获取指定末端的当前位姿（位置+旋转向量）
        :param frame_id: 末端帧ID
        :return: pos(3维), rot_vec(3维)
        """
        # 前向运动学更新
        pin.forwardKinematics(self.model, self.data, self.configuration.q)
        pin.updateFramePlacement(self.model, self.data, frame_id)
        oMf = self.data.oMf[frame_id]
        pos = oMf.translation
        rot_vec = pin.log(oMf.rotation)  # 旋转矩阵→旋转向量（6维→3维，无万向锁）
        return pos, rot_vec

    def step_ik(self, viz_step=0):
        """
        单步IK求解，更新关节配置
        :param viz_step: 当前主循环步数，用于控制可视化频率
        """
        dq = solve_ik(
            self.configuration,
            self.tasks,
            self.dt,
            solver=self.solver,
            damping=0.05,  # 阻尼项，避免奇异
        )
        # 积分关节速度，更新配置
        self.configuration.integrate_inplace(dq, self.dt)
        # 可视化
        if self.visualize and (viz_step % self.viz_freq == 0):
            self.viz.display(self.configuration.q)

    def move_to_target(self, left_pos, left_rot_vec, right_pos, right_rot_vec, pos_threshold=1e-3, ori_threshold=1e-2):
        """
        移动双臂到目标位姿（直到收敛）
        :param left_pos: 左臂目标位置
        :param left_rot_vec: 左臂目标旋转向量
        :param right_pos: 右臂目标位置
        :param right_rot_vec: 右臂目标旋转向量
        :param pos_threshold: 位置误差阈值
        :param ori_threshold: 姿态误差阈值
        :return: 最终关节角度
        """
        self.set_left_target(left_pos, left_rot_vec)
        self.set_right_target(right_pos, right_rot_vec)

        for it in range(self.max_iter):
            self.step_ik()
            # 计算误差
            err_left = self.left_task.compute_error(self.configuration)
            err_right = self.right_task.compute_error(self.configuration)
            pos_err_left = np.linalg.norm(err_left[:3])
            ori_err_left = np.linalg.norm(err_left[3:])
            pos_err_right = np.linalg.norm(err_right[:3])
            ori_err_right = np.linalg.norm(err_right[3:])
            # 收敛判断
            if (pos_err_left < pos_threshold and ori_err_left < ori_threshold and
                pos_err_right < pos_threshold and ori_err_right < ori_threshold):
                break
        return self.configuration.q.copy()

# =============================
#   3. 主程序：双臂导纳控制仿真
# =============================
if __name__ == "__main__":
    # --------------------------
    # 初始化参数
    # --------------------------
    # 1. 加载RM65双臂URDF
    urdf_path = "/home/shui/cloudfusion/DA_D03_description/urdf/rm_65_dual_without_gripper.urdf"
    ik_solver = DualArmIKSolver(urdf_path, "left_ee_link", "right_ee_link", visualize=True)
    dt = ik_solver.dt  # 时间步同步：100Hz（0.01s）
    sim_time = 20  # 总仿真时间20s
    steps = int(sim_time / dt)  # 总步数

    # 2. 导纳控制参数（左右臂共用，可单独设置）
    # 平移：惯性2，阻尼40，刚度80；旋转：惯性0.6，阻尼4，刚度8
    M = np.diag([2, 2, 2, 0.6, 0.6, 0.6])
    D = np.diag([40, 40, 40, 4, 4, 4])
    K = np.diag([80, 80, 80, 8, 8, 8])

    # 3. 获取左右臂初始位姿（位置+旋转向量，6维）
    left_pos_init, left_rot_vec_init = ik_solver.get_ee_pose(ik_solver.left_frame_id)
    right_pos_init, right_rot_vec_init = ik_solver.get_ee_pose(ik_solver.right_frame_id)
    x_init_left = np.hstack([left_pos_init, left_rot_vec_init])  # 左臂初始位姿（6维）
    x_init_right = np.hstack([right_pos_init, right_rot_vec_init])  # 右臂初始位姿（6维）

    # 4. 初始化导纳控制器（左右臂各一个）
    adm_left = AdmittanceController(M, D, K, dt, init_x=x_init_left)
    adm_right = AdmittanceController(M, D, K, dt, init_x=x_init_right)

    # 5. 数据记录（用于绘图）
    log_left_x = []  # 左臂当前位姿
    log_left_x_ref = []  # 左臂参考位姿
    log_left_x_des = []  # 左臂导纳输出位姿
    log_left_force = []  # 左臂外部力
    log_right_x = []  # 右臂当前位姿
    log_right_x_ref = []  # 右臂参考位姿
    log_right_x_des = []  # 右臂导纳输出位姿
    log_right_force = []  # 右臂外部力

    # --------------------------
    # 定义参考轨迹和外部力
    # --------------------------
    def reference_traj_left(t):
        """左臂参考轨迹：初始位姿 + y方向正弦摆动"""
        x_ref = x_init_left.copy()
        x_ref[1] += 0.1 * np.sin(0.5 * t)  # y方向：振幅0.1m，频率0.5Hz
        return x_ref

    def reference_traj_right(t):
        """右臂参考轨迹：初始位姿 + x方向正弦摆动"""
        x_ref = x_init_right.copy()
        x_ref[0] += 0.1 * np.sin(0.5 * t)  # x方向：振幅0.1m，频率0.5Hz
        return x_ref

    def external_force_left(t):
        """左臂外部力：3~10秒时，z方向正弦力（5N）"""
        if 3 < t < 10:
            return np.array([0, 0, 5 * np.sin(3 * t), 0, 0, 0])  # z方向力
        return np.zeros(6)

    def external_force_right(t):
        """右臂外部力：无外力"""
        return np.zeros(6)

    # --------------------------
    # 仿真主循环
    # --------------------------
    print("Simulation start ...")
    t = 0.0
    for i in range(steps):
        # 打印进度（每100步打印一次）
        if i % 100 == 0:
            print(f"Progress: {i}/{steps} steps ({i/steps*100:.1f}%)")
        # 1. 获取当前左右臂末端位姿
        left_pos_curr, left_rot_vec_curr = ik_solver.get_ee_pose(ik_solver.left_frame_id)
        right_pos_curr, right_rot_vec_curr = ik_solver.get_ee_pose(ik_solver.right_frame_id)
        x_curr_left = np.hstack([left_pos_curr, left_rot_vec_curr])
        x_curr_right = np.hstack([right_pos_curr, right_rot_vec_curr])

        # 2. 获取参考轨迹和外部力
        x_ref_left = reference_traj_left(t)
        x_ref_right = reference_traj_right(t)
        F_ext_left = external_force_left(t)
        F_ext_right = external_force_right(t)

        # 3. 导纳控制器更新：得到期望位姿
        x_des_left = adm_left.update(x_ref_left, F_ext_left)
        x_des_right = adm_right.update(x_ref_right, F_ext_right)

        # 4. 拆分期望位姿为位置和旋转向量
        left_pos_des = x_des_left[:3]
        left_rot_vec_des = x_des_left[3:]
        right_pos_des = x_des_right[:3]
        right_rot_vec_des = x_des_right[3:]

        # 5. IK求解：移动双臂到期望位姿
        # ik_solver.move_to_target(left_pos_des, left_rot_vec_des, right_pos_des, right_rot_vec_des)
        # 5. IK求解：设置目标位姿 + 单步IK更新（核心优化：移除收敛循环）
        ik_solver.set_left_target(left_pos_des, left_rot_vec_des)
        ik_solver.set_right_target(right_pos_des, right_rot_vec_des)
        ik_solver.step_ik(viz_step=i)  # 单步IK，传入步数控制可视化频率
        
        
        # 6. 记录数据
        log_left_x.append(x_curr_left)
        log_left_x_ref.append(x_ref_left)
        log_left_x_des.append(x_des_left)
        log_left_force.append(F_ext_left)
        log_right_x.append(x_curr_right)
        log_right_x_ref.append(x_ref_right)
        log_right_x_des.append(x_des_right)
        log_right_force.append(F_ext_right)

        # 7. 更新时间
        t += dt
        # 控制循环频率（可选）
        # time.sleep(dt * 0.1)

    print("Simulation done.")

    # --------------------------
    # 结果可视化
    # --------------------------
    
    # 1. 定义图片保存路径（可自定义，比如保存到当前目录的rm65_admittance_plots文件夹）
    save_dir = "./rm65_admittance_plots/"  # 本地相对路径，也可以用绝对路径如"/home/shui/figs/"
    # 2. 创建保存文件夹（如果不存在则创建）
    os.makedirs(save_dir, exist_ok=True)  # exist_ok=True：文件夹已存在时不报错
    
    # 转换为numpy数组
    log_left_x = np.array(log_left_x)
    log_left_x_ref = np.array(log_left_x_ref)
    log_left_x_des = np.array(log_left_x_des)
    log_left_force = np.array(log_left_force)
    t_array = np.arange(steps) * dt
    
    

    # 绘制左臂z方向偏移和外力
    plt.figure(figsize=(10, 6))
    # 偏移：导纳输出 - 参考轨迹（z方向）
    offset_z_left = (log_left_x_des[:, 2] - log_left_x_ref[:, 2]) * 10  # 放大10倍便于观察
    plt.plot(t_array, offset_z_left, label="Left Arm Z Offset (×10)")
    plt.plot(t_array, log_left_force[:, 2], label="Left Arm Z External Force")
    plt.title("RM65 Left Arm Admittance Control (Z Axis)")
    plt.xlabel("Time (s)")
    plt.ylabel("Offset (m) / Force (N)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "left_arm_z_axis_admittance.png"), bbox_inches='tight', dpi=300)
    plt.show()

    # 绘制左臂y方向参考轨迹与导纳输出
    plt.figure(figsize=(10, 6))
    plt.plot(t_array, log_left_x_ref[:, 1], label="Left Arm Y Reference")
    plt.plot(t_array, log_left_x_des[:, 1], label="Left Arm Y Desired (Admittance)")
    plt.title("RM65 Left Arm Y Position Trajectory")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "left_arm_y_trajectory_admittance.png"), bbox_inches='tight', dpi=300)
    plt.show()