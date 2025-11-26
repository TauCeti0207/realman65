import time
import numpy as np
from scipy.spatial.transform import Rotation as R

# 假设 ArmIKSolver 保存在 realman65/my_robot/pink_ik.py 中
# 如果是在同一个文件中运行，请确保 ArmIKSolver 类定义在上方
from realman65.my_robot.pink_ik import ArmIKSolver

# 初始化单臂 Solver
# 注意：根据你的代码，这里必须指向正确的单臂 URDF 路径
urdf_path = "/home/shui/cloudfusion/DA_D03_description/urdf/rm_65_without_gripper.urdf"
solver = ArmIKSolver(
    urdf_path,
    ee_frame="ee_link",
    visualize=False  # 性能测试通常关闭可视化以减少渲染开销
)

def sample_target(i: int):
    """
    生成单臂的目标位姿
    逻辑保持原双臂测试中的 'Left Arm' 轨迹逻辑
    """
    base = np.array([-0.25, 0.191, 0.229])
    radius = 0.05
    
    # 计算位置：在 YZ 平面画圆 (保留了原代码直接用 i 做弧度的逻辑)
    pos = base + np.array([0.0, radius * np.cos(i), radius * np.sin(i)])
    
    # 计算姿态：保持原有的固定姿态
    quat = R.from_euler('zyx', [3.05727925, -0.02256491, 0.66418082]).as_quat("xyzw")
    
    return pos, quat

warmup = 10
runs = 50

print(f"Starting Warmup ({warmup} iter)...")
for i in range(warmup):
    # 使用 * 解包 pos 和 quat
    solver.move_to_pose_and_get_joints(*sample_target(i), debug_print=False)

print(f"Starting Benchmark ({runs} iter)...")
times = []
for i in range(runs):
    t0 = time.perf_counter()
    
    # 调用单臂 IK 接口
    # move_to_pose_and_get_joints 内部包含了 while 循环迭代直到收敛
    solver.move_to_pose_and_get_joints(*sample_target(i), debug_print=False)
    
    t1 = time.perf_counter()
    times.append(t1 - t0)

# 打印统计结果
print(f"Single Arm IK: min={np.min(times):.4f}s avg={np.mean(times):.4f}s "
      f"p95={np.percentile(times,95):.4f}s max={np.max(times):.4f}s")