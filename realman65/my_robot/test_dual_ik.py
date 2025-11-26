import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from realman65.my_robot.dual_pink_ik import DualArmIKSolver

solver = DualArmIKSolver(
    "/home/shui/cloudfusion/DA_D03_description/urdf/rm_65_dual_without_gripper.urdf",
    "left_ee_link",
    "right_ee_link",
    visualize=False
)

def sample_target(i: int):
    base = np.array([-0.25, 0.191, 0.229])
    radius = 0.05
    left_pos = base + np.array([0.0, radius * np.cos(i), radius * np.sin(i)])
    left_quat = R.from_euler('zyx', [3.05727925, -0.02256491, 0.66418082]).as_quat("xyzw")
    right_pos = left_pos.copy()
    right_pos[1] += 0.5
    right_quat = left_quat.copy()
    return left_pos, left_quat, right_pos, right_quat

warmup = 10
runs = 50

for i in range(warmup):
    solver.move_dual_arm(*sample_target(i), debug_print=False)

times = []
for i in range(runs):
    t0 = time.perf_counter()
    solver.move_dual_arm(*sample_target(i), debug_print=True)
    t1 = time.perf_counter()
    times.append(t1 - t0)

print(f"min={np.min(times):.4f}s avg={np.mean(times):.4f}s p95={np.percentile(times,95):.4f}s max={np.max(times):.4f}s")