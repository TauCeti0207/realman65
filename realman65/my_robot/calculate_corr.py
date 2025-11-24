
import numpy as np

def normalize(vec):
    vec = np.asarray(vec, dtype=float)
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        raise ValueError("向量长度太小，重新采样")
    return vec / norm

def average_direction(pairs):
    """pairs: [(start_xyz, end_xyz), ...]"""
    diffs = [normalize(np.array(end) - np.array(start)) for start, end in pairs]
    return normalize(np.mean(diffs, axis=0))

def rotation_between(a, b):
    """返回把向量 a 旋转到 b 的 3x3 矩阵"""
    a = normalize(a)
    b = normalize(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    if np.linalg.norm(v) < 1e-8:
        # 同向或反向
        if c > 0:
            return np.eye(3)
        # 反向：选一个与 a 不共线的轴做 180° 旋转
        axis = normalize(np.array([1, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1, 0]))
        v = np.cross(a, axis)
        v = normalize(v)
        return rotation_about_axis(v, np.pi)
    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (np.linalg.norm(v) ** 2))

def rotation_about_axis(axis, angle):
    axis = normalize(axis)
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1 - c
    return np.array([
        [c + x*x*C,     x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s,   c + y*y*C,   y*z*C - x*s],
        [z*x*C - y*s,   z*y*C + x*s, c + z*z*C],
    ])

# ---------- 录入你的数据 ----------

'''
1、
手柄初始位置
[INFO][quest_vr] [VR-l-left_pose raw] [-0.111422   -0.0843984  -0.233896    0.42273908 -0.00800781 -0.00165358]
手柄向右移动
[INFO][quest_vr] [VR-l-left_pose raw] [-0.0211609  -0.0880452  -0.229981    0.38683784  0.04480498 -0.06835152]

机械臂初始位置
[INFO][teleop] arm_pose: [-0.438376  0.01549   0.242122 -1.449    -1.156     1.351   ]
机械臂向右移动
[INFO][teleop] arm_pose: [-0.438377  0.114711  0.242154 -1.448    -1.156     1.351   ]


2、
手柄初始位置
[INFO][quest_vr] [VR-l-left_pose raw] [-0.0446039  -0.0874234  -0.236503    0.49725249  0.02578962 -0.19753345]
手柄向上移动
[INFO][quest_vr] [VR-l-left_pose raw] [-0.0438013  -0.0293455  -0.233745    0.44593527  0.03464717 -0.29642102]


机械臂初始位置
[INFO][teleop] arm_pose: [-0.438376  0.01549   0.242122 -1.449    -1.156     1.351   ]
机械臂向上移动
[INFO][teleop] arm_pose: [-0.438374  0.015491  0.344877 -1.449    -1.156     1.351   ]



3、
手柄初始位置
[INFO][quest_vr] [VR-l-left_pose raw] [-0.0151094  -0.0480795  -0.194393    0.36030485  0.10322715 -0.01902549]
手柄向前移动
[INFO][quest_vr] [VR-l-left_pose raw] [-0.0298849  -0.0519552  -0.415178    0.31225185 -0.20338047 -0.02409479]


机械臂初始位置
[INFO][teleop] arm_pose: [-0.438376  0.01549   0.242122 -1.449    -1.156     1.351   ]
机械臂向前移动
[INFO][teleop] arm_pose: [-0.531855  0.015497  0.242165 -1.448    -1.156     1.351   ]


'''
# 每个列表中放若干 (start_xyz, end_xyz) 对，坐标只取位置 xyz 三个分量
# ------------ 录入的三轴数据 ------------
vr_lr_pairs = [
    ([-0.111422, -0.0843984, -0.233896],
     [-0.0211609, -0.0880452, -0.229981]),
]
arm_lr_pairs = [
    ([-0.438376, 0.01549, 0.242122],
     [-0.438377, 0.114711, 0.242154]),
]

vr_ud_pairs = [
    ([-0.0446039, -0.0874234, -0.236503],
     [-0.0438013, -0.0293455, -0.233745]),
]
arm_ud_pairs = [
    ([-0.438376, 0.01549, 0.242122],
     [-0.438374, 0.015491, 0.344877]),
]

vr_fb_pairs = [
    ([-0.0151094, -0.0480795, -0.194393],
     [-0.0298849, -0.0519552, -0.415178]),
]
arm_fb_pairs = [
    ([-0.438376, 0.01549, 0.242122],
     [-0.531855, 0.015497, 0.242165]),
]



# ---------- Step 1：左右轴 ----------
vr_lr_dir = average_direction(vr_lr_pairs)
arm_lr_dir = average_direction(arm_lr_pairs)
R_step1 = rotation_between(vr_lr_dir, arm_lr_dir)

# ---------- Step 2：上下轴 ----------
if vr_ud_pairs and arm_ud_pairs:
    vr_up_dir_raw = average_direction(vr_ud_pairs)
    arm_up_dir    = average_direction(arm_ud_pairs)

    vr_up_after_step1 = R_step1 @ vr_up_dir_raw

    axis = arm_lr_dir  # 只在左右轴垂直平面内调
    proj_vr = normalize(vr_up_after_step1 - np.dot(vr_up_after_step1, axis) * axis)
    proj_arm = normalize(arm_up_dir - np.dot(arm_up_dir, axis) * axis)

    angle = np.arctan2(
        np.dot(axis, np.cross(proj_vr, proj_arm)),
        np.dot(proj_vr, proj_arm)
    )
    R_step2 = rotation_about_axis(axis, angle) @ R_step1
else:
    R_step2 = R_step1

# ---------- Step 3：前后轴 ----------
if vr_fb_pairs and arm_fb_pairs:
    vr_fb_dir_raw = average_direction(vr_fb_pairs)
    arm_fb_dir    = average_direction(arm_fb_pairs)
    R_step3 = rotation_between(R_step2 @ vr_fb_dir_raw, arm_fb_dir) @ R_step2
else:
    # 没有前后实测，就用叉乘保证正交
    arm_up_dir_final = normalize(R_step2 @ vr_up_dir_raw) if vr_ud_pairs else arm_lr_dir
    arm_fb_dir_final = normalize(np.cross(arm_lr_dir, arm_up_dir_final))
    R_step3 = np.column_stack([arm_lr_dir, arm_fb_dir_final, arm_up_dir_final])

# ---------- 输出 4×4 调整矩阵 ----------
adj_mat = np.eye(4)
adj_mat[:3, :3] = R_step3

print("Final adj_mat = [")
for row in adj_mat:
    print("    [" + ", ".join(f"{val: .8f}" for val in row) + "],")
print("]")

# 若还要校平移：记录手柄/末端重合时的位置 p_vr、p_arm，再执行：
# p_vr = np.array([...])
# p_arm = np.array([...])
# adj_mat[:3, 3] = p_arm - R_step3 @ p_vr
