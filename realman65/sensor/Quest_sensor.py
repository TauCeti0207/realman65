import sys

import time

import numpy as np

from realman65.sensor.teleoperation_sensor import TeleoperationSensor
from realman65.utils.data_handler import matrix_to_xyz_rpy, compute_local_delta_pose, debug_print, euler_to_matrix, compute_rotate_matrix

from scipy.spatial.transform import Rotation as R
from typing import Callable, Optional

from oculus_reader import OculusReader

'''
QuestVR base code from:
https://github.com/rail-berkeley/oculus_reader.git
'''

def adjustment_matrix(transform):
    # print("adjustment_matrix from", __file__)
    if transform.shape != (4, 4):
        raise ValueError("Input transform must be a 4x4 numpy array.")
    
    # now = time.time()
    # last = getattr(adjustment_matrix, "_last_log", 0)
    # if now - last > 0.1:
    #     raw_pose = matrix_to_xyz_rpy(transform.copy())
    #     debug_print("quest_vr", f"[VR raw] {raw_pose}", "INFO")
    #     adjustment_matrix._last_log = now
    
    # adj_mat = np.array([
    #     [0,0,-1,0],
    #     [-1,0,0,0],
    #     [0,1,0,0],
    #     [0,0,0,1]
    # ])

    adj_mat = np.array([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1]
    ])


#     adj_mat = np.array([
#     [ 0.06685501,  0.01796897,  0.99760088,  0.00000000],
#     [ 0.99704832, -0.03903152, -0.06611493,  0.00000000],
#     [ 0.03774986,  0.99907640, -0.02052538,  0.00000000],
#     [ 0.00000000,  0.00000000,  0.00000000,  1.00000000],
# ])

    adj_mat = np.array([[0,0,1,0],[1,0,0,0],[0,1,0,0],[0,0,0,1]])


    r_adj = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    
    # r_adj = euler_to_matrix(np.array([0,0,0, -np.pi , 0, -np.pi/2]))

    # r_adj = euler_to_matrix(np.array([0,0,0, 0, np.pi, np.pi/2]))
    
    transform = adj_mat @ transform  
    
    transform = np.dot(transform, r_adj)
    # transform = transform @ r_adj  
    
    
    # if now - last > 0.5:
    #     adj_pose = matrix_to_xyz_rpy(transform.copy())
    #     debug_print("quest_vr", f"[VR adj] {adj_pose}", "INFO")
    
    return transform

class QuestSensor(TeleoperationSensor):
    def __init__(self,name):
        super().__init__()
        self.name = name
        self.base_pose = None  # 遥操作启动时的基准位姿（左、右手）
        self.last_pose = None  # 最近一次成功获取的位姿（左、右手）
        self.current_raw_pose = None  # 最近一次Quest控制器的绝对位姿（左、右手）

    def set_up(self):

        self.sensor = OculusReader()
        self.base_pose = None
        self.last_pose = None
        self.current_raw_pose = None

    def get_state(self):
        transformations, buttons = self.sensor.get_transformations_and_buttons()
        
        left_pose = None
        right_pose = None
        if transformations and transformations.get('r') is not None:
            try:
                right_pose = matrix_to_xyz_rpy(adjustment_matrix(np.asarray(transformations['r'])))
                # debug_print("quest_vr", f"[VR-r raw] {transformations['r']}", "INFO")
                # debug_print("quest_vr", f"[VR-r-right_pose raw] {right_pose}", "INFO")
                
            except Exception as exc:
                debug_print(self.name, f"Failed to convert right transform: {exc}", "WARNING")
                right_pose = None

            if transformations.get('l') is not None:
                try:
                    left_pose = matrix_to_xyz_rpy(adjustment_matrix(np.asarray(transformations['l'])))
                    # debug_print("quest_vr", f"[VR-l-transformations raw] {transformations['l']}", "INFO")
                    debug_print("quest_vr", f"[VR-l-left_pose raw] {left_pose}", "INFO")
                    
                except Exception as exc:
                    debug_print(self.name, f"Failed to convert left transform: {exc}", "WARNING")
                    left_pose = None

        if right_pose is None:
            if self.last_pose is not None:
                left_pose, right_pose = (np.array(self.last_pose[0]), np.array(self.last_pose[1]))
                self.current_raw_pose = np.concatenate([left_pose, right_pose])
                if not transformations:
                    debug_print(self.name, "No controller transforms received, using cached pose", "INFO")
                else:
                    debug_print(self.name, "Right controller transform missing, using cached pose", "INFO")
            else:
                debug_print(self.name, "Right controller data unavailable", "INFO")
                return {
                    "end_pose": None,
                    "extra": buttons,
                }
        else:
            if left_pose is None:
                if self.last_pose is not None:
                    left_pose = np.array(self.last_pose[0])
                else:
                    debug_print(self.name, "Left controller missing, using zeros", "INFO")
                    left_pose = np.zeros(6)

            self.current_raw_pose = np.concatenate([left_pose, right_pose])

        if self.base_pose is None:
            self.base_pose = (np.array(left_pose), np.array(right_pose))
            debug_print(self.name, f"Initializing base pose. Right pose: {right_pose}", "INFO")
            delta_left = np.zeros(6)
            delta_right = np.zeros(6)
            if self.current_raw_pose is None:
                self.current_raw_pose = np.concatenate([left_pose, right_pose])
        else:
            delta_left = compute_local_delta_pose(self.base_pose[0], left_pose)
            delta_right = compute_local_delta_pose(self.base_pose[1], right_pose)

        self.last_pose = (np.array(left_pose), np.array(right_pose))

        end_pose = np.concatenate([delta_left, delta_right])
        return {
            "end_pose": end_pose,
            "extra": buttons,
            "raw_pose": self.current_raw_pose,
        }

    def reset(self, buttons):
        debug_print(f"{self.name}", "reset success!", "INFO")
        return

if __name__ == "__main__":
    import time
    teleop = QuestSensor("left_pika")

    teleop.set_up()

    teleop.set_collect_info(["end_pose","extra"]) 
    
    while True:
        pose, buttons = teleop.get_state()["end_pose"]
        left_pose = pose[:6]
        right_pose = pose[-6:]

        teleop.reset(buttons)
        
        print("left_pose:\n", left_pose)
        print("right_pose:\n", right_pose)
        print("buttons:\n", buttons)
        time.sleep(0.1)
