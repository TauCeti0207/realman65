from Robotic_Arm.rm_robot_interface import *

arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
arm1 = RoboticArm()
handle = arm.rm_create_robot_arm("192.168.1.18", 8080, level=3)
handle1 = arm1.rm_create_robot_arm("192.168.2.19", 8080, level=3)
print(handle.id)
print(handle1.id)
print(arm.rm_get_current_arm_state())
print(arm1.rm_get_current_arm_state())