#!/usr/bin/env python


# from realman65.my_robot.realman_65_interface_right import Realman65Interface
from realman65.my_robot.realman_65_interface_dual import Realman65Interface
import time
from termcolor import cprint



def test_gripper():
    interface = Realman65Interface(auto_setup=False)
    interface.set_up()
    # interface.set_gripper('left_arm',1)
    interface.set_gripper('left_arm',0)
    interface.set_gripper('right_arm',0)
    

def test_get_state():
    interface = Realman65Interface(auto_setup=False)
    interface.set_up()
    interface.reset()
    ee_pose = interface.get_end_effector_pose()
    cprint(f"ee_pose: {ee_pose}", 'green')
    
    joint_angle = interface.get_joint_angles()
    cprint(f"joint_angle: {joint_angle}", 'red')
    
    gripper_state = interface.get_gripper_state()
    cprint(f"gripper_state: {gripper_state}", 'blue')
    

def test_control_arm():
    rm_interface = Realman65Interface(auto_setup=False)
    rm_interface.set_up()
    rm_interface.reset()

    left_target_pose = [-0.39, 0.1259, 0.34, 2.8, 0.17, -0.788]
    right_target_pose = [-0.39, 0.1259, 0.34, 2.8, 0.17, -0.788]
    
    rm_interface.init_ik = True
    while True:
        # rm_interface.update(left_target_pose,right_target_pose)
        rm_interface.update(left_target_pose)
        time.sleep(0.01)


def test_control_arm_dual():
    rm_interface = Realman65Interface(auto_setup=False)
    rm_interface.set_up()
    rm_interface.reset()

    left_target_pose = [-0.39, 0.1259, 0.34, 2.8, 0.17, -0.788]
    right_target_pose = [-0.39, 0.1259, 0.34, 2.8, 0.17, -0.788]
    
    # left_target_pose = rm_interface.get_end_effector_pose().get('left_arm')
    # right_target_pose = rm_interface.get_end_effector_pose().get('right_arm')
    
    right_target_pose[1] += 0.5
    
    
    rm_interface.init_ik = True
    while True:
        rm_interface.update(left_target_pose,right_target_pose)
        time.sleep(0.01)


if __name__ == "__main__":

    # interface = Realman65Interface(auto_setup=True)
    # interface.set_up()
    # interface.reset()
    
    try:
        test_gripper()
        # test_get_state()
        # test_control_arm()
        # test_control_arm_dual()
    except KeyboardInterrupt:
        interface.reset()
    time.sleep(1)