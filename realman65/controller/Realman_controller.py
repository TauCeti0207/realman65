import os
import sys
sys.path.append("./")




from Robotic_Arm.rm_robot_interface import *
import time
import numpy as np
from realman65.controller.arm_controller import ArmController

from realman65.utils.data_handler import debug_print, compute_rotate_matrix

_RM_API_INITIALIZED = False
_RM_API_THREAD_MODE = None
'''
RealMan base code from:
https://develop.realman-robotics.com/robot/apipython/getStarted/
'''

class RealmanController(ArmController):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.controller_type = "user_controller"
        self.controller = None

    def set_up(self,rm_ip, thread_mode=None,connection_level=3, teleop_setup=None,dT=0.01):
        try:
            result = initialize_robot(rm_ip, thread_mode,connection_level)
            if result is None:
                raise ConnectionError(f"Failed to connect to robot arm at {rm_ip}")
            self.controller, self.handle = result
        except Exception as e:
            raise ConnectionError(f"Failed to initialize robot arm: {str(e)}")
        self.prev_tech_state = None
        self.gripper_close = False

        try:
            # from realman65.third_party.Realman_IK.rm_ik import RM_IK
            # self.robot_IK = RM_IK("RM65B", dT)
            if teleop_setup:
                self.robot.set_up(**teleop_setup)
        except:
            debug_print(self.name, "Could not find Realman_IK in third_party, key[teleop_qpos] will not take effect.", "WARNING")

    def reset(self, start_angles):
        """Move robot to the specified start position"""
        print(f"\nMoving {self.name} arm to start position...")
        
        # Check if robot is still connected
        succ, _ = self.controller.rm_get_current_arm_state()
        if succ != 0:
            print(f"Error: {self.name} arm is not connected or responding")
            return False
        
        # Get current joint positions
        succ, state = self.controller.rm_get_current_arm_state()
        if succ == 0:
            current_joints = state['joint']
            print(f"Current {self.name} arm position: {current_joints}")
        
        # release gripper
        self.controller.rm_set_gripper_release(100, 0, 0)

        # Move to start position with error handling
        try:
            print(f"Target {self.name} arm position: {start_angles}")
            result = self.controller.rm_movej(start_angles, 30, 0, 0, 1)  # v=20%, blocking=True
            if result == 0:
                print(f"Successfully moved {self.name} arm to start position")
                # Verify current position
                succ, state = self.controller.rm_get_current_arm_state()
                if succ == 0:
                    current_joints = state['joint']
                    print(f"New {self.name} arm position: {current_joints}")
                    max_diff = max(abs(np.array(current_joints) - np.array(start_angles)))
                    if max_diff > 0.01:  # Allow small tolerance of 0.01 radians
                        print(f"Warning: {self.name} arm position differs from target by {max_diff} radians")
                else:
                    print(f"Warning: Could not verify {self.name} arm position")
                # Wait for system to stabilize
                print(f"Waiting for {self.name} arm to stabilize...")
                time.sleep(2)
                return True
            else:
                print(f"Failed to move {self.name} arm to start position. Error code: {result}")
                return False
        except Exception as e:
            print(f"Exception while moving {self.name} arm: {str(e)}")
            return False

    def get_state(self):
        # Get arm state
        succ, arm_state = self.controller.rm_get_current_arm_state()

        if succ != 0 or arm_state is None:
            raise RuntimeError("Failed to get arm state")
        state = arm_state.copy()

        # SDK does not provide qpos explicitly; reuse pose information when available
        if "qpos" not in state:
            state["qpos"] = np.array(state.get("pose")) if state.get("pose") is not None else None

        state["gripper"] = self.get_gripper()

        return state
    
    # The input gripper value is in the range [0, 1], representing the degree of opening.
    def get_gripper(self):
        try:
            succ, gripper = self.controller.rm_get_gripper_state()
            if succ != 0:
                raise RuntimeError("Failed to get gripper state")
            return gripper
        except Exception as e:
            raise RuntimeError(f"Error getting gripper state: {str(e)}")
        return gripper
    
    '''
    Unlike Libero and Driod, which rely on separate action spaces, 
    RealMan is controlled directly through state-based commands. 
    Therefore, this function is unnecessary.
    '''
    def get_action(self):
        raise NotImplementedError("get_action is not implemented")
    
    def set_joint(self, joint):
        try:
            if len(joint) != 7:
                raise ValueError(f"Invalid joint length: {len(joint)}")
            success = self.controller.rm_movej_canfd(joint.tolist() if isinstance(joint, np.ndarray) else list(joint), False)
            if success != 0:
                raise RuntimeError("Failed to set joint angles")
        except Exception as e:
            raise RuntimeError(f"Error moving robot: {str(e)}")

    def set_position(self, position):
        try:
            # Validate state length
            if len(position) != 6:  # xyz+rpy
                raise ValueError(f"Invalid state length: {len(position)}")
            # delta postion, abs angle
            success = self.controller.rm_movej_canfd(position.tolist() if isinstance(position, np.ndarray) else list(position), False)
            if success != 0:
                raise RuntimeError("Failed to set joint angles")           
        except Exception as e:
            raise RuntimeError(f"Error moving robot: {str(e)}")
    
    def set_position_teleop(self, position):
        try:
            # Validate state length
            if len(position) != 6:  # xyz+rpy
                raise ValueError(f"Invalid state length: {len(position)}")
            target_pose = compute_rotate_matrix(np.asarray(position, dtype=np.float64))

            joint_deg = np.asarray(self.get_state()["joint"], dtype=np.float64)
            joint_rad = np.deg2rad(joint_deg)

            joint_solve = self.robot_IK.solve(joint_rad, target_pose)
            joint_command = np.rad2deg(joint_solve)
            debug_print(self.name, f"Teleop target pos: {position}, joint command: {joint_command}", "DEBUG")

            success = self.controller.rm_movej_canfd(joint_command.tolist(), False)  
            if success != 0:
                raise RuntimeError("Failed to set joint angles")           
        except Exception as e:
            raise RuntimeError(f"Error moving robot: {str(e)}")

    def set_gripper(self, gripper):
        try:
            if gripper < 0.10 and not self.gripper_close:
                success = self.controller.rm_set_gripper_pick(100, 100, 0, 0)
                if success:
                    self.gripper_close = True
                else:
                    raise RuntimeError("Failed to close gripper")
            elif gripper > 0.9 and self.gripper_close:
                success = self.controller.rm_set_gripper_release(100, 0, 0)
                if success:
                    self.gripper_close = False
                else:
                    raise RuntimeError("Failed to open gripper")
        except Exception as e:
            raise RuntimeError(f"Error setting gripper pos: {str(e)}")

    def set_action(self, action):
        raise NotImplementedError("set_action is not implemented")
    
    def __del__(self):
        try:
            if hasattr(self, 'controller'):
                # Add any necessary cleanup for the arm controller
                pass
        except:
            pass

def initialize_robot(robot_ip, thread_mode=None, connection_level=3):
    """Initialize robot arm controller and establish connection"""
    print(f"\nInitializing robot at {robot_ip} with connection level {connection_level}...")
    
    # Create a new instance with the specified thread mode
    global _RM_API_INITIALIZED, _RM_API_THREAD_MODE
    if thread_mode is not None:
        if not _RM_API_INITIALIZED:
            print(f"Using thread mode: {thread_mode}")
            robot_controller = RoboticArm(thread_mode)
            _RM_API_INITIALIZED = True
            _RM_API_THREAD_MODE = thread_mode
        else:
            if _RM_API_THREAD_MODE != thread_mode:
                print(f"Thread mode already initialized as {_RM_API_THREAD_MODE}, reusing existing rm_init for requested mode {thread_mode}.")
            else:
                print(f"Reusing existing rm_init with thread mode: {_RM_API_THREAD_MODE}")
            robot_controller = RoboticArm()
    else:
        if not _RM_API_INITIALIZED:
            print("Using default thread mode")
        else:
            print("Using existing rm_init with default controller instance")
        # Default to single thread mode if none specified
        robot_controller = RoboticArm()
    
    # Try to connect with retry logic
    max_retries = 3
    for attempt in range(max_retries):
        print(f"Connecting to robot at {robot_ip}, attempt {attempt+1}/{max_retries}...")
        handle = robot_controller.rm_create_robot_arm(robot_ip, 8080, connection_level)
        
        if handle.id != -1:
            print(f"Successfully connected to robot at {robot_ip}, handle ID: {handle.id}")
            # Verify connection is active
            succ, state = robot_controller.rm_get_current_arm_state()
            if succ == 0:
                print(f"Connection verified for robot at {robot_ip}")
                print(f"Current state: {state}")
                
                # Get robot info for additional verification
                succ, info = robot_controller.rm_get_robot_info()
                if succ == 0:
                    print(f"Robot info: {info}")
                
                return robot_controller, handle
            else:
                print(f"Connection established but couldn't get state from robot at {robot_ip}. Error code: {succ}")
        else:
            print(f"Failed to create robot arm handle for {robot_ip}. Handle ID: {handle.id}")
        
        if attempt < max_retries - 1:
            print(f"Failed to connect to robot at {robot_ip}, retrying in 2 seconds...")
            time.sleep(2)
    
    print(f"Failed to connect to robot at {robot_ip} after {max_retries} attempts")
    return None

if __name__ == "__main__":
    controller = RealmanController("Realman")
    # controller.set_up("192.168.1.18", rm_thread_mode_e.RM_TRIPLE_MODE_E)
    # controller.reset([0, 0, 0, 0, 0, 0, 0])
    # time.sleep(1)
    # controller.set_gripper(0.5)
    # time.sleep(1)
