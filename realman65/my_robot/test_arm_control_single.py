#!/usr/bin/env python


from realman65.my_robot.realman_65_interface import Realman65Interface
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
    # interface.reset()
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


def test_last_joint_increment():
    """测试函数：持续发送指令，让最后一个自由度角度每次+1度"""
    # 配置参数
    ARM_NAME = "left_arm"  # 目标机械臂
    MAX_ITERATIONS = 50    # 最大迭代次数（避免无限增加）
    # 关节限位（根据实际机械臂调整，此处为示例值）
    JOINT_LIMITS = [
        (-180, 180),   # 关节0范围
        (-90, 90),     # 关节1范围
        (-180, 180),   # 关节2范围
        (-180, 180),   # 关节3范围
        (-90, 90),     # 关节4范围
        (-180, 180)    # 关节5（最后一个自由度）范围
    ]

    try:
        # 1. 初始化接口并连接机械臂
        arm_interface = Realman65Interface(
            auto_setup=True,
        )
        print(f"成功连接{ARM_NAME}，开始测试...")

        # 2. 复位机械臂到初始姿态
        print("复位机械臂到初始位置...")
        arm_interface.reset()
        time.sleep(3)  # 等待复位完成

        # 3. 获取初始关节角度（度）
        initial_joints_deg = arm_interface.get_joint_angles()[ARM_NAME]
        if initial_joints_deg is None or len(initial_joints_deg) < 6:
            raise RuntimeError("获取初始关节角度失败")
        
        print(f"初始关节角度：{initial_joints_deg}")
        
        # 转换为弧度（控制线程中低通滤波用弧度计算）
        current_joints_rad = np.radians(initial_joints_deg[:6])
        print(f"初始关节角度（弧度）：{current_joints_rad}")
        
        # 4. 启动持续控制线程
        arm_interface.start_control(ARM_NAME)
        print(f"启动持续控制，开始递增最后一个关节角度（每次+1度，共{MAX_ITERATIONS}次）...")

        # 5. 循环递增最后一个关节角度并发送
        for i in range(MAX_ITERATIONS):
            # 递增最后一个关节（索引5）的角度
            current_joints_rad[5] += np.radians(1)  # 最后一个自由度+1度

            # 检查是否超出限位（先转回角度检查，更直观）
            current_joint5_deg = np.degrees(current_joints_rad[5])
            min_limit, max_limit = JOINT_LIMITS[5]
            if current_joint5_deg > max_limit:
                print(f"警告：关节5角度已达上限{max_limit}度，停止测试")
                break
            if current_joint5_deg < min_limit:
                print(f"警告：关节5角度已达下限{min_limit}度，停止测试")
                break
            
            # 核心：直接更新共享变量（带锁保护，线程安全）
            with arm_interface.target_joint_angles_lock:
                arm_interface.target_joint_angles = current_joints_rad.tolist()
                arm_interface.init_ik = True  # 告诉控制线程可以开始发送指令



            # 打印当前状态
            if i % 5 == 0:  # 每5次打印一次
                print(f"第{i+1}次：关节5角度 = {current_joint5_deg:.1f}度")

            time.sleep(0.1)  # 控制发送频率

        print("递增测试完成")

    except KeyboardInterrupt:
        print("\n用户中断测试")
    except Exception as e:
        print(f"测试出错：{str(e)}")
    finally:
        # 6. 清理资源：停止线程并复位
        if 'arm_interface' in locals():
            if arm_interface.running:
                arm_interface.stop_control()
            print("复位机械臂到安全位置...")
            arm_interface.reset()
            time.sleep(2)
        print("测试结束")



def test_reset():
    interface = Realman65Interface(auto_setup=True)
    interface.set_up()
    # interface.set_joint_angles('left_arm', [30, 6, 52, -4, 88, -180])
    # interface.set_joint_angles('right_arm', [30, 6, 52, -4, 88, -180])
    # interface.set_gripper('left_arm', 0)
    # interface.set_gripper('right_arm', 0)
    interface.reset()


    # controller = interface._ensure_arm_ready(arm_name)
    


if __name__ == "__main__":

    
    try:
        # test_gripper()
        # test_send_single_angle_dual()
        test_get_state()
        # test_control_arm()
        # test_reset()
    except KeyboardInterrupt:
        # interface.reset()
        pass
    time.sleep(1)