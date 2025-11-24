# Realman65 项目简介

基于 `\home\shui\yzq\realman65\realman65` 的机器人控制与遥操作框架，涵盖机械臂接口、移动平台控制、传感器接入、VR 遥操作、数据采集与 IK 求解等功能，支持单臂与双臂场景，并可集成 ROS/ROS2。

## 功能特性
- 机械臂控制：`Realman_controller` 系列，支持 RM-65/75 等型号与双臂控制
- 传感器集成：Realsense、Vitac3D、Quest/VR 等驱动与数据管线
- VR/遥操作：Oculus/Quest、Unity/ROS2 遥操作示例
- 运动学与描述：URDF/Xacro、RViz 配置，Pink IK/自研 IK 库
- 数据采集：采集、校验与格式转换（含 LeRobot 数据生成）
- ROS/ROS2 互通：话题发布/订阅封装与示例

## 目录结构
- `my_robot/` 机械臂接口与示例、URDF/RViz、IK、VR 控制脚本
- `controller/` 控制器实现（Realman、移动平台、ROS/ROS2 适配）
- `sensor/` 传感器驱动与多线程采集（Realsense/Quest/Vision 等）
- `utils/` 公共工具（ROS/ROS2 发布订阅、调度、双向 socket 等）
- `data/` 数据采集与转换脚本（采集、校验、LeRobot 生成）
- `third_party/` 外部库（`Realman_IK` 的 `.dll/.so` 等）

## 环境要求
- Python `>=3.8`（建议 `3.10+`）
- 操作系统：Windows/Linux（`third_party/Realman_IK/lib` 同时提供 `.dll/.so`）
- 可选组件：`ROS/ROS2`、`Intel RealSense SDK`
- 依赖管理：`requirements.txt` 与 `setup.py` 已集成



## 安装
- 建议使用虚拟环境：
  - `python -m venv .venv && source .venv/bin/activate`（Linux）
- 方式一：作为库安装（读取 `requirements.txt` 自动安装依赖）
  - `pip install .`
- 方式二：仅安装依赖
  - `pip install -r requirements.txt`
- 依赖清单（摘录）
  - `numpy==1.26.4`（`setup.py` 默认依赖）
  - `scipy==1.15.3`, `h5py`, `pyrealsense2`, `opencv-python`, `keyboard`, `Robotic-Arm==1.1.1`, `pure-python-adb`, `pyyaml`

## 快速开始
- 连接与基础接口：
  - `python my_robot/realman_65_interface.py`
- VR 遥操作（Oculus/Quest）：
  - `python my_robot/vr_control_arm_oculus.py`
- ROS2 遥操作示例：
  - `python my_robot/vr_control_arm_ros2.py`
- 双臂控制基础：
  - `python my_robot/realman_65_dual_base.py`
- 双臂 + Unity 交互（示例）：
  - `python my_robot/vr_control_arm_unity_dual.py`
  - 依赖：`rclpy`, `tf2_ros`, `scipy`, `termcolor`
  - 说明：ROS2 TF 订阅手柄位姿，支持左右臂独立重定标与夹爪控制；左手扳机开启遥操，右手扳机停止并复位。
 - 双臂接口实现（本次主要改动）：
   - `python my_robot/realman_65_interface_dual.py`
   - 特性：
     - 独立左右臂控制线程（`start_control`/`stop_control`）
     - 双臂 IK 求解（`DualArmIKSolver`，URDF 链接位于本地路径）
     - 目标位姿更新与关节角缓存（`update`/`init_ik`）
     - 夹爪控制与状态查询（`set_gripper`/`get_gripper_state`）
     - 低通滤波接口预留（可按需启用）
- 数据采集：
  - `python data/collect_any.py`
  - 生成 LeRobot 数据：`python data/generate_lerobot.py`
- 仅测试 Realsense 管线：
  - `python my_robot/_realsense_only.py`

## ROS/ROS2 支持
- 发布/订阅封装：`utils/ros_publisher.py`, `utils/ros_subscriber.py`, `utils/ros2_publisher.py`, `utils/ros2_subscriber.py`
- 机器人描述与可视化：`my_robot/rm_description/urdf/*`, `my_robot/rm_description/rviz/*`
 - 打包说明：`setup.py` 指定 `package_dir={"realman65":"realman65"}` 并在 `find_packages(where="realman65", include=["realman65", "realman65.*"])` 下打包；安装时会读取 `requirements.txt` 作为 `install_requires`。

## 重要说明
- 运行前请确认脚本中的 IP、串口/设备号、话题名称等参数与实际硬件一致
- 如使用外部 IK：确保 `third_party/Realman_IK/lib/librman_algorithm.dll/.so` 可被系统加载
- 不同平台/传感器可能需要对应驱动和权限配置
 - 双臂接口：`my_robot/realman_65_interface_dual.py` 提供双臂 IK (`DualArmIKSolver`) 与独立关节发送线程，新增夹爪控制与状态查询 API。

## 许可
- 许可证信息未明确，请根据实际情况补充

## 致谢
- Realman 机器人硬件与 SDK
- Intel RealSense
- Pink IK 与相关社区开源组件