[![中文](https://img.shields.io/badge/中文-简体-blue)](./README.md)  
[![English](https://img.shields.io/badge/English-English-green)](./README_EN.md)

[中文WIKI(还在完善)](https://tian-nian.github.io/control_your_robot-doc/)

# 预计于2025.9.30前整理一版本稳定代码并发布,带上完整的使用教程
最近在准备一些真机实验,很快1.0稳定版代码将与大家见面了,使用过程有问题也可以添加我的微信:nian15959297867 

# 控制你的机器人!
该项目旨在与帮助各位进入具身智能领域后能快速上手一整套从控制机械臂开始, 到数据采集, 到最终VLA模型的训练与部署的流程.

## 近期预计改善
1. 添加数据可视化采集, 帮助快速检查是否存在数据跳变
2. 添加数据快速完整性检验

## 更新日志(只保留最新5条, 完整放在WIKI主页)
### 8.14
1. 添加了`Cv_sensor`, `VisionROS_sensor`, 并在部分采集数据过程中的高频轮询中加入了time.sleep(0.001), 降低了对CPU的性能占用
2. 修改了多进程的操作, 现在可以在多进程中加入ROS操作了

### 8.5
1. 新增了`Worker`类, 方便自定义多进程操作, 并提供了多进程同步的主从臂遥操示例`example/teleop/master_slave_arm_teleop.py`.  
    该组件提供了自定义接口:
    1. **handler():** 用于处理循环操作(如循环获取机械臂数据)
    2. **finish():** 用于处理进程结束时操作(如保存采集数据)
    3. **component_init():** 用于初始化组件(如果初始化机械臂, 或者急切人)
    4. **next_to():** 用于声明链式操作的下一个操作, 如果是最后一个, 则只需要将其的`next_event`交给`time_scheduler`
2. 修改了`time_scheduler`类, 将原本进程同步机制从Sem改成Event, 并添加了`end_event=`, 如果初始化了该参数, 则会使用链式操作

### 7.29
1. 添加了ROS2支持, 为其添加案例`controller/Bunker_controller` 
2. 为`collect_any`添加了关键词`move_check`, 用于自动筛除无运动部分数据,自动开启

### 7.18
1. 为`my_robot`中添加了`base_robot`基类, 便于快速导入自己的机器人, 无需重复定义多余函数
2. 添加了睿尔曼提供的自定义IK支持, 并于`arm_controller.move_controller()`添加新关键词`teleop_qpos`用于专门调用遥操接口(Canfd之类)

### 7.14
1. 添加了组件型的并行化单元`component_worker`用于多进程数据采集
2. 修改了`time_scheduler`的时间控制逻辑, 提升了时间的同步性, 并额外保存了实际采样时间间隔于config.json中

### 6.28
1. 合并了来自于`akajiejie`的关于`Reaksense_MultiThread_sensor`的分支, 里面根据部署运行更新了部分示例的代码

## 快速上手!
由于本项目实现了部分测试样例, 如机械臂测试样例, 视觉模拟样例, 完整机器人模拟样例, 因此可以在没有任何实体的情况下快速了解本项目的整体框架.
由于没涉及任何本体, 所以安装环境只需要执行:
```
 pip install -r requirements.txt
```  
本项目有特殊的调试参数, 分为:"DEBUG", "INFO", "ERROR", 如果想要完整看到数据的流程, 可以设置为"DEBUG".
```bash
export INFO_LEVEL="DEBUG"
```
或者可以在对应main函数中引入:
```python
import os
os.environ["INFO_LEVEL"] = "DEBUG" # DEBUG , INFO, ERROR
```
1. 数据采集测试
```bash
# 多进程(通过时间同步器实现更严格的等时间距采集)
python example/collect/collect_mp_robot.py
# 多进程(对每个元件单独进程采集数据)
python example/collect/collect_mp_component.py
# 单线程(会存在一些由于函数执行导致的延迟堆积)
python example/collect/collect.py
```
2. 模型部署测试
```bash
# 跑一个比较直观的部署测试代码
python example/deploy/robot_on_test.py
# 实现的通用部署脚本
bash deploy.sh
```
3. 远程部署数据传输
```bash
# 先启动服务器, 模仿推理端(允许多次连接, 监听端口)
python scripts/server.py
# 本地, 获取数据并执行指令(示例只执行了10次)
python scripts/client.py
```
4. 一些有意思的代码
```python
# 采集对应的关键点, 并且进行轨迹重演
python scripts/collect_moving_ckpt.py 
# sapien仿真, 请参考planner/README.md
```
5. 调试对应的一些代码
```bash
# 由于controller与sensor有__init__.py, 所以需要按照-m形式执行代码
python -m controller.TestArm_controller
python -m sensor.TestVision_sensor
python -m my_robot.test_robot
```

## 🚀 机器人控制开发进展

### 📅 更新记录
| 日期       | 更新内容                          | 状态     |
|------------|----------------------------------|----------|
| 2025.7.29   | 🤖 添加了ROS2接口封装    | ✅ 已发布 |
|2025.7.14   | 🦾添加了多进程的基础组件数据采集支持  | ✅ 已发布  | 
|2025.6.15   | 🦾添加了完整的PIka遥操机械臂示例  | ✅ 已发布  | 
| 2025.5.27  | 🧪 添加完善测试样例, 便于程序调试与参考 | ✅ 已发布 |
| 2025.5.26  | 💻 客户端-服务器通讯实现，远程推理与本地控制 | ✅ 已发布 |
| 2025.5.22  | 🎭 通用Planner接入,通用IK逆解   | ✅ 已发布 |
| 2025.5.10  | 🎭 Hybrid VLA，OpenVLA-oft正在路上 | ❌测试中，待发布 |
| 2025.4.10  | 🎮 遥操设备接口封装，pika设备支持 | ✅ 已发布 |
| 2025.4.3   | 🤖 agilex底盘控制与ROS接口封装    | ✅ 已发布 |
| 2025.3.25  | 🦾 agliex机械臂支持               | ✅ 已发布 |
| 2025.3.10  | 🏗️ RealMan机械臂支持              | ✅ 已发布 |
| 2025.2.15  | 📊 通用数据采集流程完成           | ✅ 已发布 |
| 2024.12.1  | 🧠 VLA模型训练框架规范化          | ✅ 已发布 |

### 🛣️ 正在路上
- [✅] ⛓️‍💥 控制器与模型推理分开，支持远程部署与本地多脚本同步，解决环境兼容问题
- [✅] 🔢 curobo高效IK逆解器封装（需URDF支持）
- [ ] 🕹️ pika遥操控制任意机械臂示例
- [ ] 📦 更多控制器与传感器支持
- [ ] 🧩 更多机器人模型集成

### 🤖 设备支持情况

#### 🎛️ 控制器
**✅ 已实现**
| 机械臂         | 底盘               | 灵巧手       | 其他       |
|----------------|--------------------|--------------|------------|
| Agilex Piper   | Agilex Tracer2.0   | 🚧 开发中    | 📦 待补充  |
| RealMan 65B    | 📦 待补充          | 📦 待补充    | 📦 待补充  |
| daran aloha    | 📦 待补充          | 📦 待补充    | 📦 待补充  |

**🚧 准备支持**
| 机械臂    | 底盘       | 灵巧手     | 其他       |
|-----------|------------|------------|------------|
| JAKA      | 📦 待补充  | 📦 待补充  | 📦 待补充  |
| Franka    | 📦 待补充  | 📦 待补充  | 📦 待补充  |
| UR5e      | 📦 待补充  | 📦 待补充  | 📦 待补充  |

#### 📡 传感器
**✅ 已实现**
| 视觉传感器       | 触觉传感器    | 其他传感器  |
|------------------|---------------|-------------|
| RealSense D435   | 🚧 开发中     | 📦 待补充   |

**🚧 准备支持**
有需要新的传感器支持请提issue，也欢迎PR你的传感器配置！

## 配置基础环境
``` bash
conda create -n my_robot python==3.10
conda activate my_robot
git clone git@github.com:Tian-Nian/control_your_robot.git
cd control_your_robot
pip install -r requirements.txt

# (可选)编译最新版lerobot
cd ..
git clone https://github.com/huggingface/lerobot.git
cd lerobot
conda install ffmpeg
pip install --no-binary=av -e .

# (可选)下载你的机械臂需要的python安装包
pip install piper_sdk
pip install Robotic_Arm

# 对于模型训练而言, RDT与openpi有自己的环境配置要求
# 请使用对应模型环境, 然后执行
cd ~/control_your_robot/
pip install -r requirements.txt

# 松灵机械臂安装SDK请参考:https://github.com/agilexrobotics
# 睿尔曼机械臂SDK请参考:https://develop.realman-robotics.com/robot/summarize/
# 大然机械臂SDK请参考:
# 所有机械臂如果涉及到原声代码编译或链接，会统一放置到./third_party/目录下
```

## 表格形式
| 目录 | 说明 | 主要内容 |
|------|------|----------|
| **📂 controller** | 机器人控制器封装 | 机械臂、底盘等设备的控制 `class` |
| **📂 sensor** | 传感器封装 | 目前仅 `RealSense` 相机封装 |
| **📂 utils** | 工具函数库 | 辅助功能封装（如数学计算、日志等） |
| **📂 data** | 数据采集模块 | 数据记录、处理的 `class` |
| **📂 my_robot** | 机器人集成封装 | 完整机器人系统的组合 `class` |
| **📂 policy** | VLA 模型策略 | Vision-Language-Action 模型相关代码 |
| **📂 scripts** | 实例化脚本 | 主要运行入口、测试代码 |
| **📂 third_party** | 第三方依赖 | 需要编译的外部库 |
| **📂 planner** | 路径规划模块 | `curobo` 规划器封装 + 仿真机械臂代码 |
| **📂 example** | 示例代码 | 数据采集、模型部署等示例 |
| **📂 docs** | 文档索引 | 机器人相关文档链接 |

## 如何控制你的机器人?
本项目将机器人的部件分为两类:  
`controller`: 拥有控制功能的部件, 如机械臂, 灵巧手, 底盘等...  
`sensor`: 只用于获取信息的部件, 如视觉传感器, 触觉传感器   
如果你的操作部件和传感器已经在controller/sensor中了, 那你可以简单调用他, 否则的话你可以提issue, 我们会尽可能收集到该款机械臂, 并进行适配工作. 如果想自己实现的话, 可以参考developer_README.md, 欢迎各位完成适配后提交PR!

注意!   
 我们希望机械臂返回的的joint angle是弧度制, 即[-pi, pi], 夹爪是归一化的张合度[0,1], 末端6D坐标中x,y,z单位为米, rx,ry,rz单位为弧度制,操控的对应数据单位相同与获取数据的单位.

在保证你所需要的部件都已经被定义后, 请模仿my_robot中的几个示例, 组装你的机器人.
在完成组装后, 你可以编写示例, 来查看几个关键函数是否被正确实现:
``` python
if __name__=="__main__":
    import time
    robot = PiperSingle()
    # 采集测试
    data_list = []
    for i in range(100):
        print(i)
        data = robot.get()
        robot.collect(data)
        time.sleep(0.1)
    robot.finish()
    # 运动测试
    move_data = {
        "left_arm":{
        "qpos":[0.057, 0.0, 0.216, 0.0, 0.085, 0.0],
        "gripper":0.2,
        },
    }
    robot.move(move_data)
```

## 如何采集数据
在实现你的机器人的时候, 其应该自带一个`self.collection = CollectAny()`, 方便我们根据一些配置来保存采集的数据.  
在`example/collect`中已经提供了一个采集的示例, 你可以参考他来实现你的示例, 如果你已经通过了上面的基础数据采集测试,那么直接换成你的机器人就行啦.

如果你没有遥操设备的话，可以考虑使用'scripts/collect_omving_skpt.py'的代码进行轨迹采集与复现。  
不过有的设备可能不允许多个脚本建立通讯，那么你可以参考脚本中下面的注释进行多线程数据采集。

## 如何转化数据
目前已经提供了转化lerobot格式与RDT需要的hdf5格式的脚本啦, 在`scripts/`中, 对于单臂/双臂, 请看对应文件的要求, 将对应参数正确映射.    
### RDT需求的hdf5格式  
该转化是默认对单任务的, 所以请逐个任务进行转化, 例如转化:`datasets/task_1/episode_*.hdf5`,请执行:  
```bash
python scripts/convert2rdt_hdf5.py datasets/task_1
# 如果希望自定义保存位置
python scripts/convert2rdt_hdf5.py datasets/task_1 your_output_path
```
默认会保存在`datasets/RDT/`下.  

### openpi需求的lerobot格式
**注意**
转化openpi lerobot需要先转化为RDT版本hdf5，该转化不需要配置RDT环境！

转化为openpi需求的lerobot格式需要先将数据转化为RDT的hdf5格式, 因为现在最新版lerobot不兼容之前的版本的格式转化脚本了.  
在完成hdf5转化后, 请将对应task的instruction移动到对应task的文件夹下, 并重命名为`instructions.json`,如:  
`datasets/RDT/task_1/instructions.json`  
注意,如果是多任务, 请如图示:  
```
datasets/  
├── RDT  
|   ├── my_task
|   |       ├──task_1
|   |       |   ├── instructions.json  
|   |       |   ├── episode_0.hdf5  
|   |       |   ├── episode_1.hdf5  
|   |       |   ├── ...  
|   |       |
│   |       ├── task_2
│   |       ├── ...
```
  
执行:
``` bash
python scripts/convert2openpi.py --raw_dir datasets/my_task --repo_id your_repo_id
```

### lerobot2.0版本
我们支持将您的数据转化为最新版lerobotdataset格式,并且支持多任务数据集生成!
注意, 如果是多任务, multi_task中数据的摆放也应该类似转化为openpi的数据拜访格式, 不过这里需要的是原始的采集数据, 不是转化为RDT的hdf5数据.
```
datasets/  
├── my_task
|       ├──task_1
|       |   ├── config.json  
|       |   ├── episode_0.hdf5  
|       |   ├── episode_1.hdf5  
|       |   ├── ...  
|       |
|       ├── task_2
|       ├── ...
```
```bash
# 单任务
python scripts/convert2lerobot.py  datasets/task_1 repo_id 
# 多任务
python scripts/convert2lerobot.py  datasets/my_task repo_id True
```

### TFDS数据格式
**注意**
与lerobot相同,您需要先转化为RDT支持的hdf5格式!  

转化TFDS格式比较特殊,可以参考`./policy/rlds_dataset_builder`中的一些转化示例.
需要如下格式:
```bash
├── ${my_dataset_example}
|   ├── ${my_dataset_example}_dataset_builder
|   |   ├── lass ${my_dataset_example}
|   ├── CITATION.bib
|   ├── __init__.py   
```
其中`__init__.py`和`CITATION.bib`是直接复制粘贴即可.

## 如何训练模型
目前项目的`policy/`已经提供了`openpi`与`RDT`官方的训练脚本, 并进行了一些修改, 便于进行训练, 里面都提供了详细的操作, 请参考里面的需求进行配置.
### RDT
需要将生成好的`datasets/RDT/my_task`手动移动到`polcy/RDT/training_data/`下!
不能使用`ln -s` 指令, 由于使用`os.walk`,会找不到软连接的文件夹内容, 导致报错.

注意, 转化为RDT需要的hdf5后, 还需要为每个task配置上编码后的语言指令, 在配置完RDT环境,并下载完所有模型后, 执行:  
```bash
cd policy/RDT
python scripts/encode_lang_batch_once.py task_name output_dir gpu_id
# 如python scripts/encode_lang_batch_once.py task_1 ./training_data/task_1 0
```

### openpi
注意openpi要求的不是lerobot2.0版本的数据, 使用lerobot2.0版本会导致数据读入报错!
生成完openpi的数据后, 按照`policy/openpi`中的`README.md`执行就行啦.注意区分单双臂.

## 如何部署模型
已经在`example/depoly`中提供了对应的部署脚本, 不过对于不同机械臂需要有一些修改.

如果本地设备不支持推理, 本项目也支持了优秀的client&sever机制,启动`scripts/client.py`与`scripts/server.py`, 修改ip与port,将里面的robot与model替换为您的robot与model即可.

你也可以通过设置`deploy.sh`中的参数, 来实现部署, 对应参数已给出示例, 模仿修改即可.
