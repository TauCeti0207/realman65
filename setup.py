'''
Descripttion: 
Author: tauceti0207
version: 
Date: 2025-11-02 15:02:46
LastEditors: tauceti0207
LastEditTime: 2025-11-03 13:48:46
'''
from setuptools import setup, find_packages
import os

# 读取项目依赖（如果有 requirements.txt）


def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if os.path.exists(req_path):
        with open(req_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return []


setup(
    # 项目基本信息
    name="realman65",  # 包名称（可自定义，小写无空格）
    version="0.1.0",  # 版本号
    author="tauceti0207",  # 作者
    description="RM65 机械臂控制接口，支持关节角与末端位姿控制",  # 项目描述
    long_description=open(
        "README.md", "r", encoding="utf-8").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    url="",  # 项目仓库地址（可选）
    
    # 新增：明确指定核心包的路径（关键！）
    # 意思是：所有包（如 realman65）都从当前目录（外层 realman65）查找
    package_dir={"realman65": "realman65"},
    
    # 核心：指定需要打包的模块和包
    packages=find_packages(
        where="realman65",  # 查找的根目录：内层 realman65
        include=["realman65", "realman65.*"],  # 包含内层 realman65 及其子模块
        exclude=[]
    ),

    # 非 .py 文件的资源（如果需要，如配置文件、数据等）
    package_data={
        # 示例：包含所有包下的 .json 和 .xml 文件（根据实际需求添加）
        # "*": ["*.json", "*.xml"],
        "": ["*.py", "*.pyc"],  # 包含所有包下的.py文件
        # "realman65.third_party.Realman_IK": ["*.py", "*.pyc"],  # 包含IK库的所有Python文件
    },

    # 依赖库（必须安装的第三方库）
    install_requires=read_requirements() or [
        "numpy==1.26.4",  # 适配你的 NumPy 版本（避免 np.mat 问题）
    ],

    # 支持的 Python 版本
    python_requires=">=3.8",

    # 入口脚本（如果需要命令行调用，可选）
    entry_points={
        # 示例：添加命令行入口（如 `realman-control` 直接运行接口）
        # "console_scripts": [
        #     "realman-control = my_robot.realman_65_interface:main",
        # ],
    },

    # 标识包使用 namespace packages（可选，复杂项目用）
    zip_safe=False,
)
