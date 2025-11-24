'''
Descripttion: 
Author: tauceti0207
version: 
Date: 2025-11-03 10:58:06
LastEditors: tauceti0207
LastEditTime: 2025-11-03 12:03:35
'''
from typing import Dict, Any
import numpy as np
from realman65.sensor.sensor import Sensor
import sys
sys.path.append("./")


class TeleoperationSensor(Sensor):
    def __init__(self):
        super().__init__()
        self.name = "teleoperation_sensor"
        self.sensor = None

    def get_information(self):
        sensor_info = {}
        state = self.get_state()
        if "end_pose" in self.collect_info:
            sensor_info["end_pose"] = state["end_pose"]
        if "velocity" in self.collect_info:
            sensor_info["velocity"] = state["velocity"]
        if "gripper" in self.collect_info:
            sensor_info["gripper"] = state["gripper"]
        if "extra" in self.collect_info:
            sensor_info["extra"] = state["extra"]
        if "raw_pose" in self.collect_info:
            sensor_info["raw_pose"] = state.get("raw_pose")
        return sensor_info
