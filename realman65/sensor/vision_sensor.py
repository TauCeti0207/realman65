'''
Descripttion: 
Author: tauceti0207
version: 
Date: 2025-11-03 10:58:06
LastEditors: tauceti0207
LastEditTime: 2025-11-03 12:01:39
'''
from realman65.sensor.sensor import Sensor
import sys
sys.path.append("./")


class VisionSensor(Sensor):
    def __init__(self):
        super().__init__()
        self.name = "vision_sensor"
        self.type = "vision_sensor"
        self.collect_info = None

    def get_information(self):
        image_info = {}
        image = self.get_image()
        if "color" in self.collect_info:
            image_info["color"] = image["color"]
        if "depth" in self.collect_info:
            image_info["depth"] = image["depth"]
        if "point_cloud" in self.collect_info:
            image_info["point_cloud"] = image["point_cloud"]

        return image_info
