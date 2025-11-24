import numpy as np
import time

class HeadsetData():
    def __init__(self):
        self.timestamp = time.monotonic_ns()
        self.poses = {
            'head': np.array([0, 0, 0, 0, 0, 0, 1]),
            'left': np.array([0, 0, 0, 0, 0, 0, 1]),
            'right': np.array([0, 0, 0, 0, 0, 0, 1]),
        }
        self._controller = {
            'thumbstick_x': 0.0,
            'thumbstick_y': 0.0,
            'index_trigger': 0.0,
            'hand_trigger': 0.0,
            'button_one': False,
            'button_two': False,
            'button_thumbstick': False
        }
        self.controllers = {
            'left': self._controller,
            'right': self._controller
        }