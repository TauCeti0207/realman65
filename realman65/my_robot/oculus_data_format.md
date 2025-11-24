# Oculus Quest Reader 数据格式说明

## 概览

Oculus Quest Reader 的 `get_transformations_and_buttons()` 方法返回一个包含两个字典的元组：

```python
(transformations, buttons) = oculus_reader.get_transformations_and_buttons()
```

## 1. 变换矩阵字典 (Transformations)

### 数据结构
```python
{
    'l': numpy.array([[4x4 matrix]]),  # 左手控制器变换矩阵
    'r': numpy.array([[4x4 matrix]])   # 右手控制器变换矩阵
}
```

### 4×4变换矩阵格式
```
[[  R11   R12   R13   Tx  ]
 [  R21   R22   R23   Ty  ]
 [  R31   R32   R33   Tz  ]
 [  0.0   0.0   0.0   1.0 ]]
```

- **R11-R33**: 3×3旋转矩阵，表示控制器的朝向
- **Tx, Ty, Tz**: 位置向量，表示控制器在3D空间中的坐标
- **最后一行**: 齐次坐标，固定为 [0, 0, 0, 1]

### 示例
```python
{
    'l': array([[ 0.875052  ,  0.331145  ,  0.353026  ,  0.324097  ],
                [ 0.127164  ,  0.546447  , -0.827783  ,  0.00842464],
                [-0.467026  ,  0.769246  ,  0.43606   , -0.479309  ],
                [ 0.        ,  0.        ,  0.        ,  1.        ]]),
    'r': array([[ 0.990979 , -0.0969746,  0.092504 ,  0.0803483],
                [ 0.111676 ,  0.215928 , -0.970002 , -0.182173 ],
                [ 0.0740913,  0.971582 ,  0.22481  , -0.257265 ],
                [ 0.       ,  0.       ,  0.       ,  1.       ]])
}
```

## 2. 按钮状态字典 (Buttons)

### 数据结构
```python
{
    # 数字按钮 (布尔值)
    'A': bool,      # A按钮
    'B': bool,      # B按钮  
    'X': bool,      # X按钮
    'Y': bool,      # Y按钮
    
    # 摇杆和握持按钮 (布尔值)
    'RThU': bool,   # 右摇杆按下
    'LThU': bool,   # 左摇杆按下
    'RJ': bool,     # 右摇杆点击
    'LJ': bool,     # 左摇杆点击
    'RG': bool,     # 右握持按钮
    'LG': bool,     # 左握持按钮
    'RTr': bool,    # 右扳机按钮
    'LTr': bool,    # 左扳机按钮
    
    # 模拟输入 (数值)
    'leftJS': (x, y),        # 左摇杆坐标 (-1.0 到 1.0)
    'rightJS': (x, y),       # 右摇杆坐标 (-1.0 到 1.0)
    'leftTrig': (pressure,), # 左扳机压力 (0.0 到 1.0)
    'rightTrig': (pressure,),# 右扳机压力 (0.0 到 1.0)
    'leftGrip': (pressure,), # 左握持压力 (0.0 到 1.0)
    'rightGrip': (pressure,) # 右握持压力 (0.0 到 1.0)
}
```

### 按钮映射说明

| 键名 | 全称 | 描述 |
|------|------|------|
| A, B, X, Y | Face Buttons | 主要操作按钮 |
| RThU/LThU | Right/Left Thumb Up | 摇杆按下 |
| RJ/LJ | Right/Left Joystick | 摇杆点击 |
| RG/LG | Right/Left Grip | 侧面握持按钮 |
| RTr/LTr | Right/Left Trigger | 食指扳机 |

### 示例
```python
{
    'A': False, 'B': False, 'X': False, 'Y': False,
    'RThU': False, 'LThU': True,  # 左摇杆被按下
    'RJ': False, 'LJ': False,
    'RG': False, 'LG': False, 
    'RTr': False, 'LTr': False,
    'leftJS': (0.0, 0.0),        # 摇杆居中
    'rightJS': (0.0, 0.0),       # 摇杆居中
    'leftTrig': (0.0,),          # 扳机未按压
    'rightTrig': (0.0,),         # 扳机未按压
    'leftGrip': (0.0,),          # 未握持
    'rightGrip': (0.0,)          # 未握持
}
```

## 3. 完整输出示例

```python
# 典型的输出格式
(
    # 变换矩阵字典
    {
        'l': array([[ 0.875052  ,  0.331145  ,  0.353026  ,  0.324097  ],
                    [ 0.127164  ,  0.546447  , -0.827783  ,  0.00842464],
                    [-0.467026  ,  0.769246  ,  0.43606   , -0.479309  ],
                    [ 0.        ,  0.        ,  0.        ,  1.        ]]),
        'r': array([[ 0.990979 , -0.0969746,  0.092504 ,  0.0803483],
                    [ 0.111676 ,  0.215928 , -0.970002 , -0.182173 ],
                    [ 0.0740913,  0.971582 ,  0.22481  , -0.257265 ],
                    [ 0.       ,  0.       ,  0.       ,  1.       ]])
    },
    # 按钮状态字典
    {
        'A': False, 'B': False, 'RThU': False, 'RJ': False, 
        'RG': False, 'RTr': False, 'X': False, 'Y': False, 
        'LThU': True, 'LJ': False, 'LG': False, 'LTr': False, 
        'leftJS': (0.0, 0.0), 'leftTrig': (0.0,), 'leftGrip': (0.0,), 
        'rightJS': (0.0, 0.0), 'rightTrig': (0.0,), 'rightGrip': (0.0,)
    }
)
```

## 4. 使用示例

### 基本数据获取
```python
from oculus_reader import OculusReader

# 创建读取器
oculus_reader = OculusReader()

# 获取数据
transforms, buttons = oculus_reader.get_transformations_and_buttons()

# 获取左手控制器位置
if 'l' in transforms:
    left_position = transforms['l'][:3, 3]  # [x, y, z]
    left_rotation = transforms['l'][:3, :3] # 3x3旋转矩阵

# 检查按钮状态
if buttons['A']:
    print("A按钮被按下")

# 检查摇杆输入
left_stick_x, left_stick_y = buttons['leftJS']
if abs(left_stick_x) > 0.1 or abs(left_stick_y) > 0.1:
    print(f"左摇杆移动: ({left_stick_x}, {left_stick_y})")
```

### 实际应用场景
```python
def process_controller_data(transforms, buttons):
    """处理控制器数据的示例函数"""
    
    # 机器人臂控制
    if 'r' in transforms:
        right_controller = transforms['r']
        # 提取位置和旋转用于机器人末端执行器控制
        position = right_controller[:3, 3]
        rotation = right_controller[:3, :3]
        
    # 抓取控制
    grip_pressure = buttons['rightGrip'][0]
    if grip_pressure > 0.5:
        # 触发抓取动作
        print("执行抓取")
        
    # 触发器控制
    trigger_pressure = buttons['rightTrig'][0]
    if trigger_pressure > 0.8:
        # 高精度操作
        print("精确控制模式")
```

## 5. 注意事项

1. **坐标系**: 变换矩阵使用右手坐标系
2. **单位**: 位置单位通常为米(m)
3. **频率**: 数据更新频率约为60-90Hz
4. **空数据**: 当控制器未连接时，对应的键可能不存在
5. **精度**: 位置精度约为毫米级，角度精度约为度级

## 6. 故障排除

- **空字典 `{}`**: 控制器未连接或APK未正常运行
- **缺少键**: 特定控制器可能未被检测到
- **数据不更新**: 检查Oculus Quest连接状态和权限设置