# 第一课：仿真如何从这一刻走到下一刻

目标：能指出模型、状态、控制输入分别在哪里，并解释一次 `mj_step` 做了什么。

## 先看我们现有的模型

`models/pendulum.xml` 描述了地面、支架、转轴、杆和球。转轴只能绕 Y 轴旋转，因此只有一个自由度。
原文件没有 `<actuator>`，即没有电机。此前窗口中的摆动由重力引起；它还没有控制器，更没有强化学习。

本课在内存中复制模型，添加下面的电机，原文件保持原样：

```xml
<actuator>
  <motor name="hinge_motor" joint="hinge" gear="1"
         ctrllimited="true" ctrlrange="-2 2"/>
</actuator>
```

本课副本还将支点的装饰球设为不参与碰撞（contype=0、conaffinity=0）。它与杆在转轴处重叠，若作为碰撞体会引入额外接触力。三组实验都使用这一相同副本。

这个模型采用米、千克、秒，hinge 配合 gear=1 的 motor，因此本课 `ctrl[0]=1` 对应电机施加 +1 N·m 的关节力矩。
关节还受到重力、阻尼等作用，电机力矩不等于所有力矩之和。其他 actuator 类型的 ctrl 可能表示目标位置或目标速度。

## 四个对象

| 代码 | 本课中的含义 |
|---|---|
| `model = mujoco.MjModel.from_xml_string(xml)` | 编译模型：形状、质量、关节、电机、重力和步长等 |
| `data = mujoco.MjData(model)` | 创建这一次仿真的状态与计算缓存 |
| `data.qpos[0]`、`data.qvel[0]` | 当前关节角 q（rad）和角速度（rad/s） |
| `data.ctrl[0]` | 电机控制输入；在本课配置中对应力矩（N·m） |

本课 `model.nq=1`、`model.nv=1`、`model.nu=1`，分别表示位置坐标数、速度坐标数、控制输入数。
原始无电机模型的 `nu=0`，不能给它写 `data.ctrl[0]`。

你学过状态空间：对这个单摆，可写成 x=[q, qdot]，再用 x_next=F(x,u) 描述一步离散演化。
MuJoCo 根据完整物理模型计算 F；以后换成小鸭子，关节和状态变多，但这条逻辑仍然适用。
复杂机器人的 qpos 与 qvel 维数不一定相同，例如自由关节姿态常用四元数表示。

## 一次循环

```python
data.ctrl[0] = torque
mujoco.mj_step(model, data)
q = float(data.qpos[0])
qdot = float(data.qvel[0])
```

`mj_step` 计算动力学并积分一次。XML 中的 timestep=0.002，因此 1000 次调用对应 2 秒仿真时间。
程序实际花了多少秒执行、窗口每秒显示多少帧，是另外两个量。`sleep` 可以改变播放节奏，不会直接改变物理步长。

脚本中另有 `mj_forward(model,data)`：它刷新摆球世界坐标等派生量，便于画面与当前 q 对齐；它不推进时间。

## 一个容易混淆的零点

q=0 是 XML 定义的参考姿态。我们的杆在这个姿态下已经倾斜：从支点指向 (0.55,0,-0.75)。
所以 q=0 不表示竖直。竖直向下约对应 q=0.632749 rad（36.254°）。不要把画面上相对竖直的角度直接当作 qpos。

## 三组实验：只改变重力或输入

三组都从 q=0、qdot=0 出发，具有相同质量、几何、阻尼和步长。

1. 无重力、力矩为 0：静止。
2. 有重力、力矩为 0：开始摆动。这说明控制输入为 0，并不代表系统一定不动。
3. 无重力、力矩为 +1 N·m：电机驱动它运动。因有阻尼，不能简单断言严格匀加速。

这些是物理仿真与开环输入实验，还没有观测设计、奖励、策略或训练。

## 自己运行

在 Ubuntu 终端中：

```bash
cd ~/projects/duck-rl
source .venv/bin/activate
python lessons/01_simulation_step.py
```

每组应输出 `"checks": "passed"`。结果在 `artifacts/lesson01/`：

- `pendulum_motor.xml`：本课添加电机后的模型副本。
- `checks.json`：实际运行结果及物理一致性检查。
- `traces.json`：三个实验每个时间步的状态和摆球位置，用于同步比较。

脚本检查零输入静止、重力初始运动方向、电机力矩、刚性圆轨迹、时间推进及有限数值。
三组分别新建 MjData，避免上一组的速度影响下一组。

## 检查自己是否理解

如果想仿真 1 秒而保持 timestep=0.002，应该调用多少次 `mj_step`？
如果把重力保留而令 ctrl=0，为什么摆仍可能运动？

## 官方依据

- Python API：https://mujoco.readthedocs.io/en/stable/python.html
- 仿真循环：https://mujoco.readthedocs.io/en/stable/programming/simulation.html
- motor 定义：https://mujoco.readthedocs.io/en/stable/XMLreference.html#actuator-motor
- joint 定义：https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-joint
