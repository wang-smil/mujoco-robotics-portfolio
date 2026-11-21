# 第三课：停稳不等于到达目标——重力、静差和补偿

沿用前两课：同一模型、q=0 和 dq=0 初态、目标 q=0.5 rad、Kp=4、Kd=3、电机力矩限制 ±2 N·m。本课两组都恢复重力 [0,0,-9.81]。

## 先看三维过程

在 Ubuntu 终端中运行：

```bash
cd ~/projects/duck-rl
source .venv/bin/activate
python lessons/03_gravity_compensation.py --view
```

点击三维窗口，让它接收按键：空格开始/暂停，1 从头运行 PD，2 从同一初态运行 PD+重力补偿，R 重看当前组。每组运行 8 个模拟秒后保留终态。绿色显示目标，黄色为实际球。注意同时看 q、error 和 dq：球停住了，不代表误差为零。

不加 --view 则运行两组数值实验，将逐步轨迹和检查结果保存到 artifacts/lesson03。CSV 每行的状态、各项力矩和输入都对应 mj_step 之前的同一时刻。

## 今天新增的核心代码

先看 make_model()、control_torque() 和 run_case()。show_control() 大部分沿用上一课的显示与按键逻辑，数值实验和查看器共用 control_torque()。

```python
model.opt.gravity[:] = [0, 0, -9.81]
mujoco.mj_forward(model, data)
pd = KP * (TARGET - data.qpos[0]) - KD * data.qvel[0]
compensation = data.qfrc_bias[0] if compensate else 0.0
torque = np.clip(pd + compensation, lower, upper)
```

这不是关闭重力：两组 model.opt.gravity 完全相同，只改变电机输入。

## 为什么 PD 会停在错误的位置

对这个固定转轴、常转动惯量的单摆，动力学可写成：

\[
I\ddot q+b\dot q+G(q)=u
\]

G(q) 是把重力项放在方程左侧时的符号约定；重力实际对关节施加的力矩是 -G(q)。在某个姿态静止所需的电机力矩则是 G(q)。

纯 PD 为 u=Kp*(q_target-q)-Kd*dq。若已经静止，dq=ddq=0，因此：

\[
K_p(q_{target}-q)=G(q)
\]

目标 q=0.5 处的 G(q) 不为零，所以“误差为零、速度为零”时，纯 PD 输出为零，无法平衡重力。它最终需要留下非零位置误差，让 P 项提供持续电机力矩。D 项在静止时为零，不能承担这部分静态力矩。

这叫稳态误差（静差）。本课中纯 PD 会停在约 q=0.5884 rad，误差 target-q 约 -0.0884 rad；这是本参数下的稳定平衡点，不是发散。

## 加入重力补偿后

\[
u=K_p(q_{target}-q)-K_d\dot q+G(q)
\]

在模型准确且输入未饱和的条件下，代回动力学会消去 G(q)：

\[
I\ddot q+(b+K_d)\dot q+K_p(q-q_{target})=0
\]

到达目标并静止时，PD 部分接近零，但电机仍输出约 -1.0567 N·m 来平衡重力。保持一个姿态，不等于电机不出力。

## qfrc_bias 的含义和符号

MuJoCo 的 qfrc_bias 一般包含重力、科氏和离心项，并不普遍等于“重力”。本课只有一个固定轴转动自由度，I 不随 q 变化，没有科氏项，所以可把 qfrc_bias[0] 用作上述 G(q)。先调用 mj_forward，使该量对应当前状态，而不是上一步的缓存。

按 MuJoCo 的动力学约定，补偿用的是 **加上** qfrc_bias；不要看到“抵消重力”就机械地加负号。本课在目标位置 qfrc_bias 本身是负数。

官方参考：[动力学与 bias](https://mujoco.readthedocs.io/en/stable/computation/index.html#general-framework)。

## 为什么还要限幅

目标处需要约 -1.0567 N·m，电机 ±2 N·m 可以提供。但初始 q=0 处 G(q) 约 -4.7211 N·m，加上 PD 的 +2 后请求约 -2.7211，超出范围，会裁剪为 -2。因此初始阶段不会完全按“消去重力后的方程”运动；目标可维持，也不代表沿途每个控制请求都能满足。

补偿依赖正确的模型。真实机器人上质量、重心等有误差时，补偿也可能有误差。不要据此把“PD+重力补偿”当作任何模型、任何目标都能精确跟踪的保证。

## 看完后回答这一题

当第二组最终 q≈0.5、dq≈0 时，PD 项接近零，电机为什么仍有约 -1.06 N·m 的输出？如果此时把这个输出也改成零，会怎样？

下一课开始把“读取状态—输出动作—推进仿真”封装成 reset/step，并加入奖励和回合，进入强化学习环境的组织方式。
