# 第二课：从固定力矩到 PD 闭环控制

目标：沿用第一课的单摆电机模型，关闭重力，从 q=0、dq=0 出发，让关节停在 q=0.5 rad（约 28.65°）。q 是相对 XML 初始姿态的转角；q=0 仍是原来那幅倾斜姿态。

## 在 Ubuntu 中运行

```bash
cd ~/projects/duck-rl
source .venv/bin/activate
python lessons/02_pd_control.py
```

模型来自第一课保存的 artifacts/lesson01/pendulum_motor.xml。如果提示模型不存在，先运行第一课脚本。

## 在 MuJoCo 三维窗口看控制过程

```bash
python lessons/02_pd_control.py --view
```

默认从 P 控制开始，窗口打开后先暂停，按空格开始。点击三维窗口，让它接收键盘输入。

- `1`：切到 P（Kp=4、Kd=0），从 q=0、dq=0 重新开始。
- `2`：切到 PD（Kp=4、Kd=3），从相同初态重新开始。
- `R`：用当前控制器从头开始。
- 空格：暂停/继续。每次实验运行到 8 个模拟秒后保留终态，按 R 可重看。

黄色球和青色杆是实际机构，绿色球形和细杆标记目标姿态。绿色标记只是 viewer.user_scn 中的显示几何，没有质量和碰撞，不会影响物理。窗口左上角显示控制器、时间、角度、角速度和电机输入。

也可用 `python lessons/02_pd_control.py --view --controller PD` 从 PD 开始。

新加的 show_control() 使用官方 mujoco.viewer.launch_passive。真实控制公式与 run_case() 一致：每 2 毫秒读一次状态、重新算力矩、调用 mj_step；约每 8 步通过 viewer.sync() 更新一次画面。mj_forward 只刷新派生量，sync 只负责显示同步，time.sleep 只调节播放速度。窗口版会持续在线计算控制和物理，并非读取 CSV 播放。打开窗口不会覆盖原有数值实验结果。

本课暂停/切换/重启使用上面的快捷键；查看器默认隐藏左右面板，以免与脚本控制混淆。

## 先读闭环的四行

```python
error = TARGET - q
requested_torque = KP * error - kd * dq
data.ctrl[0] = np.clip(requested_torque, lower, upper)
mujoco.mj_step(model, data)
```

与第一课固定 ctrl=1 的区别：每一个物理步都重新读取当前 q、dq，再决定这个步施加多少力矩。

目标速度是零，因此一般形式 Kp*(q_target-q) + Kd*(dq_target-dq) 简化成 Kp*(q_target-q) - Kd*dq。P 项根据位置误差拉向目标，D 项根据速度提供制动。接近目标时，即使误差仍为正，也可能需要先施加负力矩减速。

本课 motor 的 gear=1，因此 ctrl 的数值对应关节电机力矩，单位 N·m；输入限制为 [-2, 2]。Kp 的单位是 N·m/rad，Kd 是 N·m·s/rad。

## 两组实验

| 实验 | Kp | Kd | 要观察的现象 |
| --- | --- | --- | --- |
| P | 4 | 0 | 越过目标，持续来回摆动；原有关节阻尼仍在缓慢耗能 |
| PD | 4 | 3 | 根据角速度主动制动，更快接近目标并停稳 |

关闭重力后，这个单自由度模型满足 I*ddq = torque - 0.04*dq。未限幅时，代入 PD 就是 I*ddq = 4*(0.5-q) - (3+0.04)*dq。

XML 的 damping=0.04 是机构被动阻尼；代码的 -Kd*dq 是控制器让电机主动施加的制动力矩。物理作用可叠加，但后者受到电机力矩限制。

比较时两组都从独立的相同初态开始，运行 8 个模拟秒。不要只看最后一个位置：经过目标不代表停稳，要同时检查最后一整秒的位置误差和角速度。

## 读结果

终端打印两组结果；artifacts/lesson02/p_only.csv 和 pd.csv 保存轨迹。每行依次是时间、角度、角速度、位置误差、请求力矩和限幅后的实际输入，全部对应调用 mj_step 之前的同一时刻。最后一个步的终态单独在 checks.json 中报告。

可以重点查 PD 轨迹中 requested_torque_Nm 首次变负的行：当时角度是否已超过目标？这能检验你是否理解提前制动。

关闭重力是本课实验条件。后续加回重力时，目标处可能需要持续力矩，纯 PD 可能出现稳态误差，还要考虑电机是否有足够力矩；不能直接把这一课的结论照搬。

## 与小鸭子和强化学习的关系

MuJoCo 根据模型和输入推进物理状态。这里决定输入的是手写 PD 控制器；在强化学习任务中，输入会由策略决定，或者由策略给出关节目标，再由 PD 等底层控制器转换成力矩。具体采用哪一层动作，要看任务设计。
