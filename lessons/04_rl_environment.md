# 第 4 课：把单摆封装成强化学习环境

今天开始进入 RL 的组织方式。MuJoCo、模型、重力、电机仍沿用前面的内容；新增 Gymnasium 的标准 reset/step 接口。代码分两份：pendulum_env.py 是环境；04_rl_environment.py 是检查和演示。

## 运行

在 Ubuntu 中：

```bash
cd ~/projects/duck-rl
source .venv/bin/activate
python lessons/04_rl_environment.py
python lessons/04_rl_environment.py --view
```

第一条运行官方接口检查、边界检查，以及随机动作/已知控制器两个回合。轨迹与结果在 artifacts/lesson04。第二条打开官方 MuJoCo 窗口：空格开始/暂停，1 从头执行随机动作，2 从头执行 PD+重力补偿，R 重来。两者从同一个 q=0、dq=0 开始，窗口初始暂停。

## 先读环境的 reset() 和 step()

```python
obs, info = env.reset(seed=42)
while True:
    action = policy(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

这段循环以后可以直接换成学习算法提供的策略；环境不需要知道动作来自随机数、PD 还是神经网络。

| 名词 | 本课的具体含义 |
| --- | --- |
| observation / obs，观察 | [目标角度−当前角度，当前角速度]，float32，长度 2 |
| action，动作 | 长度 1 的数组，数值在 [-1,1]；力矩 u=2*action，单位 N·m |
| reward，奖励 | 执行动作后，按位置误差、角速度、输入力矩计算的分数 |
| episode，回合 | 从一次 reset 开始，直到任务失败或时间上限 |
| policy，策略 | 输入观察、输出动作的规则；本课只有随机策略和已知控制器 |

目标固定为 0.5 rad，所以由 error 可以恢复 q；给策略的是这两个数，不是全部 mjData，也没有直接提供 qfrc_bias。环境里面没有暗中加入 PD 或重力补偿。演示中的 PD+补偿控制器在环境外面，且使用已知模型的 bias，作为对照。

## reset：从哪里开始

super().reset(seed=seed) 初始化 Gymnasium 的随机数发生器。mj_resetData 复位模拟时间和动态状态，再设置 q、dq；返回初始 obs、info。默认初态有很小随机扰动，用同一 seed 可以复现。演示通过 options 显式指定 q=0、dq=0。

## step：一个动作后发生什么

1. 检查并限幅 action，把它映射成电机力矩。
2. 同一力矩保持 10 个物理步，每步 mj_step 推进 0.002 秒。
3. 更新当前观察，在动作之后的状态上计算奖励。
4. 判断是否结束，返回五个值。

因此 **一个 env.step 是 0.02 秒，一个 mj_step 是 0.002 秒**。策略每秒决策 50 次，物理每秒积分 500 次。本课外部 PD 也每 0.02 秒计算一次动作，与前一课每 0.002 秒反馈略有不同；数值轨迹不必逐点完全一样。

## 奖励由我们定义，不是 MuJoCo 自动给的

\[
r=1-4e^2-0.05\dot q^2-0.001u^2-10\,\mathbf{1}_{\text{失败}}
\]

较小误差、较小速度和较小力矩通常对应更高分。各项权重是本教学任务的设计选择，不是物理定律，也不是小鸭子的最终奖励。奖励只用来评价动作，不会作为力施加给单摆，也不会自行更新控制器。

一个回合的 return 是所有 step 奖励的累加；它与“某一步的 reward”不同。下一课 PPO 才会用采样数据来更新策略参数。当前 PD 得分较高说明这组对照在这个评分标准下表现更好，不表示随机策略已经学会了。

## 两种结束信号

- terminated=True：本任务定义的失败，例如绝对误差超过 1.2 rad 或速度超过 8 rad/s。
- truncated=True：没有先失败，但达到了人为设定的 400 步上限，即 8 个模拟秒。

到达目标不会立即结束：任务要求继续保持。回合结束后要先 reset，再开始下一回合。info 保存时间、真实角度、输入力矩和分项代价，便于检查，但不直接作为策略观察。

## 读代码顺序

1. pendulum_env.py：__init__ 的 observation_space/action_space。
2. reset()：复位、随机初态和 seed。
3. step()：动作、物理子步、奖励、结束信号。
4. 04_rl_environment.py：choose_action() 和 rollout()，看策略怎样与环境交互。
5. render()/show_demo() 最后再看，它们只用于显示与按键演示。

## 本课理解检查

如果 action=[0.5]，电机输入是多少 N·m？执行一次 env.step 后经过多少模拟时间？奖励更高会不会让随机策略自动变聪明？

官方接口参考：[Gymnasium Env](https://gymnasium.farama.org/api/env/)、[创建自定义环境](https://gymnasium.farama.org/introduction/create_custom_env/)。

## 接到小鸭子的课程安排

第 4 课：环境、观察、动作、奖励、回合。本课完成后进入第 5 课：策略网络、PPO 采样/更新、训练与评估，先用这个小环境跑通。第 6 课读取小鸭子模型与任务环境，把相同概念映射到多关节机器人；第 7 课开始基线训练、保存与回放。预计再 3～4 课可开始小鸭子训练，走路质量和完整调试能力需要继续实验；课程次数不是训练收敛的保证。

当前 Open Duck Playground 使用 JAX/MuJoCo Playground 的环境实现，其接口写法与本课 Gymnasium 不完全相同，但同样有 reset、step、观察、动作、奖励和回合。当前小鸭子任务的 ctrl_dt=0.02、sim_dt=0.002，也采用控制与物理两种时间尺度。第 6 课会逐项对照真实仓库，而不是把本课文件直接当成小鸭子环境。[小鸭子任务源码](https://github.com/apirrone/Open_Duck_Playground/blob/main/playground/open_duck_mini_v2/joystick.py)
