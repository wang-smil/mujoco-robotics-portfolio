# MuJoCo 机器人控制与双足运动仿真

项目包含两个已完成的仿真阶段。单摆部分实现动力学建模、PD 控制、重力补偿和 PPO 训练。Open Duck Mini v2 部分实现并行训练、运动指令跟踪、多种子评测及交互控制，覆盖站立、前进、原地转向、行进弧线和低速后退。

[28 秒弧线与后退对照回放](media/moving_reverse_comparison.mp4) · [36 秒完整手柄路线](media/assisted_switch_comparison.mp4) · [20 秒头颈限速对照](media/neck-ablation.mp4)

[![同一个种子的双画面对照回放预览](media/moving_reverse_comparison.png)](media/moving_reverse_comparison.mp4)

## 工程成果

| 阶段 | 实现与验证范围 | 实验结果 |
|---|---|---|
| 01 · 单摆 | MJCF 模型 → P/PD → 重力补偿 → Gymnasium → SB3 PPO | 目标 `0.5 rad` 时，重力补偿将纯 PD 的稳态误差从 `0.0884 rad` 降到约 `3.4×10⁻⁷ rad`。单摆 PPO 采集 `200,704` 步，最佳模型在 10 个测试种子上的平均回报 `375.90`。 |
| 02 · 双足 | 环境与课程设计 → MJX/Brax PPO → 示范监督 → 原生 MuJoCo 评测与手柄 | 头颈方案在 80 局基础命令中均达标，头颈关节速度 RMS 从 `1.2473` 降到 `0.3481 rad/s`。同种子的直走/后退互换从 `1/20` 提高到 `15/20` 达标，新版有 `18/20` 局跑满。 |

**后退过渡对照。** 第 22 轮 PPO 的同一检查点，后退阶跃命令 `−0.05 m/s` 的完成率为 `8/8`，缓变到同一目标时为 `0/8`。交互控制器结合 Actor、姿态反馈与相位切入，标准路线实际平均后退速度为 `−0.0508 m/s`。复杂换向的偏航误差和跌倒次数见 [分路线评测](docs/02-duck.md#最终同种子对照)。

## 阅读和运行

- [阶段 01 · 单摆与 PPO](docs/01-pendulum.md)
- [阶段 02 · Open Duck Mini 工程和训练](docs/02-duck.md)
- [实验时间线与逐轮配置索引](docs/03-timeline.md)
- [安装、运行与复现边界](docs/reproduce.md)
- [上游来源与贡献范围](docs/credits.md)

源码入口为 [`lessons/01_simulation_step.py`](lessons/01_simulation_step.py)、[`lessons/05_ppo.py`](lessons/05_ppo.py) 和 [`lessons/07_duck_ppo.py`](lessons/07_duck_ppo.py)。双足环境扩展见 [`extensions/`](extensions/)，实验数据见 [`evidence/`](evidence/)，验证好的权重和交互演示见 [`checkpoints/`](checkpoints/) 与 [`demo/`](demo/)。双足数据来自平地仿真。上游机器人设计、环境基础和公开模型见 [Open Duck Mini v2](https://github.com/apirrone/Open_Duck_Mini/tree/v2) 与 [固定版 Open Duck Playground](https://github.com/apirrone/Open_Duck_Playground/tree/b9be205ac64488c23504ca42e5ec790337adeec3)。
