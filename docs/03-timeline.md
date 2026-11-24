# 实验日志

按核心问题记录修改、代码和结果。部分后训练从更早的检查点开分支，轮次步数之和不能作为单个策略的训练总步数。

## 单摆

| 日期 | 核心问题 | 代码修改 | 实验结果 |
|---|---|---|---|
| 2026-10-29 至 10-31 | 如何确认重力、力矩和步进生效？ | 单自由度 MJCF 加力矩执行器，分别测试零输入、有重力和恒力矩。[代码](../lessons/01_simulation_step.py) | 无重力零输入时静止保持。有重力零输入时初始角加速度约 `6.374 rad/s²`。[检查记录](../evidence/lesson01/checks.json) |
| 10-31 至 11-01 | 怎么减小振荡和负载下的稳态误差？ | 比较 P 与 PD，再用当前位置计算重力补偿力矩。[PD](../lessons/02_pd_control.py) · [重力补偿](../lessons/03_gravity_compensation.py) | 无重力 P/PD 对照的最后一秒最大误差为 `0.388 / 5.69×10⁻⁷ rad`。重力补偿将稳态误差从 `0.0884` 降到 `3.44×10⁻⁷ rad`。[P/PD 轨迹](../evidence/lesson02/) · [补偿数据](../evidence/lesson03/checks.json) |
| 11-02 | 能否从同一环境训练并重载 PPO？ | 封装 Gymnasium 环境，用随机和 PD 验证奖励与终止，再训练 4 环境 SB3 PPO。[环境](../lessons/pendulum_env.py) · [训练](../lessons/05_ppo.py) | PPO 实采 `200,704` 步，所选模型在 10 个独立种子上平均回报 `375.90`。PD 平均回报 `370.32`，尾段角误差更小。[环境检查](../evidence/lesson04/checks.json) · [逐种子结果](../evidence/lesson05/evaluation.json) |

## 小鸭子

| 日期 / 轮次 | 核心问题 | 代码修改 | 实验结果 |
|---|---|---|---|
| 11-03 至 11-11 · 上游复现 | 模型、动作和训练能否贯通？ | 加载上游 `home` 和执行器地址，固定环境依赖，接入 MJX/Brax PPO 保存和回载。[模型验收](../lessons/06_duck_model.py) · [训练入口](../lessons/07_duck_ppo.py) | 模型为 `nq=21 / nv=20 / nu=14`。发布版 GPU 小批量验证完成 32 环境、1,280 步，回载动作最大误差 `0.0`。[模型记录](../evidence/lesson06/model_report.json) · [GPU 验证](../evidence/duck/publication_gpu_smoke.json) |
| 11-12 · 前进 | 能否稳定站立并跟踪低速前进？ | 将任务集中到站立和前进，加入线速度、偏航跟踪和接触稳定项。[任务与奖励](../extensions/playground/open_duck_mini_v2/locomotion.py) | 前进分支追加约 `102.4M` 步，所选检查点基础验收 `32/32`。更强联合扰动下 `2/16` 达标、`14/16` 发生物理失败。[当时的实验摘要](https://github.com/wang-smil/mujoco-robotics-portfolio/blob/3de35dae3491960121c85bb160095850993fc4f2/docs/03-timeline.md) |
| 11-17 · 头颈 | 减少头晃能否保住跟踪？ | 限制头颈目标变化率为 `0.5 rad/s`，按同种子比较原策略、直接限速、限速后续训。[代码](../extensions/playground/open_duck_mini_v2/profiles/locomotion_neck.py) | 三组均 `80/80` 通过。头颈速度 RMS 为 `1.2473 / 0.3601 / 0.3481 rad/s`，平面跟踪误差 RMS 为 `0.0044 / 0.0064 / 0.0042 m/s`。[消融数据](../evidence/duck/neck/final_metrics.json) |
| 11-18 · 第 1–4 轮 | 如何加入原地转向？ | 采样正负偏航命令，加转向跟踪配置。[代码](../extensions/playground/open_duck_mini_v2/profiles/locomotion_turn_tracking.py) | 第 4 轮按独立种子分别评估站立、前进与双向转向，站立和两档前进各 `20/20` 达标。[逐项评测](../evidence/duck/round4_heldout.json) |
| 11-18 · 第 5–8 轮 | 为什么有策略能跑满，却几乎不后退？ | 尝试后退速度跟踪、超速惩罚和参考重置。[历史代码](../history/environment_profiles/) | 一些策略跑满时接近原地。后续将跑满与跟踪达标分开统计。[逐轮记录](https://github.com/wang-smil/mujoco-robotics-portfolio/releases/tag/experiment-archive) |
| 11-18 · 第 9–12 轮 | 能否同时跟踪前向速度和偏航？ | 加入复合弧线命令，并采样直行、转向与弧线切换。[代码](../extensions/playground/open_duck_mini_v2/profiles/locomotion_arc_switches.py) | 第 11 轮记录直行、原地转向、弧线与完整切换。后退进入仍需单独处理。[独立原生评测](../evidence/duck/round11_native_heldout.json) |
| 11-18 至 11-19 · 第 13–17 轮 | 后退示范能否改善学习？ | 给匹配观测加示范动作监督，以较低探索量接续 PPO。[示范损失](../lessons/ppo_demonstration_anchor.py) | 固定后退得到可用样本，缓变进入仍不稳。[示范验证](../evidence/duck/reverse_teacher_refined_validated.json) · [训练归档](https://github.com/wang-smil/mujoco-robotics-portfolio/releases/tag/experiment-archive) |
| 11-19 · 第 18–22 轮 | 怎样让训练覆盖真实换向状态？ | 补足平衡观测和定向换向采样，用实测快照恢复目标、动作历史与步态相位。[平衡观测](../extensions/playground/open_duck_mini_v2/profiles/locomotion_combined_balance.py) · [快照](../extensions/playground/open_duck_mini_v2/locomotion_snapshot_reset.py) | 第 22 轮用 2,048 环境追加 `10,076,160` 步。单 Actor 直接后退 `8/8` 跑满、均速约 `−0.053 m/s`，缓变命令 `0/8` 跑满。[直接](../evidence/duck/round22_native_final.json) · [缓变](../evidence/duck/round22_native_final_ramp.json) |
| 11-19 · 手柄对照 | 如何改善从行进切入后退？ | 在示范初始化 Actor 上加入运行时等待、参考相位和髋踝反馈。[过渡代码](../demo/probe_moving_reverse.py) · [反馈实现](../demo/fit_reverse_entry.py) | 对比上一版辅助控制器，直走与后退互换达标 `1/20 → 15/20`，跑满 `20/20 → 18/20`。偏航增益另测后未采用。[本地对照](../evidence/duck/reverse_moving_entry_matched.json) · [作者 ONNX 对照](../evidence/duck/author_moving_entry_matched.json) · [偏航复测](../evidence/duck/reverse_moving_yaw_probe.json) |

## 原始记录

- [逐轮配置索引](../evidence/duck/round-index.csv) · [第 2–21 轮 Release 归档](https://github.com/wang-smil/mujoco-robotics-portfolio/releases/tag/experiment-archive)
- 第 22 轮 [配置](../evidence/duck/rounds/round-22/config.json) · [进度 JSONL](../evidence/duck/rounds/round-22/progress.jsonl) · [验收](../evidence/duck/rounds/round-22/checks.json) · [资源日志](../evidence/duck/guards/)
- [实验数据](../evidence/) · [发布检查点](../checkpoints/)
