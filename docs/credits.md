# 来源与贡献范围

- **机器人、XML 和基础训练接口**来自 Antoine Pirrone 与 Open Duck Mini 贡献者维护的项目。采用 Open Duck Playground 固定提交 `b9be205ac64488c23504ca42e5ec790337adeec3`。安装脚本从指定提交获取 `playground/`，核对参考动作数据后叠加本项目扩展。 [Open Duck Mini v2](https://github.com/apirrone/Open_Duck_Mini/tree/v2) · [Open Duck Playground](https://github.com/apirrone/Open_Duck_Playground/tree/b9be205ac64488c23504ca42e5ec790337adeec3)
- **作者权重基线**为 `BEST_WALK_ONNX_2.onnx`，按固定提交和 SHA-256 获取。仓库保存本地执行对照后导出的数值。 [源权重](https://github.com/apirrone/Open_Duck_Mini/blob/b23317a485b3cec7d8417f352478778b3475173c/BEST_WALK_ONNX_2.onnx) · [上游许可](https://github.com/apirrone/Open_Duck_Mini/blob/v2/LICENSE)
- **项目实现**包括单摆动力学与控制实验、双足任务扩展、训练入口、示范损失、交互控制器、评测脚本和训练检查点。视频由实测仿真轨迹生成，展示的机器人形态来自 Open Duck Mini。
- **依赖**主要为 MuJoCo、Gymnasium、Stable-Baselines3、JAX、MJX、Brax 和 Orbax。安装各自使用对应的上游发布包。环境获取和训练结果复核见 [复现说明](reproduce.md)。
