"""验证导入、物理仿真与真实图形渲染；不需要训练模型。"""

import json
import platform
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image


root = Path(__file__).resolve().parent
model = mujoco.MjModel.from_xml_path(str(root / "models" / "pendulum.xml"))
data = mujoco.MjData(model)
angles = []
for _ in range(2000):
    mujoco.mj_step(model, data)
    angles.append(float(data.qpos[0]))

assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
assert abs(data.time - 4.0) < 1e-8, f"Unexpected simulated time: {data.time}"
assert np.ptp(angles) > 0.1, "The pendulum did not move as expected"

# 固定相机便于检查画面；使用 WSLg 的 OpenGL 显示路径。
camera = mujoco.MjvCamera()
camera.lookat[:] = [0.1, 0.0, 0.85]
camera.distance = 3.1
camera.azimuth = 90
camera.elevation = -15
with mujoco.Renderer(model, height=480, width=640) as renderer:
    renderer.update_scene(data, camera=camera)
    pixels = renderer.render()
assert pixels.shape == (480, 640, 3)
assert float(pixels.std()) > 5.0, "Rendered image appears blank"

artifacts = root / "artifacts"
artifacts.mkdir(exist_ok=True)
Image.fromarray(pixels).save(artifacts / "pendulum.png")
report = {
    "python": platform.python_version(),
    "python_executable": sys.executable,
    "mujoco": mujoco.__version__,
    "simulated_seconds": data.time,
    "simulation_steps": 2000,
    "pendulum_angle_range_rad": float(np.ptp(angles)),
    "render_size": [640, 480],
    "image": str(artifacts / "pendulum.png"),
    "physics_check": "passed",
    "render_check": "passed",
}
(artifacts / "environment-check.json").write_text(
    json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)
print(json.dumps(report, indent=2, ensure_ascii=False))
