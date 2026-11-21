"""第一课：模型 + 状态 + 控制输入 -> 下一个状态。"""

import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "lesson01"
CASES = (
    ("no_gravity", "无重力 · 力矩 0", 0.0, 0.0),
    ("gravity", "有重力 · 力矩 0", -9.81, 0.0),
    ("motor", "无重力 · 力矩 1 N·m", 0.0, 1.0),
)
STEPS = 1000


def make_motor_xml():
    """原模型没有 actuator。仅给本课副本添加一个转矩电机。"""
    xml = ET.parse(ROOT / "models" / "pendulum.xml").getroot()
    marker = xml.find("./worldbody/body[@name='pivot']/geom")
    if marker is None:
        raise ValueError("The pivot marker was not found; review the lesson model")
    # 支点球只是标记，与杆重叠但不应产生接触力。
    marker.set("name", "pivot_marker")
    marker.set("contype", "0")
    marker.set("conaffinity", "0")
    if xml.find("actuator") is not None:
        raise ValueError("The source model already contains actuators; review the lesson")
    actuator = ET.SubElement(xml, "actuator")
    ET.SubElement(
        actuator, "motor", name="hinge_motor", joint="hinge", gear="1",
        ctrllimited="true", ctrlrange="-2 2",
    )
    ET.indent(xml)
    return ET.tostring(xml, encoding="unicode")


def snapshot(data):
    # geom_xpos 是派生的世界坐标；调用方先用 mj_forward 刷新它。
    position = data.geom("bob").xpos
    return [float(data.qpos[0]), float(data.qvel[0]), float(position[0]), float(position[2])]


def run_case(xml, case):
    name, label, gravity_z, torque = case
    model = mujoco.MjModel.from_xml_string(xml)
    model.opt.gravity[:] = [0, 0, gravity_z]
    data = mujoco.MjData(model)  # 每组实验从独立的初始状态开始。
    data.qpos[0] = 0.0
    data.qvel[0] = 0.0
    data.ctrl[0] = torque
    mujoco.mj_forward(model, data)
    initial_acceleration = float(data.qacc[0])
    samples = [snapshot(data)]
    max_actuator_error = 0.0
    started = time.perf_counter()

    for _ in range(STEPS):
        data.ctrl[0] = torque         # 1. 写入本步电机控制量。
        mujoco.mj_step(model, data)   # 2. 物理时间推进 model.opt.timestep。
        mujoco.mj_forward(model, data)  # 3. 刷新显示坐标；不推进时间。
        samples.append(snapshot(data))
        max_actuator_error = max(max_actuator_error, abs(float(data.qfrc_actuator[0]) - torque))
        assert data.ncon == 0, "An unexpected contact changes this simple experiment"

    elapsed = time.perf_counter() - started
    values = np.asarray(samples)
    q, dq, x, z = values.T
    assert (model.nq, model.nv, model.nu) == (1, 1, 1)
    assert np.isfinite(values).all()
    assert abs(data.time - STEPS * model.opt.timestep) < 1e-10
    assert max_actuator_error < 1e-10
    # 检查读出的世界坐标确实对应当前 q，而不是上一步的缓存。
    assert np.allclose(x, 0.55 * np.cos(q) - 0.75 * np.sin(q), atol=1e-10)
    assert np.allclose(z, 1.6 - 0.55 * np.sin(q) - 0.75 * np.cos(q), atol=1e-10)
    if name == "no_gravity":
        assert np.max(np.abs(q)) < 1e-10 and np.max(np.abs(dq)) < 1e-10
    elif name == "gravity":
        assert initial_acceleration > 0 and dq[1] > 0 and z[1] < z[0]
    else:
        assert initial_acceleration > 0 and dq[-1] > 0 and np.all(np.diff(q) > 0)

    summary = {
        "case": name, "gravity_z": gravity_z, "ctrl": torque,
        "nq": model.nq, "nv": model.nv, "nu": model.nu,
        "dt": model.opt.timestep, "steps": STEPS,
        "simulated_seconds": round(float(data.time), 9),
        "computation_seconds": round(elapsed, 6),
        "initial_qacc": initial_acceleration,
        "final_qpos_rad": float(q[-1]), "final_qvel_rad_s": float(dq[-1]),
        "checks": "passed",
    }
    trace = {
        "id": name, "label": label, "gravity": gravity_z, "torque": torque,
        "samples": [[round(v, 9) for v in row] for row in samples],
    }
    print(json.dumps(summary, ensure_ascii=False))
    return trace, summary


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    xml = make_motor_xml()
    (OUTPUT / "pendulum_motor.xml").write_text(xml + "\n", encoding="utf-8")
    results = [run_case(xml, case) for case in CASES]
    traces = {"dt": 0.002, "steps": STEPS, "pivot": [0, 1.6], "cases": [r[0] for r in results]}
    (OUTPUT / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    (OUTPUT / "checks.json").write_text(json.dumps([r[1] for r in results], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
