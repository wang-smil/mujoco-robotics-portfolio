"""第二课：每个物理步读状态、算力矩，对比 P 和 PD 闭环控制。"""

import argparse
import csv
import json
import queue
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
# 第一课 main() 保存的带 motor、支点标记不碰撞的模型。
MODEL_PATH = ROOT / "artifacts" / "lesson01" / "pendulum_motor.xml"
OUTPUT = ROOT / "artifacts" / "lesson02"
TARGET = 0.5       # rad；相对 XML 初始姿态，不是相对竖直方向。
KP = 4.0           # N·m/rad
DURATION = 8.0     # 模拟秒；运行计算本身不必等待 8 秒。


def run_case(name, kd):
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    model.opt.gravity[:] = 0.0  # 本课先隔离反馈控制的作用。
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    dt = float(model.opt.timestep)
    steps = round(DURATION / dt)
    lower, upper = model.actuator_ctrlrange[0]
    rows = []

    for _ in range(steps):
        # 闭环的核心：本步状态 -> 本步力矩 -> 下一步状态。
        q = float(data.qpos[0])
        dq = float(data.qvel[0])
        error = TARGET - q
        requested_torque = KP * error - kd * dq
        torque = float(np.clip(requested_torque, lower, upper))
        data.ctrl[0] = torque
        # 同一行都是 mj_step 之前的状态和刚计算出的输入。
        rows.append([float(data.time), q, dq, error, requested_torque, torque])
        mujoco.mj_step(model, data)

        assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
        assert data.ncon == 0, "Unexpected contact changes this experiment"

    values = np.asarray(rows)
    tail = values[values[:, 0] >= DURATION - 1.0 - dt / 2]
    summary = {
        "case": name, "target_rad": TARGET, "kp": KP, "kd": kd,
        "passive_damping": float(model.dof_damping[0]),
        "dt": dt, "steps": steps,
        "simulated_seconds": round(float(data.time), 9),
        "final_q_rad": float(data.qpos[0]),
        "final_dq_rad_s": float(data.qvel[0]),
        "last_second_max_error_rad": float(np.max(np.abs(tail[:, 3]))),
        "last_second_max_speed_rad_s": float(np.max(np.abs(tail[:, 2]))),
        "crossed_target": bool(np.any(values[:, 1] > TARGET)),
        "max_abs_torque_Nm": float(np.max(np.abs(values[:, 5]))),
    }
    assert abs(data.time - DURATION) < 1e-9
    assert summary["max_abs_torque_Nm"] <= max(abs(lower), abs(upper)) + 1e-12
    with (OUTPUT / f"{name}.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["time_s", "q_rad", "dq_rad_s", "error_rad",
                         "requested_torque_Nm", "applied_torque_Nm"])
        writer.writerows(rows)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", action="store_true", help="打开 MuJoCo 三维控制窗口")
    parser.add_argument("--controller", choices=("P", "PD"), default="P",
                        help="查看器初始控制器（默认 P）")
    args = parser.parse_args()
    if not MODEL_PATH.is_file():
        raise FileNotFoundError("请先运行 python lessons/01_simulation_step.py，生成电机模型。")
    if args.view:
        show_control(args.controller)
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    p = run_case("p_only", kd=0.0)
    pd = run_case("pd", kd=3.0)
    # 看最后一整秒，避免把偶然经过目标的瞬间误判成稳定。
    assert pd["last_second_max_error_rad"] < 0.002
    assert pd["last_second_max_speed_rad_s"] < 0.005
    assert p["crossed_target"]
    assert p["last_second_max_error_rad"] > 10 * pd["last_second_max_error_rad"]
    for result in (p, pd):
        print(json.dumps(result, ensure_ascii=False, indent=2))
    (OUTPUT / "checks.json").write_text(
        json.dumps({"checks": "passed", "cases": [p, pd]}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("检查通过。轨迹和结果保存到：", OUTPUT)


def show_control(controller):
    """实时闭环控制；sync 负责显示，mj_step 负责推进物理。"""
    import mujoco.viewer

    xml = ET.parse(MODEL_PATH).getroot()
    xml.set("model", "Lesson 02 - P and PD control")
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
    model.opt.gravity[:] = 0.0
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    lower, upper = model.actuator_ctrlrange[0]
    dt = float(model.opt.timestep)
    steps = round(DURATION / dt)

    # 用另一个 data 计算目标球心。它只用于画标记，不参与控制或物理。
    target_data = mujoco.MjData(model)
    target_data.qpos[0] = TARGET
    mujoco.mj_forward(model, target_data)
    target_position = target_data.geom("bob").xpos.copy()
    pivot_position = data.body("pivot").xpos.copy()

    keys = queue.SimpleQueue()
    paused = True  # 窗口打开后按空格开始，避免错过起始过程。
    completed_steps = 0
    torque = 0.0

    # 键盘回调只传递事件；模型和 data 全部由主循环修改。
    with mujoco.viewer.launch_passive(
        model, data, key_callback=keys.put,
        show_left_ui=False, show_right_ui=False,
    ) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = [0.25, 0.0, 1.15]
            viewer.cam.distance = 2.7
            viewer.cam.azimuth = 90
            viewer.cam.elevation = -10
            # user_scn 是纯显示几何，不会给模型增加碰撞或质量。
            sphere, line = viewer.user_scn.geoms[:2]
            mujoco.mjv_initGeom(
                sphere, mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([0.15, 0.0, 0.0]), target_position, np.eye(3).ravel(),
                np.array([0.2, 0.9, 0.35, 0.25], dtype=np.float32),
            )
            mujoco.mjv_initGeom(
                line, mujoco.mjtGeom.mjGEOM_CAPSULE,
                np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                np.array([0.2, 0.9, 0.35, 0.5], dtype=np.float32),
            )
            mujoco.mjv_connector(line, mujoco.mjtGeom.mjGEOM_CAPSULE,
                                 0.01, pivot_position, target_position)
            viewer.user_scn.ngeom = 2
        print("查看器已打开：空格开始/暂停；1=P；2=PD；R=重新开始。", flush=True)

        while viewer.is_running():
            frame_started = time.perf_counter()
            with viewer.lock():
                while not keys.empty():
                    key = keys.get_nowait()
                    if key == 32:
                        paused = not paused
                        print(f"VIEWER {'PAUSED' if paused else 'RUNNING'} t={data.time:.3f}", flush=True)
                    elif key in (49, 50, 82, 114):  # 1、2、R
                        if key in (49, 50):
                            controller = "P" if key == 49 else "PD"
                        mujoco.mj_resetData(model, data)
                        data.qpos[:] = 0.0
                        data.qvel[:] = 0.0
                        data.ctrl[:] = 0.0
                        viewer.perturb.active = 0
                        mujoco.mj_forward(model, data)
                        completed_steps, torque, paused = 0, 0.0, False
                        print(f"VIEWER RESTART {controller} q=0 dq=0", flush=True)

                kd = 0.0 if controller == "P" else 3.0
                if not paused and completed_steps < steps:
                    # 约 60 帧/秒；每个画面之间仍计算 8 个独立的物理/控制步。
                    for _ in range(min(8, steps - completed_steps)):
                        q, dq = float(data.qpos[0]), float(data.qvel[0])
                        torque = float(np.clip(KP * (TARGET - q) - kd * dq, lower, upper))
                        data.ctrl[0] = torque
                        mujoco.mj_step(model, data)
                        completed_steps += 1
                    # 刷新最后一步的几何坐标；不增加模拟时间。
                    mujoco.mj_forward(model, data)
                    if completed_steps == steps:
                        print(json.dumps({"viewer_case": controller,
                                          "simulated_seconds": round(float(data.time), 9),
                                          "q_rad": float(data.qpos[0]),
                                          "dq_rad_s": float(data.qvel[0])}), flush=True)

                status = "FINISHED - R to restart" if completed_steps == steps else (
                    "PAUSED - Space to start" if paused else "RUNNING")
                values = (f"{controller}   Kp={KP:g}   Kd={kd:g}\n{status}\n"
                          f"t={data.time:.2f} / {DURATION:g} s\n"
                          f"q={data.qpos[0]:+.4f} rad\n"
                          f"dq={data.qvel[0]:+.4f} rad/s\n"
                          f"target={TARGET:g} rad   last torque={torque:+.3f} Nm\n"
                          "Green = target; yellow = actual bob\n"
                          "1: P   2: PD   R: restart   Space: pause")
            viewer.set_texts((mujoco.mjtFont.mjFONT_BIG,
                              mujoco.mjtGridPos.mjGRID_TOPLEFT, values, ""))
            viewer.sync()
            # 只限制画面播放速度；物理步长保持 dt，不用等待时间代替它。
            time.sleep(max(0.0, 8 * dt - (time.perf_counter() - frame_started)))


if __name__ == "__main__":
    main()
