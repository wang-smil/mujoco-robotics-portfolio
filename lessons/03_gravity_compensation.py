"""第三课：恢复重力，对比 PD 与 PD + 重力补偿；--view 打开三维窗口。"""

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
MODEL_PATH = ROOT / "artifacts" / "lesson01" / "pendulum_motor.xml"
OUTPUT = ROOT / "artifacts" / "lesson03"
TARGET, KP, KD = 0.5, 4.0, 3.0
DURATION = 8.0


def make_model():
    xml = ET.parse(MODEL_PATH).getroot()
    xml.set("model", "Lesson 03 - Gravity compensation")
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
    model.opt.gravity[:] = [0, 0, -9.81]  # 两组始终都有重力。
    assert (model.nq, model.nv, model.nu) == (1, 1, 1)
    return model


def control_torque(model, data, compensate):
    """两种运行模式共用的控制律，全部基于当前状态。"""
    mujoco.mj_forward(model, data)  # 先刷新当前 q、dq 对应的 qfrc_bias。
    error = TARGET - float(data.qpos[0])
    pd = KP * error - KD * float(data.qvel[0])
    # 本单摆 I 为常数、没有科氏项，因此 bias 就是方程左侧的 G(q)。
    # 它是抵消重力所需的力矩；一般多关节模型的 bias 还含科氏/离心项。
    bias = float(data.qfrc_bias[0])
    compensation = bias if compensate else 0.0
    requested = pd + compensation
    lower, upper = model.actuator_ctrlrange[0]
    applied = float(np.clip(requested, lower, upper))  # 总力矩求和后再限幅。
    return {"pd": pd, "bias": bias, "compensation": compensation,
            "requested": requested, "applied": applied}


def run_case(name, compensate):
    model = make_model()
    data = mujoco.MjData(model)  # 每组重新创建，初始 q=dq=0。
    steps = round(DURATION / float(model.opt.timestep))
    rows, clipped_steps = [], 0
    max_actuator_error = 0.0
    for _ in range(steps):
        terms = control_torque(model, data, compensate)
        data.ctrl[0] = terms["applied"]
        rows.append([float(data.time), float(data.qpos[0]), float(data.qvel[0]),
                     TARGET - float(data.qpos[0]), terms["pd"], terms["bias"],
                     terms["compensation"], terms["requested"], terms["applied"]])
        clipped_steps += abs(terms["requested"] - terms["applied"]) > 1e-10
        mujoco.mj_step(model, data)
        max_actuator_error = max(max_actuator_error,
                                abs(float(data.qfrc_actuator[0]) - terms["applied"]))
        assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
        assert data.ncon == 0, "Unexpected contact changes this experiment"

    # 在终态重新算输入和力，检查“停住”时的力矩平衡。
    final_terms = control_torque(model, data, compensate)
    data.ctrl[0] = final_terms["applied"]
    mujoco.mj_forward(model, data)
    net_torque = float(data.qfrc_actuator[0] + data.qfrc_passive[0] - data.qfrc_bias[0])
    target_data = mujoco.MjData(model)
    target_data.qpos[0] = TARGET
    mujoco.mj_forward(model, target_data)
    target_hold = float(target_data.qfrc_bias[0])
    values = np.asarray(rows)
    tail = values[values[:, 0] >= DURATION - 1.0 - model.opt.timestep / 2]
    summary = {
        "case": name, "gravity_z": float(model.opt.gravity[2]),
        "target_rad": TARGET, "kp": KP, "kd": KD,
        "simulated_seconds": round(float(data.time), 9),
        "final_q_rad": float(data.qpos[0]), "final_dq_rad_s": float(data.qvel[0]),
        "final_error_rad": TARGET - float(data.qpos[0]),
        "last_second_max_error_rad": float(np.max(np.abs(tail[:, 3]))),
        "last_second_min_error_magnitude_rad": float(np.min(np.abs(tail[:, 3]))),
        "last_second_max_speed_rad_s": float(np.max(np.abs(tail[:, 2]))),
        "final_pd_torque_Nm": final_terms["pd"],
        "final_compensation_Nm": final_terms["compensation"],
        "final_motor_torque_Nm": final_terms["applied"],
        "final_net_torque_Nm": net_torque,
        "required_hold_torque_at_target_Nm": target_hold,
        "clipped_steps": int(clipped_steps),
    }
    assert abs(data.time - DURATION) < 1e-9 and max_actuator_error < 1e-10
    assert np.max(np.abs(values[:, 8])) <= 2.0 + 1e-12
    assert abs(target_hold) < 2.0, "The target cannot be held by this motor"
    assert summary["last_second_max_speed_rad_s"] < 0.005
    assert abs(net_torque) < 5e-5
    if compensate:
        assert clipped_steps > 0  # 初始请求超过 ±2；补偿受真实输入限制。
        assert summary["last_second_max_error_rad"] < 0.002
        assert abs(final_terms["pd"]) < 0.001
        assert abs(final_terms["applied"] - target_hold) < 0.001
    else:
        assert abs(summary["final_q_rad"] - 0.5884281) < 0.002
        assert summary["last_second_min_error_magnitude_rad"] > 0.05
        assert final_terms["compensation"] == 0.0
    with (OUTPUT / f"{name}.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["time_s", "q_rad", "dq_rad_s", "error_rad", "pd_torque_Nm",
                         "gravity_bias_Nm", "compensation_Nm", "requested_torque_Nm",
                         "applied_torque_Nm"])
        writer.writerows(rows)
    return summary


def show_control():
    import mujoco.viewer

    model = make_model()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    target_data = mujoco.MjData(model)
    target_data.qpos[0] = TARGET
    mujoco.mj_forward(model, target_data)
    target_position = target_data.geom("bob").xpos.copy()
    pivot_position = data.body("pivot").xpos.copy()
    dt = float(model.opt.timestep)
    steps = round(DURATION / dt)
    keys = queue.SimpleQueue()
    compensate, paused, completed_steps = False, True, 0
    last_terms = {"pd": 0.0, "compensation": 0.0, "applied": 0.0}

    with mujoco.viewer.launch_passive(
        model, data, key_callback=keys.put, show_left_ui=False, show_right_ui=False,
    ) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = [0.25, 0.0, 1.15]
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.7, 90, -10
            sphere, line = viewer.user_scn.geoms[:2]
            mujoco.mjv_initGeom(sphere, mujoco.mjtGeom.mjGEOM_SPHERE,
                               np.array([0.15, 0, 0]), target_position, np.eye(3).ravel(),
                               np.array([0.2, 0.9, 0.35, 0.25], dtype=np.float32))
            mujoco.mjv_initGeom(line, mujoco.mjtGeom.mjGEOM_CAPSULE,
                               np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                               np.array([0.2, 0.9, 0.35, 0.5], dtype=np.float32))
            mujoco.mjv_connector(line, mujoco.mjtGeom.mjGEOM_CAPSULE,
                                 0.01, pivot_position, target_position)
            viewer.user_scn.ngeom = 2  # 纯显示标记，不影响物理。
        print("查看器已打开：1=PD，2=PD+重力补偿，R=重来，空格=开始/暂停。", flush=True)
        while viewer.is_running():
            frame_started = time.perf_counter()
            with viewer.lock():
                while not keys.empty():
                    key = keys.get_nowait()
                    if key == 32:
                        paused = not paused
                        print(f"VIEWER {'PAUSED' if paused else 'RUNNING'} t={data.time:.3f}", flush=True)
                    elif key in (49, 50, 82, 114):
                        if key in (49, 50):
                            compensate = key == 50
                        mujoco.mj_resetData(model, data)
                        data.qpos[:], data.qvel[:], data.ctrl[:] = 0.0, 0.0, 0.0
                        viewer.perturb.active = 0
                        mujoco.mj_forward(model, data)
                        completed_steps, paused = 0, False
                        last_terms = {"pd": 0.0, "compensation": 0.0, "applied": 0.0}
                        print(f"VIEWER RESET compensation={compensate} paused={paused} q=0 dq=0", flush=True)
                if not paused and completed_steps < steps:
                    for _ in range(min(8, steps - completed_steps)):
                        last_terms = control_torque(model, data, compensate)
                        data.ctrl[0] = last_terms["applied"]
                        mujoco.mj_step(model, data)
                        completed_steps += 1
                    mujoco.mj_forward(model, data)
                    if completed_steps == steps:
                        print(json.dumps({"viewer_compensated": compensate,
                                          "simulated_seconds": round(float(data.time), 9),
                                          "q_rad": float(data.qpos[0]),
                                          "dq_rad_s": float(data.qvel[0])}), flush=True)
                mode = "PD + gravity compensation" if compensate else "PD only"
                status = "FINISHED - R to restart" if completed_steps == steps else (
                    "PAUSED - Space to start" if paused else "RUNNING")
                text = (f"{mode}   gravity=-9.81 m/s2\n{status}\n"
                        f"t={data.time:.2f} / {DURATION:g} s   target={TARGET:g} rad\n"
                        f"q={data.qpos[0]:+.4f} rad   error={TARGET-data.qpos[0]:+.4f} rad\n"
                        f"dq={data.qvel[0]:+.4f} rad/s\n"
                        f"Last PD={last_terms['pd']:+.3f}   comp={last_terms['compensation']:+.3f} Nm\n"
                        f"Last motor={last_terms['applied']:+.3f} Nm   limit=+/-2\n"
                        "Green=target; yellow=actual bob\n"
                        "1: PD   2: PD+g   R: restart   Space: pause")
            viewer.set_texts((mujoco.mjtFont.mjFONT_BIG,
                              mujoco.mjtGridPos.mjGRID_TOPLEFT, text, ""))
            viewer.sync()
            time.sleep(max(0.0, 8 * dt - (time.perf_counter() - frame_started)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", action="store_true", help="打开 MuJoCo 三维演示")
    args = parser.parse_args()
    if not MODEL_PATH.is_file():
        raise FileNotFoundError("请先运行 python lessons/01_simulation_step.py")
    if args.view:
        show_control()
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = [run_case("pd_only", False), run_case("pd_gravity", True)]
    for result in results:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    (OUTPUT / "checks.json").write_text(
        json.dumps({"checks": "passed", "cases": results}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print("检查通过。结果保存到：", OUTPUT)


if __name__ == "__main__":
    main()
