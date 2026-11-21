"""第 6 课：官方小鸭子模型、关节地址与位置执行器；本课不训练 RL。"""

import argparse
import json
from pathlib import Path
import queue
import subprocess
import time

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "third_party" / "Open_Duck_Playground"
SCENE = REPO / "playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml"
OUTPUT = ROOT / "artifacts" / "lesson06"
CONTROL_DT = 0.02
MODE_NAMES = {7: "HOME POSE - forward only", 8: "HEAD MOTION - kinematics only",
              9: "HOME TARGETS - physics, no RL"}


def actuator_addresses(model):
    """从执行器找到所驱动关节的 qpos / qvel 地址，不把两种地址混用。"""
    joints = model.actuator_trnid[:, 0]
    return model.jnt_qposadr[joints].copy(), model.jnt_dofadr[joints].copy()


def reset_home(model, data):
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key < 0:
        raise ValueError("官方场景缺少 home keyframe")
    mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)


def load_model():
    if not SCENE.is_file():
        raise FileNotFoundError(f"找不到官方模型：{SCENE}，请先按第 6 课讲义下载模型。")
    # 仅修改编译对象中的窗口标题，磁盘上的官方 XML 保持不变。
    spec = mujoco.MjSpec.from_file(str(SCENE))
    spec.modelname = "Lesson 06 - Open Duck Mini V2"
    model = spec.compile()
    data = mujoco.MjData(model)
    reset_home(model, data)
    return model, data


def kinematic_pose(model, data, seconds):
    """手动给定头部角度并更新几何；不积分、不推进 data.time。"""
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    data.qpos[:] = model.key_qpos[key]
    data.qvel[:] = 0
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "head_yaw")
    address = model.jnt_qposadr[joint]
    data.qpos[address] += 0.7 * np.sin(2 * np.pi * 0.35 * seconds)
    mujoco.mj_forward(model, data)


def configure_camera(camera):
    camera.lookat[:] = [-0.015, 0, 0.20]
    camera.distance, camera.azimuth, camera.elevation = 0.78, 40, -15


def check_model():
    """使用真实模型检查地址映射、位置执行器、运动学与动力学。"""
    model, data = load_model()
    qadr, vadr = actuator_addresses(model)
    home_q, home_ctrl = data.qpos.copy(), data.ctrl.copy()
    assert (model.nq, model.nv, model.nu) == (21, 20, 14)
    assert np.all(model.jnt_type[model.actuator_trnid[:, 0]] == mujoco.mjtJoint.mjJNT_HINGE)
    assert np.allclose(home_q[qadr], home_ctrl)
    assert np.isclose(model.opt.timestep, 0.002)

    rows = []
    for i in range(model.nu):
        joint = int(model.actuator_trnid[i, 0])
        rows.append({"action_index": i, "joint": model.joint(joint).name,
                     "qpos_address": int(qadr[i]), "qvel_address": int(vadr[i]),
                     "home_target_rad": float(home_ctrl[i]),
                     "target_range_rad": model.actuator_ctrlrange[i].tolist(),
                     "force_range_Nm": model.actuator_forcerange[i].tolist(),
                     "kp": float(model.actuator_gainprm[i, 0])})

    # 只改位置目标，立即产生跟踪力矩；此时尚未执行物理积分。
    head = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "head_yaw")
    data.ctrl[head] += 0.1
    mujoco.mj_forward(model, data)
    tracking_torque = float(data.actuator_force[head])
    assert np.isclose(tracking_torque, 13.37 * 0.1)
    assert np.array_equal(data.qpos, home_q) and data.time == 0

    reset_home(model, data)
    data.ctrl[:] = 0
    mujoco.mj_forward(model, data)
    zero_target_torque = float(np.max(np.abs(data.actuator_force)))
    assert zero_target_torque > 0  # 目标角为零，并不表示关闭执行器。

    reset_home(model, data)
    kinematic_pose(model, data, 0.5)
    assert data.time == 0 and not np.allclose(data.qpos, home_q)
    assert np.array_equal(data.qpos[:7], home_q[:7])

    reset_home(model, data)
    assert np.array_equal(data.qpos, home_q) and np.all(data.qvel == 0)
    for _ in range(2000):  # 固定 home 目标，4 秒真实动力学。
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    assert np.isclose(data.time, 4.0)
    assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
    assert all(int(item.number) == 0 for item in data.warning)

    checks = {"passed": True, "nq": model.nq, "nv": model.nv, "nu": model.nu,
              "physics_dt_seconds": model.opt.timestep,
              "home_targets_match_joint_positions": True,
              "target_offset_rad": 0.1, "resulting_head_torque_Nm": tracking_torque,
              "zero_targets_max_torque_Nm": zero_target_torque,
              "kinematics_keeps_physics_time_zero": True,
              "physics_seconds": float(data.time), "finite_state": True,
              "physics_final_base_xyz": data.qpos[:3].tolist(),
              "scope": "model and actuator checks; no learned walking policy"}
    commit = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    report = {"source": "https://github.com/apirrone/Open_Duck_Playground",
              "commit": commit, "scene": str(SCENE), "mujoco": mujoco.__version__,
              "nq": model.nq, "nv": model.nv, "nu": model.nu, "joints": rows,
              "base_qpos": "xyz + quaternion(wxyz)", "base_qvel": "linear + angular velocity"}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, payload in (("model_report.json", report), ("checks.json", checks)):
        (OUTPUT / name).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for row in rows:
        print(f"action[{row['action_index']:2}] {row['joint']:17} "
              f"qpos[{row['qpos_address']:2}] qvel[{row['qvel_address']:2}] "
              f"home={row['home_target_rad']:+.3f} rad")
    print(json.dumps(checks, ensure_ascii=False), flush=True)
    return checks


def snapshot():
    model, data = load_model()
    # 官方模型默认离屏画布只有 640x480，截图尺寸需要同步设置画布。
    model.vis.global_.offwidth, model.vis.global_.offheight = 960, 720
    camera, option = mujoco.MjvCamera(), mujoco.MjvOption()
    configure_camera(camera)
    option.geomgroup[2], option.geomgroup[3] = 1, 0
    with mujoco.Renderer(model, height=720, width=960) as renderer:
        renderer.update_scene(data, camera=camera, scene_option=option)
        pixels = renderer.render()
    from PIL import Image
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "duck_home.png"
    Image.fromarray(pixels).save(path)
    print("SNAPSHOT", path, flush=True)


def play():
    from mujoco import viewer as mj_viewer
    model, data = load_model()
    keys = queue.SimpleQueue()
    mode, paused, animation_time, physics_frames = 7, True, 0.0, 0
    substeps = round(CONTROL_DT / model.opt.timestep)
    with mj_viewer.launch_passive(model, data, key_callback=keys.put,
                                 show_left_ui=False, show_right_ui=False) as handle:
        with handle.lock():
            configure_camera(handle.cam)
            handle.opt.geomgroup[2], handle.opt.geomgroup[3] = 1, 0
        print("小鸭子：7=初姿态，8=摇头(运动学)，9=位置保持(动力学)，R=重置，空格=暂停。", flush=True)
        while handle.is_running():
            started = time.perf_counter()
            with handle.lock():
                while not keys.empty():
                    key = keys.get_nowait()
                    if key == 32:
                        paused = not paused
                    elif key in (55, 56, 57, 82, 114):
                        if key in (55, 56, 57):
                            mode = key - 48
                        reset_home(model, data)
                        handle.perturb.active = 0
                        animation_time, physics_frames, paused = 0.0, 0, mode == 7
                        print(json.dumps({"mode": mode, "label": MODE_NAMES[mode],
                                          "physics_time": data.time}), flush=True)
                if not paused:
                    if mode == 8:
                        animation_time += CONTROL_DT
                        kinematic_pose(model, data, animation_time)
                    elif mode == 9 and physics_frames < 400:
                        # 保持 reset_home 所恢复的 14 个位置目标；不读取任何 RL 模型。
                        for _ in range(substeps):
                            mujoco.mj_step(model, data)
                        mujoco.mj_forward(model, data)
                        physics_frames += 1
                        if physics_frames == 400:
                            print(json.dumps({"physics_finished_seconds": data.time,
                                              "base_xyz": data.qpos[:3].tolist()}), flush=True)
                status = "STATIC" if mode == 7 else ("PAUSED" if paused else "RUNNING")
                if mode == 9 and physics_frames == 400:
                    status = "ENDED - R to restart"
                text = (f"{MODE_NAMES[mode]} | {status}\n"
                        f"14 actuators: 5 left leg + 4 head/neck + 5 right leg\n"
                        f"nq=21  nv=20 | physics t={data.time:.2f}s\n"
                        f"base z={data.qpos[2]:.3f}m | animation t={animation_time:.2f}s\n"
                        "7: home pose  8: head motion  9: physics\n"
                        "R: reset  Space: pause | No RL policy in this lesson")
            handle.set_texts((mujoco.mjtFont.mjFONT_BIG,
                              mujoco.mjtGridPos.mjGRID_TOPLEFT, text, ""))
            handle.sync()
            time.sleep(max(0.0, CONTROL_DT - (time.perf_counter() - started)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", action="store_true", help="打开小鸭子原生 MuJoCo 窗口")
    parser.add_argument("--snapshot", action="store_true", help="完成数值检查并保存真实模型渲染图")
    args = parser.parse_args()
    if args.view and args.snapshot:
        parser.error("--view 与 --snapshot 请分别运行")
    if args.view:
        play()
    else:
        check_model()
        if args.snapshot:
            snapshot()


if __name__ == "__main__":
    main()
