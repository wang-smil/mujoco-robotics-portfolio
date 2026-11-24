"""用两组交错的腿在平地行走，记录每个周期的位移和接触。"""

from math import cos, pi, radians, sin

import mujoco
import numpy as np

from spider_standing import build_model, floor_contact_geoms, torso_tilt_degrees


GROUPS = (
    ("left_1", "left_3", "right_2", "right_4"),
    ("left_2", "left_4", "right_1", "right_3"),
)
LIFT_ANGLE = radians(12)
SWEEP_ANGLE = 0.12
PERIOD = 1.2
STANCE_FRACTION = 0.60
CYCLES = 3
SETTLE_SECONDS = 0.6
CONTROL_SUBSTEPS = 10
ACTION_LIMIT = 0.35
TARGET_SPEED = 0.034
SPEED_REWARD_WIDTH = 0.05
FALL_TILT_DEGREES = 45


def gait_at_phase(phase):
    if phase < STANCE_FRACTION:
        progress = phase / STANCE_FRACTION
        # 支撑脚在机身坐标系里从前往后。
        return cos(pi * progress), 0.0
    progress = (phase - STANCE_FRACTION) / (1 - STANCE_FRACTION)
    # 抬脚后从后往前摆，在两端把抬脚量降为零。
    return -cos(pi * progress), sin(pi * progress) ** 2


def make_expert(model, home):
    hips = [[model.actuator(name + "_hip_position").id for name in group] for group in GROUPS]
    knees = [[model.actuator(name + "_knee_position").id for name in group] for group in GROUPS]
    knee_signs = [np.array([-1 if name.startswith("left_") else 1 for name in group]) for group in GROUPS]

    def write_targets(ctrl, stride):
        ramp = min(2 * stride, 1.0)
        for index, (hip_ids, knee_ids, signs) in enumerate(zip(hips, knees, knee_signs)):
            sweep, lift = gait_at_phase((stride + 0.5 * index) % 1)
            ctrl[hip_ids] = home[hip_ids] + ramp * LIFT_ANGLE * lift
            ctrl[knee_ids] = home[knee_ids] + ramp * SWEEP_ANGLE * sweep * signs

    return write_targets


def read_observation(data, qpos_addresses, qvel_addresses, feet, contacts, phase):
    root = data.joint("root")
    return np.concatenate((
        root.qpos[3:],
        root.qvel,
        data.qpos[qpos_addresses],
        data.qvel[qvel_addresses],
        [foot in contacts for foot in feet],
        [sin(2 * pi * phase), cos(2 * pi * phase)],
    ))


def observation_names(actuator_names):
    return (
        [f"root_quat_{axis}" for axis in "wxyz"]
        + [f"root_qvel_{index}" for index in range(6)]
        + [f"angle_{name}" for name in actuator_names]
        + [f"velocity_{name}" for name in actuator_names]
        + [f"contact_{name}" for group in GROUPS for name in group]
        + ["phase_sin", "phase_cos"]
    )


def main():
    model = build_model()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    write_targets = make_expert(model, data.ctrl.copy())
    feet = {model.geom(name + "_foot").id for group in GROUPS for name in group}
    torso = data.body("torso")
    floor = model.geom("floor").id
    shell = model.geom("torso_shell").id

    dt = model.opt.timestep
    settle_steps = round(SETTLE_SECONDS / dt)
    cycle_steps = round(PERIOD / dt)
    last_position = torso.xpos[:2].copy()
    total_displacement = np.zeros(2)
    min_support, max_tilt = len(feet), 0.0
    cycle_advances = []
    print(f"cycles={CYCLES}  period={cycle_steps * dt:.2f} s")
    for step in range(settle_steps + CYCLES * cycle_steps):
        walk_step = step - settle_steps
        if walk_step >= 0:
            write_targets(data.ctrl, walk_step / cycle_steps)

        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        contacts = floor_contact_geoms(data, floor)
        if shell in contacts:
            raise RuntimeError("Torso touched the floor")
        if walk_step == -1:
            last_position = torso.xpos[:2].copy()
        if walk_step >= 0:
            min_support = min(min_support, len(contacts & feet))
            max_tilt = max(max_tilt, torso_tilt_degrees(torso))
            if (walk_step + 1) % cycle_steps == 0:
                displacement = torso.xpos[:2] - last_position
                cycle_advances.append(displacement[0])
                total_displacement += displacement
                last_position = torso.xpos[:2].copy()
                print(f"cycle {len(cycle_advances)}: dx={displacement[0]:+.4f} m  dy={displacement[1]:+.4f} m")

    print(f"total_dx={total_displacement[0]:+.4f} m  total_dy={total_displacement[1]:+.4f} m")
    print(f"min_support={min_support}  max_tilt={max_tilt:.2f} deg")
    if (
        len(cycle_advances) != CYCLES
        or not np.all(np.asarray(cycle_advances) > 0.01)
        or not abs(total_displacement[1]) < 0.02
        or min_support < 4
        or not max_tilt < 5
        or not np.isclose(data.time, dt * (settle_steps + cycle_steps * CYCLES))
        or any(warning.number for warning in data.warning)
    ):
        raise RuntimeError("Walking check failed")
    print("walk: PASS")


if __name__ == "__main__":
    main()
