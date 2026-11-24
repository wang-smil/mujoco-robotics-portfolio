"""采集直走专家示范，训练按步态相位输出关节目标的行为克隆基线。"""

from pathlib import Path

import mujoco
import numpy as np

from spider_standing import build_model, floor_contact_geoms, torso_tilt_degrees
from spider_walk import CYCLES, GROUPS, PERIOD, SETTLE_SECONDS, make_expert, observation_names, read_observation


OUTPUT = Path(__file__).resolve().parents[1] / "artifacts" / "spider_imitation"
HARMONICS = 12
RIDGE = 1e-6


def phase_features(phase):
    angles = 2 * np.pi * np.asarray(phase)[..., None] * np.arange(1, HARMONICS + 1)
    return np.concatenate((np.ones_like(angles[..., :1]), np.cos(angles), np.sin(angles)), axis=-1)


def rollout(model, clone_weights=None):
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    home = data.ctrl.copy()
    expert = make_expert(model, home)
    torso = data.body("torso")
    floor = model.geom("floor").id
    shell = model.geom("torso_shell").id
    leg_names = [name for group in GROUPS for name in group]
    feet = [model.geom(name + "_foot").id for name in leg_names]
    actuator_names = [model.actuator(index).name for index in range(model.nu)]
    joint_ids = model.actuator_trnid[:, 0]
    qpos_addresses = model.jnt_qposadr[joint_ids]
    qvel_addresses = model.jnt_dofadr[joint_ids]

    dt = model.opt.timestep
    cycle_steps = round(PERIOD / dt)
    warmup_steps = round(SETTLE_SECONDS / dt) + cycle_steps
    for step in range(warmup_steps):
        stride_step = step - (warmup_steps - cycle_steps)
        if stride_step >= 0:
            expert(data.ctrl, stride_step / cycle_steps)
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)

    start_xy = torso.xpos[:2].copy()
    previous_x = float(start_xy[0])
    advances = []
    observations, offsets, phases = [], [], []
    min_support, max_tilt = len(feet), 0.0
    contacts = floor_contact_geoms(data, floor)
    for step in range(CYCLES * cycle_steps):
        phase = (step % cycle_steps) / cycle_steps
        if clone_weights is None:
            observations.append(read_observation(
                data, qpos_addresses, qvel_addresses, feet, contacts, phase,
            ))
            expert(data.ctrl, 1 + step / cycle_steps)
            offsets.append(data.ctrl - home)
            phases.append(phase)
        else:
            data.ctrl[:] = home + phase_features(phase) @ clone_weights

        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        contacts = floor_contact_geoms(data, floor)
        if shell in contacts:
            raise RuntimeError("Torso touched the floor during the scored window")
        min_support = min(min_support, sum(foot in contacts for foot in feet))
        max_tilt = max(max_tilt, torso_tilt_degrees(torso))
        if (step + 1) % cycle_steps == 0:
            advances.append(float(torso.xpos[0]) - previous_x)
            previous_x = float(torso.xpos[0])

    dx, dy = torso.xpos[:2] - start_xy
    name = "expert" if clone_weights is None else "clone"
    print(f"{name}: dx={dx:+.4f} m  dy={dy:+.4f} m  min_support={min_support}  max_tilt={max_tilt:.2f} deg")
    if (
        not np.all(np.asarray(advances) > 0.01)
        or abs(dy) >= 0.02
        or min_support < 4
        or max_tilt >= 5
        or not np.isclose(data.time, (warmup_steps + CYCLES * cycle_steps) * dt)
        or any(warning.number for warning in data.warning)
    ):
        raise RuntimeError(f"{name} walking check failed")

    if clone_weights is None:
        return {
            "obs": np.asarray(observations),
            "action": np.asarray(offsets),
            "phase": np.asarray(phases),
            "obs_names": np.asarray(observation_names(actuator_names)),
            "actuator_names": np.asarray(actuator_names),
            "home_ctrl": home,
            "dt": dt,
            "cycle_steps": cycle_steps,
        }


def main():
    model = build_model()
    demonstration = rollout(model)
    actions = demonstration["action"]
    features = phase_features(demonstration["phase"])
    # 前两个稳定周期拟合，第三个周期检查动作拟合误差。
    split = 2 * demonstration["cycle_steps"]
    train = features[:split]
    weights = np.linalg.solve(train.T @ train + RIDGE * np.eye(train.shape[1]), train.T @ actions[:split])
    rmse = np.sqrt(np.mean((features[split:] @ weights - actions[split:]) ** 2))
    print(f"samples={len(actions)}  obs={demonstration['obs'].shape[1]}  actions={actions.shape[1]}")
    print(f"held_out_action_rmse={rmse:.6f} rad")
    if not rmse < 0.002:
        raise RuntimeError("Action fit check failed")
    rollout(model, clone_weights=weights)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT / "demonstration.npz", **demonstration)
    np.savez_compressed(
        OUTPUT / "phase_clone.npz",
        weights=weights,
        harmonics=HARMONICS,
        cycle_steps=demonstration["cycle_steps"],
        dt=demonstration["dt"],
        home_ctrl=demonstration["home_ctrl"],
        actuator_names=demonstration["actuator_names"],
    )
    print("imitation: PASS")


if __name__ == "__main__":
    main()
