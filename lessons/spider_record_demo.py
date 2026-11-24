"""用 PPO 将要调用的 SpiderEnv 录制 50 Hz 专家示范。"""

from pathlib import Path

import numpy as np

from spider_env import SpiderEnv
from spider_walk import CYCLES, make_expert, observation_names


OUTPUT = Path(__file__).resolve().parents[1] / "artifacts" / "spider_imitation" / "demonstration_50hz.npz"


def main():
    env = SpiderEnv()
    obs, info = env.reset()
    expert = make_expert(env.model, env.home_ctrl)
    targets = env.home_ctrl.copy()
    observations, actions, phases, times = [], [], [], []
    min_support, max_tilt = len(env.feet), 0.0
    start_xy = env.torso.xpos[:2].copy()

    # 专家从 home 起步，第一周期热身，随后录下三个完整周期。
    for step in range((1 + CYCLES) * env.cycle_steps):
        recording = step >= env.cycle_steps
        if step == env.cycle_steps:
            start_xy = env.torso.xpos[:2].copy()
        expert(targets, step / env.cycle_steps)
        action = (targets - env.home_ctrl).astype(np.float32)
        if not env.action_space.contains(action):
            raise RuntimeError("Expert action is outside the environment bounds")
        if recording:
            observations.append(obs)
            actions.append(action)
            phases.append(info["phase"])
            times.append(info["simulated_seconds"])

        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated or not env.observation_space.contains(obs):
            raise RuntimeError("The environment did not complete the demonstration")
        if recording:
            min_support = min(min_support, info["support_feet"])
            max_tilt = max(max_tilt, info["torso_tilt_deg"])

    dx, dy = env.torso.xpos[:2] - start_xy
    if dx <= 0.05 or abs(dy) >= 0.02 or min_support < 4 or max_tilt >= 5:
        raise RuntimeError("The recorded expert did not pass the walking check")

    actuator_names = [env.model.actuator(index).name for index in range(env.model.nu)]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT,
        obs=np.asarray(observations),
        action=np.asarray(actions),
        phase=np.asarray(phases),
        time=np.asarray(times),
        obs_names=np.asarray(observation_names(actuator_names)),
        actuator_names=np.asarray(actuator_names),
        home_ctrl=env.home_ctrl,
        control_dt=env.control_dt,
        physics_dt=env.model.opt.timestep,
        cycle_steps=env.cycle_steps,
    )
    print(f"samples={len(actions)}  control_dt={env.control_dt:.3f} s  physics_dt={env.model.opt.timestep:.3f} s")
    print(f"expert: dx={dx:+.4f} m  dy={dy:+.4f} m  min_support={min_support}  max_tilt={max_tilt:.2f} deg")
    print("demonstration: PASS")


if __name__ == "__main__":
    main()
