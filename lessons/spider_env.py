"""小蜘蛛平地直走的 Gymnasium 环境。直接运行用于验收专家回放。"""

from math import exp

import gymnasium as gym
import mujoco
import numpy as np

from spider_standing import build_model, floor_contact_geoms, torso_tilt_degrees
from spider_walk import (
    ACTION_LIMIT, CONTROL_SUBSTEPS, CYCLES, FALL_TILT_DEGREES, GROUPS, PERIOD,
    SPEED_REWARD_WIDTH, TARGET_SPEED, make_expert, read_observation,
)


class SpiderEnv(gym.Env):
    def __init__(self, max_episode_steps=500):
        super().__init__()
        if max_episode_steps < 1:
            raise ValueError("max_episode_steps must be positive")
        self.model = build_model()
        self.data = mujoco.MjData(self.model)
        self.frame_skip = CONTROL_SUBSTEPS
        self.control_dt = self.frame_skip * float(self.model.opt.timestep)
        self.cycle_steps = round(PERIOD / self.control_dt)
        self.max_episode_steps = max_episode_steps
        self.target_speed = TARGET_SPEED
        self.action_space = gym.spaces.Box(-ACTION_LIMIT, ACTION_LIMIT, shape=(self.model.nu,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(52,), dtype=np.float32)

        self.home_key = self.model.key("home").id
        self.home_ctrl = self.model.key_ctrl[self.home_key].copy()
        self.floor = self.model.geom("floor").id
        self.shell = self.model.geom("torso_shell").id
        self.feet = [self.model.geom(name + "_foot").id for group in GROUPS for name in group]
        driven_joints = self.model.actuator_trnid[:, 0]
        self.qpos_addresses = self.model.jnt_qposadr[driven_joints]
        self.qvel_addresses = self.model.jnt_dofadr[driven_joints]
        self.torso = self.data.body("torso")
        self.elapsed_steps = 0
        self._needs_reset = True

    def _observe(self, contacts, forward_speed):
        phase = (self.elapsed_steps % self.cycle_steps) / self.cycle_steps
        obs = read_observation(
            self.data, self.qpos_addresses, self.qvel_addresses, self.feet, contacts, phase,
        ).astype(np.float32)
        info = {
            "simulated_seconds": float(self.data.time),
            "phase": phase,
            "forward_speed": float(forward_speed),
            "target_speed": self.target_speed,
            "support_feet": sum(foot in contacts for foot in self.feet),
            "torso_tilt_deg": torso_tilt_degrees(self.torso),
        }
        if not np.isfinite(obs).all():
            raise RuntimeError("Non-finite spider state")
        return obs, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_key)
        mujoco.mj_forward(self.model, self.data)
        self.elapsed_steps = 0
        self._needs_reset = False
        return self._observe(floor_contact_geoms(self.data, self.floor), forward_speed=0.0)

    def step(self, action):
        if self._needs_reset:
            raise RuntimeError("Call reset() before step(), including after an episode ends")
        action = np.asarray(action, dtype=np.float32)
        if action.shape != self.action_space.shape or not np.isfinite(action).all():
            raise ValueError(f"Action must be finite with shape {self.action_space.shape}")

        # action 是相对 home 的目标角偏移，单位 rad，保持 10 个物理步。
        self.data.ctrl[:] = self.home_ctrl + np.clip(action, self.action_space.low, self.action_space.high)
        start_x = float(self.torso.xpos[0])
        shell_contact = False
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
            contacts = floor_contact_geoms(self.data, self.floor)
            shell_contact |= self.shell in contacts
        self.elapsed_steps += 1

        forward_speed = (self.torso.xpos[0] - start_x) / self.control_dt
        obs, info = self._observe(contacts, forward_speed)
        terminated = shell_contact or info["torso_tilt_deg"] > FALL_TILT_DEGREES or any(
            warning.number for warning in self.data.warning
        )
        truncated = self.elapsed_steps >= self.max_episode_steps and not terminated
        speed_error = (forward_speed - self.target_speed) / SPEED_REWARD_WIDTH
        reward = exp(-speed_error**2) - float(terminated)
        self._needs_reset = terminated or truncated
        return obs, reward, terminated, truncated, info


def check_expert():
    env = SpiderEnv()
    obs, info = env.reset()
    expert = make_expert(env.model, env.home_ctrl)
    targets = env.home_ctrl.copy()
    start_xy = env.torso.xpos[:2].copy()
    max_tilt = 0.0
    print(f"reset: obs={obs.shape}  action={env.action_space.shape}  t={info['simulated_seconds']:.3f} s")
    for step in range(CYCLES * env.cycle_steps):
        expert(targets, step / env.cycle_steps)
        obs, reward, terminated, truncated, info = env.step(targets - env.home_ctrl)
        if terminated or truncated or not env.observation_space.contains(obs):
            raise RuntimeError("Unexpected end or invalid observation during expert check")
        if step == 0:
            print(f"first step: t={info['simulated_seconds']:.3f} s  reward={reward:.3f}")
        max_tilt = max(max_tilt, info["torso_tilt_deg"])
    dx, dy = env.torso.xpos[:2] - start_xy
    print(f"expert: cycles={CYCLES}  dx={dx:+.4f} m  dy={dy:+.4f} m  max_tilt={max_tilt:.2f} deg")
    if dx <= 0.05 or abs(dy) >= 0.02 or max_tilt >= 5:
        raise RuntimeError("Expert did not pass the environment walking check")
    print("env: PASS")


if __name__ == "__main__":
    check_expert()
