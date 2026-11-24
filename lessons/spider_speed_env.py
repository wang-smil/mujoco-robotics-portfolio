"""沿用平地动力学，给 Actor 加入前后速度指令。"""

import jax
import jax.numpy as jp
from mujoco import mjx
import numpy as np

from spider_full_env import SpiderFullEnv
from spider_speed_command import TRACKING_WIDTH_M_S, forward_speed, scale_action, slew_speed
from spider_walk import TARGET_SPEED


class SpiderSpeedEnv(SpiderFullEnv):
    def __init__(self, profiles, episode_steps, line_search_iterations):
        super().__init__(None, reward_mode="forward")
        profiles = np.asarray(profiles, np.float32)
        if (profiles.ndim != 2 or profiles.size == 0 or not np.isfinite(profiles).all()
                or episode_steps < profiles.shape[1] or episode_steps % profiles.shape[1]):
            raise ValueError("Provide finite command profiles that evenly divide the episode")
        self.profiles = jp.asarray(profiles)
        self.segment_steps = episode_steps // profiles.shape[1]
        self.hips = jp.array([self.mj_model.actuator(i).name.endswith("_hip_position") for i in range(self.action_size)])
        if self.mj_model.opt.ls_iterations != line_search_iterations:
            self.mj_model.opt.ls_iterations = line_search_iterations
            self._mjx_model = mjx.put_model(self.mj_model)

    @property
    def observation_size(self):
        return 55

    def _command(self, profile, time):
        tick = jp.rint(time / self.dt).astype(jp.int32)
        segment = jp.minimum(tick // self.segment_steps, self.profiles.shape[1] - 1)
        return self.profiles[profile, segment]

    def reset(self, rng):
        state = super().reset(rng)
        profile = jax.random.randint(rng, (), 0, self.profiles.shape[0])
        request = self._command(profile, state.data.time)
        idle = jp.zeros_like(request)
        obs = jp.concatenate((state.obs, jp.stack((request / TARGET_SPEED, idle))))
        metrics = {**state.metrics, "speed_tracking_abs_error": idle}
        return state.replace(obs=obs, info={**state.info, "speed_profile": profile}, metrics=metrics)

    def step(self, state, action):
        request = state.obs[-2] * TARGET_SPEED
        applied = slew_speed(state.obs[-1] * TARGET_SPEED, request, self.dt, jp)
        valid_action = jp.all(jp.isfinite(action))
        motor_action = scale_action(action, self.hips, applied, jp)
        # 保留基类的无效动作处理，包括比例为零时的情况。
        motor_action = jp.where(valid_action, motor_action, jp.nan)
        updated = super().step(state, motor_action)
        delta = updated.data.qpos[:2] - state.data.qpos[:2]
        speed = forward_speed(delta, updated.data.xmat[self.torso_id, :2, 0], self.dt, jp)
        error = applied - speed
        valid = (jp.isfinite(error) & valid_action & jp.all(jp.isfinite(updated.data.qpos))
                 & jp.all(jp.isfinite(updated.data.qvel)))
        done = jp.maximum(updated.done, (~valid).astype(updated.done.dtype))
        reward = jp.where(valid, jp.exp(-jp.square(error / TRACKING_WIDTH_M_S)) - done, -1.0)
        next_request = self._command(state.info["speed_profile"], updated.data.time)
        obs = jp.concatenate((updated.obs, jp.stack((next_request, applied)) / TARGET_SPEED))
        metrics = {
            **updated.metrics,
            "forward_speed": jp.where(valid, speed, 0.0),
            "speed_tracking_abs_error": jp.where(valid, jp.abs(error), 0.0),
        }
        return updated.replace(obs=obs, reward=reward, done=done, metrics=metrics)
