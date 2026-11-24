"""小蜘蛛的完整直走奖励和相对 home 的横向位置观测。"""

from math import isfinite

import jax
import jax.numpy as jp
from mujoco import mjx

from spider_mjx_env import SpiderMJXEnv
from spider_walk import SPEED_REWARD_WIDTH, TARGET_SPEED


WEIGHTS = dict(
    lateral_speed=0.10,
    yaw_rate=0.10,
    heading=0.10,
    slip=0.10,
    reference_pose=0.08,
    centerline=0.20,
    action_rate=0.05,
    effort=0.025,
    tilt=0.05,
    alive=0.06,
)
METRIC_TERMS = ("action_rate_cost", "effort_cost", "tilt_cost", "alive_bonus")
Y_OBSERVATION_SCALE = 0.05
CENTERLINE_REWARD_WIDTH = 0.015
CENTERLINE_CORRIDOR = 0.10


class SpiderFullEnv(SpiderMJXEnv):
    def __init__(
        self, reference_qpos, start_offset=0.0, centerline_weight=WEIGHTS["centerline"], reward_mode="full",
    ):
        if reward_mode not in ("full", "forward", "straight"):
            raise ValueError(f"Unknown reward mode: {reward_mode}")
        self.reward_mode = reward_mode
        self._centerline_metric = "centerline_bonus" if reward_mode == "straight" else "centerline_cost"
        weight = 0.0 if reward_mode == "forward" else 1.0
        super().__init__(
            reference_qpos=reference_qpos,
            reference_weight=weight * WEIGHTS["reference_pose"],
            lateral_weight=weight * WEIGHTS["lateral_speed"],
            yaw_rate_weight=weight * WEIGHTS["yaw_rate"],
            heading_weight=weight * WEIGHTS["heading"],
            slip_weight=weight * WEIGHTS["slip"],
        )
        self._start_offset = float(start_offset)
        self.centerline_weight = float(centerline_weight)
        if not isfinite(self.centerline_weight) or self.centerline_weight < 0:
            raise ValueError("Centerline weight must be finite and nonnegative")
        self._home_y = self._home_data.qpos[self._root_pos + 1]

    @property
    def observation_size(self):
        return 53

    def cross_track_error(self, data):
        return data.qpos[self._root_pos + 1] - self._home_y

    def _observe(self, data, foot_contacts):
        obs = super()._observe(data, foot_contacts)
        y = jp.clip(self.cross_track_error(data) / Y_OBSERVATION_SCALE, -10.0, 10.0)
        return jp.concatenate((obs, jp.atleast_1d(y)))

    def reset_at_y_offset(self, rng, offset):
        state = super().reset(rng)
        qpos = state.data.qpos.at[self._root_pos + 1].set(self._home_y + offset)
        data = mjx.forward(self.mjx_model, state.data.replace(qpos=qpos))
        contacts = self._floor_contact(data, self._feet)
        zero = jp.zeros_like(data.time)
        return state.replace(
            data=data,
            obs=self._observe(data, contacts),
            metrics={**state.metrics, **{name: zero for name in (*METRIC_TERMS, self._centerline_metric)}},
        )

    def reset(self, rng):
        offset = 0.0
        if self._start_offset:
            offset = jax.random.uniform(rng, (), minval=-self._start_offset, maxval=self._start_offset)
        return self.reset_at_y_offset(rng, offset)

    def step(self, state, action):
        updated = super().step(state, action)
        if self.reward_mode == "forward":
            return updated
        active = updated.done == 0
        target_delta = updated.data.ctrl - state.data.ctrl
        angle_xy = updated.metrics["torso_tilt_deg"]
        cross_track = self.cross_track_error(updated.data)
        error_squared = jp.square(cross_track / CENTERLINE_REWARD_WIDTH)
        if self.reward_mode == "straight":
            speed_score = jp.exp(-jp.square((updated.metrics["forward_speed"] - TARGET_SPEED) / SPEED_REWARD_WIDTH))
            centerline = jp.where(jp.abs(cross_track) <= CENTERLINE_CORRIDOR, speed_score * jp.exp(-error_squared), 0)
        else:
            centerline = self.centerline_weight * jp.expm1(-error_squared)
        terms = {
            "action_rate_cost": WEIGHTS["action_rate"] * jp.expm1(-jp.mean(jp.square(target_delta / 0.020))),
            "effort_cost": WEIGHTS["effort"] * jp.expm1(-jp.mean(jp.square(updated.data.actuator_force / 0.40))),
            "tilt_cost": WEIGHTS["tilt"] * jp.expm1(-jp.square(angle_xy / 3.0)),
            "alive_bonus": jp.full_like(updated.reward, WEIGHTS["alive"]),
        }
        terms[self._centerline_metric] = centerline
        terms = {name: jp.where(active, value, 0.0) for name, value in terms.items()}
        return updated.replace(reward=updated.reward + sum(terms.values()), metrics={**updated.metrics, **terms})
