"""可由 Brax 批量运行的小蜘蛛平地环境。直接运行用于验收专家回放。"""

from math import cos, radians

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
from mujoco_playground._src import mjx_env
import numpy as np

from spider_standing import LEG_TEMPLATE, TORSO_GYRO, build_model
from spider_walk import (
    ACTION_LIMIT, CONTROL_SUBSTEPS, CYCLES, FALL_TILT_DEGREES, GROUPS, PERIOD,
    SPEED_REWARD_WIDTH, TARGET_SPEED, make_expert,
)


YAW_RATE_REWARD_WIDTH = 0.20
HEADING_REWARD_WIDTH = radians(2.0)
SLIP_REWARD_WIDTH = 0.05


class SpiderMJXEnv(mjx_env.MjxEnv):
    def __init__(
        self, reference_qpos=None, reference_weight=0.0, lateral_weight=0.0,
        yaw_rate_weight=0.0, heading_weight=0.0, slip_weight=0.0, *, model=None,
    ):
        self._mj_model = build_model() if model is None else model
        sim_dt = float(self._mj_model.opt.timestep)
        super().__init__(config_dict.create(sim_dt=sim_dt, ctrl_dt=sim_dt * CONTROL_SUBSTEPS))
        self._mjx_model = mjx.put_model(self._mj_model)
        self.cycle_steps = round(PERIOD / self.dt)
        self.action_scale = ACTION_LIMIT
        self._reference_weight = float(reference_weight)
        self._lateral_weight = float(lateral_weight)
        self._yaw_rate_weight = float(yaw_rate_weight)
        self._heading_weight = float(heading_weight)
        self._slip_weight = float(slip_weight)
        if not np.isfinite(self._reference_weight) or self._reference_weight < 0:
            raise ValueError("Reference weight must be finite and nonnegative")
        if not np.isfinite(self._lateral_weight) or self._lateral_weight < 0:
            raise ValueError("Lateral weight must be finite and nonnegative")
        if not np.isfinite(self._yaw_rate_weight) or self._yaw_rate_weight < 0:
            raise ValueError("Yaw-rate weight must be finite and nonnegative")
        if not np.isfinite(self._heading_weight) or self._heading_weight < 0:
            raise ValueError("Heading weight must be finite and nonnegative")
        if not np.isfinite(self._slip_weight) or self._slip_weight < 0:
            raise ValueError("Slip weight must be finite and nonnegative")
        if reference_qpos is not None:
            reference_qpos = np.asarray(reference_qpos, dtype=np.float32)
            if reference_qpos.shape != (self.cycle_steps, self.action_size) or not np.isfinite(reference_qpos).all():
                raise ValueError("Reference must contain one finite joint pose per control step in a gait cycle")
        elif self._reference_weight:
            raise ValueError("A nonzero reference weight needs a joint trajectory")
        self._reference_qpos = None if reference_qpos is None else jp.asarray(reference_qpos)

        self.torso_id = self.mj_model.body("torso").id
        self._gyro_z_adr = int(self.mj_model.sensor(TORSO_GYRO).adr[0]) + 2
        self._floor = self.mj_model.geom("floor").id
        self._shell = self.mj_model.geom("torso_shell").id
        feet = [self.mj_model.geom(name + "_foot").id for group in GROUPS for name in group]
        self._feet = jp.array(feet)
        self._foot_bodies = self.mj_model.geom_bodyid[feet]
        self._foot_roots = self.mj_model.body_rootid[self._foot_bodies]
        joint_ids = self.mj_model.actuator_trnid[:, 0]
        self._qpos_addresses = self.mj_model.jnt_qposadr[joint_ids]
        self._qvel_addresses = self.mj_model.jnt_dofadr[joint_ids]
        root = self.mj_model.joint("root")
        self._root_pos = int(root.qposadr[0])
        self._root_vel = int(root.dofadr[0])

        native_data = mujoco.MjData(self.mj_model)
        mujoco.mj_resetDataKeyframe(self.mj_model, native_data, self.mj_model.key("home").id)
        mujoco.mj_forward(self.mj_model, native_data)
        self.home_ctrl = jp.asarray(native_data.ctrl)
        self._home_data = mjx.put_data(self.mj_model, native_data)
        self._home_heading = self._yaw(self._home_data)

    @property
    def xml_path(self):
        return str(LEG_TEMPLATE)

    @property
    def mj_model(self):
        return self._mj_model

    @property
    def mjx_model(self):
        return self._mjx_model

    @property
    def action_size(self):
        return self.mj_model.nu

    def _floor_contact_mask(self, data, geom_ids):
        # MJX 为每种碰撞对预留槽位，用距离和几何 ID 筛出真实接触。
        first, second = data.contact.geom.T
        touching = data.contact.dist <= data.contact.includemargin
        return touching & (
            ((first == self._floor) & (second == geom_ids[..., None]))
            | ((second == self._floor) & (first == geom_ids[..., None]))
        )

    def _floor_contact(self, data, geom_ids):
        return jp.any(self._floor_contact_mask(data, geom_ids), axis=-1)

    def foot_contact_speed_squared(self, data):
        """地面接触点的切向速度平方，以及对应足端是否触地。"""
        matches = self._floor_contact_mask(data, self._feet)
        count = jp.sum(matches, axis=1)
        contact_sum = jp.sum(jp.where(matches[..., None], data.contact.pos[None, ...], 0.0), axis=1)
        contact_point = contact_sum / jp.maximum(count[:, None], 1)
        # cvel 是世界轴下、以运动树质心为参考点的空间速度。
        omega = data.cvel[self._foot_bodies, :3]
        linear = data.cvel[self._foot_bodies, 3:]
        displacement = contact_point - data.subtree_com[self._foot_roots]
        point_velocity = linear + jp.cross(omega, displacement)
        squared_speed = jp.sum(jp.square(point_velocity[:, :2]), axis=1)
        return squared_speed, count > 0

    def _stance_slip_penalty(self, previous, current):
        speed_squared, touching = self.foot_contact_speed_squared(current)
        stance = touching & self._floor_contact(previous, self._feet)
        cost = -jp.expm1(-speed_squared / SLIP_REWARD_WIDTH**2)
        return self._slip_weight * jp.sum(jp.where(stance, cost, 0)) / jp.maximum(jp.sum(stance), 1)

    def _phase_index(self, data):
        # 相位从物理时间计算，Brax 自动重置时会随 data.time 一起归零。
        control_step = jp.rint(data.time / self.dt).astype(jp.int32)
        return control_step % self.cycle_steps

    def _observe(self, data, foot_contacts):
        phase = self._phase_index(data) / self.cycle_steps
        angle = 2 * jp.pi * phase
        return jp.concatenate((
            data.qpos[self._root_pos + 3:self._root_pos + 7],
            data.qvel[self._root_vel:self._root_vel + 6],
            data.qpos[self._qpos_addresses],
            data.qvel[self._qvel_addresses],
            foot_contacts.astype(data.qpos.dtype),
            jp.stack((jp.sin(angle), jp.cos(angle))),
        ))

    def _metrics(self, data, contacts, speed):
        upright = data.xmat[self.torso_id, 2, 2]
        return {
            "forward_speed": jp.nan_to_num(speed, nan=0.0, posinf=0.0, neginf=0.0),
            "support_feet": jp.sum(contacts).astype(data.qpos.dtype),
            "torso_tilt_deg": jp.degrees(jp.arccos(jp.clip(jp.nan_to_num(upright, nan=0.0), -1, 1))),
        }

    def gyro_z(self, data):
        """机身局部 Z 轴角速度，单位 rad/s。"""
        return data.sensordata[self._gyro_z_adr]

    def _yaw(self, data):
        rotation = data.xmat[self.torso_id]
        return jp.arctan2(rotation[1, 0], rotation[0, 0])

    def heading_error(self, data):
        """相对 home 航向的最短有符号角差，单位 rad。"""
        delta = self._yaw(data) - self._home_heading
        return jp.arctan2(jp.sin(delta), jp.cos(delta))

    def _reset_data(self, rng):
        return self._home_data

    def reset(self, rng):
        data = self._reset_data(rng)
        contacts = self._floor_contact(data, self._feet)
        zero = jp.zeros_like(data.time)
        return mjx_env.State(
            data=data,
            obs=self._observe(data, contacts),
            reward=zero,
            done=zero,
            metrics=self._metrics(data, contacts, zero),
            info={},
        )

    def step(self, state, action):
        # Brax 的 [-1, 1] 动作转换成相对 home 的目标角偏移（rad）。
        valid_action = jp.all(jp.isfinite(action))
        ctrl = self.home_ctrl + self.action_scale * jp.clip(jp.where(jp.isfinite(action), action, 0.0), -1, 1)
        data = state.data.replace(ctrl=ctrl)
        shell = jp.asarray(self._shell)

        def advance(_, carry):
            current, shell_contact = carry
            current = mjx.step(self.mjx_model, current)
            return current, shell_contact | self._floor_contact(current, shell)

        data, shell_contact = jax.lax.fori_loop(0, self.n_substeps, advance, (data, jp.asarray(False)))
        data = mjx.forward(self.mjx_model, data)
        shell_contact |= self._floor_contact(data, shell)

        contacts = self._floor_contact(data, self._feet)
        speed = (data.qpos[self._root_pos] - state.data.qpos[self._root_pos]) / self.dt
        lateral_speed = (data.qpos[self._root_pos + 1] - state.data.qpos[self._root_pos + 1]) / self.dt
        obs = self._observe(data, contacts)
        upright = data.xmat[self.torso_id, 2, 2]
        slip_penalty = self._stance_slip_penalty(state.data, data) if self._slip_weight else 0.0
        finite = (
            valid_action & jp.all(jp.isfinite(obs)) & jp.isfinite(speed)
            & jp.isfinite(lateral_speed) & jp.isfinite(upright) & jp.isfinite(slip_penalty)
        )
        done = (shell_contact | (upright < cos(radians(FALL_TILT_DEGREES))) | ~finite).astype(data.qpos.dtype)
        return state.replace(
            data=data,
            obs=jp.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0),
            reward=self._reward(data, speed, done, finite, lateral_speed) - jp.where(finite, slip_penalty, 0),
            done=done,
            metrics={**state.metrics, **self._metrics(data, contacts, speed)},
        )

    def _reward(self, data, speed, done, finite, lateral_speed=0.0):
        error = (speed - TARGET_SPEED) / SPEED_REWARD_WIDTH
        reward = jp.exp(-jp.square(error)) - done
        if self._lateral_weight:
            # 目标 vy=0，左右对称，惩罚上限等于 lateral_weight。
            lateral_error = lateral_speed / SPEED_REWARD_WIDTH
            reward += self._lateral_weight * jp.expm1(-jp.square(lateral_error))
        if self._yaw_rate_weight:
            yaw_rate = self.gyro_z(data)
            reward += self._yaw_rate_weight * jp.expm1(-jp.square(yaw_rate / YAW_RATE_REWARD_WIDTH))
        if self._heading_weight:
            yaw_error = self.heading_error(data)
            reward += self._heading_weight * jp.expm1(-jp.square(yaw_error / HEADING_REWARD_WIDTH))
        if self._reference_weight:
            reference = self._reference_qpos[self._phase_index(data)]
            pose_error = (data.qpos[self._qpos_addresses] - reference) / 0.15
            gait_bonus = jp.exp(-jp.mean(jp.square(pose_error)))
            reward += (1 - done) * self._reference_weight * gait_bonus
        return jp.where(finite, reward, -1.0)


def check_expert():
    env = SpiderMJXEnv()
    start = jax.jit(env.reset)(jax.random.key(0))
    home = np.asarray(env.home_ctrl)
    expert = make_expert(env.mj_model, home)
    targets = home.copy()
    actions = []
    for step in range(CYCLES * env.cycle_steps):
        expert(targets, step / env.cycle_steps)
        actions.append((targets - home) / env.action_scale)

    @jax.jit
    def replay(state, actions):
        def advance(current, action):
            current = env.step(current, action)
            return current, (current.done, current.metrics["support_feet"], current.metrics["torso_tilt_deg"])

        return jax.lax.scan(advance, state, actions)

    end, (done, support, tilt) = replay(start, jp.asarray(actions))
    dx, dy = np.asarray(end.data.qpos[:2] - start.data.qpos[:2])
    min_support_after_settle = int(jp.min(support[round(0.5 / env.dt):]))
    max_tilt = float(jp.max(tilt))
    elapsed = float(end.data.time)
    print(f"reset: obs={start.obs.shape}  action_size={env.action_size}  control_dt={env.dt:.3f} s")
    print(f"expert: t={elapsed:.3f} s  dx={dx:+.4f} m  dy={dy:+.4f} m")
    print(f"min_support_after_0.5s={min_support_after_settle}  max_tilt={max_tilt:.2f} deg")
    if (
        start.obs.shape != (52,) or env.action_size != 16 or bool(jp.any(done))
        or not np.isclose(elapsed, CYCLES * PERIOD, atol=0.0001)
        or dx <= 0.05 or abs(dy) >= 0.02 or max_tilt >= 5 or min_support_after_settle < 4
    ):
        raise RuntimeError("MJX expert did not pass the environment walking check")
    print("mjx env: PASS")


if __name__ == "__main__":
    check_expert()
