"""Start selected training episodes in physically observed reverse-entry states."""
from pathlib import Path

import jax
import jax.numpy as jp
from mujoco_playground._src import mjx_env
from mujoco_playground._src.collision import geoms_colliding
import numpy as np

from playground.open_duck_mini_v2 import locomotion_gait, locomotion_reversal_pairs


def default_config():
    cfg = locomotion_reversal_pairs.default_config()
    cfg.reverse_snapshot_probability = .30
    return cfg


class SnapshotReset(locomotion_reversal_pairs.ReversalPairs):
    def __init__(self, task='flat_terrain', config=None, config_overrides=None):
        super().__init__(task=task, config=default_config() if config is None else config,
                         config_overrides=config_overrides)
        source = Path(__file__).with_name('data') / 'balance_reverse_snapshots.npz'
        shapes = dict(qpos=(21,), qvel=(20,), ctrl=(14,), action_history=(3, 14), phase=(), command=(7,))
        with np.load(source, allow_pickle=False) as snapshots:
            count = len(snapshots['qpos'])
            if count < 1000 or not np.isclose(float(snapshots['cadence']), self._config.reverse_cadence, atol=1e-6):
                raise ValueError('Reverse reset data length or cadence does not match the environment')
            self.reverse_snapshots = {name: jp.asarray(snapshots[name]) for name in shapes}
            for name, shape in shapes.items():
                values = snapshots[name]
                if values.shape != (count, *shape) or not np.isfinite(values).all():
                    raise ValueError(f'Invalid reverse reset array: {name}')

    def reset(self, rng):
        if not self._config.reverse_snapshot_probability:
            return super().reset(rng)
        home_key, choice, row_key = jax.random.split(rng, 3)
        normal = super().reset(home_key)
        use_saved = jax.random.bernoulli(choice, self._config.reverse_snapshot_probability)
        index = jax.random.randint(row_key, (), 0, len(self.reverse_snapshots['qpos']))
        return jax.lax.cond(use_saved, lambda state: self._snapshot_reset(state, index), lambda state: state, normal)

    def _snapshot_reset(self, normal, index):
        saved = {name: values[index] for name, values in self.reverse_snapshots.items()}
        qpos = saved['qpos'].at[:2].set(normal.data.qpos[:2])
        data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=saved['qvel'], ctrl=saved['ctrl'])
        feet = jp.array([geoms_colliding(data, geom, self._floor_geom_id) for geom in self._feet_geom_id])
        history = saved['action_history']
        delay = self._config.noise_config.action_max_delay
        phase = saved['phase']
        theta = phase / self.PRM.nb_steps_in_period * (2 * jp.pi)
        command = saved['command']
        filtered = jp.concatenate((self.get_local_linvel(data)[:2], self.get_gyro(data)[2:3]))
        info = {key: value for key, value in normal.info.items() if key != '_episode_reset'}
        info['imu_history'] = jp.zeros_like(info['imu_history'])
        info.update(command=command, command_target=jp.zeros_like(command).at[0].set(-.05),
            command_ramp_active=jp.asarray(True), imitation_i=phase, imitation_phase=jp.array([jp.cos(theta), jp.sin(theta)]),
            current_reference_motion=self.PRM.get_reference_motion(*command[:3], phase), motor_targets=saved['ctrl'],
            last_contact=feet, last_act=history[0], last_last_act=history[1], last_last_last_act=history[2],
            action_history=history[jp.minimum(jp.arange(delay), 2)].reshape(-1), filtered_twist=filtered)
        observation = self._get_obs(data, info, feet)
        info['_episode_reset'] = {**info}
        state = normal.replace(data=data, info=info, obs=locomotion_gait.patch_filtered_observation(observation, filtered))
        state = self._observe_balance(state)
        return jax.lax.cond(self._get_termination(data), lambda: normal, lambda: state)
