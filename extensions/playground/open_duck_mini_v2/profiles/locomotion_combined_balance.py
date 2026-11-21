"""Expose body tilt and planar velocity to the simulated locomotion Actor."""
import jax.numpy as jp

from playground.open_duck_mini_v2 import locomotion_combined_slew_stable


def default_config():
    cfg = locomotion_combined_slew_stable.default_config()
    cfg.balance_observations = True
    return cfg


class BalanceLocomotion(locomotion_combined_slew_stable.StableReverseLocomotion):
    def __init__(self, task='flat_terrain', config=None, config_overrides=None):
        super().__init__(task=task, config=default_config() if config is None else config,
                         config_overrides=config_overrides)

    def _observe_balance(self, state):
        data = state.data
        features = jp.concatenate((self.get_gravity(data)[:2], self.get_local_linvel(data)[:2]))
        # This profile uses the four unused head-command inputs. The command and reward stay separate.
        obs = {name: value.at[9:13].set(features) for name, value in state.obs.items()}
        return state.replace(obs=obs)

    def reset(self, rng):
        return self._observe_balance(super().reset(rng))

    def step(self, state, action):
        return self._observe_balance(super().step(state, action))
