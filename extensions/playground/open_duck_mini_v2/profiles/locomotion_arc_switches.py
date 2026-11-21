"""Expose the arc policy to command transitions more often during training."""
import jax.numpy as jp

from playground.open_duck_mini_v2 import locomotion_arc


def default_config():
    config = locomotion_arc.default_config()
    config.turn_command_hold_steps = 200
    return config


class ArcSwitches(locomotion_arc.ArcCurriculum):
    def __init__(self, task='flat_terrain', config=None, config_overrides=None):
        super().__init__(task=task, config=default_config() if config is None else config,
                         config_overrides=config_overrides)

    def step(self, state, action):
        command_timer = state.info['step']
        due = command_timer >= self._config.turn_command_hold_steps - 1
        # Joystick samples after incrementing its command counter beyond 500.
        info = {**state.info, 'step': jp.where(due, 500, command_timer)}
        return super().step(state.replace(info=info), action)
