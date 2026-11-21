"""Add low-speed forward turns alongside the existing commands."""
import jax
import jax.numpy as jp

from playground.open_duck_mini_v2 import locomotion_turn_tracking


def default_config():
    config = locomotion_turn_tracking.default_config()
    config.turn_yaw_error_multiplier = 4.0
    config.arc_probability = .12
    config.arc_forward_speed = [.075, .125]
    return config


class ArcCurriculum(locomotion_turn_tracking.TurnTracking):
    def __init__(self, task='flat_terrain', config=None, config_overrides=None):
        super().__init__(task=task, config=default_config() if config is None else config,
                         config_overrides=config_overrides)

    def sample_command(self, rng):
        inherited, selection, forward, turn, direction = jax.random.split(rng, 5)
        command = super().sample_command(inherited)
        forward_probability = .9 - self._config.turn_probability
        use_arc = (command[0] > 0) & (jax.random.uniform(selection) < self._config.arc_probability / forward_probability)
        vx = jax.random.uniform(forward, minval=self._config.arc_forward_speed[0], maxval=self._config.arc_forward_speed[1])
        turn_speed = jax.random.uniform(turn, minval=self._config.turn_yaw_speed[0], maxval=self._config.turn_yaw_speed[1])
        yaw = jp.where(jax.random.bernoulli(direction), turn_speed, -turn_speed)
        return jp.where(use_arc, command.at[0].set(vx).at[2].set(yaw), command)
