"""Add conservative reverse walking to the retained turn curriculum."""
import jax
import jax.numpy as jp

from playground.open_duck_mini_v2 import locomotion_turn_tracking


def default_config():
    config = locomotion_turn_tracking.default_config()
    config.turn_yaw_error_multiplier = 4.0
    config.reverse_probability = .10
    config.reverse_speed = [.035, .065]
    return config


class ReverseCurriculum(locomotion_turn_tracking.TurnTracking):
    def __init__(self, task='flat_terrain', config=None, config_overrides=None):
        super().__init__(task=task, config=default_config() if config is None else config,
                         config_overrides=config_overrides)

    def sample_command(self, rng):
        inherited, selection, speed_key = jax.random.split(rng, 3)
        command = super().sample_command(inherited)
        forward_probability = .9 - self._config.turn_probability
        reverse = (command[0] > 0) & (jax.random.uniform(selection) < self._config.reverse_probability / forward_probability)
        speed = jax.random.uniform(speed_key, minval=self._config.reverse_speed[0], maxval=self._config.reverse_speed[1])
        return command.at[0].set(jp.where(reverse, -speed, command[0]))
