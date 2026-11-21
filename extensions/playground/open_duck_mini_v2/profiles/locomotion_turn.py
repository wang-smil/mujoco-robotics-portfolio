"""First turn curriculum on the established neck-limited walking task."""
import jax
import jax.numpy as jp

from playground.open_duck_mini_v2 import locomotion_neck


def default_config():
    config = locomotion_neck.default_config()
    config.turn_yaw_speed = [.15, .25]
    config.turn_probability = .20
    return config


class ConservativeTurn(locomotion_neck.NeckLimitedGait):
    def __init__(self, task='flat_terrain', config=None, config_overrides=None):
        super().__init__(task=task, config=default_config() if config is None else config,
                         config_overrides=config_overrides)

    def sample_command(self, rng):
        inherited, category, magnitude, direction = jax.random.split(rng, 4)
        base_command = super().sample_command(inherited)
        draw = jax.random.uniform(category)
        speed = jax.random.uniform(magnitude, minval=self._config.turn_yaw_speed[0],
                                   maxval=self._config.turn_yaw_speed[1])
        yaw = jp.where(jax.random.bernoulli(direction), speed, -speed)
        turn_command = jp.zeros_like(base_command).at[2].set(yaw)
        # Keep the parent's 10% stand events and draw turn events from the
        # remaining walking events. Thus 20% of all commands are turns.
        turn = (base_command[0] > 0) & (draw < self._config.turn_probability / .9)
        return jp.where(turn, turn_command, base_command)
