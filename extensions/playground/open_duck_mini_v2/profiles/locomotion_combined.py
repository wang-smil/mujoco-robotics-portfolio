"""Combined command distribution with the validated reverse reference cadence."""
import jax
import jax.numpy as jp

from playground.open_duck_mini_v2 import locomotion_arc_switches


def default_config():
    config = locomotion_arc_switches.default_config()
    config.reverse_probability = .15
    config.reverse_speed = [.045, .055]
    config.reverse_cadence = 1.1447
    return config


class CombinedLocomotion(locomotion_arc_switches.ArcSwitches):
    def __init__(self, task='flat_terrain', config=None, config_overrides=None):
        super().__init__(task=task, config=default_config() if config is None else config,
                         config_overrides=config_overrides)

    def sample_command(self, rng):
        inherited, selector, magnitude = jax.random.split(rng, 3)
        command = super().sample_command(inherited)
        forward_probability = .9 - self._config.turn_probability - self._config.arc_probability
        straight = (command[0] > 0) & (command[2] == 0.)
        use_reverse = straight & (jax.random.uniform(selector) < self._config.reverse_probability / forward_probability)
        speed = jax.random.uniform(magnitude, minval=self._config.reverse_speed[0], maxval=self._config.reverse_speed[1])
        return command.at[0].set(jp.where(use_reverse, -speed, command[0]))

    def reset(self, rng):
        state = super().reset(rng)
        phase = jp.asarray(state.info['imitation_i'], dtype=jp.float32)
        snapshot = {**state.info['_episode_reset'], 'imitation_i': phase}
        return state.replace(info={**state.info, 'imitation_i': phase, '_episode_reset': snapshot})

    def step(self, state, action):
        # Restore wrapper-owned task history before changing its phase.
        from playground.open_duck_mini_v2.locomotion_gait import coherent_task_info
        info = coherent_task_info(state.info, state.obs['state'][-2:])
        reverse = info['command'][0] < -.01
        info['imitation_i'] += jp.where(reverse, self._config.reverse_cadence - 1., 0.)
        return super().step(state.replace(info=info), action)
