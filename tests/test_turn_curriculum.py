"""The first curriculum must exercise both turning directions without mixing axes."""
import unittest

import jax
import numpy as np
from playground.open_duck_mini_v2 import locomotion_neck, locomotion_turn


class TurnCurriculumTest(unittest.TestCase):
    def test_command_coverage(self):
        config = locomotion_turn.default_config()
        env = object.__new__(locomotion_turn.ConservativeTurn)
        env._config = config
        keys = jax.random.split(jax.random.PRNGKey(61), 4096)
        commands = np.asarray(jax.jit(jax.vmap(env.sample_command))(keys))
        forward, yaw = commands[:, 0], commands[:, 2]
        np.testing.assert_array_equal(commands[:, [1, 3, 4, 5, 6]], 0)
        self.assertTrue(np.all((forward == 0) | (yaw == 0)))
        self.assertTrue(np.all((forward == 0) | ((forward >= .05) & (forward <= .15))))
        self.assertTrue(np.all((yaw == 0) | ((abs(yaw) >= .15) & (abs(yaw) <= .25))))
        for mask, expected in [((forward == 0) & (yaw == 0), .1), (yaw > 0, .1),
                               (yaw < 0, .1), (forward > 0, .7)]:
            self.assertAlmostEqual(mask.mean(), expected, delta=.03)

    def test_parent_settings_are_retained(self):
        config = locomotion_turn.default_config().to_dict()
        config.pop('turn_yaw_speed')
        config.pop('turn_probability')
        self.assertEqual(config, locomotion_neck.default_config().to_dict())


if __name__ == '__main__':
    unittest.main()
