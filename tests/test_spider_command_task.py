from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lessons"))
from spider_command_task import advance_command, body_velocity, command_authority, heading_error, observation


class SpiderCommandTaskTest(unittest.TestCase):
    def test_body_velocity_is_independent_of_world_heading(self):
        np.testing.assert_allclose(body_velocity(np.array([0., .001]), np.pi / 2, .02, np), [.05, 0.], atol=1e-8)

    def test_heading_error_wraps_at_pi(self):
        self.assertAlmostEqual(heading_error(-np.pi + .02, np.pi - .03, np), .05)

    def test_command_rate_limit_and_stop(self):
        np.testing.assert_allclose(advance_command(np.zeros(3), np.array([.03, -.01, .25]), .02, np),
                                   [.0016, -.0012, .024])
        self.assertEqual(command_authority(np.zeros(3), np), 0.)
        self.assertEqual(command_authority(np.array([0., -.012, 0.]), np), 1.)

    def test_observation_adds_body_commands_without_moving_legacy_vx_slots(self):
        base = np.r_[1., np.zeros(51)]
        obs = observation(base, 0, np.array([.034, -.012, .25]), np.zeros(3), 0, np)
        self.assertEqual(obs.shape, (61,))
        np.testing.assert_allclose(obs[53:], [1, 0, -1, 1, 0, 0, 0, 0])


if __name__ == "__main__":
    unittest.main()
