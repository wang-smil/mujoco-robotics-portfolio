from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lessons"))
from spider_speed_command import forward_speed, scale_action, slew_speed


class SpiderSpeedCommandTest(unittest.TestCase):
    def test_forward_baseline_stop_and_reverse(self):
        action = np.array([0.3, -0.2, -0.1, 0.4])
        hips = np.array([True, False, True, False])
        np.testing.assert_allclose(scale_action(action, hips, .034, np), action)
        np.testing.assert_allclose(scale_action(action, hips, 0, np), np.zeros(4))
        np.testing.assert_allclose(scale_action(action, hips, -.017, np), [.15, .1, -.05, -.2])
        np.testing.assert_allclose(scale_action(action, hips, .051, np), [.3, -.3, -.1, .6])

    def test_slew_respects_rate_and_exact_target(self):
        self.assertAlmostEqual(slew_speed(.034, -.024, .02, np), .0324)
        self.assertAlmostEqual(slew_speed(.001, 0, .02, np), 0)

    def test_forward_speed_projects_onto_current_heading(self):
        self.assertAlmostEqual(forward_speed(np.array([.0002, -.0006]), np.array([0, .9]), .02, np), -.03)


if __name__ == "__main__":
    unittest.main()
