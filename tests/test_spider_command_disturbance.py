from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lessons"))
from spider_command_disturbance import push_history, sensor_observation


class SpiderCommandDisturbanceTest(unittest.TestCase):
    def test_fifo_has_exact_control_tick_delays(self):
        for delay in range(3):
            history = np.full((3, 1), -1.)
            delivered = []
            for value in range(4):
                history, selected = push_history(history, np.array([value]), delay, np)
                delivered.append(selected.item())
            self.assertEqual(delivered, [-1] * delay + list(range(4 - delay)))

    def test_delay_moves_sensors_and_preserves_command_clock(self):
        current = np.r_[1., np.zeros(49), np.arange(11.)]
        delayed = current.copy()
        delayed[6:10] = -2.
        delayed[42] = 1.
        delayed[59:61] = [.0, .0]
        noise = np.zeros_like(current)
        noise[7:10] = 1.
        result = sensor_observation(current, delayed, noise, np)
        np.testing.assert_allclose(result[:4], [1., 0., 0., 0.])
        np.testing.assert_allclose(result[7:10], [-1.982, -1.982, -1.982])
        self.assertEqual(result[42], 1.)
        np.testing.assert_array_equal(result[50:59], current[50:59])
        np.testing.assert_array_equal(result[59:61], [0., 0.])


if __name__ == "__main__":
    unittest.main()
