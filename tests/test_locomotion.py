"""Behavioral regressions: wrong-direction rewards, stand gating and hidden fall cost."""
import sys
import unittest
from pathlib import Path

import jax.numpy as jp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import locomotion
except ImportError:
    try:
        from playground.open_duck_mini_v2 import locomotion
    except ImportError:
        locomotion = None


class RewardBehavior(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(locomotion, 'locomotion reward implementation is missing')

    def test_matching_lateral_command_outscores_standing_and_wrong_direction(self):
        command = jp.array([0., .1, 0.])
        still, _ = locomotion.tracking_rewards(command, jp.zeros(3), jp.zeros(3))
        matching, _ = locomotion.tracking_rewards(command, jp.array([0., .1, 0.]), jp.zeros(3))
        wrong, _ = locomotion.tracking_rewards(command, jp.array([0., -.1, 0.]), jp.zeros(3))
        self.assertAlmostEqual(float(matching), 1.)
        self.assertLess(float(still), .03)
        self.assertLess(float(wrong), float(still))

    def test_yaw_must_match_direction(self):
        _, good = locomotion.tracking_rewards(jp.array([0.,0.,.5]), jp.zeros(3), jp.array([0.,0.,.5]))
        _, wrong = locomotion.tracking_rewards(jp.array([0.,0.,.5]), jp.zeros(3), jp.array([0.,0.,-.5]))
        self.assertAlmostEqual(float(good), 1.)
        self.assertLess(float(wrong), .001)

    def test_standing_pose_and_motion_costs_only_under_zero_twist(self):
        args = dict(velocity=jp.array([.1,0.,0.]), gyro=jp.array([0.,0.,.3]), up=jp.array([0.,0.,1.]), pose_error=jp.ones(10)*.2, foot_velocity=jp.zeros((2,3)), contact=jp.array([True,True]))
        still = locomotion.stability_costs(jp.zeros(3), **args)
        moving = locomotion.stability_costs(jp.array([.1,0,0]), **args)
        turning = locomotion.stability_costs(jp.array([0,0,.5]), **args)
        self.assertAlmostEqual(float(still['stand_pose']), .04, places=6)
        self.assertAlmostEqual(float(still['stand_motion']), .10, places=6)
        for result in (moving, turning):
            self.assertEqual(float(result['stand_pose']), 0.)
            self.assertEqual(float(result['stand_motion']), 0.)

    def test_contact_slip_excludes_swing_foot_and_upright_cost_increases(self):
        args = dict(command=jp.zeros(3), velocity=jp.zeros(3), gyro=jp.zeros(3), pose_error=jp.zeros(10), foot_velocity=jp.array([[1.,0,0],[2.,0,0]]), contact=jp.array([True,False]))
        straight = locomotion.stability_costs(up=jp.array([0.,0.,1.]), **args)
        tilted = locomotion.stability_costs(up=jp.array([.8,0.,.6]), **args)
        self.assertAlmostEqual(float(straight['feet_slip']), 1.)
        self.assertEqual(float(straight['orientation']), 0.)
        self.assertGreater(float(tilted['orientation']), .15)

    def test_failure_cost_stays_negative(self):
        self.assertAlmostEqual(float(locomotion.signed_reward({'alive':1., 'termination':1.}, {'alive':.5, 'termination':-10.}, .02)), -.19, places=6)

    def test_failure_before_upside_down_and_nonfinite_state(self):
        q, v = jp.zeros(21), jp.zeros(20)
        self.assertFalse(bool(locomotion.is_failure(1., .16, q, v)))
        self.assertTrue(bool(locomotion.is_failure(.49, .16, q, v)))
        self.assertTrue(bool(locomotion.is_failure(1., .09, q, v)))
        self.assertTrue(bool(locomotion.is_failure(1., .16, q, v.at[0].set(jp.nan))))

    def test_command_patch_preserves_sensor_samples_and_updates_critic_reference(self):
        self.assertTrue(hasattr(locomotion,'patch_command_observation'),'pure observation patch missing')
        actor=jp.arange(101,dtype=float)
        critic=jp.concatenate([actor,jp.arange(111,dtype=float)])
        obs=dict(state=actor,privileged_state=critic)
        command=jp.array([.1,0,0,0,0,0,0]); reference=jp.ones(40)*.25
        patched=locomotion.patch_command_observation(obs,command,reference)
        np.testing.assert_array_equal(patched['state'][:6],actor[:6])
        np.testing.assert_array_equal(patched['state'][13:],actor[13:])
        np.testing.assert_array_equal(patched['state'][6:13],command)
        np.testing.assert_array_equal(patched['privileged_state'][:101],patched['state'])
        np.testing.assert_array_equal(patched['privileged_state'][-43:-3],reference)
        np.testing.assert_array_equal(obs['state'],actor)

    def test_nonfinite_terms_become_finite_failure_only(self):
        self.assertTrue(hasattr(locomotion,'sanitize_terms'),'finite terminal reward sanitizer missing')
        terms,invalid=locomotion.sanitize_terms({'tracking_lin_vel':jp.nan,'termination':0.},False)
        self.assertTrue(bool(invalid))
        self.assertEqual(float(terms['tracking_lin_vel']),0.)
        self.assertEqual(float(terms['termination']),1.)


if __name__ == '__main__':
    unittest.main()
