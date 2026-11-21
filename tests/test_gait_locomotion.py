"""Regressions for reference-grid bias and penalizing normal cyclic COM sway."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import jax
import jax.numpy as jp
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
try:
    import locomotion_gait as gait
except ImportError:
    try:from playground.open_duck_mini_v2 import locomotion_gait as gait
    except ImportError:gait=None


class GaitTracking(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(gait,'gait-aware implementation missing')

    def source(self):
        axis=[-1.,1.]
        coefficients=np.zeros((2,2,2,40,2),np.float32)
        for x in range(2):
            for y in range(2):
                for z in range(2):
                    coefficients[x,y,z,:,0]=5.
                    coefficients[x,y,z,:,1]=2*axis[x]-3*axis[y]+4*axis[z]
        return SimpleNamespace(dxs=axis,dys=axis,dthetas=axis,nb_steps_in_period=4,fps=50.,period=.08,
            data_array=jp.array(coefficients),sample_polynomial=lambda t,c:jax.vmap(lambda row:jp.polyval(row,t))(c))

    def test_interpolation_reproduces_linear_motion_and_zero_between_grid_nodes(self):
        reference=gait.InterpolatedReference(self.source())
        result=reference.get_reference_motion(.2,.3,.4,1)
        np.testing.assert_allclose(result,2.35,atol=2e-6)
        np.testing.assert_allclose(reference.get_reference_motion(0.,0.,0.,0),0.,atol=1e-6)
        np.testing.assert_allclose(reference.get_reference_motion(.2,.3,.4,5),result,atol=1e-6)

    def test_filter_keeps_mean_and_reduces_zero_mean_gait_sway(self):
        filtered=jp.zeros(3)
        raw_errors=[];filtered_errors=[]
        for i in range(270):
            twist=jp.array([.1,.14*np.sin(2*np.pi*i/27),0.])
            filtered=gait.filter_twist(filtered,twist,.02,.2)
            if i>=54:
                raw_errors.append(float(twist[1]**2))
                filtered_errors.append(float(filtered[1]**2))
        self.assertAlmostEqual(float(filtered[0]),.1,places=5)
        self.assertLess(np.mean(filtered_errors),.25*np.mean(raw_errors))
        biased=jp.zeros(3)
        for _ in range(200):biased=gait.filter_twist(biased,jp.array([0.,.04,0.]),.02,.2)
        self.assertAlmostEqual(float(biased[1]),.04,places=5)

    def test_filtered_critic_fields_do_not_change_actor_or_reference(self):
        actor=jp.arange(101,dtype=float);critic=jp.arange(212,dtype=float)
        obs=dict(state=actor,privileged_state=critic)
        patched=gait.patch_filtered_observation(obs,jp.array([.1,.2,.3]))
        np.testing.assert_array_equal(patched['state'],actor)
        np.testing.assert_allclose(patched['privileged_state'][167:169],[.1,.2])
        self.assertAlmostEqual(float(patched['privileged_state'][209]),.3,places=6)
        np.testing.assert_array_equal(patched['privileged_state'][169:209],critic[169:209])

    def test_zero_command_does_not_hide_instantaneous_wobble_in_filter(self):
        self.assertTrue(hasattr(gait,'tracking_twist'),'zero-command raw twist gate missing')
        measurement=jp.array([0.,.1,.2]);filtered=jp.zeros(3)
        np.testing.assert_allclose(gait.tracking_twist(jp.zeros(3),filtered,measurement),measurement)
        np.testing.assert_allclose(gait.tracking_twist(jp.array([.1,0,0]),filtered,measurement),filtered)

    def test_auto_reset_restores_task_history_but_preserves_rng_and_wrapper_metrics(self):
        self.assertTrue(hasattr(gait,'coherent_task_info'),'auto-reset task-history synchronization missing')
        initial=dict(imitation_phase=jp.zeros(2),command=jp.zeros(7),filtered_twist=jp.zeros(3),rng=jp.array([1,2]))
        info=dict(imitation_phase=jp.array([1.,0.]),command=jp.ones(7),filtered_twist=jp.ones(3),rng=jp.array([3,4]),
                  episode_metrics={'reward':jp.array(12.)},_episode_reset=initial)
        fixed=gait.coherent_task_info(info,jp.zeros(2))
        np.testing.assert_array_equal(fixed['command'],jp.zeros(7))
        np.testing.assert_array_equal(fixed['filtered_twist'],jp.zeros(3))
        np.testing.assert_array_equal(fixed['rng'],jp.array([3,4]))
        self.assertEqual(float(fixed['episode_metrics']['reward']),12.)
        continuing=gait.coherent_task_info(info,jp.array([1.,0.]))
        np.testing.assert_array_equal(continuing['filtered_twist'],jp.ones(3))


if __name__=='__main__':unittest.main()
