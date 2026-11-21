"""Prevent time-limit success being labeled a fall or post-fall motion being scored."""
import importlib.util
import json
from pathlib import Path
import unittest
import numpy as np

base=Path(__file__).resolve().parents[1]
path=base/'evaluate_locomotion.py'
if not path.exists():path=base/'lessons/evaluate_locomotion.py'
evaluation=None
if path.exists():
    spec=importlib.util.spec_from_file_location('evaluation_metrics',path)
    evaluation=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluation)


class PhysicalMetrics(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(evaluation,'physical evaluator missing')

    def fixture(self, done, velocity):
        n=len(done)
        return dict(done=np.array(done),velocity=np.array(velocity),gyro=np.zeros((n,3)),
            up=np.tile([0.,0.,1.],(n,1)),position=np.zeros((n,3)),
            feet_velocity=np.zeros((n,2,3)),contact=np.ones((n,2),bool))

    def test_time_limit_is_not_failure_and_matching_speed_has_zero_error(self):
        trace=self.fixture([False,False,False],[[.1,0,0]]*3)
        result=evaluation.summarize_episode(trace,[.1,0,0],dt=.02,settle_seconds=0)
        self.assertFalse(result['terminated'])
        self.assertAlmostEqual(result['duration_s'],.06)
        self.assertAlmostEqual(result['rms_planar_error'],0.)

    def test_post_failure_values_do_not_contaminate_tracking(self):
        trace=self.fixture([False,True,True],[[.1,0,0],[.1,0,0],[9.,0,0]])
        result=evaluation.summarize_episode(trace,[.1,0,0],dt=.02,settle_seconds=0)
        self.assertTrue(result['terminated'])
        self.assertEqual(result['steps'],2)
        self.assertAlmostEqual(result['mean_body_velocity'][0],.1)

    def test_nonfinite_failure_remains_serializable_and_no_contact_is_unavailable(self):
        trace=self.fixture([True],[[float('nan'),0,0]])
        trace['position'][:]=np.nan;trace['contact'][:]=False
        result=evaluation.summarize_episode(trace,[.1,0,0],settle_seconds=0)
        self.assertFalse(result['finite'])
        self.assertIsNone(result['contact_slip_speed'])
        self.assertIsNone(result['world_xy_displacement_m'])
        json.dumps(result,allow_nan=False)

    def test_surviving_while_ignoring_walk_command_is_not_task_success(self):
        self.assertTrue(hasattr(evaluation,'basic_task_pass'),'command-compliance acceptance is missing')
        episode=dict(case='forward',terminated=False,finite=True,mean_body_velocity=[0.,0.,0.],rms_planar_error=.1,rms_yaw_error=0.,world_xy_displacement_m=[0.,0.])
        self.assertFalse(evaluation.basic_task_pass(episode))
        good=dict(episode,mean_body_velocity=[.1,0.,0.],rms_planar_error=.02)
        self.assertTrue(evaluation.basic_task_pass(good))
        self.assertFalse(evaluation.basic_task_pass(dict(good,terminated=True)))

    def test_normal_zero_mean_gait_sway_is_separate_from_average_tracking_error(self):
        n=270;t=np.arange(n)
        trace=self.fixture(np.zeros(n,bool),np.column_stack([np.ones(n)*.1,.14*np.sin(2*np.pi*t/27),np.zeros(n)]))
        result=evaluation.summarize_episode(trace,[.1,0,0],dt=.02,settle_seconds=0)
        self.assertIn('cycle_rms_planar_error',result,'gait-cycle average metric missing')
        self.assertGreater(result['rms_planar_error'],.09)
        self.assertLess(result['cycle_rms_planar_error'],1e-6)
        self.assertTrue(evaluation.basic_task_pass(dict(case='forward',**result)))

    def test_acceptance_uses_requested_speed_not_always_point_one(self):
        trace=self.fixture(np.zeros(100,bool),[[.05,0.,0.]]*100)
        result=evaluation.summarize_episode(trace,[.05,0,0],settle_seconds=0)
        self.assertTrue(evaluation.basic_task_pass(dict(case='forward',command=[.05,0,0],**result)))

    def test_half_speed_does_not_pass_low_speed_command(self):
        trace=self.fixture(np.zeros(100,bool),[[.025,0.,0.]]*100)
        result=evaluation.summarize_episode(trace,[.05,0,0],settle_seconds=0)
        self.assertFalse(evaluation.basic_task_pass(dict(case='forward',command=[.05,0,0],**result)))


if __name__=='__main__':unittest.main()
