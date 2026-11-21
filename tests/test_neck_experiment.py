"""Real behavioral contracts for rate limits and fair multi-seed comparison."""
from pathlib import Path
import importlib.util
import inspect
import sys
import unittest
import numpy as np
import json

ROOT=Path(__file__).resolve().parents[1]
def load(name, filename):
    p=ROOT/filename
    if not p.exists():p=ROOT/'lessons'/filename
    spec=importlib.util.spec_from_file_location(name,p)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

entry=load('neck_entry','07_duck_ppo.py')
evaluation=load('neck_evaluation','evaluate_locomotion.py')
sys.path.insert(0,str(ROOT))
try:
    import locomotion_neck as neck
except ImportError:
    try:from playground.open_duck_mini_v2 import locomotion_neck as neck
    except ImportError:neck=None


class NeckExperiment(unittest.TestCase):
    def test_head_bound_does_not_slow_leg_targets(self):
        self.assertIsNotNone(neck,'neck limiter missing')
        limits=np.asarray(neck.motor_velocity_limits(6.,.5))
        before=np.zeros(14);requested=np.ones(14)
        after=np.clip(requested,before-limits*.02,before+limits*.02)
        np.testing.assert_allclose(after,[.12,.12,.12,.12,.12,.01,.01,.01,.01,.12,.12,.12,.12,.12])
        after=np.clip(-requested,before-limits*.02,before+limits*.02)
        np.testing.assert_allclose(after[5:9],[-.01]*4)

    def test_saved_vector_keeps_existing_leg_specific_bounds(self):
        self.assertIsNotNone(neck,'neck limiter missing')
        original=np.arange(1,15,dtype=float)
        limits=np.asarray(neck.motor_velocity_limits(original,.25))
        np.testing.assert_allclose(limits,[1,2,3,4,5,.25,.25,.25,.25,10,11,12,13,14])
        np.testing.assert_array_equal(original,np.arange(1,15))

    def test_invalid_speed_and_wrong_actuator_count_fail_before_simulation(self):
        self.assertIsNotNone(neck,'neck limiter missing')
        for base,head in [(5.,0.),(5.,-1.),(5.,np.nan),(np.inf,.5),([1.,2.],.5)]:
            with self.subTest(base=base,head=head),self.assertRaises(ValueError):
                neck.motor_velocity_limits(base,head)

    def test_config_overrides_change_executed_limit_and_cannot_disable_it(self):
        self.assertTrue(hasattr(neck,'configured_limits'),'rate overrides not resolved before validation')
        original=neck.default_config()
        cfg=neck.configured_limits(original,{'head_max_motor_velocity':.2,'max_motor_velocity':4.})
        np.testing.assert_allclose(np.asarray(cfg.max_motor_velocity)[5:9],[.2]*4)
        self.assertAlmostEqual(float(cfg.max_motor_velocity[0]),4.)
        self.assertEqual(original.head_max_motor_velocity,.5)
        with self.assertRaises(ValueError):neck.configured_limits(original,{'head_max_motor_velocity':0.})

    def test_8192_is_accepted_and_seed_reaches_ppo_settings(self):
        try:args=entry.parse_args(['--train','--envs','8192','--steps','100000000','--seed','7','--task-profile','neck_forward'])
        except SystemExit:self.fail('8192/seed/neck_forward CLI unavailable')
        self.assertIn('seed',inspect.signature(entry.ppo_settings).parameters)
        settings=entry.ppo_settings(args.envs,preset='official',seed=args.seed)
        self.assertEqual(settings['seed'],7)
        self.assertEqual(settings['num_envs'],8192)
        self.assertEqual(settings['batch_size']*settings['num_minibatches'],8192)

    def test_transfer_preset_is_explicit_and_keeps_predecessor_network_compatible(self):
        try:args=entry.parse_args(['--train','--preset','transfer'])
        except SystemExit:self.fail('explicit transfer preset missing')
        self.assertEqual(entry.network_of(args.preset),entry.network_of('official'))

    def test_comparison_restores_each_saved_controller_and_disturbance_override_is_detached(self):
        self.assertTrue(hasattr(evaluation,'evaluation_environment_config'),'comparison environment resolver missing')
        new={'task_profile':'neck_forward','environment':{'reset_pose_scale':.2,'noise_config':{'level':.3},'push_config':{'enable':False}}}
        old={'task_profile':'gait_forward','environment':{'reset_pose_scale':.2,'noise_config':{'level':.3},'push_config':{'enable':False}}}
        cfg=evaluation.evaluation_environment_config(new,old,True,False)
        self.assertEqual(cfg['task_profile'],'gait_forward')
        self.assertEqual(evaluation.evaluation_environment_config(new,old,False,False)['task_profile'],'neck_forward')
        cfg=evaluation.evaluation_environment_config(new,old,True,True)
        self.assertEqual(cfg['environment']['noise_config']['level'],1.)
        self.assertTrue(cfg['environment']['push_config']['enable'])
        self.assertFalse(old['environment']['push_config']['enable'])

    def test_head_metrics_distinguish_target_rate_from_actual_joint_speed_and_world_rotation(self):
        self.assertTrue(hasattr(evaluation,'head_metrics'),'head measurements missing')
        n=4;dt=.02
        qpos=np.zeros((n,21));qvel=np.zeros((n,20));ctrl=np.zeros((n,14))
        qpos[:,12]=[0,.01,.02,.03];qvel[:,11:15]=2.
        ctrl[:,5:9]=np.arange(n)[:,None]*.01
        angles=np.arange(n)*dt
        quat=np.column_stack([np.cos(angles/2),np.zeros(n),np.zeros(n),np.sin(angles/2)])
        metrics=evaluation.head_metrics(dict(qpos=qpos,qvel=qvel,ctrl=ctrl,head_quaternion=quat),dt)
        self.assertAlmostEqual(metrics['target_rate_max_rad_s'],.5)
        self.assertAlmostEqual(metrics['joint_velocity_rms_rad_s'],2.)
        self.assertAlmostEqual(metrics['world_angular_speed_rms_rad_s'],1.,places=5)
        self.assertAlmostEqual(metrics['joint_range_deg'][0],.03*180/np.pi)
        # A quaternion sign change represents the same orientation.
        quat[1::2]*=-1
        self.assertAlmostEqual(evaluation.head_metrics(dict(qpos=qpos,qvel=qvel,ctrl=ctrl,head_quaternion=quat),dt)['world_angular_speed_rms_rad_s'],1.,places=5)

    def test_bad_head_measurement_remains_serializable(self):
        trace=dict(qpos=np.full((2,21),np.nan),qvel=np.full((2,20),np.nan),ctrl=np.zeros((2,14)),head_quaternion=np.zeros((2,4)))
        metrics=evaluation.head_metrics(trace,.02)
        self.assertIn('measurement_finite',metrics,'missing invalid-measurement status')
        self.assertFalse(metrics['measurement_finite'])
        self.assertIsNone(metrics['joint_velocity_rms_rad_s'])
        self.assertIsNone(metrics['world_angular_speed_rms_rad_s'])
        json.dumps(metrics,allow_nan=False)

    def test_target_bound_counts_reset_jump_and_initial_transient(self):
        self.assertTrue(hasattr(evaluation,'whole_target_rate'),'whole-episode bound missing')
        ctrl=np.zeros((100,14));ctrl[:,5:9]=.01;ctrl[0,5:9]=.1
        result=evaluation.whole_target_rate(ctrl,np.zeros(14),.02)
        self.assertAlmostEqual(result,5.)

    def test_numerical_rollback_is_not_mislabeled_as_a_finite_physical_fall(self):
        n=2
        trace=dict(done=np.array([False,True]),numerical_failure=np.array([False,True]),velocity=np.zeros((n,3)),gyro=np.zeros((n,3)),up=np.tile([0.,0.,1.],(n,1)),position=np.zeros((n,3)),feet_velocity=np.zeros((n,2,3)),contact=np.ones((n,2),bool))
        result=evaluation.summarize_episode(trace,[.1,0,0],settle_seconds=0)
        self.assertFalse(result['finite'],'rollback hid numerical failure')

    def test_batching_preserves_every_case_order_and_discards_padding(self):
        self.assertTrue(hasattr(evaluation,'run_batched_rollout'),'bounded evaluation batching missing')
        # This toy rollout encodes each independent key/command pair directly.
        # A wrong slice, pairing or merge produces an observable wrong record.
        def rollout(keys,commands):
            value=np.asarray(keys)+np.asarray(commands)[:,0]
            return {'value':np.stack([value,value+100])},np.asarray(keys)[:,None]
        keys=np.arange(6);commands=np.array([[10.],[20.],[30.],[40.],[50.],[60.]])
        trace,initial=evaluation.run_batched_rollout(rollout,keys,commands,4)
        np.testing.assert_array_equal(trace['value'],[[10,21,32,43,54,65],[110,121,132,143,154,165]])
        np.testing.assert_array_equal(initial[:,0],[0,1,2,3,4,5])

    def test_bad_batch_size_and_empty_case_list_are_rejected(self):
        self.assertTrue(hasattr(evaluation,'run_batched_rollout'),'bounded evaluation batching missing')
        with self.assertRaises(ValueError):evaluation.run_batched_rollout(None,np.arange(2),np.zeros((2,1)),0)
        with self.assertRaises(ValueError):evaluation.run_batched_rollout(None,np.array([]),np.zeros((0,1)),4)


if __name__=='__main__':unittest.main()
