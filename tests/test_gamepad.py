import importlib.util
from pathlib import Path
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
path=ROOT/'08_duck_gamepad.py'
if not path.exists():path=ROOT/'lessons/08_duck_gamepad.py'
game=None
if path.exists():
    spec=importlib.util.spec_from_file_location('duck_gamepad',path)
    game=importlib.util.module_from_spec(spec);spec.loader.exec_module(game)


class GamepadBehavior(unittest.TestCase):
    def setUp(self):self.assertIsNotNone(game,'virtual gamepad missing')

    def test_forward_stop_and_pause_have_distinct_meanings(self):
        state=game.ControlState(timeout=2.)
        state.apply('forward',speed=.1,now=1.)
        s=state.snapshot(now=1.1)
        self.assertEqual(s['target'],[.1,0.,0.]);self.assertFalse(s['paused'])
        state.apply('stop',now=1.2);s=state.snapshot(now=1.2)
        self.assertEqual(s['target'],[0.,0.,0.]);self.assertFalse(s['paused'])
        state.apply('pause',now=1.3);self.assertTrue(state.snapshot(now=1.3)['paused'])

    def test_untrained_commands_are_available_and_identified(self):
        state=game.ControlState(timeout=2.)
        state.apply('forward_left',speed=.1,now=1.)
        s=state.snapshot(now=1.)
        self.assertEqual(s['target'],[.1,0.,.3]);self.assertTrue(s['experimental'])
        state.apply('backward',speed=.15,now=1.)
        self.assertEqual(state.snapshot(now=1.)['target'],[-.1,0.,0.])
        with self.assertRaises(ValueError):state.apply('run_shell',now=1.)
        with self.assertRaises(ValueError):state.apply('forward',speed=float('nan'),now=1.)

    def test_disconnect_pauses_and_clears_latched_movement(self):
        state=game.ControlState(timeout=1.)
        state.apply('forward',speed=.1,now=2.)
        state.heartbeat(now=2.5)
        self.assertFalse(state.snapshot(now=3.)['paused'])
        s=state.snapshot(now=3.6)
        self.assertTrue(s['paused']);self.assertEqual(s['target'],[0.,0.,0.])

    def test_failure_requires_reset_and_reset_stops_before_restarting(self):
        state=game.ControlState(timeout=2.)
        state.fail('fell')
        state.apply('forward',now=1.)
        self.assertTrue(state.snapshot(now=1.)['paused'])
        state.apply('reset',now=1.)
        s=state.snapshot(now=1.)
        self.assertFalse(s['failed']);self.assertTrue(s['paused']);self.assertEqual(s['reset_id'],1)
        state.apply('forward',now=1.)
        self.assertFalse(state.snapshot(now=1.)['paused'])

    def test_old_terminal_result_cannot_cancel_a_new_reset(self):
        state=game.ControlState(timeout=2.)
        before=state.snapshot(now=1.)['reset_id']
        state.apply('reset',now=1.)
        self.assertIn('reset_id',__import__('inspect').signature(state.fail).parameters,'terminal events have no generation check')
        state.fail('old step fell',reset_id=before)
        state.apply('forward',now=1.)
        self.assertFalse(state.snapshot(now=1.)['failed'])
        self.assertFalse(state.snapshot(now=1.)['paused'])

    def test_command_ramp_is_bounded_without_overshoot(self):
        np.testing.assert_allclose(game.ramp_command([0,0,0],[.1,0,.3],.02),[.006,0,.02])
        np.testing.assert_allclose(game.ramp_command([.099,0,.299],[.1,0,.3],.02),[.1,0,.3])

    def test_live_command_patch_preserves_sensors_rng_and_phase(self):
        import jax.numpy as jp
        from types import SimpleNamespace
        class Reference:
            def get_reference_motion(self,x,y,w,i):return jp.ones(40)*(x+i)
        actor=jp.arange(101,dtype=float);critic=jp.arange(212,dtype=float)
        info=dict(imitation_i=jp.array(7),step=jp.array(500),rng=jp.array([11,22]),imitation_phase=jp.array([.2,.8]),command=jp.zeros(7),current_reference_motion=jp.zeros(40))
        state=SimpleNamespace(info=info,obs=dict(state=actor,privileged_state=critic))
        state.replace=lambda **kw:SimpleNamespace(**{**vars(state),**kw})
        changed=game.patch_live_command(state,jp.array([.1,0.,.3]),Reference())
        np.testing.assert_allclose(changed.obs['state'][6:13],[.1,0,.3,0,0,0,0])
        np.testing.assert_array_equal(changed.obs['state'][:6],actor[:6])
        np.testing.assert_array_equal(changed.info['rng'],[11,22])
        np.testing.assert_allclose(changed.info['imitation_phase'],[.2,.8])
        self.assertEqual(int(changed.info['step']),0)
        self.assertEqual(int(info['step']),500)
        np.testing.assert_allclose(changed.obs['privileged_state'][-43:-3],7.1,atol=1e-6)


if __name__=='__main__':unittest.main()
