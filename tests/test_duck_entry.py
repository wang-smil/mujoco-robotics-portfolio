"""Catch silent replay drift, path escapes and fixed-command CLI errors."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

base = Path(__file__).resolve().parents[1]
source = base / '07_duck_ppo.py'
if not source.is_file():
    source = base / 'lessons/07_duck_ppo.py'
spec = importlib.util.spec_from_file_location('duck_entry', source)
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


class EntryBehavior(unittest.TestCase):
    def test_saved_environment_is_restored_without_mutating_original(self):
        config = {'environment': {'ctrl_dt':.02,'noise_config':{'level':.17}}, 'task_profile':'forward', 'task':'flat_terrain'}
        self.assertTrue(hasattr(entry, 'environment_settings'), 'saved-environment resolver is missing')
        profile, task, cfg = entry.environment_settings(config)
        self.assertEqual(profile, 'forward')
        self.assertEqual(cfg['noise_config']['level'], .17)
        cfg['noise_config']['level'] = 1.
        self.assertEqual(config['environment']['noise_config']['level'], .17)

    def test_legacy_profile_is_upstream_and_unknown_profile_rejected(self):
        self.assertTrue(hasattr(entry, 'environment_settings'), 'saved-environment resolver is missing')
        self.assertEqual(entry.environment_settings({'environment':{'ctrl_dt':.02}})[0], 'upstream')
        with self.assertRaises(ValueError):
            entry.environment_settings({'task_profile':'typo', 'environment':{}})

    def test_fixed_command_and_checkpoint_are_accepted(self):
        args = entry.parse_args(['--view','--command','.1','0','0','--checkpoint','step_20'])
        self.assertEqual(args.command, [.1,0.,0.])
        self.assertEqual(args.checkpoint, 'step_20')
        with self.assertRaises(SystemExit):
            entry.parse_args(['--view','--command','nan','0','0'])

    def test_changed_reward_source_is_rejected(self):
        self.assertTrue(hasattr(entry, 'validate_sources'), 'source validation is missing')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            source=root/'reward.py'; source.write_text('v1')
            recorded={'source_sha256':{'reward.py':hashlib.sha256(b'v1').hexdigest()}}
            entry.validate_sources(recorded, root)
            source.write_text('v2')
            with self.assertRaises(RuntimeError):
                entry.validate_sources(recorded, root)

    def test_versioned_profile_requires_complete_manifest(self):
        with self.assertRaises(RuntimeError):
            entry.validate_sources({'task_profile':'forward'})
        with self.assertRaises(RuntimeError):
            entry.validate_sources({'task_profile':'forward','source_sha256':{'not-enough.py':'abc'}})

    def test_warm_start_preserves_lineage_and_rejects_different_network(self):
        self.assertTrue(hasattr(entry,'warm_start_metadata'),'checkpoint warm-start resolver missing')
        with tempfile.TemporaryDirectory() as directory:
            run=Path(directory)
            checkpoint=run/'checkpoints/step_20000000';checkpoint.mkdir(parents=True)
            network=dict(policy_hidden=[64,64],value_hidden=[64,64],policy_obs_key='state',value_obs_key='privileged_state')
            (run/'config.json').write_text(json.dumps({'network':network,'prior_environment_steps':10000000}))
            metadata=entry.warm_start_metadata(checkpoint,network)
            self.assertEqual(metadata['prior_environment_steps'],30000000)
            self.assertFalse(metadata['optimizer_state_restored'])
            with self.assertRaises(ValueError):
                entry.warm_start_metadata(checkpoint,dict(network,policy_hidden=[32,32]))

    def test_warm_start_only_allowed_for_training(self):
        args=entry.parse_args(['--train','--warm-start','checkpoints/step_20'])
        self.assertTrue(args.warm_start.is_absolute())
        with self.assertRaises(SystemExit):
            entry.parse_args(['--view','--warm-start','checkpoints/step_20'])

    def test_warm_start_inherits_task_and_saved_parameters_unless_explicitly_changed(self):
        self.assertTrue(hasattr(entry,'training_environment_source'),'warm-start task inheritance missing')
        with tempfile.TemporaryDirectory() as directory:
            run=Path(directory);checkpoint=run/'checkpoints/step_20';checkpoint.mkdir(parents=True)
            (run/'config.json').write_text(json.dumps({'task_profile':'forward','environment':{'noise_config':{'level':.17}}}))
            profile,saved=entry.training_environment_source(None,checkpoint)
            self.assertEqual(profile,'forward')
            self.assertEqual(saved['environment']['noise_config']['level'],.17)
            profile,saved=entry.training_environment_source('joystick',checkpoint)
            self.assertEqual(profile,'joystick')
            self.assertIsNone(saved)


if __name__ == '__main__':
    unittest.main()
