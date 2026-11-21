"""Use real short child processes; only hardware readings are injected."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

base=Path(__file__).resolve().parents[1]
path=base/'capacity_guard.py'
if not path.exists():
    path=base/'lessons/capacity_guard.py'
guard=None
if path.exists():
    spec=importlib.util.spec_from_file_location('capacity_guard',path)
    guard=importlib.util.module_from_spec(spec); spec.loader.exec_module(guard)

healthy=dict(gpu_total_mib=6144, gpu_used_mib=1000, host_available_mib=5000)
full=dict(gpu_total_mib=6144, gpu_used_mib=5800, host_available_mib=5000)


class CapacityGuard(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(guard, 'bounded own-child runner is missing')

    def test_healthy_child_completes_and_log_is_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            result=guard.run_guarded([sys.executable,'-c','print("done")'],Path(directory),sampler=lambda:healthy,interval=.01,timeout=2)
            self.assertEqual(result['status'],'passed')
            self.assertEqual(result['exit_code'],0)
            self.assertIn('done',(Path(directory)/'process.log').read_text())

    def test_insufficient_headroom_prevents_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            marker=Path(directory)/'started'
            result=guard.run_guarded([sys.executable,'-c',f'from pathlib import Path; Path({str(marker)!r}).touch()'],Path(directory),sampler=lambda:full,interval=.01,timeout=2)
            self.assertEqual(result['status'],'gpu_headroom')
            self.assertFalse(result['child_started'])
            self.assertFalse(marker.exists())

    def test_midrun_guard_terminates_only_its_own_child(self):
        with tempfile.TemporaryDirectory() as directory:
            marker=Path(directory)/'other_completed'
            other=subprocess.Popen([sys.executable,'-c',f'import time; from pathlib import Path; time.sleep(.2); Path({str(marker)!r}).touch()'])
            calls=[healthy,full]
            result=guard.run_guarded([sys.executable,'-c','import time; time.sleep(10)'],Path(directory),sampler=lambda:calls.pop(0) if calls else full,interval=.01,timeout=2)
            self.assertEqual(result['status'],'gpu_headroom')
            self.assertNotEqual(result['exit_code'],0)
            self.assertEqual(other.wait(timeout=2),0)
            self.assertTrue(marker.exists())


if __name__=='__main__':
    unittest.main()
