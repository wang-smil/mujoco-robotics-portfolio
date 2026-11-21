"""Start the reverse curriculum only after the turn hold-out gate passes."""
import json
from pathlib import Path
import subprocess
import time


ROOT = Path.cwd()
STAGE = Path(__file__).resolve().parent
GUARD = STAGE.parent / 'lessons/capacity_guard.py'
PYTHON = ROOT / '.venv/bin/python'


def wait_for_guard(name, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            result = json.loads((STAGE / name / 'guard.json').read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(2)
            continue
        if result['status'] != 'running':
            return result
        time.sleep(2)
    raise TimeoutError(name)


def main():
    outcome = wait_for_guard('round4_heldout_guard', 940)
    if outcome['status'] != 'passed':
        raise RuntimeError(f"Turn hold-out evaluation ended with {outcome['status']}")
    report = json.loads((STAGE / 'round4_heldout.json').read_text())
    results = report['summary']
    passed = all(row['count'] == row['survived'] == 20 and row['passed'] >= 19 for row in results.values())
    verification = dict(passed=passed, source=report['checkpoint'], minimum_passed=19, required_survival=20,
                        results=results)
    (STAGE / 'round4_gate.json').write_text(json.dumps(verification, indent=2) + '\n')
    if not passed:
        print('Hold-out criteria not met. Continuing to reserve the reverse stage until a passing checkpoint.', flush=True)
        return
    wait_for_guard('round4_native_guard', 500)
    source = Path(report['checkpoint'])
    assert (source.parents[1] / 'config.json').is_file()
    assert not (STAGE / 'round5_guard/process.log').exists()
    print(f'Hold-out gate passed. Starting low-speed reverse training from {source}.', flush=True)
    subprocess.run([str(PYTHON), '-B', str(GUARD), '--output', str(STAGE / 'round5_guard'), '--timeout', '1500', '--',
        str(ROOT / '.venv-duck/bin/python'), '-u', '-B', 'lessons/07_duck_ppo.py', '--train', '--steps', '10000000',
        '--envs', '4096', '--evals', '3', '--preset', 'transfer', '--task-profile', 'neck_reverse', '--seed', '2',
        '--warm-start', str(source), '--skip-rollouts'], check=True)
    subprocess.run([str(PYTHON), '-u', '-B', str(STAGE / 'check_round2.py'), '--round', '5', '--actions', 'reverse',
                    '--held-out-on-pass'], check=True)


if __name__ == '__main__':
    main()
