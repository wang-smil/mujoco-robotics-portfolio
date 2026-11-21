"""Refine the cloned single actor with conservative PPO updates."""
import argparse
import json
from pathlib import Path
import re
import subprocess

from check_arc_curriculum import gate as arc_gate
from continue_reverse import ROOT, STAGE, PYTHON, wait_for_guard
from run_reverse_tracking import guarded


def tracks_command(segment):
    vx, vy, yaw = segment['mean_twist']
    command_x, _, command_yaw = segment['command']
    speed_tolerance = {'reverse': .018, 'stop': .015, 'left': .025, 'right': .025}
    turn_tolerance = .06 if segment['segment'] == 'stop' else .08
    return (not segment['terminated'] and abs(vx - command_x) <= speed_tolerance.get(segment['segment'], .03)
            and abs(vy) <= .025 and abs(yaw - command_yaw) <= turn_tolerance)


def combined_gate(path):
    report = json.loads(path.read_text())
    return (all(row['survived'] == row['cases'] for row in report['summary'].values())
            and all(tracks_command(segment) for row in report['episodes'] for segment in row['segments']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--migrated', action='store_true')
    args = parser.parse_args()
    training_guard = 'round12_retry_guard' if args.migrated else 'round12_guard'
    for label in ('final', 'mid'):
        result = wait_for_guard(f'round11_native_{label}_guard', 520)
        assert result['status'] == 'passed'
    if any(arc_gate(STAGE / f'round11_native_{label}.json') for label in ('final', 'mid')):
        first = wait_for_guard('round11_mid_guard', 520)
        if first['status'] == 'passed':
            final = wait_for_guard('round11_final_guard', 520)
            if final['status'] == 'passed':
                for label in ('mid', 'final'):
                    summary = json.loads((STAGE / f'round11_{label}.json').read_text())['summary']
                    if all(row['passed'] == row['count'] for row in summary.values()):
                        wait_for_guard('round11_heldout_guard', 520)
                        break
    run_file = 'reverse_imitation_gpu_run.txt' if args.migrated else 'reverse_imitation_run.txt'
    source_run = Path((STAGE / run_file).read_text().strip())
    source = source_run / 'checkpoints/initial'
    baseline = json.loads((STAGE / 'imitation_native.json').read_text())['summary']['reverse']
    assert baseline['survived'] == 8 and abs(baseline['mean_twist'][0] + .05) <= .015
    assert not (STAGE / training_guard / 'process.log').exists()
    executable = str(ROOT / '.venv-duck/bin/python')
    guarded(training_guard, 1500, [executable, '-u', '-B', 'lessons/07_duck_ppo.py', '--train',
        '--steps', '10000000', '--envs', '4096', '--evals', '3', '--preset', 'transfer', '--seed', '9',
        '--learning-rate', '.0001', '--warm-start', str(source), '--skip-rollouts'])
    log = (STAGE / training_guard / 'process.log').read_text()
    run = Path(re.search(r'^RUN_DIRECTORY: (.+)$', log, flags=re.MULTILINE)[1])
    assert run.is_relative_to(ROOT / 'artifacts/lesson07')
    checks = json.loads((run / 'checks.json').read_text())
    assert checks['status'] == 'passed' and checks['training']['restore_max_action_error'] <= 1e-6
    config = json.loads((run / 'config.json').read_text())
    (STAGE / 'round12_run.txt').write_text(str(run) + '\n')
    checkpoints = [('final', Path(config['final_checkpoint']).name), ('mid', 'step_5079040')]
    for label, checkpoint in checkpoints:
        subprocess.run(['/usr/bin/env', 'JAX_PLATFORMS=cpu', 'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1',
            executable, '-u', '-B', str(STAGE / 'benchmark_author.py'), '--run', str(run), '--checkpoint', checkpoint,
            '--cpu-restore', '--paired-ramp', '--output', str(STAGE / f'round12_native_{label}.json')], check=True, timeout=480)
    if any(combined_gate(STAGE / f'round12_native_{label}{suffix}.json')
           for label, _ in checkpoints for suffix in ('', '_ramp')):
        subprocess.run([str(PYTHON), '-u', '-B', str(STAGE / 'check_round2.py'), '--round', '12', '--actions',
                        'all', '--held-out-on-pass', '--training-guard', training_guard], check=True)


if __name__ == '__main__':
    main()
