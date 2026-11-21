"""Compare the arc branch with the author's native settings, then test MJX."""
import argparse
import json
from pathlib import Path
import re
import subprocess

from continue_reverse import ROOT, STAGE, PYTHON, wait_for_guard
from run_reverse_tracking import guarded


def close_to_command(twist, target_yaw):
    return abs(twist[0] - .10) <= .03 and abs(twist[2] - target_yaw) <= .08


def gate(path):
    report = json.loads(path.read_text())
    summary = report['summary']
    if not all(row['survived'] == row['cases'] for row in summary.values()):
        return False
    for label, yaw in [('walk_left', .20), ('walk_right', -.20)]:
        if not close_to_command(summary[label]['mean_twist'], yaw):
            return False
    tracked = 0
    for row in report['episodes']:
        if row['case'] != 'switches' or row['segments'][-1]['terminated']:
            continue
        segments = [segment for segment in row['segments'] if segment['segment'] in ('walk_left', 'walk_right')]
        tracked += (len(segments) == 2 and all(close_to_command(segment['mean_twist'], segment['command'][2])
                                                for segment in segments))
    return tracked >= 7


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--round', type=int, default=9)
    args = parser.parse_args()
    prefix = f'round{args.round}'
    result = wait_for_guard(f'{prefix}_guard', 1520)
    if result['status'] != 'passed' or result['exit_code'] != 0:
        raise RuntimeError(f"Arc training ended with {result['status']}")
    log = (STAGE / f'{prefix}_guard/process.log').read_text()
    run = Path(re.search(r'^RUN_DIRECTORY: (.+)$', log, flags=re.MULTILINE)[1])
    assert run.is_relative_to(ROOT / 'artifacts/lesson07')
    checks = json.loads((run / 'checks.json').read_text())
    assert checks['status'] == 'passed' and checks['training']['restore_max_action_error'] <= 1e-6
    config = json.loads((run / 'config.json').read_text())
    (STAGE / f'{prefix}_run.txt').write_text(str(run) + '\n')
    executable = str(ROOT / '.venv-duck/bin/python')
    checkpoints = [('final', Path(config['final_checkpoint']).name), ('mid', 'step_5079040')]
    for label, name in checkpoints:
        guarded(f'{prefix}_native_{label}_guard', 480, ['/usr/bin/env', 'JAX_PLATFORMS=cuda,cpu', executable, '-u', '-B',
            str(STAGE / 'benchmark_author.py'), '--run', str(run), '--checkpoint', name, '--arc-switches',
            '--output', str(STAGE / f'{prefix}_native_{label}.json')])
    if any(gate(STAGE / f'{prefix}_native_{label}.json') for label, _ in checkpoints):
        subprocess.run([str(PYTHON), '-u', '-B', str(STAGE / 'check_round2.py'), '--round', str(args.round), '--actions', 'arc',
                        '--held-out-on-pass'], check=True)


if __name__ == '__main__':
    main()
