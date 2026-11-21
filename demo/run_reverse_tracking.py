"""Train the next reverse candidate from the matched round-five checkpoint."""
import json
from pathlib import Path
import re
import subprocess

from continue_reverse import ROOT, STAGE, GUARD, PYTHON, wait_for_guard


def guarded(directory, timeout, command):
    subprocess.run([str(PYTHON), '-B', str(GUARD), '--output', str(STAGE / directory),
                    '--timeout', str(timeout), '--', *command], check=True)


def main():
    wait_for_guard('round5_reverse_probe_guard', 340)
    prior = json.loads((STAGE / 'round5_native_final.json').read_text())
    source = Path(prior['model'])
    assert prior['summary']['reverse']['survived'] == 0
    assert prior['summary']['reverse']['mean_twist'][0] < -.08
    assert not (STAGE / 'round6_guard/process.log').exists()
    executable = str(ROOT / '.venv-duck/bin/python')
    guarded('round6_guard', 1500, [executable, '-u', '-B', 'lessons/07_duck_ppo.py', '--train',
        '--steps', '10000000', '--envs', '4096', '--evals', '3', '--preset', 'transfer', '--seed', '3',
        '--task-profile', 'neck_reverse_tracking', '--warm-start', str(source), '--skip-rollouts'])
    log = (STAGE / 'round6_guard/process.log').read_text()
    run = Path(re.search(r'^RUN_DIRECTORY: (.+)$', log, flags=re.MULTILINE)[1])
    checks = json.loads((run / 'checks.json').read_text())
    assert checks['status'] == 'passed' and checks['training']['restore_max_action_error'] <= 1e-6
    config = json.loads((run / 'config.json').read_text())
    checkpoint = Path(config['final_checkpoint']).name
    (STAGE / 'round6_run.txt').write_text(str(run) + '\n')
    guarded('round6_native_guard', 480, ['/usr/bin/env', 'JAX_PLATFORMS=cuda,cpu', executable, '-u', '-B',
        str(STAGE / 'benchmark_author.py'), '--run', str(run), '--checkpoint', checkpoint, '--reverse-switches',
        '--output', str(STAGE / 'round6_native.json')])
    probe = json.loads((STAGE / 'round6_native.json').read_text())
    if probe['summary']['reverse']['survived'] == 8 and probe['summary']['switches']['survived'] == 8:
        subprocess.run([str(PYTHON), '-u', '-B', str(STAGE / 'check_round2.py'), '--round', '6', '--actions', 'reverse',
                        '--held-out-on-pass'], check=True)
    else:
        print('Reverse gate remains open; latest native measurements saved.', flush=True)


if __name__ == '__main__':
    main()
