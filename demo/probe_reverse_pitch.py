"""Measure a small pitch-feedback correction around the learned policy in MuJoCo."""
import argparse
import json
from pathlib import Path

import numpy as np

from benchmark_author import STAGE, make_local_env, rollout
from probe_direction_experts import bridge_reverse
from run_combined_refine import tracks_command


class PitchFeedback:
    def __init__(self, env, policy, gain, damping, limit, deadband=None):
        self.env = env
        self.policy = policy
        self.gain, self.damping, self.limit = gain, damping, limit
        self.deadband = deadband

    def infer(self, obs):
        action = self.policy.infer(obs)
        if obs[6] >= 0.:
            return action
        w, x, y, z = self.env.data.qpos[3:7]
        pitch = np.arcsin(np.clip(2. * (w * y - x * z), -1., 1.))
        correction = self.gain * (pitch + .045) + self.damping * obs[1]
        if self.deadband is not None:
            correction = min(correction + self.gain * self.deadband, 0.)
        correction = np.clip(correction, -self.limit, self.limit) / self.env.action_scale
        action = action.copy()
        action[[2, 11]] += correction
        action[[4, 13]] -= .5 * correction
        return np.clip(action, -1., 1.)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--deadband', action='store_true')
    mode.add_argument('--switch-phase', action='store_true')
    parser.add_argument('--run-pointer', type=Path, default=STAGE / 'reverse_entry_imitation_run.txt')
    parser.add_argument('--checkpoint', default='initial')
    parser.add_argument('--prefix', default='reverse_pitch')
    args = parser.parse_args()
    if Path(args.prefix).name != args.prefix:
        parser.error('--prefix must be a simple filename prefix')
    run = Path(args.run_pointer.read_text().strip())
    env, _ = make_local_env(run, args.checkpoint, cpu_restore=True)
    actor = env.policy
    schedules = {
        'home': [('reverse', 1000)],
        'stand': [('stop', 100), ('reverse', 200), ('stop', 120)],
        'motion': [('stop', 100), ('forward_010', 180), ('reverse', 220), ('walk_left', 160), ('stop', 100)],
    }
    if args.switch_phase:
        schedules['standard'] = [(action, 200) for action in ('stop', 'forward_010', 'left', 'forward_010',
                                  'right', 'reverse', 'walk_left', 'walk_right', 'stop')]

    def trial(settings, case, seed, ramp, phase=None, pause=0):
        env.policy = PitchFeedback(env, actor, *settings)
        schedule = bridge_reverse(schedules[case], pause)
        rows = rollout(env, seed, schedule, ramp=ramp, reverse_entry_phase=phase)
        full = len(rows) == len(schedule) and not rows[-1]['terminated']
        passed = full and all(tracks_command(row) for row in rows if row['steps'] >= 100)
        return dict(case=case, seed=seed, ramp=ramp, complete=full, passed=passed, segments=rows)

    if args.switch_phase:
        pitch = (1.8, .108, .16, .04)
        candidates = []
        for phase in (None, 0., .17, .33, .5, .66, .83):
            for pause in (0, 20):
                rows = [trial(pitch, case, seed, ramp, phase, pause) for case in schedules
                        for seed in (3831, 3832) for ramp in (False, True)]
                score = (sum(row['passed'] for row in rows), sum(row['complete'] for row in rows))
                print('Reverse phase', phase, 'pause', pause, 'tracking and survival:', score, flush=True)
                candidates.append(dict(phase=phase, pause=pause, score=score, cases=rows))
        best = max(candidates, key=lambda row: row['score'])
        verified = [trial(pitch, case, seed, ramp, best['phase'], best['pause'])
                    for case in schedules for seed in range(3871, 3883) for ramp in (False, True)]
        counts = {f'{case}:{ramp}': sum(row['passed'] for row in verified if row['case'] == case and row['ramp'] == ramp)
                  for case in schedules for ramp in (False, True)}
        print('Independent validation', best['phase'], best['pause'], counts, flush=True)
        (STAGE / f'{args.prefix}_phase_probe.json').write_text(json.dumps(
            dict(source=str(run), candidates=candidates, counts=counts, verified=verified), indent=2, allow_nan=False) + '\n')
        return

    candidates = [(0., 0., 0.)]
    if args.deadband:
        candidates += [(gain, gain * lookahead, .16, tilt) for gain in (.55, 1.1, 1.8)
                       for lookahead in (.06, .10) for tilt in (.04, .08, .12)]
        output = STAGE / f'{args.prefix}_deadband_probe.json'
        screen_seeds, validation_seeds = (3621, 3622, 3623), range(3671, 3679)
    else:
        candidates += [(gain, gain * period, limit) for gain in (-.8, -.35, .35, .8)
                       for period in (.025, .085) for limit in (.12, .22)]
        output = STAGE / f'{args.prefix}_probe.json'
        screen_seeds, validation_seeds = (3601, 3602, 3603), range(3651, 3659)
    screened = []
    for settings in candidates:
        rows = [trial(settings, case, seed, True) for case in schedules for seed in screen_seeds]
        tracked = sum(row['passed'] for row in rows)
        survived = sum(row['complete'] for row in rows)
        screened.append(dict(settings=settings, tracked=tracked, survived=survived, rows=rows))
        print('Pitch settings', settings, 'tracked', tracked, 'ran full', survived, flush=True)
    shortlist = sorted(screened, key=lambda row: (-row['tracked'], -row['survived']))[:3]
    verified = []
    for candidate in shortlist:
        rows = [trial(candidate['settings'], case, seed, ramp) for case in schedules
                for seed in validation_seeds for ramp in (False, True)]
        counts = {f'{case}:{ramp}': sum(row['passed'] for row in rows if row['case'] == case and row['ramp'] == ramp)
                  for case in schedules for ramp in (False, True)}
        verified.append(dict(settings=candidate['settings'], counts=counts, rows=rows))
        print('Independent validation', candidate['settings'], counts, flush=True)
    output.write_text(json.dumps(
        dict(source=str(run), screening=screened, verified=verified), indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
