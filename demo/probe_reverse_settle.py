"""Test whether a measured settling condition improves repeated backward switches."""
import argparse
import json
from pathlib import Path

import numpy as np

from benchmark_author import STAGE, TARGETS, make_local_env, rollout
from fit_reverse_entry import ReverseEntry, make_reverse_reference
from probe_steered_reverse import CalibratedYaw
from run_combined_refine import tracks_command


class SettleBeforeReverse:
    def __init__(self, speed_address, max_wait, threshold):
        self.speed_address = speed_address
        self.max_wait, self.threshold = max_wait, threshold
        self.waiting = False
        self.transitioned = False
        self.previous_yaw = 0.
        self.releases = []

    def __call__(self, env, target, previous, tick):
        if tick == 0:
            self.waiting = target[0] < -.01 and previous[0] >= -.01 and bool(np.any(previous))
            self.transitioned = self.waiting
            self.previous_yaw = previous[2]
        if self.waiting:
            start = self.speed_address
            motion = env.data.sensordata[start:start + 2]
            settled = (np.linalg.norm(motion) < self.threshold
                       and abs(env.get_gyro(env.data)[2]) < .12)
            if tick < self.max_wait and (tick < 6 or not settled):
                return np.zeros(3)
            self.waiting = False
            self.releases.append(tick)
        return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--direction-aware', action='store_true')
    args = parser.parse_args()
    base = Path((STAGE / 'round11_run.txt').read_text().strip())
    env, _ = make_local_env(base, 'step_5079040', cpu_restore=True)
    positive = env.policy
    teacher = json.loads((STAGE / 'reverse_teacher_refined_validated.json').read_text())[0]
    backward = make_reverse_reference(env, teacher['parameters'])
    fit = json.loads((STAGE / 'reverse_entry_fit.json').read_text())
    gain, ticks, trigger, pause, _ = fit['selection']
    assert pause == 0
    calibration = json.loads((STAGE / 'reverse_yaw_calibration_probe.json').read_text())['best']
    positive = CalibratedYaw(positive, calibration['gain'])
    velocity = int(env.model.sensor('local_linvel').adr[0])
    routes = {
        'repeated_arcs': ('stop', 'walk_right', 'reverse', 'walk_left', 'reverse', 'walk_right', 'stop'),
        'cycle': ('stop', 'forward_005', 'walk_left', 'reverse', 'forward_015', 'right', 'reverse', 'stop'),
        'standard': ('stop', 'forward_010', 'left', 'forward_010', 'right', 'reverse', 'walk_left', 'walk_right', 'stop'),
        'mirror': ('stop', 'forward_010', 'right', 'forward_010', 'left', 'reverse', 'walk_right', 'walk_left', 'stop'),
        'start': ('stop', 'reverse', 'walk_left', 'stop'),
    }

    def trial(params, name, order, seed, ramp):
        max_wait, speed_limit, moving_phase, left_offset, right_offset = params
        release = SettleBeforeReverse(velocity, max_wait, speed_limit)
        env.policy = ReverseEntry(env, positive, backward, gain, ticks, trigger)

        def phase(controller):
            if not release.transitioned:
                return calibration['phase']
            offset = left_offset if release.previous_yaw > .01 else right_offset if release.previous_yaw < -.01 else 0.
            return (moving_phase + offset) % 1.

        ticks_per_action = 1000 if len(order) == 1 else 200
        series = [(action, ticks_per_action) for action in order]
        segments = rollout(env, seed, series, ramp=ramp, reverse_entry_phase=phase, command_filter=release)
        complete = len(segments) == len(series) and not segments[-1]['terminated']
        tracked = sum(tracks_command(row) for row in segments)
        return dict(case=name, seed=seed, ramp=ramp, complete=complete, passed=complete and tracked == len(series),
                    tracked=tracked, release_ticks=release.releases, segments=segments)

    if args.direction_aware:
        previous_best = json.loads((STAGE / 'reverse_settle_probe.json').read_text())['best']
        wait, speed, phase = previous_best[:3]
        candidates = [(wait, speed, phase, 0., 0.)]
        candidates += [(wait, speed, start, left, right) for start in (.0, .33, .5, .66)
                       for left, right in ((.5, 0.), (0., .5), (.17, -.17), (-.17, .17))]
        probe_seeds, holdout_seeds = (4491, 4492), range(4501, 4513)
        filename = 'reverse_settle_direction_probe.json'
    else:
        candidates = [(0, .03, .5, 0., 0.)]
        candidates += [(wait, speed, phase, 0., 0.) for wait in (12, 20, 28) for speed in (.03, .06)
                       for phase in (0., .5, .66)]
        probe_seeds, holdout_seeds = (4291, 4292), range(4301, 4313)
        filename = 'reverse_settle_probe.json'
    results = []
    for setting in candidates:
        cases = [trial(setting, name, order, seed, ramp) for name, order in routes.items() for seed in probe_seeds
                 for ramp in (False, True)]
        score = (sum(row['passed'] for row in cases), sum(row['tracked'] for row in cases),
                 sum(row['complete'] for row in cases))
        results.append(dict(setting=setting, score=score, cases=cases))
        print('Reverse settling settings:', setting, 'whole-route passes:', score[0], '/', len(cases), flush=True)
    best = max(results, key=lambda result: result['score'])
    routes.update({name: (name,) for name in TARGETS})
    heldout = [trial(best['setting'], name, order, seed, ramp) for name, order in routes.items()
               for seed in holdout_seeds for ramp in (False, True)]
    counts = {f'{name}:{ramp}': sum(row['passed'] for row in heldout if row['case'] == name and row['ramp'] == ramp)
              for name in routes for ramp in (False, True)}
    print('Independent settling validation:', best['setting'], counts, flush=True)
    (STAGE / filename).write_text(json.dumps(
        dict(source=str(base), best=best['setting'], pilots=results, heldout=heldout, counts=counts),
        indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
