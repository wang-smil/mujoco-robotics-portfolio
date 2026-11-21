"""Check whether reducing turn command before reversing improves full switching."""
import argparse
import json
from pathlib import Path

import numpy as np

from benchmark_author import STAGE, TARGETS, make_local_env, rollout
from fit_reverse_entry import ReverseEntry, make_reverse_reference
from run_combined_refine import tracks_command


class CalibratedYaw:
    def __init__(self, policy, gain):
        self.policy = policy
        self.gain = gain

    def infer(self, observation):
        if observation[8] < 0. and self.gain != 1.:
            observation = observation.copy()
            observation[8] = max(-.25, self.gain * observation[8])
        return self.policy.infer(observation)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--right-calibration', action='store_true')
    args = parser.parse_args()
    source = Path((STAGE / 'round11_run.txt').read_text().strip())
    env, _ = make_local_env(source, 'step_5079040', cpu_restore=True)
    positive = env.policy
    reference_fit = json.loads((STAGE / 'reverse_teacher_refined_validated.json').read_text())[0]
    reverse = make_reverse_reference(env, reference_fit['parameters'])
    original = json.loads((STAGE / 'reverse_entry_refined_fit.json').read_text())
    gain, ticks, trigger, pause, _ = original['selection']
    assert pause == 0
    routes = {
        'standard': ('stop', 'forward_010', 'left', 'forward_010', 'right', 'reverse', 'walk_left', 'walk_right', 'stop'),
        'mirrored': ('stop', 'forward_010', 'right', 'forward_010', 'left', 'reverse', 'walk_right', 'walk_left', 'stop'),
        'arc_directions': ('stop', 'walk_right', 'reverse', 'walk_left', 'reverse', 'walk_right', 'stop'),
        'longer_cycle': ('stop', 'forward_005', 'walk_left', 'reverse', 'forward_015', 'right', 'reverse', 'stop'),
        'stand_to_reverse': ('stop', 'reverse', 'stop'),
    }

    def run_case(phase, yaw_gate, yaw_gain, name, order, seed, ramp, collect=False):
        env.policy = ReverseEntry(env, CalibratedYaw(positive, yaw_gain), reverse, gain, ticks, trigger)
        schedule = [(name, 1000) for name in order] if len(order) == 1 else [(name, 200) for name in order]
        segments = rollout(env, seed, schedule, ramp=ramp, reverse_entry_phase=phase, reverse_yaw_gate=yaw_gate)
        complete = len(segments) == len(schedule) and not segments[-1]['terminated']
        target_passes = sum(tracks_command(segment) for segment in segments)
        reverse_segments = [row for row in segments if row['segment'] == 'reverse']
        reversing = sum(tracks_command(segment) for segment in reverse_segments)
        result = dict(case=name, seed=seed, ramp=ramp, complete=complete,
                      passed=complete and target_passes == len(schedule), passed_segments=target_passes,
                      reverse_passes=reversing, segments=segments)
        if collect and result['passed']:
            observations.extend(env.policy.observations)
            actions.extend(env.policy.actions)
        return result

    candidates = []
    if args.right_calibration:
        variants = [(phase, None, scale) for phase in (0., .5, .66) for scale in (1., 1.10, 1.20, 1.25)]
        screen_seed, check_seed = 4011, 4051
        compared_ramps = (False, True)
        output = STAGE / 'reverse_yaw_calibration_probe.json'
    else:
        variants = [(phase, gate, 1.) for phase in (0., .33, .5, .60, .66, .75) for gate in (None, .075, .020)]
        screen_seed, check_seed = 3711, 3751
        compared_ramps = (True,)
        output = STAGE / 'reverse_steering_probe.json'
    for phase, gate, yaw_gain in variants:
        screening = [run_case(phase, gate, yaw_gain, name, order, seed, ramp)
                     for name, order in routes.items() for seed in range(screen_seed, screen_seed + 4)
                     for ramp in compared_ramps]
        score = (sum(row['passed'] for row in screening), sum(row['reverse_passes'] for row in screening),
                 sum(row['passed_segments'] for row in screening))
        candidates.append(dict(phase=phase, gate=gate, gain=yaw_gain, score=score, screening=screening))
        print('Phase', phase, 'yaw gate', gate, 'negative yaw gain', yaw_gain,
              'whole-schedule passes', score[0], '/', len(screening), flush=True)
    selected = max(candidates, key=lambda row: row['score'])
    observations, actions = [], []
    sequence_routes = dict(routes)
    routes.update({action: (action,) for action in TARGETS})
    verified = [run_case(selected['phase'], selected['gate'], selected['gain'], name, order, seed, ramp,
                         collect=name in sequence_routes)
                for name, order in routes.items() for seed in range(check_seed, check_seed + 12) for ramp in (False, True)]
    counts = {f'{name}:{ramp}': sum(row['passed'] for row in verified if row['case'] == name and row['ramp'] == ramp)
              for name in routes for ramp in (False, True)}
    print('Held-out complete and tracking passes:', counts, flush=True)
    output.write_text(json.dumps(
        dict(source=str(source), candidates=candidates,
             best=dict(phase=selected['phase'], gate=selected['gate'], gain=selected['gain']),
             holdout=verified, counts=counts, samples=len(observations)), indent=2, allow_nan=False) + '\n')
    switch_results = [count for label, count in counts.items() if label.split(':')[0] in sequence_routes]
    if sum(switch_results) >= 100 and min(switch_results) >= 8 and observations:
        name = 'reverse_yaw_calibration' if args.right_calibration else 'reverse_steered'
        np.savez_compressed(STAGE / f'{name}_demo_data.npz', obs=np.stack(observations), action=np.stack(actions),
                            cadence=np.asarray(reverse.cadence, dtype=np.float32))


if __name__ == '__main__':
    main()
