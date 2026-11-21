"""Measure whether command-conditioned Actor blending protects existing motions."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from benchmark_author import STAGE, TARGETS, make_local_env, rollout
from run_combined_refine import tracks_command


class BlendedActor:
    def __init__(self, original, reverse, width):
        self.original = original
        self.reverse = reverse
        self.width = width

    def infer(self, observation):
        reverse_fraction = np.clip(-observation[6] / self.width, 0., 1.)
        if reverse_fraction == 0.:
            return self.original.infer(observation)
        if reverse_fraction == 1.:
            return self.reverse.infer(observation)
        blend = reverse_fraction**2 * (3. - 2. * reverse_fraction)
        start = self.original.infer(observation)
        end = self.reverse.infer(observation)
        return start + blend * (end - start)


def bridge_reverse(schedule, pause):
    if not pause:
        return schedule
    order = []
    previous = 'stop'
    for name, steps in schedule:
        if (name == 'reverse') != (previous == 'reverse') and name != 'stop' and previous != 'stop':
            order.append(('stop', pause))
        order.append((name, steps))
        previous = name
    return order


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase-screen', action='store_true')
    args = parser.parse_args()
    root = Path.cwd()
    source = Path((STAGE / 'round11_run.txt').read_text().strip())
    env, _ = make_local_env(source, 'step_5079040', cpu_restore=True)
    original = env.policy
    os.chdir(root)
    reverse_pointer = 'refined_transition_imitation_run.txt' if args.phase_screen else 'straight_imitation_run.txt'
    reverse_source = Path((STAGE / reverse_pointer).read_text().strip())
    reverse_env, _ = make_local_env(reverse_source, 'initial', cpu_restore=True)
    backward = reverse_env.policy
    np.testing.assert_array_equal(env.max_motor_velocity, reverse_env.max_motor_velocity)
    env.reverse_cadence = reverse_env.reverse_cadence
    switches = {
        'standard': ('stop', 'forward_010', 'left', 'forward_010', 'right', 'reverse', 'walk_left', 'walk_right', 'stop'),
        'arcs': ('stop', 'walk_left', 'reverse', 'walk_right', 'reverse', 'forward_005', 'stop'),
        'reverse_start': ('stop', 'reverse', 'forward_010', 'reverse', 'left', 'stop'),
    }
    schedules = {case: [(case, 1000)] for case in TARGETS}
    schedules.update({case: [(name, 200) for name in sequence] for case, sequence in switches.items()})

    def compare(width, pause, seed_start, seeds, entry_phase=None, only_switches=False):
        env.policy = BlendedActor(original, backward, width)
        result = dict(width=width, pause_seconds=pause * .02, entry_phase=entry_phase, seeds=seeds, trials=[], passed=0)
        for case, base_schedule in schedules.items():
            if only_switches and case not in switches:
                continue
            schedule = bridge_reverse(base_schedule, pause)
            for ramp in (False, True):
                for seed in range(seed_start, seed_start + seeds):
                    segments = rollout(env, seed, schedule, ramp=ramp, reverse_entry_phase=entry_phase)
                    passed = (len(segments) == len(schedule) and not segments[-1]['terminated']
                              and all(tracks_command(segment) for segment in segments if segment['steps'] >= 100))
                    result['passed'] += passed
                    result['trials'].append(dict(case=case, ramp=ramp, seed=seed, passed=passed, segments=segments))
        return result

    pilots = []
    if args.phase_screen:
        variants = [(.006, pause, phase) for pause in (0, 25, 50) for phase in (0., .25, .5, .75)]
        output = STAGE / 'direction_expert_phase_probe.json'
    else:
        variants = [(width, pause, None) for width in (.006, .025, .045) for pause in (0, 50)]
        output = STAGE / 'direction_expert_probe.json'
    for width, pause, phase in variants:
        outcome = compare(width, pause, 2901 if args.phase_screen else 2801, 4, phase, args.phase_screen)
        pilots.append(outcome)
        switches_passed = sum(row['passed'] for row in outcome['trials'] if row['case'] in switches)
        print('Blend width', width, 'pause', pause, 'phase', phase, 'all passes', outcome['passed'], '/', len(outcome['trials']),
              'switches', switches_passed, '/24', flush=True)
    best = max(pilots, key=lambda row: row['passed'])
    result = dict(sources=[str(source), str(reverse_source)], pilots=pilots)
    if best['passed'] == len(best['trials']):
        result['heldout'] = compare(best['width'], round(best['pause_seconds'] / .02),
                                    2951 if args.phase_screen else 2851, 20, best['entry_phase'])
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
