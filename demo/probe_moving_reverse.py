"""Keep the tested handpad controller and try a specialized transition from moving gaits."""
import argparse
import json
from pathlib import Path

import numpy as np

from benchmark_author import STAGE, make_local_env, rollout
from fit_reverse_entry import ReverseEntry, make_reverse_reference
from probe_adaptive_reverse import assisted_policy
from probe_direction_experts import bridge_reverse
from probe_reverse_settle import SettleBeforeReverse
from probe_steered_reverse import CalibratedYaw
from run_combined_refine import tracks_command


ROUTES = {
    'standard': ('stop', 'forward_010', 'left', 'forward_010', 'right', 'reverse', 'walk_left', 'walk_right', 'stop'),
    'straight': ('stop', 'forward_010', 'reverse', 'forward_010', 'stop'),
    'arcs': ('stop', 'walk_right', 'reverse', 'walk_left', 'reverse', 'walk_right', 'stop'),
    'other_arcs': ('stop', 'walk_left', 'reverse', 'walk_right', 'reverse', 'walk_left', 'stop'),
}


class MovingEntry:
    def __init__(self, ordinary, procedural, speed, wait, threshold, arcs_only, neutral_limit, enabled=True):
        self.ordinary, self.procedural = ordinary, procedural
        self.settle = SettleBeforeReverse(speed, wait, threshold)
        self.enabled, self.arcs_only = enabled, arcs_only
        self.neutral_limit = neutral_limit
        self.reset()

    def reset(self):
        self.use_procedure = False
        self.neutral_ticks = 0
        self.last_motion = np.zeros(3)
        self.original = np.zeros(3)
        self.procedural.count = 0
        self.procedural.filtered_x = 0.
        self.settle.releases.clear()

    def filter(self, env, target, _previous, tick):
        if tick == 0:
            self.original = self.last_motion.copy()
            if target[0] < -.01:
                forward = self.original[0] > .03 and self.neutral_ticks <= self.neutral_limit
                turning = abs(self.original[2]) > .05
                self.use_procedure = self.enabled and forward and (turning or not self.arcs_only)
        if np.any(target):
            self.last_motion = target.copy()
            self.neutral_ticks = 0
        else:
            self.neutral_ticks += 1
        if self.use_procedure and target[0] < -.01:
            value = self.settle(env, target, self.original, tick)
            self.settle.releases.clear()
            return value
        return target

    def infer(self, observation):
        if self.use_procedure and observation[6] < -.002:
            return self.procedural.infer(observation)
        # A fresh entry should start its action blend from its first reverse frame.
        self.procedural.count = 0
        self.procedural.filtered_x = self.procedural.env.data.sensordata[self.procedural.speed_address]
        return self.ordinary.infer(observation)


def prepare_reference(env):
    teacher = json.loads((STAGE / 'reverse_teacher_refined_validated.json').read_text())[0]
    cadence, forward_cadence = env.reverse_cadence, env.phase_frequency_factor
    reference = make_reverse_reference(env, teacher['parameters'])
    env.reverse_cadence, env.phase_frequency_factor = cadence, forward_cadence
    parameters = json.loads((STAGE / 'reverse_entry_fit.json').read_text())['selection'][:3]
    return reference, parameters


def moving_policy(env, actor, settings, parameters, reference, gait_params):
    wait, phase, arcs_only, enable = parameters
    ordinary, standard_phase = assisted_policy(env, actor, settings)
    procedural = ReverseEntry(env, CalibratedYaw(actor, settings[4]), reference, *gait_params, record=False)
    speed = int(env.model.sensor('local_linvel').adr[0])
    policy = MovingEntry(ordinary, procedural, speed, wait, .03, arcs_only, settings[3], enable)

    def select(controller):
        return phase if policy.use_procedure else standard_phase(controller)

    return policy, select


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matched', action='store_true', help='Compare both controllers on the same new seeds')
    args = parser.parse_args()
    refined = json.loads((STAGE / 'reverse_adaptive_refine.json').read_text())
    run = Path(refined['source'])
    env, _ = make_local_env(run, 'initial', cpu_restore=True)
    actor = env.policy
    settings = refined['best']
    reference, gait_params = prepare_reference(env)
    routes = ROUTES

    def trial(parameters, case, order, seed, ramp):
        env.policy, phase = moving_policy(env, actor, settings, parameters, reference, gait_params)
        plan = bridge_reverse([(action, 200) for action in order], settings[3])
        segments = rollout(env, seed, plan, ramp=ramp, reverse_entry_phase=phase, command_filter=env.policy.filter)
        complete = len(segments) == len(plan) and not segments[-1]['terminated']
        tracked = sum(tracks_command(row) for row in segments if row['steps'] >= 100)
        return dict(case=case, seed=seed, ramp=ramp, complete=complete, passed=complete and tracked == len(order),
                    tracked=tracked, segments=segments)

    if args.matched:
        selected = json.loads((STAGE / 'reverse_moving_entry_probe.json').read_text())['selected']
        matched = {}
        for name, route in routes.items():
            reference_rows = [trial((0, 0., False, False), name, route, seed, True) for seed in range(6001, 6021)]
            improved = [trial(selected, name, route, seed, True) for seed in range(6001, 6021)]
            if name == 'standard':
                assert reference_rows == improved, 'The moving-entry change affected the standard route'
            counts = [sum(row['passed'] for row in result) for result in (reference_rows, improved)]
            matched[name] = dict(original=reference_rows, candidate=improved, counts=counts)
            print('Matched handpad comparison', name, counts, '/20', flush=True)
        (STAGE / 'reverse_moving_entry_matched.json').write_text(json.dumps(
            dict(source=str(run), selected=selected, matched=matched), indent=2, allow_nan=False) + '\n')
        return

    options = [(0, 0., False, False)]
    options += [(wait, phase, arcs_only, True) for wait in (10, 24) for phase in (.0, .5, .66)
                for arcs_only in (False, True)]
    results = []
    for params in options:
        cases = [trial(params, case, order, seed, ramp) for case, order in routes.items()
                 for seed in (5901, 5902) for ramp in (False, True)]
        score = (sum(row['passed'] for row in cases), sum(row['tracked'] for row in cases),
                 sum(row['complete'] for row in cases))
        print('Moving-entry variant:', params, 'whole-route passes:', score[0], '/', len(cases), flush=True)
        results.append(dict(parameters=params, score=score, cases=cases))
    winner = max(results, key=lambda row: row['score'])
    observed = [trial(winner['parameters'], case, order, seed, ramp) for case, order in routes.items()
                for seed in range(5941, 5953) for ramp in (False, True)]
    counts = {f'{case}:{ramp}': sum(row['passed'] for row in observed if row['case'] == case and row['ramp'] == ramp)
              for case in routes for ramp in (False, True)}
    print('Moving-entry holdout:', winner['parameters'], counts, flush=True)
    (STAGE / 'reverse_moving_entry_probe.json').write_text(json.dumps(
        dict(source=str(run), selected=winner['parameters'], screening=results, tested=observed, counts=counts),
        indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
