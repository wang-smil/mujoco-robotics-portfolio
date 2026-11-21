"""Evaluate posture-based reverse entry around a single trained Actor."""
import argparse
import json
from pathlib import Path

import numpy as np

from benchmark_author import STAGE, TARGETS, make_local_env, rollout
from probe_direction_experts import bridge_reverse
from probe_reverse_pitch import PitchFeedback
from probe_steered_reverse import CalibratedYaw
from run_combined_refine import tracks_command


def assisted_policy(env, actor, params):
    home_phase, rest_phase, moving_phase, _, yaw_gain = params[:5]
    gain, damping, deadband = params[5:] if len(params) > 5 else (1.8, .108, .04)
    velocity = int(env.model.sensor('local_linvel').adr[0])
    policy = PitchFeedback(env, CalibratedYaw(actor, yaw_gain), gain, damping, .16, deadband)

    def select_phase(controller):
        data = controller.data
        if data.time < .35:
            return home_phase
        speed = np.linalg.norm(data.sensordata[velocity:velocity + 2])
        yaw_rate = abs(controller.get_gyro(data)[2])
        return rest_phase if speed <= .03 and yaw_rate <= .10 else moving_phase

    return policy, select_phase


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--confirm', action='store_true', help='Repeat the selected controller on untouched seeds')
    mode.add_argument('--refine', action='store_true', help='Refine only the remaining starting and stopping errors')
    mode.add_argument('--extended-confirm', action='store_true', help='Validate the released preset on harder sequences')
    args = parser.parse_args()
    source = Path((STAGE / 'balance_imitation_run.txt').read_text().strip())
    env, _ = make_local_env(source, 'initial', cpu_restore=True)
    actor = env.policy
    routes = {
        'stand': ('stop', 'reverse', 'stop'),
        'forward': ('stop', 'forward_010', 'reverse', 'forward_010', 'stop'),
        'mirror': ('stop', 'right', 'reverse', 'left', 'walk_right', 'stop'),
        'arcs': ('stop', 'walk_right', 'reverse', 'walk_left', 'reverse', 'walk_right', 'stop'),
        'standard': ('stop', 'forward_010', 'left', 'forward_010', 'right', 'reverse', 'walk_left', 'walk_right', 'stop'),
    }

    def trial(params, name, actions, seed, ramp):
        pause = params[3]
        env.policy, select_phase = assisted_policy(env, actor, params)
        ticks = 1000 if len(actions) == 1 else 200
        original = [(action, ticks) for action in actions]
        schedule = bridge_reverse(original, pause)
        result = rollout(env, seed, schedule, ramp=ramp, reverse_entry_phase=select_phase)
        complete = len(result) == len(schedule) and not result[-1]['terminated']
        tracked = sum(tracks_command(row) for row in result if row['steps'] >= 100)
        return dict(case=name, seed=seed, ramp=ramp, complete=complete,
                    passed=complete and tracked == len(original), tracked=tracked, segments=result)

    if args.extended_confirm:
        configuration = json.loads((STAGE / 'reverse_adaptive_refine.json').read_text())
        assert configuration['source'] == str(source)
        settings = configuration['best']
        tested = [trial(settings, name, actions, seed, ramp) for name, actions in routes.items()
                  for seed in range(5801, 5821) for ramp in (False, True)]
        results = {f'{name}:{ramp}': sum(row['passed'] for row in tested if row['case'] == name and row['ramp'] == ramp)
                   for name in routes for ramp in (False, True)}
        print('Harder routes on untouched seeds:', results, flush=True)
        (STAGE / 'reverse_adaptive_extended.json').write_text(json.dumps(
            dict(source=str(source), best=settings, counts=results, verified=tested),
            indent=2, allow_nan=False) + '\n')
        return

    if args.confirm:
        previous = json.loads((STAGE / 'reverse_adaptive_probe.json').read_text())
        assert previous['source'] == str(source)
        settings = previous['best']
        protocol = {'standard': routes['standard'], **{name: (name,) for name in TARGETS}}
        independent = [trial(settings, name, actions, seed, ramp) for name, actions in protocol.items()
                       for seed in range(5601, 5621) for ramp in (False, True)]
        counts = {f'{name}:{ramp}': sum(row['passed'] for row in independent if row['case'] == name and row['ramp'] == ramp)
                  for name in protocol for ramp in (False, True)}
        print('Untouched seeds:', counts, flush=True)
        (STAGE / 'reverse_adaptive_confirm.json').write_text(json.dumps(
            dict(source=str(source), settings=settings, counts=counts, verified=independent), indent=2, allow_nan=False) + '\n')
        return

    if args.refine:
        selected = json.loads((STAGE / 'reverse_adaptive_probe.json').read_text())['best']
        _, rest, moving, pause, yaw_gain = selected
        settings = [tuple(selected)]
        settings += [(home, rest, moving, pause, yaw, gain, damping, band)
                     for home in (None, .0, .5) for yaw in (yaw_gain, 1.12)
                     for gain, damping, band in ((1.8, .108, .04), (2.1, .126, .03), (1.6, .112, .04))]
        focused = {'standard': routes['standard'], 'reverse': ('reverse',)}
        results = []
        for variant in settings:
            cases = [trial(variant, name, actions, seed, True) for name, actions in focused.items()
                     for seed in range(5661, 5666)]
            score = (sum(row['passed'] for row in cases), sum(row['complete'] for row in cases))
            results.append(dict(configuration=variant, score=score, cases=cases))
            print('Refined assisted controller', variant, 'passes / survival', score, flush=True)
        best = max(results, key=lambda row: row['score'])['configuration']
        protocol = {'standard': routes['standard'], **{name: (name,) for name in TARGETS}}
        independent = [trial(best, name, actions, seed, ramp) for name, actions in protocol.items()
                       for seed in range(5701, 5721) for ramp in (False, True)]
        counts = {f'{name}:{ramp}': sum(row['passed'] for row in independent if row['case'] == name and row['ramp'] == ramp)
                  for name in protocol for ramp in (False, True)}
        print('Refined controller confirmed:', best, counts, flush=True)
        (STAGE / 'reverse_adaptive_refine.json').write_text(json.dumps(
            dict(source=str(source), best=best, counts=counts, screening=results, verified=independent),
            indent=2, allow_nan=False) + '\n')
        return

    variants = [(.5, .5, .5, 20, 1.)]
    variants += [(.5, rest, move, 20, 1.2) for rest in (0., .17, .33, .5, .66) for move in (.17, .5, .66)]
    variants += [(.5, .17, .5, pause, 1.2) for pause in (8, 32, 45)]
    results = []
    for params in variants:
        samples = [trial(params, name, actions, seed, ramp) for name, actions in routes.items()
                   for seed in (4201, 4202) for ramp in (False, True)]
        score = (sum(item['passed'] for item in samples), sum(item['tracked'] for item in samples),
                 sum(item['complete'] for item in samples))
        results.append(dict(settings=params, score=score, cases=samples))
        print('Adaptive reverse settings:', params, 'full tracking:', score[0], '/', len(samples), flush=True)
    best = max(results, key=lambda result: result['score'])
    routes.update({name: (name,) for name in TARGETS})
    independent = [trial(best['settings'], name, actions, seed, ramp) for name, actions in routes.items()
                   for seed in range(4231, 4243) for ramp in (False, True)]
    counts = {f'{name}:{ramp}': sum(row['passed'] for row in independent if row['case'] == name and row['ramp'] == ramp)
              for name in routes for ramp in (False, True)}
    print('Validated:', best['settings'], counts, flush=True)
    (STAGE / 'reverse_adaptive_probe.json').write_text(json.dumps(
        dict(source=str(source), pilots=results, best=best['settings'], validated=independent, counts=counts),
        indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
