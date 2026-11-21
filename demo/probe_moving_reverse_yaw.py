"""Check whether reverse gait yaw feedback improves moving reversals."""
import json
from pathlib import Path

import numpy as np

from benchmark_author import STAGE, make_local_env, rollout
from probe_direction_experts import bridge_reverse
from probe_moving_reverse import ROUTES, moving_policy, prepare_reference
from run_combined_refine import tracks_command


def main():
    matched = json.loads((STAGE / 'reverse_moving_entry_matched.json').read_text())
    settings = json.loads((STAGE / 'reverse_adaptive_refine.json').read_text())['best']
    env, _ = make_local_env(Path(matched['source']), 'initial', cpu_restore=True)
    actor = env.policy
    reference, gait_params = prepare_reference(env)
    baseline = (reference.yaw_bias, reference.yaw_gain)
    schedules = {name: bridge_reverse([(key, 200) for key in actions], settings[3])
                 for name, actions in ROUTES.items() if name != 'standard'}

    def test(offset, name, seed):
        reference.yaw_bias, reference.yaw_gain = np.asarray(baseline) + offset
        env.policy, phase = moving_policy(env, actor, settings, matched['selected'], reference, gait_params)
        schedule = schedules[name]
        segments = rollout(env, seed, schedule, ramp=True, reverse_entry_phase=phase, command_filter=env.policy.filter)
        complete = len(segments) == len(schedule) and not segments[-1]['terminated']
        scored = [segment for segment in segments if segment['steps'] >= 100]
        passed = complete and all(tracks_command(segment) for segment in scored)
        return dict(seed=seed, complete=complete, passed=passed, segments=segments)

    def trials(offset, seeds):
        return {name: [test(offset, name, seed) for seed in seeds] for name in schedules}

    def score(rows):
        episodes = [row for route in rows.values() for row in route]
        return (sum(row['complete'] for row in episodes), sum(row['passed'] for row in episodes),
                -sum(s['yaw_abs_mean_error'] for row in episodes for s in row['segments'] if s['segment'] == 'reverse'))

    offsets = [(0., 0.), (0., -.12), (0., -.06), (0., .06), (0., .12), (-.015, 0.), (.015, 0.)]
    screening = []
    for offset in offsets:
        result = score(trials(offset, range(6101, 6105)))
        screening.append(dict(offset=offset, score=result))
        print('Yaw screening', offset, result, flush=True)
    finalists = sorted(screening, key=lambda row: row['score'], reverse=True)[:2]
    finalists = [dict(offset=row['offset'], score=score(trials(row['offset'], range(6141, 6147))))
                 for row in finalists]
    selected = max(finalists, key=lambda row: row['score'])['offset']
    tested = {'baseline': trials(offsets[0], range(6201, 6221))}
    if selected != offsets[0]:
        tested['candidate'] = trials(selected, range(6201, 6221))
    for label, routes in tested.items():
        for name, rows in routes.items():
            print('Yaw held-out', label, name, sum(row['complete'] for row in rows),
                  sum(row['passed'] for row in rows), '/', len(rows), flush=True)
    output = STAGE / 'reverse_moving_yaw_probe.json'
    output.write_text(json.dumps(dict(selected=selected, screening=screening, finalists=finalists, tested=tested),
                                indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
