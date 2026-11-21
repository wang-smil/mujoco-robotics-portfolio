"""Fixed stop/forward tests under ONE saved environment for all candidates.

Command resampling is frozen; imitation phase, noise, action delay and push
histories keep advancing. This is a finite validation sample, not certification.
"""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import numpy as np



def evaluation_environment_config(target,candidate,saved_environments=False,disturbances=False):
    protocol=copy.deepcopy(candidate if saved_environments else target)
    if disturbances:
        protocol['environment']['noise_config']['level']=1.
        protocol['environment']['push_config']['enable']=True
        if 'reset_pose_scale' in protocol['environment']:
            protocol['environment']['reset_pose_scale']=1.
    return protocol


def whole_target_rate(ctrl,initial_ctrl,dt):
    targets=np.vstack([np.asarray(initial_ctrl)[None,5:9],np.asarray(ctrl)[:,5:9]])
    rates=np.diff(targets,axis=0)/dt
    return float(np.abs(rates).max()) if rates.size and np.isfinite(rates).all() else None


def head_metrics(trace,dt):
    q=np.asarray(trace['qpos'])[:,12:16]
    v=np.asarray(trace['qvel'])[:,11:15]
    targets=np.asarray(trace['ctrl'])[:,5:9]
    rate=np.diff(targets,axis=0)/dt
    safe=lambda value:float(value) if np.isfinite(value) else None
    finite=all(np.isfinite(x).all() for x in (q,v,targets))
    result=dict(measurement_finite=bool(finite),
        joint_range_deg=[safe(x) for x in np.ptp(q,axis=0)*180/np.pi],
        joint_velocity_rms_rad_s=safe(np.sqrt(np.mean(v*v))),
        joint_velocity_max_rad_s=safe(np.abs(v).max()),
        target_rate_max_rad_s=safe(np.abs(rate).max()) if rate.size else None)
    if 'head_quaternion' in trace:
        quat=np.asarray(trace['head_quaternion'],dtype=float)
        norms=np.linalg.norm(quat,axis=1,keepdims=True)
        valid=bool(np.isfinite(quat).all() and (norms>0).all())
        result['measurement_finite']=bool(finite and valid)
        if valid:
            quat=quat/norms
            dot=np.abs(np.sum(quat[1:]*quat[:-1],axis=1)).clip(0,1)
            speed=2*np.arctan2(np.sqrt(np.maximum(0,1-dot*dot)),dot)/dt
        else:speed=np.array([np.nan])
        result['world_angular_speed_rms_rad_s']=safe(np.sqrt(np.mean(speed*speed))) if len(speed) else None
        result['world_angular_speed_p95_rad_s']=safe(np.percentile(speed,95)) if len(speed) else None
    return result


def summarize_episode(trace, command, dt=.02, settle_seconds=1., cycle_seconds=.54):
    failures=np.flatnonzero(trace['done'])
    n=int(failures[0]+1) if len(failures) else len(trace['done'])
    warmup=int(settle_seconds/dt)
    start=warmup if n>warmup else 0
    velocity=np.asarray(trace['velocity'])[start:n]
    gyro=np.asarray(trace['gyro'])[start:n]
    position=np.asarray(trace['position'])[:n]
    up=np.asarray(trace['up'])[:n]
    feet=np.asarray(trace['feet_velocity'])[start:n,:,:2]
    contact=np.asarray(trace['contact'])[start:n]
    finite=bool(all(np.isfinite(np.asarray(v)[:n]).all() for v in trace.values()))
    finite=finite and not bool(np.asarray(trace.get('numerical_failure',False))[:n].any()) if 'numerical_failure' in trace else finite
    error=np.sum((velocity[:,:2]-np.asarray(command)[:2])**2,axis=1)
    window=min(len(velocity),max(1,int(round(cycle_seconds/dt))))
    kernel=np.ones(window)/window
    cycle_velocity=np.column_stack([np.convolve(velocity[:,axis],kernel,mode='valid') for axis in range(2)])
    cycle_yaw=np.convolve(gyro[:,2],kernel,mode='valid')
    slip=np.linalg.norm(feet,axis=-1)
    initial_position=np.asarray(trace.get('initial_position',position[0]))
    displacement=position[-1,:2]-initial_position[:2]
    contact_samples=int(contact.sum())
    return dict(steps=n,duration_s=n*dt,terminated=bool(len(failures)),finite=finite,
        settle_applied=start>0,mean_body_velocity=np.nan_to_num(velocity.mean(0)).tolist(),
        rms_planar_error=float(np.nan_to_num(np.sqrt(np.mean(error)),nan=1e6,posinf=1e6)),
        rms_yaw_error=float(np.nan_to_num(np.sqrt(np.mean((gyro[:,2]-command[2])**2)),nan=1e6,posinf=1e6)),
        cycle_window_samples=window,
        cycle_rms_planar_error=float(np.nan_to_num(np.sqrt(np.mean(np.sum((cycle_velocity-np.asarray(command)[:2])**2,axis=1))),nan=1e6,posinf=1e6)),
        cycle_rms_yaw_error=float(np.nan_to_num(np.sqrt(np.mean((cycle_yaw-command[2])**2)),nan=1e6,posinf=1e6)),
        max_tilt_degrees=float(np.nan_to_num(np.degrees(np.arccos(np.clip(up[:,2],-1,1))).max(),nan=180)),
        world_xy_displacement_m=displacement.tolist() if np.isfinite(displacement).all() else None,
        contact_samples=contact_samples,
        contact_slip_speed=float(np.nan_to_num(np.sum(slip*contact)/contact_samples,nan=1e6)) if contact_samples else None,
        contact_transitions=int(np.count_nonzero(np.diff(contact.astype(int),axis=0))),
        near_still_fraction=float(np.mean(np.linalg.norm(velocity[:,:2],axis=1)<.02)))


def basic_task_pass(episode):
    if episode['terminated'] or not episode['finite']:
        return False
    if episode['case']=='forward':
        desired=episode.get('command',[.1,0.,0.])[0]
        tolerance=min(.03,.3*abs(desired))
        return (abs(episode['mean_body_velocity'][0]-desired)<=tolerance
                and episode.get('cycle_rms_planar_error',episode['rms_planar_error'])<=tolerance
                and episode.get('cycle_rms_yaw_error',episode['rms_yaw_error'])<=.1)
    drift=episode['world_xy_displacement_m']
    return (episode['rms_planar_error']<=.02 and episode['rms_yaw_error']<=.1
            and drift is not None and np.linalg.norm(drift)<=.05)


def make_rollout(lesson,env,make_policy,steps,park_failed=False):
    jax,jp=lesson.jax,lesson.jp
    head_body_id=env._mj_model.body("head_assembly").id
    lesson.disable_assisted_resets(env)

    def reset(key,command):
        return lesson.patch_evaluation_command(env,env.reset(key),command)

    def rollout(parameters,keys,commands):
        state=jax.vmap(reset)(keys,commands)
        initial_position=state.data.qpos[:,:3]
        initial_ctrl=state.data.ctrl
        initial_state=state
        policy=make_policy(parameters,deterministic=True)
        def advance(carry,unused):
            state,failed=carry
            previous=state
            def choose(new,old):
                mask=failed.reshape((len(commands),)+(1,)*(new.ndim-1))
                return jp.where(mask,old,new)
            if park_failed:
                # Evaluate inactive lanes from the healthy reset state. Keep
                # terminal samples in the trace, not in the physics solver.
                state=jax.tree_util.tree_map(choose,state,initial_state)
            def advance_physics(value):
                # Freeze random command refresh without resetting phase or pushes.
                info = {**value.info, 'step': jp.zeros_like(value.info['step'])}
                action, _ = policy(value.obs, jax.random.PRNGKey(0))
                return jax.vmap(env.step)(value.replace(info=info), action)
            if park_failed:
                proposed = jax.lax.cond(jp.all(failed), lambda _: previous, advance_physics, state)
            else:
                state.info['step'] = jp.zeros_like(state.info['step'])
                action, _ = policy(state.obs, jax.random.PRNGKey(0))
                proposed = jax.vmap(env.step)(state, action)
            state=jax.tree_util.tree_map(choose,proposed,previous)
            done=failed | proposed.done.astype(bool)
            values=dict(done=done,numerical_failure=state.info['invalid_transition'],reset_position=initial_position,reset_ctrl=initial_ctrl,velocity=jax.vmap(env.get_local_linvel)(state.data),
                gyro=jax.vmap(env.get_gyro)(state.data),up=jax.vmap(env.get_gravity)(state.data),
                position=state.data.qpos[:,:3],contact=state.info['last_contact'],
                qpos=state.data.qpos,qvel=state.data.qvel,ctrl=state.data.ctrl,
                head_quaternion=state.data.xquat[:,head_body_id],
                feet_velocity=state.data.sensordata[:,env._foot_linvel_sensor_adr])
            return (state,done),values
        trace=jax.lax.scan(advance,(state,jp.zeros(len(commands),bool)),None,length=steps)[1]
        return trace, initial_position
    return jax.jit(rollout)


def run_batched_rollout(rollout,keys,commands,batch_cases=16):
    count=len(keys)
    if batch_cases<1 or count<1 or len(commands)!=count:
        raise ValueError('Positive batch size and matching nonempty cases required')
    traces=[];positions=[]
    for start in range(0,count,batch_cases):
        end=min(start+batch_cases,count);valid=end-start
        batch_keys=keys[start:end];batch_commands=commands[start:end]
        if valid<batch_cases:
            batch_keys=np.concatenate([batch_keys,np.repeat(np.asarray(batch_keys[-1:]),batch_cases-valid,axis=0)])
            batch_commands=np.concatenate([batch_commands,np.repeat(np.asarray(batch_commands[-1:]),batch_cases-valid,axis=0)])
        trace,initial=rollout(batch_keys,batch_commands)
        traces.append({key:np.asarray(value)[:,:valid] for key,value in trace.items()})
        positions.append(np.asarray(initial)[:valid])
    return {key:np.concatenate([trace[key] for trace in traces],axis=1) for key in traces[0]},np.concatenate(positions,axis=0)


def candidate_paths(run,config,explicit):
    names=list(explicit or [])
    if not names:
        rows=[json.loads(line) for line in (run/'progress.jsonl').read_text().splitlines()]
        rows=[row for row in rows if row['step'] and (run/'checkpoints'/f"step_{row['step']}").is_dir()]
        if rows:
            peak=max(rows,key=lambda row:row['metrics']['eval/episode_reward'])
            names.append(f"step_{peak['step']}")
        names.append(Path(config['final_checkpoint']).name)
    paths=[]
    for name in dict.fromkeys(names):
        path=(run/'checkpoints'/name).resolve()
        if not path.is_relative_to((run/'checkpoints').resolve()) or not path.is_dir():
            raise ValueError(f'Invalid checkpoint: {name}')
        paths.append(path)
    return paths


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--checkpoint',action='append')
    parser.add_argument('--compare-run',type=Path)
    parser.add_argument('--compare-checkpoint',help='Explicit baseline step_* checkpoint')
    parser.add_argument('--saved-environments',action='store_true',help='Restore each candidate controller/environment; report differences explicitly')
    parser.add_argument('--steps',type=int,default=1000)
    parser.add_argument('--batch-cases',type=int,default=16,help='Bound JIT evaluation batch; all requested cases retained')
    parser.add_argument('--seeds',type=int,nargs='+',default=[11,22,33,44])
    parser.add_argument('--speeds',type=float,nargs='+',default=[.1],help='Fixed positive forward speeds; default .1; final sweep .05 .1 .15')
    parser.add_argument('--disturbances',action='store_true',help='Override saved noise to1 and enable saved push magnitudes; report this separately')
    args=parser.parse_args()
    if args.steps<1:parser.error('steps must be positive')
    if args.batch_cases<1:parser.error('batch-cases must be positive')
    if not all(np.isfinite(v) and v>0 for v in args.speeds):parser.error('speeds must be finite and positive')
    run=args.run.resolve()
    baseline=args.compare_run.resolve() if args.compare_run else None
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('duck_lesson',root/'lessons/07_duck_ppo.py')
    lesson=importlib.util.module_from_spec(spec);spec.loader.exec_module(lesson)
    lesson.load_runtime()
    config=json.loads((run/'config.json').read_text())
    lesson.validate_sources(config)
    protocol=evaluation_environment_config(config,config,args.saved_environments,args.disturbances)
    env=lesson.make_environment(protocol)
    settings=dict(config['ppo'],num_envs=8,num_eval_envs=1,num_minibatches=1,batch_size=8)
    network=lesson.network_from_config(config)
    commands_to_test=[('stop',[0.,0.,0.])]+[('forward',[v,0.,0.]) for v in args.speeds]
    cases=[(name,seed,command) for name,command in commands_to_test for seed in args.seeds]
    keys=lesson.jp.stack([lesson.jax.random.PRNGKey(seed) for _,seed,_ in cases])
    commands=lesson.jp.array([command for _,_,command in cases])
    output=run/'evaluations'/lesson.unique_name('fixed_')
    output.mkdir(parents=True)
    shutil.copy2(Path(__file__),output/'evaluation_source.py')
    shutil.copy2(root/'lessons/07_duck_ppo.py',output/'entry_source.py')
    candidates=[('new',path,config) for path in candidate_paths(run,config,args.checkpoint)]
    if args.compare_run:
        baseconfig=json.loads((baseline/'config.json').read_text())
        if lesson.network_from_config(baseconfig)!=network:
            raise ValueError('Batch comparison requires matching network architectures')
        basepath=(candidate_paths(baseline,baseconfig,[args.compare_checkpoint])[0] if args.compare_checkpoint else lesson.latest_checkpoint(baseline,baseconfig.get('final_checkpoint')))
        candidates.append(('baseline',basepath,baseconfig))
    report=dict(protocol=('Fixed stop/forward; each candidate uses its saved controller/environment' if args.saved_environments else 'Fixed stop/forward; one common saved environment for all candidates')+'; deterministic policy, command held constant; tracking/head steady metrics discard first1s; whole target-rate bound includes reset',
        metric_version='gait-neck-v3',evaluation_batch_cases=args.batch_cases,saved_candidate_environments=args.saved_environments,
        acceptance=dict(forward_speed_and_cycle_error='min(0.03m/s,30% of requested speed)',
                        forward_cycle_yaw_error_max=.1,stop_raw_speed_rms_max=.02,
                        stop_raw_yaw_rms_max=.1,stop_drift_max_m=.05),
        environment=protocol['environment'],task_profile=protocol.get('task_profile','upstream'),
        stronger_disturbances=args.disturbances,steps_cap=args.steps,seeds=args.seeds,speeds=args.speeds,candidates=[])
    rollout=None
    for label,checkpoint,candidate_config in candidates:
        if args.saved_environments:
            lesson.validate_sources(candidate_config)
            candidate_protocol=evaluation_environment_config(config,candidate_config,True,args.disturbances)
            env=lesson.make_environment(candidate_protocol)
        else:candidate_protocol=protocol
        make_policy,parameters=lesson.restore(env,settings,checkpoint,network)
        if rollout is None or args.saved_environments:rollout=make_rollout(lesson,env,make_policy,args.steps)
        print(f'FIXED_EVAL start {label} {checkpoint.name}',flush=True)
        trace, initial_position=run_batched_rollout(
            lambda batch_keys,batch_commands:lesson.jax.block_until_ready(rollout(parameters,batch_keys,batch_commands)),
            keys,commands,args.batch_cases)
        trace={key:np.asarray(value) for key,value in trace.items()}
        filename=f'{label}_{checkpoint.name}.npz'
        np.savez_compressed(output/filename,**trace)
        episodes=[]
        for index,(name,seed,command) in enumerate(cases):
            one={key:value[:,index] for key,value in trace.items()}
            one['initial_position']=np.asarray(initial_position[index])
            episode=summarize_episode(one,command,env.dt,cycle_seconds=env.PRM.period)
            start=50 if episode['steps']>50 else 0
            measured={key:value[start:episode['steps']] for key,value in one.items() if key!='initial_position'}
            episode['head']=head_metrics(measured,env.dt)
            episode['head']['whole_episode_target_rate_max_rad_s']=whole_target_rate(one['ctrl'][:episode['steps']],one['reset_ctrl'][0],env.dt)
            episode['head']['steady_start_step']=start
            episode['finite']=episode['finite'] and episode['head']['measurement_finite']
            episode=dict(case=name,seed=seed,command=command,**episode)
            episode['basic_task_pass']=bool(basic_task_pass(episode))
            episodes.append(episode)
        record=dict(label=label,checkpoint=str(checkpoint),trajectory=filename,episodes=episodes,
            task_profile=candidate_protocol.get('task_profile','upstream'),environment=candidate_protocol['environment'],
            failures=sum(ep['terminated'] or not ep['finite'] for ep in episodes),
            task_failures=sum(not ep['basic_task_pass'] for ep in episodes),
            mean_rms_planar_error=float(np.mean([ep['rms_planar_error'] for ep in episodes])))
        record['mean_cycle_rms_planar_error']=float(np.mean([ep['cycle_rms_planar_error'] for ep in episodes]))
        report['candidates'].append(record)
        (output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print('FIXED_EVAL '+json.dumps({key:value for key,value in record.items() if key!='episodes'}),flush=True)
    new=[record for record in report['candidates'] if record['label']=='new']
    recommended=min(new,key=lambda record:(record['task_failures'],record['failures'],record['mean_cycle_rms_planar_error']))
    report['recommended_observation_checkpoint']=recommended['checkpoint']
    report['all_test_episodes_survived']=recommended['failures']==0
    report['all_basic_requirements_passed']=recommended['task_failures']==0
    report['selection_note']='Choose by task failures (including command compliance), then physical failures and tracking error; finite small test sample, not proof of robust locomotion'
    (output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f'FIXED_EVAL_REPORT: {output}',flush=True)


if __name__=='__main__':main()
