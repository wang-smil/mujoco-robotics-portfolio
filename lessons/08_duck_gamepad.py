"""Local clickable/keyboard gamepad for the saved lesson07 MuJoCo policy."""
import argparse
from contextlib import nullcontext
import importlib.util
import json
import math
import queue
import secrets
import hmac
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import threading
import time

ROOT=Path(__file__).resolve().parents[1]


class ControlState:
    def __init__(self,timeout=1.5):
        self.lock=threading.RLock();self.timeout=timeout
        self.last_input=time.monotonic();self.target=[0.,0.,0.]
        self.paused=True;self.failed=False;self.closing=False
        self.reset_id=0;self.stop_id=0;self.note='就绪，点击前进开始'
        self.telemetry={}

    def heartbeat(self,now=None):
        with self.lock:self.last_input=time.monotonic() if now is None else now

    def apply(self,action,speed=.1,now=None):
        speed=float(speed)
        if not math.isfinite(speed):raise ValueError('speed must be finite')
        speed=min(.15,max(.05,speed))
        moves={'forward':[speed,0.,0.],'backward':[-min(speed,.1),0.,0.],
            'left':[0.,0.,.3],'right':[0.,0.,-.3],
            'forward_left':[speed,0.,.3],'forward_right':[speed,0.,-.3],
            'slide_left':[0.,.05,0.],'slide_right':[0.,-.05,0.]}
        with self.lock:
            if action not in (*moves,'stop','pause','reset','close'):raise ValueError('unknown action')
            self.heartbeat(now)
            if action=='reset':
                self.reset_id+=1;self.stop_id+=1;self.target=[0.,0.,0.]
                self.failed=False;self.paused=True;self.note='已重置，点击前进开始'
            elif action=='close':self.closing=True;self.paused=True;self.target=[0.,0.,0.]
            elif self.failed:self.note='已触发物理失败，请按R重置'
            elif action=='pause':self.paused=not self.paused;self.note='已暂停' if self.paused else '继续运行'
            elif action=='stop':
                self.target=[0.,0.,0.];self.stop_id+=1
                self.note='速度命令已归零，仿真仍暂停' if self.paused else '速度命令归零，策略继续站稳'
            else:self.target=moves[action];self.paused=False;self.note='连续策略控制中'

    def fail(self,reason,reset_id=None):
        with self.lock:
            if reset_id is not None and reset_id!=self.reset_id:return False
            self.failed=True;self.paused=True;self.target=[0.,0.,0.];self.stop_id+=1
            self.note=reason+'；请按R重置'
            return True

    def update(self,**values):
        with self.lock:self.telemetry.update(values)

    def snapshot(self,now=None):
        now=time.monotonic() if now is None else now
        with self.lock:
            if now-self.last_input>self.timeout and (not self.paused or any(self.target)):
                self.paused=True;self.target=[0.,0.,0.];self.stop_id+=1;self.note='控制面板断连，已归零并暂停'
            return dict(target=list(self.target),paused=self.paused,failed=self.failed,
                closing=self.closing,reset_id=self.reset_id,stop_id=self.stop_id,note=self.note,
                experimental=self.target[0]<0 or self.target[1]!=0 or self.target[2]!=0,
                **self.telemetry)


def ramp_command(current,target,dt):
    import numpy as np
    current=np.asarray(current,dtype=float);target=np.asarray(target,dtype=float)
    return current+np.clip(target-current,-np.array([.3,.3,1.])*dt,np.array([.3,.3,1.])*dt)


def patch_live_command(state,command,reference,balance_observations=False):
    import jax.numpy as jp
    from playground.open_duck_mini_v2.locomotion import patch_command_observation
    info=dict(state.info);cmd=jp.concatenate([command,jp.zeros(4)])
    ref=reference.get_reference_motion(*command,info['imitation_i'])
    info.update(command=cmd,current_reference_motion=ref,step=jp.zeros_like(info['step']))
    if 'command_target' in info:
        info.update(command_target=cmd,command_ramp_active=jp.asarray(False))
    obs=patch_command_observation(state.obs,cmd,ref)
    if balance_observations:
        obs={name: value.at[9:13].set(state.obs[name][9:13]) for name,value in obs.items()}
    return state.replace(info=info,obs=obs)


def start_server(control,html,port=0):
    token=secrets.token_urlsafe(24)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def reply(self,status,value,kind='application/json; charset=utf-8'):
            body=value if isinstance(value,bytes) else json.dumps(value,ensure_ascii=False,allow_nan=False).encode()
            self.send_response(status);self.send_header('Content-Type',kind)
            self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)))
            self.end_headers();self.wfile.write(body)
        def authorized(self):
            return hmac.compare_digest(self.headers.get('X-Control-Token',''),token)
        def do_GET(self):
            if self.path=='/':return self.reply(200,html,'text/html; charset=utf-8')
            if self.path!='/api/state':return self.reply(404,{'error':'not found'})
            if not self.authorized():return self.reply(403,{'error':'control token required'})
            self.reply(200,control.snapshot())
        def do_POST(self):
            if self.path!='/api/control':return self.reply(404,{'error':'not found'})
            if not self.authorized():return self.reply(403,{'error':'control token required'})
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=4096:raise ValueError('invalid body size')
                body=json.loads(self.rfile.read(length))
                action=body.get('action')
                if action=='heartbeat':control.heartbeat()
                else:control.apply(action,body.get('speed',.1))
            except (ValueError,TypeError,AttributeError) as error:return self.reply(400,{'error':str(error)})
            self.reply(200,control.snapshot())
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    server.daemon_threads=True
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    url=f'http://127.0.0.1:{server.server_port}/#token={token}'
    return server,url


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path)
    parser.add_argument('--checkpoint')
    parser.add_argument('--seed',type=int,default=11)
    parser.add_argument('--port',type=int,default=0)
    parser.add_argument('--status-file',type=Path,default=ROOT/'artifacts/gamepad/status.json')
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--seconds',type=float,help='Finite integration session; default unlimited')
    args=parser.parse_args()
    if args.seconds is not None and (not math.isfinite(args.seconds) or args.seconds<=0):parser.error('seconds must be positive')
    args.status_file=args.status_file.resolve()
    lesson_path=ROOT/'lessons/07_duck_ppo.py'
    spec=importlib.util.spec_from_file_location('duck_lesson',lesson_path)
    lesson=importlib.util.module_from_spec(spec);spec.loader.exec_module(lesson)
    run=(args.run or Path(lesson.LATEST.read_text().strip())).resolve()
    cfg=json.loads((run/'config.json').read_text());lesson.validate_sources(cfg)
    checkpoint=lesson.latest_checkpoint(run,cfg.get('final_checkpoint')) if not args.checkpoint else (run/'checkpoints'/args.checkpoint).resolve()
    if not checkpoint.is_relative_to((run/'checkpoints').resolve()):raise ValueError('checkpoint escapes run')
    print('Loading saved policy:',checkpoint,flush=True)
    lesson.load_runtime();env=lesson.make_environment(cfg)
    if "reverse_snapshot_probability" in env._config:
        env._config.reverse_snapshot_probability=0.
    make_policy,parameters=lesson.restore(env,cfg['ppo'],checkpoint,lesson.network_from_config(cfg))
    policy=lesson.deterministic(make_policy,parameters)
    reset=lesson.jax.jit(env.reset)
    balance_observations=env._config.get('balance_observations',False)
    def advance(state,command):
        state=patch_live_command(state,command,env.PRM,balance_observations)
        action,_=policy(state.obs,lesson.jax.random.PRNGKey(0))
        state=env.step(state,action)
        return state,env.get_local_linvel(state.data),env.get_gyro(state.data)
    advance=lesson.jax.jit(advance)
    state=reset(lesson.jax.random.PRNGKey(args.seed))
    print('Compiling live controller...',flush=True)
    warm,_,_=lesson.jax.block_until_ready(advance(state,lesson.jp.zeros(3)))
    lesson.jax.block_until_ready(advance(warm,lesson.jp.zeros(3)))
    control=ControlState();events=queue.SimpleQueue()
    html=(ROOT/'lessons/duck_gamepad.html').read_bytes()
    server,url=start_server(control,html,args.port)
    model=lesson.mujoco.MjModel.from_xml_path(str(lesson.WORK/'playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml'))
    display=lesson.mujoco.MjData(model)
    if not args.headless:
        import mujoco.viewer
        context=mujoco.viewer.launch_passive(model,display,key_callback=events.put,show_left_ui=False,show_right_ui=False)
    else:context=nullcontext(None)
    status=dict(url=url,run=str(run),checkpoint=str(checkpoint),seed=args.seed,ready=True,pid=__import__('os').getpid())
    args.status_file.parent.mkdir(parents=True,exist_ok=True)
    def write_status(value):
        temporary=args.status_file.with_suffix('.tmp')
        temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2))
        temporary.replace(args.status_file)
    write_status(status)
    print('GAMEPAD_URL:',url,flush=True)
    keys={87:'forward',83:'backward',65:'left',68:'right',81:'forward_left',69:'forward_right',32:'stop',80:'pause',82:'reset'}
    applied=lesson.np.zeros(3);last_reset=0;last_stop=0;frames=0;started=time.monotonic()
    origin=lesson.np.asarray(state.data.qpos[:2]);last_yaw=None;heading=0.
    velocity=env.get_local_linvel(state.data);gyro=env.get_gyro(state.data)
    def publish():
        q=lesson.np.asarray(state.data.qpos);position=q[:2]-origin
        control.update(applied=applied.tolist(),actual_velocity=lesson.np.asarray(velocity).tolist(),
            actual_yaw_rate=float(gyro[2]),position=position.tolist(),distance=float(lesson.np.linalg.norm(position)),
            heading_deg=heading*180/math.pi,sim_seconds=float(state.data.time),frames=frames,seed=args.seed)
        write_status({**status,**control.snapshot()})
    try:
        with context as handle:
            if handle:
                with handle.lock():
                    handle.cam.type=lesson.mujoco.mjtCamera.mjCAMERA_TRACKING
                    handle.cam.trackbodyid=model.body('trunk_assembly').id
                    handle.cam.distance=.85;handle.cam.azimuth=40;handle.cam.elevation=-15
                    handle.opt.geomgroup[3]=0
            last_publish=0.
            while not control.snapshot()['closing'] and (handle is None or handle.is_running()):
                tick=time.monotonic()
                if args.seconds and tick-started>=args.seconds:break
                while not events.empty():
                    key=events.get()
                    if key in keys:control.apply(keys[key])
                snapshot=control.snapshot()
                if snapshot['reset_id']!=last_reset:
                    state=reset(lesson.jax.random.PRNGKey(args.seed));last_reset=snapshot['reset_id']
                    applied[:]=0;frames=0;heading=0;last_yaw=None;velocity=env.get_local_linvel(state.data);gyro=env.get_gyro(state.data)
                    origin=lesson.np.asarray(state.data.qpos[:2])
                if snapshot['stop_id']!=last_stop:applied[:]=0;last_stop=snapshot['stop_id']
                if not snapshot['paused']:
                    applied=ramp_command(applied,snapshot['target'],env.dt)
                    state,velocity,gyro=advance(state,lesson.jp.asarray(applied,dtype=lesson.jp.float32))
                    frames+=1
                    q=lesson.np.asarray(state.data.qpos);w,x,y,z=q[3:7]
                    yaw=math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
                    if last_yaw is not None:heading+=math.atan2(math.sin(yaw-last_yaw),math.cos(yaw-last_yaw))
                    last_yaw=yaw
                    if bool(state.done):control.fail('触发物理失败',reset_id=snapshot['reset_id'])
                if handle:
                    with handle.lock():
                        display.qpos[:]=lesson.np.asarray(state.data.qpos);display.qvel[:]=lesson.np.asarray(state.data.qvel)
                        display.ctrl[:]=lesson.np.asarray(state.data.ctrl);display.time=float(state.data.time)
                        lesson.mujoco.mj_forward(model,display)
                    handle.sync()
                if tick-last_publish>=.1:publish();last_publish=tick
                time.sleep(max(0.,env.dt-(time.monotonic()-tick)))
    finally:
        control.apply('close');status['ready']=False;publish();server.shutdown();server.server_close()
        print('Gamepad session closed.',flush=True)


if __name__=='__main__':main()
