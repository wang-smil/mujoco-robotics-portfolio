"""Run one owned child with a JAX allocator budget and measured GPU headroom.

Sampling is not a guarantee against brief peaks or allocation failures. Never
find/kill other Python jobs; termination is limited to the Popen object we own.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time


def sample_resources():
    result=subprocess.run(['/usr/lib/wsl/lib/nvidia-smi',
        '--query-gpu=memory.total,memory.used','--format=csv,noheader,nounits'],
        capture_output=True,text=True,check=True,timeout=3)
    total,used=(int(value.strip()) for value in result.stdout.splitlines()[0].split(','))
    memory={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
    return dict(gpu_total_mib=total,gpu_used_mib=used,host_available_mib=memory['MemAvailable']//1024)


def run_guarded(command, directory, sampler=sample_resources, interval=.5,
                timeout=1800, gpu_headroom_mib=768, host_headroom_mib=1024):
    directory.mkdir(parents=True,exist_ok=True)
    result=dict(command=command,status='running',child_started=False,exit_code=None,
        observed_peak_gpu_mib=0,minimum_host_available_mib=None,
        gpu_headroom_mib=gpu_headroom_mib,host_headroom_mib=host_headroom_mib,
        allocator_fraction='0.65',sampling_interval_s=interval,
        started_utc=datetime.now(timezone.utc).isoformat())
    started=time.monotonic()
    child=None

    def save():
        result['elapsed_s']=round(time.monotonic()-started,3)
        (directory/'guard.json').write_text(json.dumps(result,indent=2)+'\n')

    def check_sample():
        sample=sampler()
        result['last_sample']=sample
        result['observed_peak_gpu_mib']=max(result['observed_peak_gpu_mib'],sample['gpu_used_mib'])
        available=result['minimum_host_available_mib']
        result['minimum_host_available_mib']=sample['host_available_mib'] if available is None else min(available,sample['host_available_mib'])
        with (directory/'memory.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(elapsed_s=time.monotonic()-started,**sample))+'\n')
        if sample['gpu_total_mib']-sample['gpu_used_mib']<gpu_headroom_mib:
            return 'gpu_headroom'
        if sample['host_available_mib']<host_headroom_mib:
            return 'host_headroom'
        return None

    try:
        breach=check_sample()
        if breach:
            result['status']=breach
        else:
            env=dict(os.environ,JAX_PLATFORMS='cuda',OMP_NUM_THREADS='4',
                XLA_PYTHON_CLIENT_PREALLOCATE='false',XLA_PYTHON_CLIENT_MEM_FRACTION='0.65')
            with (directory/'process.log').open('w') as stream:
                child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=env)
                result.update(child_started=True,pid=child.pid)
                while child.poll() is None:
                    breach=check_sample()
                    if breach or time.monotonic()-started>timeout:
                        result['status']=breach or 'timeout'
                        child.terminate()
                        try:
                            child.wait(timeout=8)
                        except subprocess.TimeoutExpired:
                            child.kill(); child.wait(timeout=8)
                        break
                    save()
                    time.sleep(interval)
                result['exit_code']=child.wait()
                if result['status']=='running':
                    result['status']='passed' if child.returncode==0 else 'child_failed'
    except BaseException as error:
        result.update(status='monitor_error',error=f'{type(error).__name__}: {error}')
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=8)
            except subprocess.TimeoutExpired:
                child.kill(); child.wait(timeout=8)
        if child is not None:
            result['exit_code']=child.returncode
    finally:
        save()
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--timeout',type=float,default=1800)
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    command=args.command[1:] if args.command[:1]==['--'] else args.command
    if not command:
        parser.error('Provide the owned child command after --')
    result=run_guarded(command,args.output,timeout=args.timeout)
    print(json.dumps(result,indent=2),flush=True)
    raise SystemExit(0 if result['status']=='passed' else 1)


if __name__=='__main__':
    main()
