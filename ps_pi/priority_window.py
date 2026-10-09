"""Temporarily defer our own cohort managers; finish active children and always resume."""
import argparse
import json
import os
from pathlib import Path
import signal
import time
from .common import ROOT,atomic_json

def identity(pid,expected_module):
    proc=Path(f'/proc/{pid}')
    command=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    if f'-m {expected_module} ' not in command+' ':raise RuntimeError(f'Unexpected process {pid}')
    if (proc/'cwd').resolve()!=ROOT:raise RuntimeError(f'Unexpected working directory for {pid}')
    # Field 22 is the process start tick; protect against PID reuse.
    started=(proc/'stat').read_text().rsplit(')',1)[1].split()[19]
    return command,started

def main():
    p=argparse.ArgumentParser();p.add_argument('--priority-pid',type=int,required=True)
    p.add_argument('--background-pids',type=int,nargs='+',required=True)
    p.add_argument('--priority-module',default='ps_pi.joint_residual_screen',choices=['ps_pi.joint_residual_screen','ps_pi.joint_residual_calibrated_screen'])
    p.add_argument('--log-directory',default='artifacts/logs/cub_joint_residual_v6')
    p.add_argument('--seconds',type=int,default=600);args=p.parse_args()
    priority_identity=identity(args.priority_pid,args.priority_module)
    background={pid:identity(pid,'ps_pi.joint_screen') for pid in args.background_pids}
    path=ROOT/args.log_directory/'priority_window.json'
    stopped=[];started=time.monotonic();reason='unknown'
    try:
        for pid in background:
            os.kill(pid,signal.SIGSTOP);stopped.append(pid)
        atomic_json({'phase':'priority_window','pid':os.getpid(),'priority_pid':args.priority_pid,
                     'deferred_managers':stopped,'active_children_finish_naturally':True,
                     'maximum_seconds':args.seconds,'time':time.time()},path)
        while time.monotonic()-started<args.seconds:
            try:current=identity(args.priority_pid,args.priority_module)
            except (FileNotFoundError,RuntimeError):reason='priority_manager_terminal';break
            if current!=priority_identity:reason='priority_identity_changed';break
            status_path=path.parent/'status.json'
            # A newly started manager may still be importing before its first
            # status write. Its verified live PID remains authoritative.
            status=json.loads(status_path.read_text()) if status_path.exists() else {}
            if status.get('phase') in ['complete','failed']:reason=status['phase'];break
            time.sleep(1)
        else:reason='priority_window_elapsed'
    finally:
        resumed=[]
        for pid in stopped:
            try:
                if identity(pid,'ps_pi.joint_screen')==background[pid]:
                    os.kill(pid,signal.SIGCONT);resumed.append(pid)
            except (FileNotFoundError,RuntimeError):pass
        atomic_json({'phase':'resumed','reason':reason,'resumed_managers':resumed,
                     'elapsed_seconds':time.monotonic()-started,'time':time.time()},path)
        print('Resumed original managers:',resumed,reason,flush=True)

if __name__=='__main__':main()
