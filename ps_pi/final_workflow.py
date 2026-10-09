"""Finish the registered replication, then evaluate locked validation and test sets."""
import json
import os
import subprocess
import sys
import time
from .common import ROOT, atomic_json
from .final_lock import verify_lock


def main():
    verify_lock();logs=ROOT/'artifacts/logs/final_workflow_v1';logs.mkdir(parents=True,exist_ok=True)
    training=ROOT/'artifacts/logs/final_v1/status.json'
    while True:
        state=json.loads(training.read_text())
        if state['phase']=='complete':break
        pid=state['pid'];cmd=ROOT.parent.parent/'unused'
        from pathlib import Path
        proc=Path(f'/proc/{pid}/cmdline')
        if not proc.exists() or b'ps_pi.final_run' not in proc.read_bytes():raise RuntimeError('Registered training controller is no longer live')
        atomic_json({'phase':'await_training','pid':os.getpid(),'training':state,'time':time.time()},logs/'status.json')
        time.sleep(5)
    commands=[['ps_pi.final_evaluate','--split','val'],
              ['ps_pi.final_test_prepare','--dataset','cub'],
              ['ps_pi.final_test_prepare','--dataset','cifar10'],
              ['ps_pi.final_test_prepare','--dataset','cifar100'],
              ['ps_pi.final_evaluate','--split','test']]
    for index,arguments in enumerate(commands):
        with (logs/f'{index}_{arguments[0]}.log').open('a') as stream:
            process=subprocess.Popen([sys.executable,'-u','-m',*arguments],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            atomic_json({'phase':'evaluation','pid':os.getpid(),'child_pid':process.pid,'command':arguments,'index':index,'time':time.time()},logs/'status.json')
            if process.wait():raise RuntimeError(f'Locked evaluation failed: {arguments}')
    atomic_json({'phase':'complete','pid':os.getpid(),'test_used_for_selection':False,'time':time.time()},logs/'status.json')


if __name__=='__main__':main()
