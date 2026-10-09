"""Checkpointed first baseline; exits on failure before any subsequent stage."""
import os
import subprocess
import sys
import time
from .common import ROOT, atomic_json

def main():
    logdir=ROOT/'artifacts/logs/first_baseline';logdir.mkdir(parents=True,exist_ok=True)
    for name,module in [('prepare','ps_pi.prepare'),('baseline','ps_pi.baseline')]:
        atomic_json({'phase':name,'pid':os.getpid(),'time':time.time()},logdir/'status.json')
        with (logdir/f'{name}.log').open('a') as stream:
            process=subprocess.Popen([sys.executable,'-u','-m',module,'--dataset','cifar10'],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            atomic_json({'phase':name,'pid':os.getpid(),'child_pid':process.pid,'time':time.time()},logdir/'status.json')
            code=process.wait()
        if code:
            atomic_json({'phase':'failed','step':name,'returncode':code,'time':time.time()},logdir/'status.json')
            raise SystemExit(code)
    atomic_json({'phase':'complete','test_used':False,'time':time.time()},logdir/'status.json')
if __name__=='__main__':main()
