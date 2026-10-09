"""Complete observed-view controls, then prepare independent full baseline seeds."""
import os
import subprocess
import sys
import time
from .common import ROOT, atomic_json


def main():
    logs=ROOT/'artifacts/logs/observed_views_and_baselines';logs.mkdir(exist_ok=True)
    for module in ['joint_observed_views_screen','baseline_replicates']:
        with (logs/f'{module}.log').open('a') as stream:
            child=subprocess.Popen([sys.executable,'-u','-m',f'ps_pi.{module}'],cwd=ROOT,
                                   stdout=stream,stderr=subprocess.STDOUT)
            atomic_json({'phase':'running','module':module,'pid':os.getpid(),'child_pid':child.pid,
                         'time':time.time()},logs/'status.json')
            result=child.wait()
        if result:
            atomic_json({'phase':'failed','module':module,'returncode':result,'time':time.time()},logs/'status.json')
            raise SystemExit(result)
    atomic_json({'phase':'complete','test_used':False,'final_evidence':False,'time':time.time()},logs/'status.json')


if __name__=='__main__':main()
