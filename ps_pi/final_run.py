"""Run only the seven locked controls across five independent CUB baselines."""
import os
import subprocess
import sys
import time
from .common import ROOT, atomic_json
from .final_lock import register, verify_lock, SEEDS, MODES
from .final_geometry import prepare


def main():
    register();logs=ROOT/'artifacts/logs/final_v1';logs.mkdir(parents=True,exist_ok=True)
    for seed in SEEDS:prepare(seed)
    for i,baseline_seed in enumerate(SEEDS):
        for mode in MODES:
            verify_lock()
            with (logs/f'seed{baseline_seed}_{mode}.log').open('a') as stream:
                process=subprocess.Popen([sys.executable,'-u','-m','ps_pi.final_train',
                    '--baseline-seed',str(baseline_seed),'--seed',str(i),'--mode',mode,
                    '--output',f'outputs/cub/final_v1/seed{baseline_seed}'],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
                atomic_json({'phase':'training','baseline_seed':baseline_seed,'mode':mode,
                    'pid':os.getpid(),'child_pid':process.pid,'time':time.time()},logs/'status.json')
                if process.wait():raise RuntimeError(f'Final training failed: {baseline_seed} {mode}')
    atomic_json({'phase':'complete','models':35,'test_used_for_training':False,'time':time.time()},logs/'status.json')


if __name__=='__main__':main()
