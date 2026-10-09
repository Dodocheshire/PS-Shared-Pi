"""Wait for a live, specific cross-fit process, then run matched CUB noise controls."""
import argparse
import json
import os
import subprocess
import sys
import time
from .common import ROOT,atomic_json,digest

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-pid',type=int,required=True)
    args=parser.parse_args();logs=ROOT/'artifacts/logs/cub_crossfit_v4';logs.mkdir(parents=True,exist_ok=True)
    preparation=ROOT/'artifacts/cub/crossfit_v1'
    while True:
        status=json.loads((preparation/'status.json').read_text()) if (preparation/'status.json').exists() else {}
        if status.get('phase')=='complete' and (preparation/'train_concepts.pt').exists():break
        proc=__import__('pathlib').Path(f'/proc/{args.prepare_pid}')
        if not proc.exists() or 'ps_pi.crossfit' not in (proc/'cmdline').read_bytes().decode(errors='ignore'):
            raise RuntimeError('Expected cross-fit process stopped before preparation completed')
        atomic_json({'phase':'waiting_for_verified_process','pid':os.getpid(),'dependency_pid':args.prepare_pid,'time':time.time()},logs/'status.json')
        time.sleep(2)
    methods=['mean_only','adapter_matched','diag_noise','shared_noise']
    cases=[{'mode':mode,'concept_weight':cw} for cw in [.1,1.] for mode in methods]
    cases.append({'mode':'fixed_noise','concept_weight':.1})
    plan={'iteration':4,'dataset':'cub','training_input':'3-fold held-out CBL predictions','inference':'full frozen CBL',
          'baseline_seed':42,'seed':0,'epochs':60,'cases':cases,'mixup_alpha':0.,'energy_weight':.01,
          'initial_scale':.1,'max_scale':1.,'crossfit_sha256':digest(preparation/'train_concepts.pt'),'test_access':False,
          'source_sha256':{name:digest(ROOT/'ps_pi'/name) for name in ['crossfit.py','adapt_crossfit.py','noise.py','noise_v2.py','screen_crossfit.py']}}
    previous=logs/'plan.json'
    if previous.exists() and json.loads(previous.read_text())!=plan:raise RuntimeError('Cross-fit adaptation plan changed')
    atomic_json(plan,previous)
    for index,case in enumerate(cases):
        command=[sys.executable,'-u','-m','ps_pi.adapt_crossfit','--mode',case['mode'],'--concept-weight',str(case['concept_weight'])]
        with (logs/f'{case["mode"]}_cw{case["concept_weight"]:g}.log').open('a') as stream:
            process=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            atomic_json({'phase':'running','pid':os.getpid(),'child_pid':process.pid,'index':index,'total':len(cases),'case':case,'time':time.time()},logs/'status.json')
            code=process.wait()
        if code:
            atomic_json({'phase':'failed','case':case,'returncode':code,'time':time.time()},logs/'status.json');raise SystemExit(code)
    atomic_json({'phase':'complete','completed':len(cases),'test_used':False,'time':time.time()},logs/'status.json')
if __name__=='__main__':main()
