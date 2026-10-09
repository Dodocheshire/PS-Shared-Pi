"""Third screen: predictive Bernoulli concept loss, matched nonlinear controls."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .common import ROOT,atomic_json,digest

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',default='cifar10',choices=['cifar10','cifar100','cub'])
    parser.add_argument('--output',default=None);parser.add_argument('--logs',default=None)
    args=parser.parse_args();output=args.output or f'outputs/{args.dataset}/validation_predictive_v3'
    logs=ROOT/(args.logs or f'artifacts/logs/{args.dataset}_predictive_v3');logs.mkdir(parents=True,exist_ok=True)
    methods=['mean_only','adapter_matched','diag_noise','shared_noise']
    cases=[{'mode':mode,'mixup_alpha':alpha} for alpha in [0.,.2] for mode in methods]
    plan={'iteration':3,'dataset':args.dataset,'baseline_seed':42,'adaptation_seed':0,'epochs':60,'cases':cases,
          'concept_weight':.1,'energy_weight':.01,'initial_scale':.1,'max_scale':1.,'train_mc':8,'eval_mc':64,
          'selection':'eligible gates first, accuracy then NLL','gates':{'NLL_vs_clean':0.,'ECE_vs_clean':.002,'predictive_concept_BCE_vs_clean':0.,'drift_limit':.3},
          'test_access':False,'source_sha256':{name:digest(ROOT/'ps_pi'/name) for name in ['noise.py','noise_v2.py','adapt.py','adapt_mixup.py','adapt_predictive.py','screen_predictive.py']}}
    previous=logs/'plan.json'
    if previous.exists() and json.loads(previous.read_text())!=plan:raise RuntimeError('Screen plan changed; use a fresh study directory')
    atomic_json(plan,previous)
    for index,case in enumerate(cases):
        folder=f'{output}/alpha{case["mixup_alpha"]:g}'
        command=[sys.executable,'-u','-m','ps_pi.adapt_predictive','--dataset',args.dataset,'--mode',case['mode'],
                 '--mixup-alpha',str(case['mixup_alpha']),'--output',folder]
        with (logs/f'{case["mode"]}_alpha{case["mixup_alpha"]:g}.log').open('a') as stream:
            process=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            atomic_json({'phase':'running','index':index,'total':len(cases),'case':case,'pid':os.getpid(),'child_pid':process.pid,'time':time.time()},logs/'status.json')
            code=process.wait()
        if code:
            atomic_json({'phase':'failed','case':case,'returncode':code,'time':time.time()},logs/'status.json');raise SystemExit(code)
    atomic_json({'phase':'complete','completed':len(cases),'test_used':False,'time':time.time()},logs/'status.json')
if __name__=='__main__':main()
