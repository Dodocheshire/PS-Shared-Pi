"""Pre-registered CIFAR-10 validation screen followed by remaining baseline preparation."""
import argparse
import os
import subprocess
import sys
import time
from .common import ROOT,atomic_json,digest

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='outputs/cifar10/validation_v1')
    parser.add_argument('--logs',default='artifacts/logs/validation_v1');parser.add_argument('--prepare-remaining',action='store_true')
    args=parser.parse_args();logs=ROOT/args.logs;logs.mkdir(parents=True,exist_ok=True)
    cases=[]
    for cw,ew in [(.1,.01),(1.,.1)]:
        for mode in ['mean_only','adapter_matched','diag_noise','shared_noise']:
            cases.append(dict(mode=mode,concept_weight=cw,energy_weight=ew))
    cases.append(dict(mode='fixed_noise',concept_weight=.1,energy_weight=.01))
    plan={'dataset':'cifar10','split':'validation only','baseline_seed':42,'adaptation_seed':0,'epochs':40,
          'selection':'accuracy then NLL','cases':cases,'test_access':False,
          'source_sha256':{name:digest(ROOT/'ps_pi'/name) for name in ['noise.py','adapt.py','screen.py']}}
    atomic_json(plan,logs/'plan.json')
    commands=[]
    for case in cases:
        name=f'{case["mode"]}_cw{case["concept_weight"]:g}_ew{case["energy_weight"]:g}'
        command=[sys.executable,'-u','-m','ps_pi.adapt','--output',args.output,
                 '--mode',case['mode'],'--concept-weight',str(case['concept_weight']),'--energy-weight',str(case['energy_weight'])]
        commands.append((name,command))
    if args.prepare_remaining:
        for dataset in ['cifar100','cub']:
            for stage in ['prepare','baseline']:
                commands.append((f'{dataset}_{stage}',[sys.executable,'-u','-m',f'ps_pi.{stage}','--dataset',dataset]))
    for index,(name,command) in enumerate(commands):
        with (logs/f'{name}.log').open('a') as stream:
            process=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            atomic_json({'phase':'running','index':index,'total':len(commands),'step':name,'pid':os.getpid(),'child_pid':process.pid,'time':time.time()},logs/'status.json')
            code=process.wait()
        if code:
            atomic_json({'phase':'failed','step':name,'returncode':code,'time':time.time()},logs/'status.json')
            raise SystemExit(code)
    atomic_json({'phase':'complete','completed':len(commands),'test_used':False,'time':time.time()},logs/'status.json')
if __name__=='__main__':main()
