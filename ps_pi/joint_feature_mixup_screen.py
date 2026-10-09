"""Registered residual-direction joint-training cohort with eight matched controls."""
# V5 implementation preserved as a separate version to protect running protocols.
import argparse
import json
import os
import subprocess
import sys
import time
from .common import ROOT,atomic_json,digest
from .joint_noise import MODES
from .calibrated_scale import scale_from_training_diagnostic

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',default='cub',choices=['cifar10','cifar100','cub'])
    p.add_argument('--epochs',type=int,default=40)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--output',default=None)
    args=p.parse_args()
    output=args.output or f'outputs/{args.dataset}/validation_joint_feature_mixup_v10'
    logs=ROOT/'artifacts/logs'/f'{args.dataset}_joint_feature_mixup_v10'
    logs.mkdir(parents=True,exist_ok=True)
    cases=[{'mode':m,'concept_weight':cw,'cbl_lr':lr}
           for cw,lr in [(1.,1e-5),(5.,3e-6)] for m in MODES]
    diagnostic_path=ROOT/'artifacts'/args.dataset/'joint_feature_residual_v6/train_sensitivity.json'
    calibration=scale_from_training_diagnostic(json.loads(diagnostic_path.read_text()))
    plan={'iteration':10,'dataset':args.dataset,'seed':args.seed,'baseline_seed':42,
          'epochs':args.epochs,'mixup_alpha':0.2,'calibration':calibration,'calibration_sha256':digest(diagnostic_path),'output':output,'cases':cases,'test_access':False,
          'causal_contrast':'shared_noise vs shared_no_sampling: same generator, prior, budget; training sampling toggled',
          'source_sha256':{n:digest(ROOT/'ps_pi'/n) for n in ['joint_noise.py','joint_feature_mixup_train.py','joint_feature_mixup_screen.py','calibrated_scale.py','residual_diagnostic.py','directions.py','residual_directions.py','mixup_likelihood.py']}}
    path=logs/'plan.json'
    if path.exists() and json.loads(path.read_text())!=plan:
        raise RuntimeError('Registered cohort changed; use a new log directory')
    atomic_json(plan,path)
    for index,case in enumerate(cases):
        command=[sys.executable,'-u','-m','ps_pi.joint_feature_mixup_train','--dataset',args.dataset,
                 '--mode',case['mode'],'--concept-weight',str(case['concept_weight']),
                 '--cbl-lr',str(case['cbl_lr']),'--seed',str(args.seed),
                 '--epochs',str(args.epochs),'--output',output,
                 '--initial-scale',str(calibration['initial_scale']),'--min-scale',str(calibration['min_scale']),
                 '--max-scale',str(calibration['max_scale'])]
        name=f'{case["mode"]}_cw{case["concept_weight"]:g}_lr{case["cbl_lr"]:g}'
        with (logs/f'{name}.log').open('a') as stream:
            process=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            atomic_json({'phase':'running','pid':os.getpid(),'child_pid':process.pid,
                         'index':index,'total':len(cases),'case':case,'time':time.time()},logs/'status.json')
            code=process.wait()
        if code:
            atomic_json({'phase':'failed','case':case,'returncode':code,'time':time.time()},logs/'status.json')
            raise SystemExit(code)
    atomic_json({'phase':'complete','completed':len(cases),'test_used':False,'time':time.time()},logs/'status.json')

if __name__=='__main__':main()
