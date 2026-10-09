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
from .augmentation_calibration import measured_scale

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',default='cub',choices=['cub'])
    p.add_argument('--epochs',type=int,default=40)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--output',default=None)
    args=p.parse_args()
    output=args.output or f'outputs/{args.dataset}/validation_joint_observed_views_v14'
    logs=ROOT/'artifacts/logs'/f'{args.dataset}_joint_observed_views_v14'
    logs.mkdir(parents=True,exist_ok=True)
    cases=[{'mode':m,'concept_weight':cw,'cbl_lr':lr}
           for cw,lr in [(1.,1e-5),(5.,3e-6)] for m in MODES]
    diagnostic_path=ROOT/'artifacts'/args.dataset/'joint_feature_augmented_v13/train_sensitivity.json'
    calibration=measured_scale(json.loads(diagnostic_path.read_text()))
    plan={'iteration':14,'dataset':args.dataset,'seed':args.seed,'baseline_seed':42,
          'epochs':args.epochs,'mixup_alpha':0.2,'calibration':calibration,'calibration_sha256':digest(diagnostic_path),'output':output,'cases':cases,'test_access':False,
          'training_change':'uniform original/weak input view, frozen-view conditioning, original-image concept consistency; all controls matched',
          'causal_contrast':'shared_noise vs shared_no_sampling: same generator, prior, budget; training sampling toggled',
          'source_sha256':{n:digest(ROOT/'ps_pi'/n) for n in ['joint_noise.py','joint_observed_views_train.py','joint_observed_views_screen.py','calibrated_scale.py','augmentation_calibration.py','augmentation_diagnostic.py','directions.py','augmentation_directions.py','mixup_likelihood.py','training_views.py']}}
    path=logs/'plan.json'
    if path.exists() and json.loads(path.read_text())!=plan:
        raise RuntimeError('Registered cohort changed; use a new log directory')
    atomic_json(plan,path)
    for index,case in enumerate(cases):
        command=[sys.executable,'-u','-m','ps_pi.joint_observed_views_train','--dataset',args.dataset,
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
    for module in ['joint_observed_views_audit','expected_augmented_audit']:
        atomic_json({'phase':'audit','module':module,'completed_training':16,'pid':os.getpid(),'time':time.time()},logs/'status.json')
        with (logs/f'{module}.log').open('a') as stream:
            subprocess.run([sys.executable,'-u','-m',f'ps_pi.{module}','--dataset',args.dataset,'--input',output],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
    atomic_json({'phase':'complete','completed':len(cases),'test_used':False,'time':time.time()},logs/'status.json')

if __name__=='__main__':main()
