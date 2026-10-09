"""V22 registered protected-channel label-free concept Gaussian cohort."""
import argparse
import json
import os
import subprocess
import sys
import time
from .common import ROOT,atomic_json,digest
from .joint_noise import MODES

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',default='cub',choices=['cub'])
    p.add_argument('--epochs',type=int,default=40)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--output',default=None)
    args=p.parse_args()
    output=args.output or f'outputs/{args.dataset}/validation_anchored_signed_v22'
    logs=ROOT/'artifacts/logs'/f'{args.dataset}_anchored_signed_v22'
    logs.mkdir(parents=True,exist_ok=True)
    cases=[{'mode':m,'concept_weight':cw,'cbl_lr':lr}
           for cw,lr in [(1.,1e-5),(5.,3e-6)] for m in MODES]
    plan={'iteration':22,'dataset':args.dataset,'seed':args.seed,'baseline_seed':42,
          'epochs':args.epochs,'mixup_alpha':0.2,'output':output,'cases':cases,'test_access':False,
          'native_training_supervision':False,
          'training_change':'only replace fully frozen CBL trunk with adaptive trunk plus frozen native-output reference branch; keep signed covariance, mean decoder, all controls/objectives/gates; match two CBL forwards for every control',
          'mean_bound':0.3,'initial_scale':0.1,'min_scale':0.02,'max_scale':0.3,
          'quality_gates':{'nll_tolerance':0.,'ece_tolerance':0.,'concept_bce_tolerance':0.,'drift_limit':0.3,'native_ba_tolerance':0.,'native_auroc_tolerance':0.},
          'compact_checkpoint_sha256':digest(ROOT/'outputs/cub/compact_baselines/seed42/model.pt'),
          'native_protocol_sha256':digest(ROOT/'artifacts/cub/native_attributes_replicates/seed42/protocol.json'),
          'mean_capacity':'adapter_matched/diag/shared/no_sampling have identical all-output mean mapping and allocated parameters; mean_only retains weak first-output control',
          'causal_contrast':'shared_noise vs shared_no_sampling: same architecture, prior, amplitude bounds, initialization, order and budget; training sampling toggled',
          'signed_geometry_sha256':digest(ROOT/'artifacts/cub/signed_shared_geometry_v21/bank.pt'),
          'signed_geometry_diagnostic_sha256':digest(ROOT/'artifacts/cub/signed_shared_geometry_v21/diagnostic.json'),
          'training_risk_diagnostic_sha256':digest(ROOT/'artifacts/cub/concept_risk_v19/diagnostic.json'),
          'explanation_protection':'identical 161 native outputs from frozen reference encoder for every control; adaptive unprotected trunk can train',
          'scope':'single baseline42/adapter0 validation exploration; no final five-seed or test evidence; compaction itself is not noise gain',
          'source_sha256':{n:digest(ROOT/'ps_pi'/n) for n in ['anchored_signed_screen.py','anchored_signed_train.py','anchored_signed_audit.py','protected_concept_forward.py','protected_concepts.py','all_output_mean_noise.py','signed_shared_noise.py','anchored_cbl.py','signed_shared_geometry.py','signed_concept_forward.py','compact_concept_forward.py','compact_outputs.py','bottleneck_noise.py','joint_noise.py','directions.py','noise.py','common.py','baseline.py','adapt_predictive.py','mixup_likelihood.py','joint_observed_views_audit.py']}}
    path=logs/'plan.json'
    if path.exists() and json.loads(path.read_text())!=plan:
        raise RuntimeError('Registered cohort changed; use a new log directory')
    atomic_json(plan,path)
    for index,case in enumerate(cases):
        if any(digest(ROOT/'ps_pi'/name)!=sha for name,sha in plan['source_sha256'].items()):
            raise RuntimeError('Registered native-training source changed')
        command=[sys.executable,'-u','-m','ps_pi.anchored_signed_train','--dataset',args.dataset,
                 '--mode',case['mode'],'--concept-weight',str(case['concept_weight']),
                 '--cbl-lr',str(case['cbl_lr']),'--seed',str(args.seed),
                 '--epochs',str(args.epochs),'--output',output]
        name=f'{case["mode"]}_cw{case["concept_weight"]:g}_lr{case["cbl_lr"]:g}'
        with (logs/f'{name}.log').open('a') as stream:
            process=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            atomic_json({'phase':'running','pid':os.getpid(),'child_pid':process.pid,
                         'index':index,'total':len(cases),'case':case,'time':time.time()},logs/'status.json')
            code=process.wait()
        if code:
            atomic_json({'phase':'failed','case':case,'returncode':code,'time':time.time()},logs/'status.json')
            raise SystemExit(code)
    for module in ['anchored_signed_audit']:
        atomic_json({'phase':'audit','module':module,'completed_training':16,'pid':os.getpid(),'time':time.time()},logs/'status.json')
        with (logs/f'{module}.log').open('a') as stream:
            subprocess.run([sys.executable,'-u','-m',f'ps_pi.{module}','--dataset',args.dataset,'--input',output],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
    atomic_json({'phase':'complete','completed':len(cases),'test_used':False,'time':time.time()},logs/'status.json')

if __name__=='__main__':main()
