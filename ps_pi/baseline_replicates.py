"""Prepare independently trained full baselines without evaluating test features."""
import argparse
import json
import os
import subprocess
import sys
import time
from .common import ROOT, atomic_json, digest, verify_official


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--datasets',nargs='+',default=['cub','cifar10','cifar100'],choices=['cub','cifar10','cifar100'])
    p.add_argument('--seeds',nargs='+',type=int,default=[43,44,45,46])
    args=p.parse_args()
    if 42 in args.seeds or len(set(args.seeds))!=len(args.seeds):
        raise ValueError('Seed42 already exists; new seeds must be unique')
    logs=ROOT/'artifacts/logs/full_baseline_replicates';logs.mkdir(exist_ok=True)
    cases=[{'dataset':dataset,'seed':seed} for dataset in args.datasets for seed in args.seeds]
    for dataset in args.datasets:
        original=ROOT/'outputs'/dataset/'baseline/seed42'
        assert json.loads((original/'status.json').read_text())['phase']=='complete'
        for variant in ['official_final_state','corrected_best_validation']:
            assert (original/f'{variant}.pt').is_file()
            assert not json.loads((original/f'{variant}.json').read_text())['smoke']
    sources={name:digest(ROOT/'ps_pi'/name) for name in ['baseline.py','common.py','baseline_replicates.py']}
    plan={'stage':'independent_full_baseline_preparation','reference_seed':42,'new_seeds':args.seeds,
          'datasets':args.datasets,'cases':cases,'source_sha256':sources,'official_commit':verify_official(),
          'configs':{dataset:json.loads((ROOT/'configs'/f'{dataset}_rn50.json').read_text()) for dataset in args.datasets},
          'data_sha256':{dataset:{name:digest(ROOT/'artifacts'/dataset/name) for name in [
              'manifest.json','train_target.pt','val_target.pt','train_concept_labels.pt','val_concept_labels.pt','concept_bank.pt']}
              for dataset in args.datasets},
          'variants':['official_final_state','corrected_best_validation'],
          'test_access':False,'locked_noise_candidate':False,'final_evidence':False,
          'scope':'Independent CBL and original SAGA-head training; fixed official split, pretrained features and deterministic training-only concept bank',
          'limits':'Preparatory baseline replication. No selected Pi protocol, paired noise/control runs or final test evidence yet.'}
    guard=logs/'plan.json'
    if guard.exists() and json.loads(guard.read_text())!=plan:
        raise RuntimeError('Full baseline replication protocol changed')
    atomic_json(plan,guard)
    for index,case in enumerate(cases):
        if any(digest(ROOT/'ps_pi'/name)!=sha for name,sha in sources.items()):
            raise RuntimeError('Registered baseline source changed')
        command=[sys.executable,'-u','-m','ps_pi.baseline','--dataset',case['dataset'],'--seed',str(case['seed'])]
        with (logs/f"{case['dataset']}_seed{case['seed']}.log").open('a') as stream:
            child=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            atomic_json({'phase':'running','index':index,'total':len(cases),'case':case,
                         'pid':os.getpid(),'child_pid':child.pid,'time':time.time()},logs/'status.json')
            result=child.wait()
        if result:
            atomic_json({'phase':'failed','case':case,'returncode':result,'time':time.time()},logs/'status.json')
            raise SystemExit(result)
        folder=ROOT/'outputs'/case['dataset']/'baseline'/f"seed{case['seed']}"
        assert json.loads((folder/'status.json').read_text())['phase']=='complete'
        for variant in plan['variants']:assert (folder/f'{variant}.pt').exists()
    atomic_json({'phase':'complete','new_full_baselines':len(cases),'new_models':2*len(cases),
                 'seeds':[42]+args.seeds,'test_used':False,'final_evidence':False,'time':time.time()},logs/'status.json')


if __name__=='__main__':main()
