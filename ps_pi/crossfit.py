"""Cross-fit only the CBL so Pi sees predictions on images that CBL did not fit."""
import argparse
import json
import os
import time
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from .baseline import snapshot,validation_metrics
from .common import ROOT,atomic_json,atomic_save,digest,gpu_lease,official_module,seed_all

def fold_plan(labels,folds,seed):
    labels=np.asarray(labels)
    if folds<2 or np.unique(labels,return_counts=True)[1].min()<folds:raise ValueError('Every class needs at least one sample per fold')
    splits=list(StratifiedKFold(folds,shuffle=True,random_state=seed).split(np.arange(len(labels)),labels))
    held=[]
    for fit,out in splits:
        if set(fit)&set(out) or len(fit)+len(out)!=len(labels):raise RuntimeError('Invalid cross-fit partition')
        held.extend(out.tolist())
    if sorted(held)!=list(range(len(labels))):raise RuntimeError('Each sample must be held out exactly once')
    return splits

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',default='cub',choices=['cifar10','cifar100','cub'])
    parser.add_argument('--folds',type=int,default=3);parser.add_argument('--seed',type=int,default=17)
    args=parser.parse_args();torch.set_num_threads(8)
    directory=ROOT/'artifacts'/args.dataset;out=directory/'crossfit_v1';out.mkdir(parents=True,exist_ok=True)
    config=json.loads((ROOT/'configs'/f'{args.dataset}_rn50.json').read_text())
    data=torch.load(directory/'train_target.pt',weights_only=True);pseudo=torch.load(directory/'train_concept_labels.pt',weights_only=True)
    bank=torch.load(directory/'concept_bank.pt',weights_only=False)
    splits=fold_plan(data['labels'].numpy(),args.folds,310)
    protocol={'dataset':args.dataset,'folds':args.folds,'seed':args.seed,'split_seed':310,'config':config,
              'checkpoint_selection':'fixed official final step; no held-fold supervision used',
              'crossfit_scope':'CBL fitting only; concept bank precomputed on full original training split',
              'train_features_sha256':digest(directory/'train_target.pt'),'pseudo_sha256':digest(directory/'train_concept_labels.pt'),
              'bank_sha256':digest(directory/'concept_bank.pt'),'source_sha256':digest(__file__),'test_opened':False}
    previous=out/'protocol.json'
    if previous.exists() and json.loads(previous.read_text())!=protocol:raise RuntimeError('Cross-fit protocol changed')
    atomic_json(protocol,previous)
    CBL=official_module('model.cbl').ConceptBottleneckLayer
    with gpu_lease(f'PS-CBM cross-fit {args.dataset}'):
        features=data['features'].cuda();targets=pseudo.cuda()
        for fold,(fit,held) in enumerate(splits):
            destination=out/f'fold{fold}.pt'
            if destination.exists():
                saved=torch.load(destination,weights_only=False)
                if saved['protocol']!=protocol:raise RuntimeError('Fold protocol mismatch')
                continue
            seed_all(args.seed+fold)
            model=CBL(features.shape[1],len(bank['concepts']),config['cbl_layer_num'],config['cbl_bias']).cuda()
            optimizer=torch.optim.Adam(model.parameters(),lr=config['cbl_lr'],weight_decay=config['weight_decay'])
            fit=torch.tensor(fit);held=torch.tensor(held);begin=0
            resume=out/f'fold{fold}_resume.pt'
            if resume.exists():
                state=torch.load(resume,weights_only=False)
                if state['protocol']!=protocol:raise RuntimeError('Fold resume mismatch')
                model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
                begin=state['next_step'];torch.set_rng_state(state['rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
            started=time.monotonic()
            for step in range(begin,config['cbl_steps']):
                ids=fit[torch.randperm(len(fit))[:config['cbl_batch_size']]].cuda()
                optimizer.zero_grad();loss=torch.nn.functional.binary_cross_entropy_with_logits(model(features[ids]),targets[ids])
                loss.backward();optimizer.step()
                if step%1000==0 or step==config['cbl_steps']-1:
                    atomic_save({'protocol':protocol,'model':snapshot(model),'optimizer':optimizer.state_dict(),'next_step':step+1,
                                 'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},resume)
                    atomic_json({'phase':'train','fold':fold,'step':step,'folds':args.folds,'pid':os.getpid(),'time':time.time()},out/'status.json')
                    print(f'fold {fold} step {step}/{config["cbl_steps"]}: train BCE {loss.item():.5f}; {time.monotonic()-started:.1f}s',flush=True)
            with torch.no_grad():
                raw=model(features[held.cuda()]).cpu()
                raw_fit=model(features[fit.cuda()]);fold_mean=raw_fit.mean(0,keepdim=True).cpu();fold_std=raw_fit.std(0,keepdim=True).cpu()
            atomic_save({'protocol':protocol,'fit_rows':fit,'held_rows':held,'held_source_indices':data['indices'][held],
                         'raw_logits':raw,'fold_mean':fold_mean,'fold_std':fold_std,'cbl_state':snapshot(model)},destination)
        combined=torch.empty(len(data['features']),len(bank['concepts']));seen=torch.zeros(len(combined),dtype=torch.int)
        for fold in range(args.folds):
            result=torch.load(out/f'fold{fold}.pt',weights_only=False);rows=result['held_rows']
            combined[rows]=result['raw_logits'];seen[rows]+=1
        if not torch.equal(seen,torch.ones_like(seen)):raise RuntimeError('Cross-fit rows do not cover training exactly once')
        baseline=ROOT/'outputs'/args.dataset/'baseline/seed42/corrected_best_validation.pt'
        pack=torch.load(baseline,weights_only=False)
        concepts=(combined-pack['train_mean'])/pack['train_std']
        metrics=validation_metrics(torch.nn.functional.linear(concepts,pack['weight'],pack['bias']),data['labels'])
        atomic_save({'features':concepts,'raw_logits':combined,'labels':data['labels'],'indices':data['indices'],
                     'protocol':protocol,'normalization':'full-baseline training mean/std','baseline_sha256':digest(baseline)},out/'train_concepts.pt')
        atomic_json({'phase':'complete','rows':len(combined),'frozen_head_oof_metrics':metrics,'test_used':False,'time':time.time()},out/'status.json')
        print('Cross-fit complete',metrics,flush=True)
if __name__=='__main__':main()
