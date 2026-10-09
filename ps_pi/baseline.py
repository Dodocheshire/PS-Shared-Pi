"""Official CBL/SAGA budget, with final-state and corrected checkpoint controls."""
import argparse
import copy
import json
import os
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from .common import ROOT, atomic_json, atomic_save, digest, gpu_lease, official_module, seed_all

def snapshot(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

@torch.no_grad()
def validation_metrics(logits, y):
    probabilities=logits.softmax(-1); confidence,pred=probabilities.max(-1)
    correct=pred.eq(y)
    ece=logits.new_zeros(())
    boundaries=torch.linspace(0,1,16,device=logits.device)
    for lo, hi in zip(boundaries[:-1],boundaries[1:]):
        mask=(confidence>lo)&(confidence<=hi)
        if mask.any(): ece += mask.float().mean()*(correct[mask].float().mean()-confidence[mask].mean()).abs()
    return {'accuracy':correct.float().mean().item(),'nll':F.cross_entropy(logits,y).item(),'ece15':ece.item()}

def run(args):
    config=json.loads((ROOT/'configs'/f'{args.dataset}_rn50.json').read_text()); config['seed']=args.seed
    if args.smoke:
        config.update(cbl_steps=100,n_iters=2)
    directory=ROOT/'artifacts'/args.dataset
    output=ROOT/'outputs'/args.dataset/('smoke' if args.smoke else 'baseline')/f'seed{args.seed}'
    output.mkdir(parents=True,exist_ok=True)
    atomic_json(config,output/'config.json')
    train=torch.load(directory/'train_target.pt',weights_only=True)
    val=torch.load(directory/'val_target.pt',weights_only=True)
    targets=torch.load(directory/'train_concept_labels.pt',weights_only=True)
    val_targets=torch.load(directory/'val_concept_labels.pt',weights_only=True)
    bank=torch.load(directory/'concept_bank.pt',weights_only=False)
    assert train['features'].shape[1] == val['features'].shape[1] == 2048
    assert len(train['features']) == len(targets)
    manifest=json.loads((directory/'manifest.json').read_text())
    assert torch.equal(train['indices'],torch.tensor(manifest['indices']['train']))
    assert torch.equal(val['indices'],torch.tensor(manifest['indices']['val']))
    expected={'config':config,'manifest_sha256':digest(directory/'manifest.json'),
              'bank_sha256':digest(directory/'concept_bank.pt'),'test_used':False}
    protocol_path=output/'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text())!=expected:
        raise RuntimeError('Protocol mismatch; never overwrite a different run')
    atomic_json(expected,protocol_path)
    CBL=official_module('model.cbl').ConceptBottleneckLayer
    saga=official_module('glm_saga.elasticnet')
    seed_all(args.seed); torch.set_num_threads(8)
    with gpu_lease(f'PS-CBM baseline {args.dataset} seed {args.seed}'):
        cbl=CBL(2048,len(bank['concepts']),config['cbl_layer_num'],config['cbl_bias']).cuda()
        optimizer=torch.optim.Adam(cbl.parameters(),lr=config['cbl_lr'],weight_decay=config['weight_decay'])
        tx=train['features'].cuda(); vx=val['features'].cuda(); tc=targets.cuda(); vc=val_targets.cuda()
        best=float('inf'); best_state=None; best_step=None; history=[]; start=0
        resume=output/'cbl_resume.pt'
        if resume.exists():
            state=torch.load(resume,weights_only=False)
            if state['protocol']!=expected: raise RuntimeError('Resume protocol mismatch')
            cbl.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
            best,best_state,best_step=state['best'],state['best_state'],state['best_step']
            history,start=state['history'],state['next_step']
            torch.set_rng_state(state['rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
        began=time.monotonic()
        for step in range(start,config['cbl_steps']):
            # Keep official CPU randperm sequence; resident immutable tensors save transfers.
            idx=torch.randperm(len(tx))[:config['cbl_batch_size']].cuda()
            optimizer.zero_grad();loss=F.binary_cross_entropy_with_logits(cbl(tx[idx]),tc[idx])
            loss.backward();optimizer.step()
            if step % 50 == 0 or step==config['cbl_steps']-1:
                with torch.no_grad(): value=F.binary_cross_entropy_with_logits(cbl(vx),vc).item()
                if value<best: best=value;best_state=snapshot(cbl);best_step=step
                history.append({'step':step,'train_bce':loss.item(),'val_bce':value})
            if step % 500 == 0 or step==config['cbl_steps']-1:
                atomic_save({'protocol':expected,'model':snapshot(cbl),'optimizer':optimizer.state_dict(),
                             'best':best,'best_state':best_state,'best_step':best_step,'history':history,
                             'next_step':step+1,'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},resume)
                atomic_json({'phase':'cbl','step':step,'total_steps':config['cbl_steps'],'best_val_bce':best,
                             'best_step':best_step,'pid':os.getpid(),'time':time.time()},output/'status.json')
                print(f'CBL {step}/{config["cbl_steps"]}: val BCE {value:.5f}, best {best:.5f}; {time.monotonic()-began:.1f}s',flush=True)
        final_state=snapshot(cbl)
        atomic_save({'best':best_state,'final':final_state,'best_step':best_step,'history':history},output/'cbl_states.pt')
        for label,state in [('official_final_state',final_state),('corrected_best_validation',best_state)]:
            checkpoint=output/f'{label}.pt'
            if checkpoint.exists(): continue
            cbl.load_state_dict(state)
            with torch.no_grad():
                raw=cbl(tx); mean=raw.mean(0,keepdim=True); std=raw.std(0,keepdim=True)
                if not torch.isfinite(std).all() or (std<=0).any(): raise RuntimeError('Invalid training standard deviation')
                concept_train=((raw-mean)/std).cpu(); concept_val=((cbl(vx)-mean)/std).cpu()
            ds=saga.IndexedTensorDataset(concept_train,train['labels'])
            vs=saga.IndexedTensorDataset(concept_val,val['labels'])
            seed_all(args.seed)
            head=torch.nn.Linear(len(bank['concepts']),len(manifest['classes'])).cuda()
            with torch.no_grad():head.weight.zero_();head.bias.zero_()
            atomic_json({'phase':'saga','variant':label,'pid':os.getpid(),'time':time.time()},output/'status.json')
            result=saga.glm_saga(head,DataLoader(ds,batch_size=config['saga_batch_size'],shuffle=True),.1,
                                config['n_iters'],.99,epsilon=1,k=1,
                                val_loader=DataLoader(vs,batch_size=config['saga_batch_size']),do_zero=False,
                                metadata={'max_reg':{'nongrouped':config['lam']}},n_ex=len(ds),n_classes=len(manifest['classes']))
            solution=result['path'][0]; weight=solution['weight'].cpu();bias=solution['bias'].cpu()
            metrics=validation_metrics(F.linear(concept_val,weight,bias),val['labels'])
            metrics.update(concept_count=len(bank['concepts']),concept_bce_best=best,checkpoint_step=best_step if label.startswith('corrected') else config['cbl_steps']-1,
                           head_nonzero=int(weight.count_nonzero()),head_parameters=weight.numel(),split='val',smoke=args.smoke)
            atomic_save({'cbl_state':state,'weight':weight,'bias':bias,'train_mean':mean.cpu(),'train_std':std.cpu(),
                         'config':config,'concepts':bank['concepts'],'class_sets':bank['class_sets'],'protocol':expected,
                         'variant':label,'validation_metrics':metrics},checkpoint)
            atomic_json(metrics,output/f'{label}.json');print(label,metrics,flush=True)
        atomic_json({'phase':'complete','test_used':False,'pid':os.getpid(),'time':time.time()},output/'status.json')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',default='cifar10',choices=['cifar10','cifar100','cub'])
    parser.add_argument('--seed',type=int,default=42);parser.add_argument('--smoke',action='store_true')
    run(parser.parse_args())
if __name__=='__main__':main()
