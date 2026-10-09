"""Validation-only joint CBL training with within-class residual feature directions."""
# V5 implementation preserved as a separate version to protect running protocols.
import argparse
import math
import json
import os
import time
import torch
import numpy as np
from torch.nn import functional as F
from .baseline import snapshot
from .common import ROOT,atomic_json,atomic_save,digest,gpu_lease,official_module,seed_all
from .joint_noise import FeatureNoise,MODES,LEARNED
from .adapt_predictive import marginal_concept_nll
from .robust_likelihood import expected_soft_task_nll

def marginal_task_nll(logits,labels):
    logp=logits.log_softmax(-1)
    selected=logp.gather(-1,labels.view(1,-1,1).expand(len(logits),-1,1)).squeeze(-1)
    return -(torch.logsumexp(selected,0)-math.log(len(logits))).mean()

@torch.no_grad()
def evaluate(cbl,noise,features,context,labels,pseudo,pack,samples=32,scale_multiplier=1.,mean_multiplier=1.,eval_seed=9131,batch_size=128):
    generator=torch.Generator(device=features.device).manual_seed(eval_seed)
    probabilities=[];bce=0.;drift=0.;mean_e=0.;var_e=0.;tp=0;tn=0;positive=0;negative=0
    for start in range(0,len(features),batch_size):
        x=features[start:start+batch_size];ctx=context[start:start+len(x)]
        draws=noise.draw(x,ctx,samples,generator,mean_multiplier,scale_multiplier)
        raw=cbl(draws.reshape(-1,features.shape[1])).reshape(len(draws),len(x),-1)
        z=(raw-pack['train_mean'])/pack['train_std'];logits=F.linear(z,pack['weight'],pack['bias'])
        probabilities.append(logits.softmax(-1).mean(0).cpu())
        target=pseudo[start:start+len(x)];bce+=marginal_concept_nll(raw,target).item()*len(x)
        drift+=(z-ctx.unsqueeze(0)).square().mean().item()*len(x)
        pred=raw.sigmoid().mean(0)>=.5
        tp+=((target==1)&pred).sum().item();tn+=((target==0)&~pred).sum().item()
        positive+=target.sum().item();negative+=target.numel()-target.sum().item()
        me,ve=noise.energy(ctx);mean_e+=me.item()*len(x)*mean_multiplier**2;var_e+=ve.item()*len(x)*scale_multiplier**2
    probs=torch.cat(probabilities);y=labels.cpu();confidence,pred=probs.max(-1);correct=pred.eq(y);ece=0.
    for lo,hi in zip(torch.linspace(0,1,16)[:-1],torch.linspace(0,1,16)[1:]):
        mask=(confidence>lo)&(confidence<=hi)
        if mask.any():ece+=mask.float().mean().item()*abs(correct[mask].float().mean().item()-confidence[mask].mean().item())
    return {'accuracy':correct.float().mean().item(),'correct':int(correct.sum()),'n':len(y),
            'nll':-probs[torch.arange(len(y)),y].clamp_min(1e-30).log().mean().item(),'ece15':ece,
            'concept_bce':bce/len(y),'pseudo_concept_micro_balanced_accuracy':.5*(tp/max(positive,1)+tn/max(negative,1)),
            'concept_drift_rms':(drift/len(y))**.5,'feature_mean_energy':mean_e/len(y),'feature_variance_energy':var_e/len(y),
            'split':'val','eval_seed':eval_seed,'test_used':False},probs

def selection_key(metrics,baseline,args):
    eligible=(metrics['correct']>=baseline['correct'] and metrics['nll']<=baseline['nll']+args.nll_tolerance and
              metrics['ece15']<=baseline['ece15']+args.ece_tolerance and
              metrics['concept_bce']<=baseline['concept_bce']+args.concept_bce_tolerance and metrics['concept_drift_rms']<=args.drift_limit)
    return eligible,metrics['correct'],-metrics['nll']

def run(args):
    torch.set_num_threads(8);seed_all(args.seed)
    directory=ROOT/'artifacts'/args.dataset
    baseline_path=ROOT/'outputs'/args.dataset/'baseline'/f'seed{args.baseline_seed}'/'corrected_best_validation.pt'
    bank_path=directory/'joint_feature_residual_v6/directions.pt'
    output=ROOT/args.output/f'{args.mode}_seed{args.seed}_cw{args.concept_weight:g}_lr{args.cbl_lr:g}'
    output.mkdir(parents=True,exist_ok=True)
    protocol={'iteration':12,'config':{k:v for k,v in vars(args).items() if k!='output'},'baseline_sha256':digest(baseline_path),
              'directions_sha256':digest(bank_path),'training_calibration_sha256':digest(directory/'joint_feature_residual_v6/train_sensitivity.json'),'manifest_sha256':digest(directory/'manifest.json'),
              'source_sha256':{name:digest(ROOT/'ps_pi'/name) for name in ['joint_noise.py','joint_robust_mixup_train.py','calibrated_scale.py','residual_diagnostic.py','residual_directions.py','directions.py','noise.py','common.py','baseline.py','adapt_predictive.py','robust_likelihood.py']},
              'selection':'nondecreasing accuracy and quality gates first, then accuracy and NLL','test_access':False,
              'objective':'feature mixup (alpha 0.2) + soft-target expected task CE / MC marginal concept NLL + interpolated mean concept preservation + feature energy + scale prior',
              'trainable':'CBL and learned perturbation generator; classification head and feature directions frozen',
              'training_encoder_forward_evaluations_per_image':args.train_samples}
    p=output/'protocol.json'
    if p.exists() and json.loads(p.read_text())!=protocol:raise RuntimeError('Joint protocol changed; use a new experiment directory')
    atomic_json(protocol,p)
    if (output/'status.json').exists() and json.loads((output/'status.json').read_text()).get('phase')=='complete':return
    pack=torch.load(baseline_path,weights_only=False);bank=torch.load(bank_path,weights_only=False)
    with gpu_lease(f'PS-CBM residual joint {args.dataset} {args.mode}'):
        for name in ['train_mean','train_std','weight','bias']:pack[name]=pack[name].cuda()
        CBL=official_module('model.cbl').ConceptBottleneckLayer
        cbl=CBL(2048,len(pack['concepts']),pack['config']['cbl_layer_num'],pack['config']['cbl_bias']).cuda()
        cbl.load_state_dict(pack['cbl_state'])
        data={s:torch.load(directory/f'{s}_target.pt',weights_only=True) for s in ['train','val']}
        features={s:data[s]['features'].cuda() for s in data};labels={s:data[s]['labels'].cuda() for s in data}
        pseudo={s:torch.load(directory/f'{s}_concept_labels.pt',weights_only=True).cuda() for s in data}
        context={};cache=directory/'baseline_concepts'/protocol['baseline_sha256'];cache.mkdir(parents=True,exist_ok=True)
        for split in data:
            cached=cache/f'{split}.pt'
            if not cached.exists():
                with torch.no_grad():z=torch.cat([((cbl(x)-pack['train_mean'])/pack['train_std']).cpu() for x in features[split].split(4096)])
                atomic_save(z,cached)
            context[split]=torch.load(cached,weights_only=True).cuda()
        seed_all(args.seed)
        noise=FeatureNoise(bank,len(pack['bias']),args.mode,args.hidden,args.max_mean,args.initial_scale,args.min_scale,args.max_scale).cuda()
        clean_noise=FeatureNoise(bank,len(pack['bias']),'clean',args.hidden,args.max_mean,args.initial_scale,args.min_scale,args.max_scale).cuda()
        baseline,_=evaluate(cbl,clean_noise,features['val'],context['val'],labels['val'],pseudo['val'],pack,args.eval_samples)
        if abs(baseline['accuracy']-pack['validation_metrics']['accuracy'])>1e-7:raise RuntimeError('Joint initialization does not reproduce baseline')
        atomic_json(baseline,output/'baseline_validation.json')
        optimizer=torch.optim.Adam([{'params':cbl.parameters(),'lr':args.cbl_lr},
                                    {'params':[p for p in noise.parameters() if p.requires_grad],'lr':args.noise_lr}],weight_decay=args.weight_decay)
        sampling=torch.Generator(device='cuda').manual_seed(args.seed+3101)
        permutation=torch.Generator().manual_seed(args.seed+3102)
        mixing=np.random.default_rng(args.seed+3103)
        history=[];best=None;best_state=None;best_epoch=None;start=0
        resume=output/'resume.pt'
        if resume.exists():
            state=torch.load(resume,weights_only=False)
            if state['protocol']!=protocol:raise RuntimeError('Joint mixup resume protocol changed')
            cbl.load_state_dict(state['cbl']);noise.load_state_dict(state['noise']);optimizer.load_state_dict(state['optimizer'])
            sampling.set_state(state['sampling']);permutation.set_state(state['permutation'])
            mixing.bit_generator.state=state['mixing']
            history,best,best_state,best_epoch,start=state['history'],state['best'],state['best_state'],state['best_epoch'],state['next_epoch']
        if not history:
            initial,_=evaluate(cbl,noise,features['val'],context['val'],labels['val'],pseudo['val'],pack,args.eval_samples)
            best=selection_key(initial,baseline,args);best_state={'cbl':snapshot(cbl),'noise':snapshot(noise)};best_epoch=0
            history.append({'epoch':0,'validation':initial})
        began=time.monotonic()
        for epoch in range(start,args.epochs):
            order=torch.randperm(len(features['train']),generator=permutation);loss_sum=0.
            for ids in order.split(args.batch_size):
                ids=ids.cuda();x=features['train'][ids];ctx=context['train'][ids]
                mixed_ids=torch.as_tensor(mixing.permutation(len(ids)),device=x.device)
                mix=float(mixing.beta(args.mixup_alpha,args.mixup_alpha))
                x=mix*x+(1-mix)*x[mixed_ids]
                ctx=mix*ctx+(1-mix)*ctx[mixed_ids]
                soft=F.one_hot(labels['train'][ids],num_classes=len(pack['bias'])).float()
                soft=mix*soft+(1-mix)*soft[mixed_ids]
                target=pseudo['train'][ids]
                target=mix*target+(1-mix)*target[mixed_ids]
                draws=noise.draw(x,ctx,args.train_samples,sampling)
                # Match encoder evaluations even when the control is deterministic.
                if len(draws)==1 and args.train_samples>1:draws=draws.expand(args.train_samples,-1,-1)
                raw=cbl(draws.reshape(-1,x.shape[1])).reshape(len(draws),len(x),-1)
                concepts=(raw-pack['train_mean'])/pack['train_std']
                logits=F.linear(concepts,pack['weight'],pack['bias'])
                task=expected_soft_task_nll(logits,soft)
                teacher=marginal_concept_nll(raw,target)
                preservation=(concepts.mean(0)-ctx).square().mean()
                me,ve=noise.energy(ctx)
                loss=task+args.concept_weight*teacher+args.preservation_weight*preservation+args.energy_weight*(me+ve)+args.prior_weight*noise.scale_prior(ctx)
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(list(cbl.parameters())+list(noise.parameters()),5.);optimizer.step()
                loss_sum+=loss.item()*len(ids)
            metrics,_=evaluate(cbl,noise,features['val'],context['val'],labels['val'],pseudo['val'],pack,args.eval_samples)
            history.append({'epoch':epoch+1,'training_objective':loss_sum/len(order),'validation':metrics})
            key=selection_key(metrics,baseline,args)
            if key>best:best=key;best_state={'cbl':snapshot(cbl),'noise':snapshot(noise)};best_epoch=epoch+1
            atomic_save({'protocol':protocol,'cbl':snapshot(cbl),'noise':snapshot(noise),'optimizer':optimizer.state_dict(),
                         'sampling':sampling.get_state(),'permutation':permutation.get_state(),'mixing':mixing.bit_generator.state,'history':history,'best':best,'best_state':best_state,
                         'best_epoch':best_epoch,'next_epoch':epoch+1},resume)
            atomic_json({'phase':'train','epoch':epoch+1,'epochs':args.epochs,'best_epoch':best_epoch,'best_accuracy':best[1]/len(labels['val']),
                         'pid':os.getpid(),'time':time.time()},output/'status.json')
            if epoch%5==0 or epoch==args.epochs-1:
                print(f'{args.mode} {epoch+1}/{args.epochs}: val {100*metrics["accuracy"]:.2f}%, NLL {metrics["nll"]:.4f}, ECE {100*metrics["ece15"]:.2f}%, best {100*best[1]/len(labels["val"]):.2f}%; {time.monotonic()-began:.1f}s',flush=True)
        cbl.load_state_dict(best_state['cbl']);noise.load_state_dict(best_state['noise'])
        full,probs=evaluate(cbl,noise,features['val'],context['val'],labels['val'],pseudo['val'],pack,args.eval_samples)
        mean,mean_probs=evaluate(cbl,noise,features['val'],context['val'],labels['val'],pseudo['val'],pack,args.eval_samples,scale_multiplier=0.)
        no_perturbation,_=evaluate(cbl,noise,features['val'],context['val'],labels['val'],pseudo['val'],pack,args.eval_samples,scale_multiplier=0.,mean_multiplier=0.)
        full.update(mode=args.mode,dataset=args.dataset,seed=args.seed,baseline_seed=args.baseline_seed,best_epoch=best_epoch,selection_eligible=bool(best[0]),
                    delta_original_pp=100*(full['accuracy']-baseline['accuracy']),delta_inference_sampling_pp=100*(full['accuracy']-mean['accuracy']),
                    same_model_mean_accuracy=mean['accuracy'],concept_count=len(pack['concepts']),
                    cbl_parameters=sum(p.numel() for p in cbl.parameters()),noise_allocated_parameters=sum(p.numel() for p in noise.parameters()),
                    noise_trainable_parameters=sum(p.numel() for p in noise.parameters() if p.requires_grad))
        atomic_save({'state':best_state,'protocol':protocol,'best_epoch':best_epoch},output/'model.pt')
        atomic_save({'probabilities':probs,'mean_probabilities':mean_probs,'labels':labels['val'].cpu(),'indices':data['val']['indices']},output/'validation_predictions.pt')
        atomic_json({'metrics':full,'same_model_mean':mean,'same_cbl_no_perturbation':no_perturbation,'original_baseline':baseline,'history':history},output/'metrics.json')
        atomic_json({'phase':'complete','pid':os.getpid(),'test_used':False,'time':time.time()},output/'status.json')
        print('complete',full,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',default='cub',choices=['cifar10','cifar100','cub']);p.add_argument('--mode',choices=MODES,default='shared_noise')
    p.add_argument('--baseline-seed',type=int,default=42);p.add_argument('--seed',type=int,default=0);p.add_argument('--output',default='outputs/cub/validation_joint_robust_mixup_v12')
    p.add_argument('--epochs',type=int,default=40);p.add_argument('--batch-size',type=int,default=128);p.add_argument('--hidden',type=int,default=128)
    p.add_argument('--cbl-lr',type=float,default=1e-5);p.add_argument('--noise-lr',type=float,default=1e-3);p.add_argument('--weight-decay',type=float,default=1e-5)
    p.add_argument('--concept-weight',type=float,default=1.);p.add_argument('--preservation-weight',type=float,default=1.)
    p.add_argument('--energy-weight',type=float,default=.1);p.add_argument('--prior-weight',type=float,default=.01)
    p.add_argument('--max-mean',type=float,default=.1);p.add_argument('--initial-scale',type=float,default=.1);p.add_argument('--min-scale',type=float,default=.02);p.add_argument('--max-scale',type=float,default=.3)
    p.add_argument('--mixup-alpha',type=float,default=.2)
    p.add_argument('--train-samples',type=int,default=4);p.add_argument('--eval-samples',type=int,default=32)
    p.add_argument('--nll-tolerance',type=float,default=0.);p.add_argument('--ece-tolerance',type=float,default=.002);p.add_argument('--concept-bce-tolerance',type=float,default=0.)
    p.add_argument('--drift-limit',type=float,default=.3)
    run(p.parse_args())
if __name__=='__main__':main()
