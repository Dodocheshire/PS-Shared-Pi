"""V21 concept adaptation/noise restricted outside mapped native channels."""
import argparse
import math
import json
import os
import time
import torch
import numpy as np
from torch.nn import functional as F
from .baseline import snapshot
from .common import ROOT,atomic_json,atomic_save,digest,gpu_lease,seed_all
from .joint_noise import MODES
from .protected_concepts import FrozenConceptChannels
from .signed_shared_noise import SignedSharedNoise
from .compact_outputs import load_compact_checkpoint
from .signed_concept_forward import remapped_bank, sample_concepts, mean_concept_probability
from .adapt_predictive import marginal_concept_nll
from .mixup_likelihood import marginal_soft_task_nll
from .joint_observed_views_audit import attribute_metrics

def marginal_task_nll(logits,labels):
    logp=logits.log_softmax(-1)
    selected=logp.gather(-1,labels.view(1,-1,1).expand(len(logits),-1,1)).squeeze(-1)
    return -(torch.logsumexp(selected,0)-math.log(len(logits))).mean()

@torch.no_grad()
def evaluate_classification(cbl,noise,features,context,labels,pseudo,pack,samples=32,scale_multiplier=1.,mean_multiplier=1.,eval_seed=9131,batch_size=128):
    generator=torch.Generator(device=features.device).manual_seed(eval_seed)
    probabilities=[];bce=0.;drift=0.;mean_e=0.;var_e=0.;tp=0;tn=0;positive=0;negative=0
    for start in range(0,len(features),batch_size):
        x=features[start:start+batch_size];ctx=context[start:start+len(x)]
        z,raw,current=sample_concepts(cbl,noise,x,pack,samples,generator,mean_multiplier,scale_multiplier)
        logits=F.linear(z,pack['weight'],pack['bias'])
        probabilities.append(logits.softmax(-1).mean(0).cpu())
        target=pseudo[start:start+len(x)];bce+=marginal_concept_nll(raw,target).item()*len(x)
        drift+=(z-ctx.unsqueeze(0)).square().mean().item()*len(x)
        pred=mean_concept_probability(raw,noise)>=.5
        tp+=((target==1)&pred).sum().item();tn+=((target==0)&~pred).sum().item()
        positive+=target.sum().item();negative+=target.numel()-target.sum().item()
        me,ve=noise.energy(current);mean_e+=me.item()*len(x)*mean_multiplier**2;var_e+=ve.item()*len(x)*scale_multiplier**2
    probs=torch.cat(probabilities);y=labels.cpu();confidence,pred=probs.max(-1);correct=pred.eq(y);ece=0.
    for lo,hi in zip(torch.linspace(0,1,16)[:-1],torch.linspace(0,1,16)[1:]):
        mask=(confidence>lo)&(confidence<=hi)
        if mask.any():ece+=mask.float().mean().item()*abs(correct[mask].float().mean().item()-confidence[mask].mean().item())
    return {'accuracy':correct.float().mean().item(),'correct':int(correct.sum()),'n':len(y),
            'nll':-probs[torch.arange(len(y)),y].clamp_min(1e-30).log().mean().item(),'ece15':ece,
            'concept_bce':bce/len(y),'pseudo_concept_micro_balanced_accuracy':.5*(tp/max(positive,1)+tn/max(negative,1)),
            'concept_drift_rms':(drift/len(y))**.5,'standardized_concept_mean_energy':mean_e/len(y),'potential_standardized_concept_variance_energy':var_e/len(y),
            'split':'val','eval_seed':eval_seed,'test_used':False},probs

@torch.no_grad()
def evaluate(cbl,noise,features,context,labels,pseudo,pack,samples=32,scale_multiplier=1.,mean_multiplier=1.,eval_seed=9131,batch_size=128):
    metrics,probs=evaluate_classification(cbl,noise,features,context,labels,pseudo,pack,samples,scale_multiplier,mean_multiplier,eval_seed,batch_size)
    if '_native_validation' in pack:
        native_protocol,annotations=pack['_native_validation']
        generator=torch.Generator(device=features.device).manual_seed(9081)
        probabilities=[]
        for start in range(0,len(features),batch_size):
            x=features[start:start+batch_size];ctx=context[start:start+len(x)]
            _,raw,_=sample_concepts(cbl,noise,x,pack,samples,generator,mean_multiplier,scale_multiplier)
            probabilities.append(mean_concept_probability(raw,noise).cpu())
        native=attribute_metrics(torch.cat(probabilities),annotations,native_protocol)
        metrics['native_balanced_accuracy']=native['macro_balanced_accuracy']
        metrics['native_auroc']=native['macro_auroc']
        metrics['native_gate_samples']=samples
    return metrics,probs

def selection_key(metrics,baseline,args):
    eligible=(metrics['correct']>=baseline['correct'] and metrics['nll']<=baseline['nll']+args.nll_tolerance and
              metrics['ece15']<=baseline['ece15']+args.ece_tolerance and
              metrics['concept_bce']<=baseline['concept_bce']+args.concept_bce_tolerance and metrics['concept_drift_rms']<=args.drift_limit and
              metrics['native_balanced_accuracy']+1e-12>=baseline['native_balanced_accuracy'] and
              metrics['native_auroc']+1e-12>=baseline['native_auroc'])
    return eligible,metrics['correct'],-metrics['nll']

def run(args):
    torch.set_num_threads(8);seed_all(args.seed)
    directory=ROOT/'artifacts'/args.dataset
    baseline_path=ROOT/'outputs'/args.dataset/'baseline'/f'seed{args.baseline_seed}'/'corrected_best_validation.pt'
    compact_path=ROOT/'outputs'/args.dataset/'compact_baselines'/f'seed{args.baseline_seed}'/'model.pt'
    output=ROOT/args.output/f'{args.mode}_seed{args.seed}_cw{args.concept_weight:g}_lr{args.cbl_lr:g}'
    output.mkdir(parents=True,exist_ok=True)
    native_path=directory/f'native_attributes_replicates/seed{args.baseline_seed}/protocol.json'
    protocol={'iteration':21,'config':{k:v for k,v in vars(args).items() if k!='output'},
              'baseline_sha256':digest(baseline_path),'compact_checkpoint_sha256':digest(compact_path),
              'signed_geometry_sha256':digest(ROOT/'artifacts/cub/signed_shared_geometry_v21/bank.pt'),
              'manifest_sha256':digest(directory/'manifest.json'),
              'source_sha256':{name:digest(ROOT/'ps_pi'/name) for name in [
                  'protected_signed_train.py','protected_concept_forward.py','protected_concepts.py','all_output_mean_noise.py','signed_shared_noise.py','signed_shared_geometry.py','signed_concept_forward.py','compact_concept_forward.py','bottleneck_noise.py','compact_outputs.py',
                  'joint_noise.py','directions.py','noise.py','common.py','baseline.py','adapt_predictive.py',
                  'mixup_likelihood.py','training_views.py','joint_observed_views_audit.py']},
              'selection':'accuracy/NLL/ECE/pseudo BCE/drift and native BA/AUROC gates, then accuracy/NLL',
              'test_access':False,'native_training_supervision':False,
              'native_protocol_sha256':digest(native_path),
              'native_train_val_sha256':digest(directory/'native_attributes_v1/train_val.pt'),
              'objective':'marginal class likelihood + marginal pseudo concept NLL + original mean concept preservation + energy + scale prior',
              'trainable':'unprotected last-linear CBL rows and generator; CBL hidden trunk, mapped native rows, original classifier/scaler frozen',
              'training_encoder_forward_evaluations_per_image':1,
              'training_head_evaluations_per_image':args.train_samples,
              'covariance':'diag(sigma^2) + signed B diag(gamma^2) signed B.T; observed train covariance PCs restricted to original sharing support, original per-coordinate group variance preserved; original unsigned factor retained for mean decoder; no true class input',
              'mean_decoder':'same exact all-output map as deterministic adapter for shared/diag/no_sampling; uses first, second and group latents, no extra parameters',
              'noise_conditioning':'current clean standardized concepts detached from CBL gradient; fixed original concepts used only as preservation targets',
              'compaction':'exact nonzero head columns plus all mapped native attribute channels; identical for all controls',
              'attribute_protection':'freeze CBL hidden trunk and all mapped last-linear rows; zero perturbation on mapped coordinates; restore rows after Adam weight decay; same for all controls',
              'raw_reconstruction':'add standardized perturbation residual to original raw outputs; exact zero shift identity',
              'observed_view_sha256':[digest(directory/'augmentation_pairs_v13'/f'view{view}.pt') for view in range(2)],
              'view_sampling':'uniform original/weak view0/weak view1; mixup alpha 0.2; original pseudo/mean targets'}
    p=output/'protocol.json'
    if p.exists() and json.loads(p.read_text())!=protocol:raise RuntimeError('Joint protocol changed; use a new experiment directory')
    atomic_json(protocol,p)
    if (output/'status.json').exists() and json.loads((output/'status.json').read_text()).get('phase')=='complete':return
    original=torch.load(baseline_path,weights_only=False,map_location='cpu')
    cbl,pack=load_compact_checkpoint(compact_path)
    if pack['protocol']['baseline_sha256']!=digest(baseline_path):
        raise RuntimeError('Compact model comes from a different baseline')
    retained=pack['retained_original_ids'];bank=remapped_bank(original,pack)
    pack['config']=pack['source_config']
    with gpu_lease(f'PS-CBM compact concept {args.dataset} {args.mode}'):
        for name in ['train_mean','train_std','weight','bias']:pack[name]=pack[name].cuda()
        cbl=cbl.cuda()
        frozen_channels=FrozenConceptChannels(cbl,pack['native_protocol'])
        data={s:torch.load(directory/f'{s}_target.pt',weights_only=True) for s in ['train','val']}
        features={s:data[s]['features'].cuda() for s in data};labels={s:data[s]['labels'].cuda() for s in data}
        pseudo={s:torch.load(directory/f'{s}_concept_labels.pt',weights_only=True)[:,retained].cuda() for s in data}
        cache=directory/'baseline_concepts'/protocol['baseline_sha256']
        context={s:torch.load(cache/f'{s}.pt',weights_only=True)[:,retained].cuda() for s in data}
        native_protocol=pack['native_protocol']
        native_annotations=torch.load(directory/'native_attributes_v1/train_val.pt',weights_only=True)
        if not torch.equal(native_annotations['val']['indices'],data['val']['indices']):
            raise RuntimeError('Native validation order differs')
        pack['_native_validation']=(native_protocol,native_annotations['val'])
        view_features=[features['train']]
        for view,sha in enumerate(protocol['observed_view_sha256']):
            view_path=directory/'augmentation_pairs_v13'/f'view{view}.pt'
            item=torch.load(view_path,weights_only=False)
            assert torch.equal(item['indices'],data['train']['indices']) and torch.equal(item['labels'],data['train']['labels'])
            assert item['protocol']['split']=='train' and not item['protocol']['val_images_opened'] and not item['protocol']['test_images_opened']
            view_features.append(item['features'].cuda())
        pooled_features=torch.stack(view_features)
        augmentation=torch.Generator().manual_seed(args.seed+3104)
        seed_all(args.seed)
        noise=SignedSharedNoise(bank,len(pack['bias']),args.mode,args.hidden,args.max_mean,args.initial_scale,args.min_scale,args.max_scale).cuda()
        clean_noise=SignedSharedNoise(bank,len(pack['bias']),'clean',args.hidden,args.max_mean,args.initial_scale,args.min_scale,args.max_scale).cuda()
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
            augmentation.set_state(state['augmentation'])
            history,best,best_state,best_epoch,start=state['history'],state['best'],state['best_state'],state['best_epoch'],state['next_epoch']
        if not history:
            initial,_=evaluate(cbl,noise,features['val'],context['val'],labels['val'],pseudo['val'],pack,args.eval_samples)
            best=selection_key(initial,baseline,args);best_state={'cbl':snapshot(cbl),'noise':snapshot(noise)};best_epoch=0
            history.append({'epoch':0,'validation':initial})
        began=time.monotonic()
        for epoch in range(start,args.epochs):
            order=torch.randperm(len(features['train']),generator=permutation);loss_sum=0.
            for ids in order.split(args.batch_size):
                chosen=torch.randint(0,len(pooled_features),(len(ids),),generator=augmentation).cuda()
                ids=ids.cuda();x=pooled_features[chosen,ids]
                ctx=context['train'][ids]
                mixed_ids=torch.as_tensor(mixing.permutation(len(ids)),device=x.device)
                mix=float(mixing.beta(args.mixup_alpha,args.mixup_alpha))
                x=mix*x+(1-mix)*x[mixed_ids]
                ctx=mix*ctx+(1-mix)*ctx[mixed_ids]
                soft=F.one_hot(labels['train'][ids],num_classes=len(pack['bias'])).float()
                soft=mix*soft+(1-mix)*soft[mixed_ids]
                target=pseudo['train'][ids]
                target=mix*target+(1-mix)*target[mixed_ids]
                concepts,raw,current=sample_concepts(cbl,noise,x,pack,args.train_samples,sampling)
                # All methods evaluate CBL once and the same number of head likelihoods.
                if len(concepts)==1 and args.train_samples>1:
                    concepts=concepts.expand(args.train_samples,-1,-1)
                    raw=raw.expand(args.train_samples,-1,-1)
                logits=F.linear(concepts,pack['weight'],pack['bias'])
                task=marginal_soft_task_nll(logits,soft)
                teacher=marginal_concept_nll(raw,target)
                preservation=(concepts.mean(0)-ctx).square().mean()
                me,ve=noise.energy(current)
                loss=task+args.concept_weight*teacher+args.preservation_weight*preservation+args.energy_weight*(me+ve)+args.prior_weight*noise.scale_prior(current)
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(list(cbl.parameters())+list(noise.parameters()),5.);optimizer.step();frozen_channels.restore()
                loss_sum+=loss.item()*len(ids)
            metrics,_=evaluate(cbl,noise,features['val'],context['val'],labels['val'],pseudo['val'],pack,args.eval_samples)
            history.append({'epoch':epoch+1,'training_objective':loss_sum/len(order),'validation':metrics})
            key=selection_key(metrics,baseline,args)
            if key>best:best=key;best_state={'cbl':snapshot(cbl),'noise':snapshot(noise)};best_epoch=epoch+1
            atomic_save({'protocol':protocol,'cbl':snapshot(cbl),'noise':snapshot(noise),'optimizer':optimizer.state_dict(),
                         'sampling':sampling.get_state(),'permutation':permutation.get_state(),'mixing':mixing.bit_generator.state,'augmentation':augmentation.get_state(),'history':history,'best':best,'best_state':best_state,
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
                    cbl_parameters=sum(p.numel() for p in cbl.parameters()),
                    cbl_trainable_parameters=sum(p.numel() for p in cbl.parameters() if p.requires_grad),
                    protected_concept_count=len(frozen_channels.ids),
                    cbl_effective_trainable_parameters=(cbl.cbl[-1].out_features-len(frozen_channels.ids))*(cbl.cbl[-1].in_features+int(cbl.cbl[-1].bias is not None)),noise_allocated_parameters=sum(p.numel() for p in noise.parameters()),
                    noise_trainable_parameters=sum(p.numel() for p in noise.parameters() if p.requires_grad))
        atomic_save({'state':best_state,'protocol':protocol,'best_epoch':best_epoch},output/'model.pt')
        atomic_save({'probabilities':probs,'mean_probabilities':mean_probs,'labels':labels['val'].cpu(),'indices':data['val']['indices']},output/'validation_predictions.pt')
        atomic_json({'metrics':full,'same_model_mean':mean,'same_cbl_no_perturbation':no_perturbation,'original_baseline':baseline,'history':history},output/'metrics.json')
        atomic_json({'phase':'complete','pid':os.getpid(),'test_used':False,'time':time.time()},output/'status.json')
        print('complete',full,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',default='cub',choices=['cub']);p.add_argument('--mode',choices=MODES,default='shared_noise')
    p.add_argument('--baseline-seed',type=int,default=42);p.add_argument('--seed',type=int,default=0);p.add_argument('--output',default='outputs/cub/validation_protected_signed_v21')
    p.add_argument('--epochs',type=int,default=40);p.add_argument('--batch-size',type=int,default=128);p.add_argument('--hidden',type=int,default=128)
    p.add_argument('--cbl-lr',type=float,default=1e-5);p.add_argument('--noise-lr',type=float,default=1e-3);p.add_argument('--weight-decay',type=float,default=1e-5)
    p.add_argument('--concept-weight',type=float,default=1.);p.add_argument('--preservation-weight',type=float,default=1.)
    p.add_argument('--energy-weight',type=float,default=.1);p.add_argument('--prior-weight',type=float,default=.01)
    p.add_argument('--max-mean',type=float,default=.3);p.add_argument('--initial-scale',type=float,default=.1);p.add_argument('--min-scale',type=float,default=.02);p.add_argument('--max-scale',type=float,default=.3)
    p.add_argument('--mixup-alpha',type=float,default=.2)
    p.add_argument('--train-samples',type=int,default=4);p.add_argument('--eval-samples',type=int,default=32)
    p.add_argument('--nll-tolerance',type=float,default=0.);p.add_argument('--ece-tolerance',type=float,default=0.);p.add_argument('--concept-bce-tolerance',type=float,default=0.)
    p.add_argument('--drift-limit',type=float,default=.3)
    run(p.parse_args())
if __name__=='__main__':main()
