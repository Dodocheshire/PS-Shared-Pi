"""Device-selectable third iteration; CPU cohort is recorded separately from GPU runs."""
import argparse
import json
import os
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from .baseline import snapshot
from .common import ROOT,atomic_json,atomic_save,digest,gpu_lease,official_module,seed_all
from .noise import MODES,marginal_nll
from .noise_v2 import ConceptNoise

@torch.no_grad()
def evaluate(noise,concepts,labels,pseudo,pack,samples=64,eval_seed=8301,batch_size=512,scale_multiplier=1.,mean_multiplier=1.):
    generator=torch.Generator(device=concepts.device).manual_seed(eval_seed)
    probabilities=[]; bce_sum=0.; expected_bce_sum=0.; mean_sum=0.;var_sum=0.;tp=0.;tn=0.;pos=0.;neg=0.
    for start in range(0,len(concepts),batch_size):
        z=concepts[start:start+batch_size]
        draws=z.unsqueeze(0) if noise is None else noise.draw(z,samples,generator,mean_multiplier,scale_multiplier)
        logits=F.linear(draws,pack['weight'],pack['bias'])
        probabilities.append(logits.softmax(-1).mean(0).cpu())
        raw=draws*pack['train_std']+pack['train_mean']
        target=pseudo[start:start+len(z)]
        bce_sum+=marginal_concept_nll(raw,target).item()*len(z)
        expected_bce_sum+=F.binary_cross_entropy_with_logits(raw,target.unsqueeze(0).expand_as(raw)).item()*len(z)
        concept_prediction=raw.sigmoid().mean(0)>=.5
        tp+=((target==1)&concept_prediction).sum().item();tn+=((target==0)&~concept_prediction).sum().item()
        pos+=target.sum().item();neg+=target.numel()-target.sum().item()
        if noise is not None:
            m,v=noise.expected_energy(z);mean_sum+=m.item()*len(z)*mean_multiplier**2;var_sum+=v.item()*len(z)*scale_multiplier**2
    probs=torch.cat(probabilities); y=labels.cpu(); pred=probs.argmax(-1); confidence=probs.max(-1).values
    correct=pred.eq(y);ece=0.
    for lo,hi in zip(torch.linspace(0,1,16)[:-1],torch.linspace(0,1,16)[1:]):
        selected=(confidence>lo)&(confidence<=hi)
        if selected.any():ece+=selected.float().mean().item()*abs(correct[selected].float().mean().item()-confidence[selected].mean().item())
    mean_energy,variance_energy=mean_sum/len(y),var_sum/len(y)
    metrics={'accuracy':correct.float().mean().item(),'correct':int(correct.sum()),'n':len(y),
             'nll':-probs[torch.arange(len(y)),y].clamp_min(1e-30).log().mean().item(),'ece15':ece,
             'concept_bce':bce_sum/len(y),'concept_expected_bce':expected_bce_sum/len(y),
             'pseudo_concept_micro_balanced_accuracy':.5*(tp/max(pos,1)+tn/max(neg,1)),'mean_energy':mean_energy,'variance_energy':variance_energy,
             'concept_drift_rms':(mean_energy+variance_energy)**.5,
             'stochastic_energy_fraction':variance_energy/max(1e-30,mean_energy+variance_energy),
             'mc_samples':samples if noise is not None and scale_multiplier and noise.mode not in ['mean_only','adapter_matched'] else 1,
             'eval_seed':eval_seed,'split':'val'}
    return metrics,probs

def encode_concepts(data,pack):
    cbl=official_module('model.cbl').ConceptBottleneckLayer(data['features'].shape[1],len(pack['concepts']),
             pack['config']['cbl_layer_num'],pack['config']['cbl_bias']).to(pack['weight'].device).eval().requires_grad_(False)
    cbl.load_state_dict(pack['cbl_state'])
    with torch.no_grad():
        chunks=[((cbl(x.to(pack['weight'].device))-pack['train_mean'])/pack['train_std']).cpu() for x in data['features'].split(4096)]
    return torch.cat(chunks)

def marginal_concept_nll(raw, targets):
    # The likelihood matches the Monte Carlo mean concept probability used in inference.
    log_positive=torch.logsumexp(F.logsigmoid(raw),dim=0)-__import__('math').log(raw.shape[0])
    log_negative=torch.logsumexp(F.logsigmoid(-raw),dim=0)-__import__('math').log(raw.shape[0])
    return -(targets*log_positive+(1-targets)*log_negative).mean()

def marginal_soft_nll(logits, targets):
    log_mean=torch.logsumexp(logits.log_softmax(-1),dim=0)-__import__('math').log(logits.shape[0])
    return -(targets*log_mean).sum(-1).mean()

def selection_key(metrics,clean,args):
    eligible=(metrics['nll']<=clean['nll']+args.nll_tolerance and
              metrics['ece15']<=clean['ece15']+args.ece_tolerance and
              metrics['concept_bce']<=clean['concept_bce']+args.concept_bce_tolerance and
              metrics['concept_drift_rms']<=args.drift_limit)
    return (eligible,metrics['correct'],-metrics['nll'])

def run(args):
    device=torch.device(args.device)
    torch.set_num_threads(args.threads);seed_all(args.seed)
    baseline=ROOT/'outputs'/args.dataset/'baseline'/f'seed{args.baseline_seed}'/f'{args.baseline_variant}.pt'
    directory=ROOT/'artifacts'/args.dataset
    output=ROOT/args.output/f'{args.mode}_seed{args.seed}_cw{args.concept_weight:g}_ew{args.energy_weight:g}'
    output.mkdir(parents=True,exist_ok=True)
    source_hashes={name:digest(ROOT/'ps_pi'/name) for name in ['noise.py','noise_v2.py','adapt.py','adapt_mixup.py','adapt_predictive.py','adapt_device.py','baseline.py','common.py']}
    configuration={k:v for k,v in vars(args).items() if k!='output'}
    protocol={'iteration':3,'config':configuration,'baseline_sha256':digest(baseline),'manifest_sha256':digest(directory/'manifest.json'),
              'source_sha256':source_hashes,'split':'train+val only','test_opened':False,
              'selection':'eligible calibration and semantic gates first, then validation accuracy and NLL','objective':'task MC marginal likelihood, predictive Bernoulli concept marginal likelihood, optional mixup and energy'}
    protocol_path=output/'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text())!=protocol:raise RuntimeError('Run protocol changed; use a new output directory')
    atomic_json(protocol,protocol_path)
    if (output/'status.json').exists() and json.loads((output/'status.json').read_text()).get('phase')=='complete':
        print('reuse completed',output,flush=True);return
    pack=torch.load(baseline,weights_only=False)
    lease=gpu_lease(f'PS-CBM Pi {args.dataset} {args.mode} seed{args.seed}') if device.type=='cuda' else __import__('contextlib').nullcontext()
    with lease:
        for name in ['weight','bias','train_mean','train_std']:pack[name]=pack[name].to(device)
        data={split:torch.load(directory/f'{split}_target.pt',weights_only=True) for split in ['train','val']}
        concepts={};cache_dir=directory/'baseline_concepts'/protocol['baseline_sha256'];cache_dir.mkdir(parents=True,exist_ok=True)
        for split in ['train','val']:
            path=cache_dir/f'{split}.pt'
            if not path.exists():atomic_save(encode_concepts(data[split],pack),path)
            concepts[split]=torch.load(path,weights_only=True).to(device)
        labels={split:data[split]['labels'].to(device) for split in data}
        pseudo={split:torch.load(directory/f'{split}_concept_labels.pt',weights_only=True).to(device) for split in data}
        seed_all(args.seed)
        model=ConceptNoise(len(pack['concepts']),len(pack['bias']),pack['class_sets'],args.mode,args.hidden,
                           args.max_mean,args.initial_scale,args.max_scale).to(device)
        generator=torch.Generator(device=device).manual_seed(args.seed+101)
        mixup_generator=torch.Generator(device=device).manual_seed(args.seed+103)
        mixup_rng=__import__('numpy').random.default_rng(args.seed+104)
        permutation_generator=torch.Generator().manual_seed(args.seed+102)
        clean,_=evaluate(None,concepts['val'],labels['val'],pseudo['val'],pack,args.eval_samples)
        if abs(clean['accuracy']-pack['validation_metrics']['accuracy'])>1e-7:raise RuntimeError('Frozen clean prediction does not reproduce its baseline')
        atomic_json(clean,output/'clean_validation.json')
        history=[];best_key=None;best_state=None;best_epoch=None;start=0
        optimizer=None if args.mode=='fixed_noise' else torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
        resume=output/'resume.pt'
        if resume.exists():
            saved=torch.load(resume,weights_only=False)
            if saved['protocol']!=protocol:raise RuntimeError('Resume protocol mismatch')
            model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
            generator.set_state(saved['noise_rng']);permutation_generator.set_state(saved['permutation_rng'])
            mixup_generator.set_state(saved['mixup_torch_rng']);mixup_rng.bit_generator.state=saved['mixup_numpy_rng']
            history,best_key,best_state,best_epoch=saved['history'],saved['best_key'],saved['best_state'],saved['best_epoch']
            start=saved['next_epoch']
        initial,probs=evaluate(model,concepts['val'],labels['val'],pseudo['val'],pack,args.eval_samples)
        if not history:
            history.append({'epoch':0,'validation':initial});best_key=selection_key(initial,clean,args)
            best_state=snapshot(model);best_epoch=0
        started=time.monotonic()
        epochs=0 if args.mode=='fixed_noise' else args.epochs
        for epoch in range(start,epochs):
            order=torch.randperm(len(concepts['train']),generator=permutation_generator)
            loss_total=0.
            for ids in order.split(args.batch_size):
                ids=ids.to(device);z=concepts['train'][ids];y=labels['train'][ids]
                q=F.one_hot(y,num_classes=len(pack['bias'])).float()
                pseudo_target=pseudo['train'][ids]
                if args.mixup_alpha>0:
                    partner=torch.randperm(len(z),device=device,generator=mixup_generator)
                    weight=float(mixup_rng.beta(args.mixup_alpha,args.mixup_alpha))
                    z=weight*z+(1-weight)*z[partner]
                    q=weight*q+(1-weight)*q[partner]
                    pseudo_target=weight*pseudo_target+(1-weight)*pseudo_target[partner]
                draws=model.draw(z,args.train_samples,generator)
                task=marginal_soft_nll(F.linear(draws,pack['weight'],pack['bias']),q)
                raw=draws*pack['train_std']+pack['train_mean']
                concept_loss=marginal_concept_nll(raw,pseudo_target)
                me,ve=model.expected_energy(z)
                loss=task+args.concept_weight*concept_loss+args.energy_weight*(me+ve)
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
                loss_total+=loss.item()*len(ids)
            metrics,_=evaluate(model,concepts['val'],labels['val'],pseudo['val'],pack,args.eval_samples)
            history.append({'epoch':epoch+1,'train_objective':loss_total/len(order),'validation':metrics})
            key=selection_key(metrics,clean,args)
            if best_key is None or key>best_key:best_key=key;best_state=snapshot(model);best_epoch=epoch+1
            atomic_save({'protocol':protocol,'model':snapshot(model),'optimizer':optimizer.state_dict(),
                         'noise_rng':generator.get_state(),'permutation_rng':permutation_generator.get_state(),
                         'mixup_torch_rng':mixup_generator.get_state(),'mixup_numpy_rng':mixup_rng.bit_generator.state,
                         'history':history,'best_key':best_key,'best_state':best_state,'best_epoch':best_epoch,'next_epoch':epoch+1},resume)
            atomic_json({'phase':'train','epoch':epoch+1,'epochs':epochs,'best_epoch':best_epoch,'best_accuracy':best_key[1]/len(labels['val']),
                         'pid':os.getpid(),'time':time.time()},output/'status.json')
            if epoch%5==0 or epoch==epochs-1:
                print(f'{args.mode} epoch {epoch+1}/{epochs}: val {100*metrics["accuracy"]:.2f}%, NLL {metrics["nll"]:.4f}, best {100*best_key[1]/len(labels["val"]):.2f}%; {time.monotonic()-started:.1f}s',flush=True)
        model.load_state_dict(best_state)
        metrics,probs=evaluate(model,concepts['val'],labels['val'],pseudo['val'],pack,args.eval_samples)
        mean_metrics,mean_probs=evaluate(model,concepts['val'],labels['val'],pseudo['val'],pack,args.eval_samples,scale_multiplier=0.)
        zero_metrics,_=evaluate(model,concepts['val'],labels['val'],pseudo['val'],pack,args.eval_samples,scale_multiplier=0.,mean_multiplier=0.)
        if abs(zero_metrics['accuracy']-clean['accuracy'])>1e-7:raise RuntimeError('Zero perturbation fails to restore baseline')
        metrics.update(selection_eligible=bool(best_key[0]),mixup_alpha=args.mixup_alpha,iteration=3,best_epoch=best_epoch,concept_count=len(pack['concepts']),allocated_parameters=sum(p.numel() for p in model.parameters()),
                       trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                       shared_concept_count=int((model.factor.sum(1)>0).sum()),
                       clean_accuracy=clean['accuracy'],delta_clean_pp=100*(metrics['accuracy']-clean['accuracy']),
                       same_model_mean_accuracy=mean_metrics['accuracy'],delta_same_model_mean_pp=100*(metrics['accuracy']-mean_metrics['accuracy']),
                       mode=args.mode,seed=args.seed,baseline_seed=args.baseline_seed,dataset=args.dataset,test_used=False)
        atomic_save({'state':best_state,'protocol':protocol,'best_epoch':best_epoch},output/'model.pt')
        atomic_save({'probabilities':probs,'mean_probabilities':mean_probs,'labels':labels['val'].cpu(),'indices':data['val']['indices']},output/'validation_predictions.pt')
        atomic_json({'metrics':metrics,'same_model_mean':mean_metrics,'clean':clean,'history':history},output/'metrics.json')
        atomic_json({'phase':'complete','pid':os.getpid(),'test_used':False,'time':time.time()},output/'status.json')
        print('complete',metrics,flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',choices=['cifar10','cifar100','cub'],default='cifar10')
    parser.add_argument('--mode',choices=MODES,default='shared_noise');parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--baseline-seed',type=int,default=42)
    parser.add_argument('--baseline-variant',default='corrected_best_validation',choices=['corrected_best_validation','official_final_state'])
    parser.add_argument('--output',default='outputs/cifar10/validation_predictive_v3_cpu')
    parser.add_argument('--epochs',type=int,default=60);parser.add_argument('--batch-size',type=int,default=1024)
    parser.add_argument('--hidden',type=int,default=128);parser.add_argument('--lr',type=float,default=.001)
    parser.add_argument('--weight-decay',type=float,default=0.)
    parser.add_argument('--concept-weight',type=float,default=.1);parser.add_argument('--energy-weight',type=float,default=.01)
    parser.add_argument('--train-samples',type=int,default=8);parser.add_argument('--eval-samples',type=int,default=64)
    parser.add_argument('--max-mean',type=float,default=.5);parser.add_argument('--initial-scale',type=float,default=.1);parser.add_argument('--max-scale',type=float,default=1.)
    parser.add_argument('--mixup-alpha',type=float,default=0.)
    parser.add_argument('--nll-tolerance',type=float,default=0.)
    parser.add_argument('--ece-tolerance',type=float,default=.002)
    parser.add_argument('--concept-bce-tolerance',type=float,default=0.)
    parser.add_argument('--drift-limit',type=float,default=.3)
    parser.add_argument('--device',default='cpu',choices=['cpu','cuda'])
    parser.add_argument('--threads',type=int,default=4)
    run(parser.parse_args())
if __name__=='__main__':main()
