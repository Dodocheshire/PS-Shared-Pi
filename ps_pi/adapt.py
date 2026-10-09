"""Validation-only Pi iteration on a frozen PS-CBM; test artifacts are never opened."""
import argparse
import json
import os
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from .baseline import snapshot
from .common import ROOT,atomic_json,atomic_save,digest,gpu_lease,official_module,seed_all
from .noise import ConceptNoise,MODES,marginal_nll

@torch.no_grad()
def evaluate(noise,concepts,labels,pseudo,pack,samples=64,eval_seed=8301,batch_size=512,scale_multiplier=1.,mean_multiplier=1.):
    generator=torch.Generator(device=concepts.device).manual_seed(eval_seed)
    probabilities=[]; bce_sum=0.; mean_sum=0.;var_sum=0.
    for start in range(0,len(concepts),batch_size):
        z=concepts[start:start+batch_size]
        draws=z.unsqueeze(0) if noise is None else noise.draw(z,samples,generator,mean_multiplier,scale_multiplier)
        logits=F.linear(draws,pack['weight'],pack['bias'])
        probabilities.append(logits.softmax(-1).mean(0).cpu())
        raw=draws*pack['train_std']+pack['train_mean']
        bce_sum+=F.binary_cross_entropy_with_logits(raw,pseudo[start:start+len(z)].unsqueeze(0).expand_as(raw)).item()*len(z)
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
             'concept_bce':bce_sum/len(y),'mean_energy':mean_energy,'variance_energy':variance_energy,
             'concept_drift_rms':(mean_energy+variance_energy)**.5,
             'stochastic_energy_fraction':variance_energy/max(1e-30,mean_energy+variance_energy),
             'mc_samples':samples if noise is not None and scale_multiplier and noise.mode not in ['mean_only','adapter_matched'] else 1,
             'eval_seed':eval_seed,'split':'val'}
    return metrics,probs

def encode_concepts(data,pack):
    cbl=official_module('model.cbl').ConceptBottleneckLayer(data['features'].shape[1],len(pack['concepts']),
             pack['config']['cbl_layer_num'],pack['config']['cbl_bias']).cuda().eval().requires_grad_(False)
    cbl.load_state_dict(pack['cbl_state'])
    with torch.no_grad():
        chunks=[((cbl(x.cuda())-pack['train_mean'])/pack['train_std']).cpu() for x in data['features'].split(4096)]
    return torch.cat(chunks)

def run(args):
    torch.set_num_threads(8);seed_all(args.seed)
    baseline=ROOT/'outputs'/args.dataset/'baseline'/f'seed{args.baseline_seed}'/f'{args.baseline_variant}.pt'
    directory=ROOT/'artifacts'/args.dataset
    output=ROOT/args.output/f'{args.mode}_seed{args.seed}_cw{args.concept_weight:g}_ew{args.energy_weight:g}'
    output.mkdir(parents=True,exist_ok=True)
    source_hashes={name:digest(ROOT/'ps_pi'/name) for name in ['noise.py','adapt.py','baseline.py','common.py']}
    configuration={k:v for k,v in vars(args).items() if k!='output'}
    protocol={'config':configuration,'baseline_sha256':digest(baseline),'manifest_sha256':digest(directory/'manifest.json'),
              'source_sha256':source_hashes,'split':'train+val only','test_opened':False,
              'selection':'validation accuracy, tie break validation NLL','objective':'MC marginal negative log likelihood + expected pseudo concept BCE + noise energy'}
    protocol_path=output/'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text())!=protocol:raise RuntimeError('Run protocol changed; use a new output directory')
    atomic_json(protocol,protocol_path)
    if (output/'status.json').exists() and json.loads((output/'status.json').read_text()).get('phase')=='complete':
        print('reuse completed',output,flush=True);return
    pack=torch.load(baseline,weights_only=False)
    with gpu_lease(f'PS-CBM Pi {args.dataset} {args.mode} seed{args.seed}'):
        for name in ['weight','bias','train_mean','train_std']:pack[name]=pack[name].cuda()
        data={split:torch.load(directory/f'{split}_target.pt',weights_only=True) for split in ['train','val']}
        concepts={};cache_dir=directory/'baseline_concepts'/protocol['baseline_sha256'];cache_dir.mkdir(parents=True,exist_ok=True)
        for split in ['train','val']:
            path=cache_dir/f'{split}.pt'
            if not path.exists():atomic_save(encode_concepts(data[split],pack),path)
            concepts[split]=torch.load(path,weights_only=True).cuda()
        labels={split:data[split]['labels'].cuda() for split in data}
        pseudo={split:torch.load(directory/f'{split}_concept_labels.pt',weights_only=True).cuda() for split in data}
        seed_all(args.seed)
        model=ConceptNoise(len(pack['concepts']),len(pack['bias']),pack['class_sets'],args.mode,args.hidden,
                           args.max_mean,args.initial_scale,args.max_scale).cuda()
        generator=torch.Generator(device='cuda').manual_seed(args.seed+101)
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
            history,best_key,best_state,best_epoch=saved['history'],saved['best_key'],saved['best_state'],saved['best_epoch']
            start=saved['next_epoch']
        initial,probs=evaluate(model,concepts['val'],labels['val'],pseudo['val'],pack,args.eval_samples)
        if not history:
            history.append({'epoch':0,'validation':initial});best_key=(initial['correct'],-initial['nll'])
            best_state=snapshot(model);best_epoch=0
        started=time.monotonic()
        epochs=0 if args.mode=='fixed_noise' else args.epochs
        for epoch in range(start,epochs):
            order=torch.randperm(len(concepts['train']),generator=permutation_generator)
            loss_total=0.
            for ids in order.split(args.batch_size):
                ids=ids.cuda();z=concepts['train'][ids];y=labels['train'][ids]
                draws=model.draw(z,args.train_samples,generator)
                task=marginal_nll(F.linear(draws,pack['weight'],pack['bias']),y)
                raw=draws*pack['train_std']+pack['train_mean']
                concept_loss=F.binary_cross_entropy_with_logits(raw,pseudo['train'][ids].unsqueeze(0).expand_as(raw))
                me,ve=model.expected_energy(z)
                loss=task+args.concept_weight*concept_loss+args.energy_weight*(me+ve)
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
                loss_total+=loss.item()*len(ids)
            metrics,_=evaluate(model,concepts['val'],labels['val'],pseudo['val'],pack,args.eval_samples)
            history.append({'epoch':epoch+1,'train_objective':loss_total/len(order),'validation':metrics})
            key=(metrics['correct'],-metrics['nll'])
            if best_key is None or key>best_key:best_key=key;best_state=snapshot(model);best_epoch=epoch+1
            atomic_save({'protocol':protocol,'model':snapshot(model),'optimizer':optimizer.state_dict(),
                         'noise_rng':generator.get_state(),'permutation_rng':permutation_generator.get_state(),
                         'history':history,'best_key':best_key,'best_state':best_state,'best_epoch':best_epoch,'next_epoch':epoch+1},resume)
            atomic_json({'phase':'train','epoch':epoch+1,'epochs':epochs,'best_epoch':best_epoch,'best_accuracy':best_key[0]/len(labels['val']),
                         'pid':os.getpid(),'time':time.time()},output/'status.json')
            if epoch%5==0 or epoch==epochs-1:
                print(f'{args.mode} epoch {epoch+1}/{epochs}: val {100*metrics["accuracy"]:.2f}%, NLL {metrics["nll"]:.4f}, best {100*best_key[0]/len(labels["val"]):.2f}%; {time.monotonic()-started:.1f}s',flush=True)
        model.load_state_dict(best_state)
        metrics,probs=evaluate(model,concepts['val'],labels['val'],pseudo['val'],pack,args.eval_samples)
        mean_metrics,mean_probs=evaluate(model,concepts['val'],labels['val'],pseudo['val'],pack,args.eval_samples,scale_multiplier=0.)
        zero_metrics,_=evaluate(model,concepts['val'],labels['val'],pseudo['val'],pack,args.eval_samples,scale_multiplier=0.,mean_multiplier=0.)
        if abs(zero_metrics['accuracy']-clean['accuracy'])>1e-7:raise RuntimeError('Zero perturbation fails to restore baseline')
        metrics.update(best_epoch=best_epoch,concept_count=len(pack['concepts']),allocated_parameters=sum(p.numel() for p in model.parameters()),
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
    parser.add_argument('--output',default='outputs/cifar10/validation_v1')
    parser.add_argument('--epochs',type=int,default=40);parser.add_argument('--batch-size',type=int,default=1024)
    parser.add_argument('--hidden',type=int,default=128);parser.add_argument('--lr',type=float,default=.001)
    parser.add_argument('--weight-decay',type=float,default=0.)
    parser.add_argument('--concept-weight',type=float,default=.1);parser.add_argument('--energy-weight',type=float,default=.01)
    parser.add_argument('--train-samples',type=int,default=8);parser.add_argument('--eval-samples',type=int,default=64)
    parser.add_argument('--max-mean',type=float,default=.5);parser.add_argument('--initial-scale',type=float,default=.03);parser.add_argument('--max-scale',type=float,default=.3)
    run(parser.parse_args())
if __name__=='__main__':main()
