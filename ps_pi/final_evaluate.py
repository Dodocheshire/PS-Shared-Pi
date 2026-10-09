"""Locked five-seed evaluation: validation, held-out tests and protected attributes."""
import argparse
import json
import torch
from torch.nn import functional as F
from .common import ROOT, atomic_json, atomic_save, digest, gpu_lease, official_module
from .final_lock import verify_lock, PATH, SEEDS, MODES
from .final_geometry import remapped_bank
from .compact_outputs import load_compact_checkpoint
from .anchored_cbl import AnchoredCBL
from .signed_shared_noise import SignedSharedNoise
from .protected_concept_forward import sample_concepts
from .anchored_signed_audit import attribute_metrics
from .verify_validation_artifacts import metrics


def summarize(probabilities,labels):
    result=metrics(probabilities,labels);result.update(accuracy=result['correct']/len(labels),n=len(labels))
    return result


@torch.no_grad()
def original_predictions(model,pack,x):
    result=[];raw=[]
    for batch in x.split(128):
        values=model(batch);z=(values-pack['train_mean'])/pack['train_std']
        result.append(F.linear(z,pack['weight'],pack['bias']).softmax(-1).cpu());raw.append(values.sigmoid().cpu())
    return torch.cat(result),torch.cat(raw)


@torch.no_grad()
def predict(cbl,noise,x,pack,seed,scale=1.):
    generator=torch.Generator(device='cuda').manual_seed(seed);result=[]
    for batch in x.split(128):
        z,_,_=sample_concepts(cbl,noise,batch,pack,256,generator,scale_multiplier=scale)
        result.append(F.linear(z,pack['weight'],pack['bias']).softmax(-1).mean(0).cpu())
    return torch.cat(result)


def evaluation_lock():
    path=ROOT/'reports/final_v1/EVALUATION_LOCK.json'
    models={str(p.relative_to(ROOT)):digest(p) for d in ['cifar10','cifar100','cub'] for s in SEEDS
            for p in [ROOT/f'outputs/{d}/baseline/seed{s}/corrected_best_validation.pt',ROOT/f'outputs/{d}/baseline/seed{s}/official_final_state.pt']}
    models.update({str(p.relative_to(ROOT)):digest(p) for p in sorted((ROOT/'outputs/cub/final_v1').glob('seed*/*/model.pt'))})
    if len(models)!=65:raise RuntimeError(f'Incomplete final models: {len(models)} expected 65')
    value={'final_lock_sha256':digest(PATH),'checkpoint_sha256':models,
           'evaluation_sources':{name:digest(ROOT/'ps_pi'/name) for name in ['final_evaluate.py','final_test_prepare.py']},
           'selection_frozen':True,'test_use':'evaluation only; no selection or parameter changes'}
    if path.exists() and json.loads(path.read_text())!=value:raise RuntimeError('Evaluation lock changed')
    atomic_json(value,path);return value


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['val','test'],default='val');args=p.parse_args()
    plan=verify_lock();lock=evaluation_lock();torch.set_num_threads(8)
    output=ROOT/f'outputs/final_evaluation_v1/{args.split}';output.mkdir(parents=True,exist_ok=True)
    rows=[]
    with gpu_lease(f'Final locked evaluation {args.split}'):
        for dataset in ['cifar10','cifar100','cub']:
            directory=ROOT/'artifacts'/dataset
            data_path=directory/'val_target.pt' if args.split=='val' else directory/'final_test_v1/test_target.pt'
            data=torch.load(data_path,weights_only=True);x=data['features'].cuda();y=data['labels']
            annotations=None
            if dataset=='cub':annotations=(torch.load(directory/'native_attributes_v1/train_val.pt',weights_only=True)['val'] if args.split=='val' else torch.load(directory/'final_test_v1/native_annotations.pt',weights_only=True))
            for seed in SEEDS:
                compact_cbl,compact=load_compact_checkpoint(ROOT/f'outputs/{dataset}/compact_baselines/seed{seed}/model.pt')
                baseline_raw=None;baseline_probs=None;compact_reference_raw=None
                for variant in ['official_final_state','corrected_best_validation']:
                    destination=output/f'{dataset}/seed{seed}/{variant}';destination.mkdir(parents=True,exist_ok=True)
                    checkpoint=ROOT/f'outputs/{dataset}/baseline/seed{seed}/{variant}.pt'
                    pack=torch.load(checkpoint,weights_only=False,map_location='cpu');config=pack['config']
                    cbl=official_module('model.cbl').ConceptBottleneckLayer(2048,len(pack['concepts']),config['cbl_layer_num'],config['cbl_bias']).cuda().eval()
                    cbl.load_state_dict(pack['cbl_state'])
                    for key in ['train_mean','train_std','weight','bias']:pack[key]=pack[key].cuda()
                    probs,raw=original_predictions(cbl,pack,x)
                    if variant=='corrected_best_validation':baseline_raw=raw;baseline_probs=probs
                    native=None
                    if annotations is not None:
                        # Use each corrected model's locked training mapping/thresholds for the main baseline comparison.
                        if variant=='corrected_best_validation':
                            protocol=json.loads((directory/f'native_attributes_replicates/seed{seed}/protocol.json').read_text())
                            full_native=attribute_metrics(raw,annotations,protocol)
                            compact_reference_raw=torch.cat([compact_cbl.cuda()(batch).sigmoid().cpu() for batch in x.split(128)])
                            native=attribute_metrics(compact_reference_raw,annotations,compact['native_protocol']);native.update(split=args.split,test_used=args.split=='test')
                            full_delta=float((compact_reference_raw-raw[:,compact['retained_original_ids']]).abs().max())
                            if full_delta>2e-6:raise RuntimeError('Full-to-compact native numerical difference exceeds tolerance')
                            native['full_to_compact_probability_max_abs_delta']=full_delta
                            native['full_model_metrics']=full_native
                            native['reference']='clean compact model; identical weights/output rows, minor GPU GEMM rounding relative to full model'

                    row={'dataset':dataset,'baseline_seed':seed,'method':variant,'split':args.split,
                         **summarize(probs,y),'native_attributes':native,'checkpoint_sha256':digest(checkpoint),
                         'test_used_for_selection':False,'test_used_for_evaluation':args.split=='test'}
                    atomic_save({'probabilities':probs,'labels':y,'indices':data['indices']},destination/'predictions.pt')
                    atomic_json(row,destination/'metrics.json');rows.append(row)
                    print(dataset,seed,variant,f"{100*row['accuracy']:.2f}%",flush=True)
                if dataset!='cub':continue
                original=torch.load(ROOT/f'outputs/cub/baseline/seed{seed}/corrected_best_validation.pt',weights_only=False,map_location='cpu')
                bank=remapped_bank(original,compact)
                for key in ['train_mean','train_std','weight','bias']:compact[key]=compact[key].cuda()
                anchored=AnchoredCBL(compact_cbl,compact['native_protocol']).cuda().eval()
                protected=~bank['active_mask'].bool()
                for mode in MODES:
                    checkpoint=ROOT/f'outputs/cub/final_v1/seed{seed}/{mode}_seed{seed-42}_cw5_lr3e-06/model.pt'
                    model=torch.load(checkpoint,weights_only=False,map_location='cpu');config=model['protocol']['config']
                    anchored.load_state_dict(model['state']['cbl'])
                    noise=SignedSharedNoise(bank,len(compact['bias']),mode,config['hidden'],config['max_mean'],config['initial_scale'],config['min_scale'],config['max_scale']).cuda().eval()
                    noise.load_state_dict(model['state']['noise'])
                    # Frozen reference and protected geometry must equal this independently trained baseline.
                    for name,value in compact['cbl_state'].items():
                        if not torch.equal(anchored.state_dict()['anchor.'+name].cpu(),value):raise RuntimeError('Frozen native reference changed')
                    native_raw=torch.cat([anchored(batch).sigmoid().cpu() for batch in x.split(128)])
                    expected=compact_reference_raw[:,protected]
                    if not torch.equal(native_raw[:,protected],expected):raise RuntimeError('Protected test/validation outputs changed')
                    native=attribute_metrics(native_raw,annotations,compact['native_protocol']);native.update(split=args.split,test_used=args.split=='test')
                    # Protected perturbations are zero; attribute sigmoid responses equal clean reference exactly.
                    if noise.active_mask.cpu()[protected].ne(0).any():raise RuntimeError('Noise perturbs native coordinates')
                    native_probability=compact_reference_raw[:,protected]
                    native_ids=protected.nonzero().flatten()
                    mc=[predict(anchored,noise,x,compact,s) for s in plan['mc_seeds']]
                    probs=torch.stack(mc).mean(0);mean=predict(anchored,noise,x,compact,9071,scale=0.)
                    row={'dataset':dataset,'baseline_seed':seed,'adaptation_seed':seed-42,'method':mode,'split':args.split,
                         **summarize(probs,y),'mc_metrics':[summarize(v,y) for v in mc],
                         'same_model_mean':summarize(mean,y),'native_attributes':native,
                         'native_protected_exact':True,'checkpoint_sha256':digest(checkpoint),'best_epoch':model['best_epoch'],
                         'selection_eligible':json.loads((checkpoint.parent/'metrics.json').read_text())['metrics']['selection_eligible'],
                         'test_used_for_selection':False,'test_used_for_evaluation':args.split=='test'}
                    destination=output/f'cub/seed{seed}/{mode}';destination.mkdir(parents=True,exist_ok=True)
                    atomic_save({'probabilities':probs,'mc_probabilities':mc,'mean_probabilities':mean,'labels':y,
                                 'indices':data['indices'],'protected_probability':native_probability,'protected_ids':native_ids},destination/'predictions.pt')
                    atomic_json(row,destination/'metrics.json');rows.append(row)
                    print('cub',seed,mode,f"{100*row['accuracy']:.2f}%",'native exact',flush=True)
    atomic_json({'rows':rows,'models':len(rows),'split':args.split,'lock_sha256':digest(PATH),
                 'evaluation_lock_sha256':digest(ROOT/'reports/final_v1/EVALUATION_LOCK.json'),
                 'test_used_for_selection':False,'test_used_for_evaluation':args.split=='test'},output/'summary.json')


if __name__=='__main__':main()
