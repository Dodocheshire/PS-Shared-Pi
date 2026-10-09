"""Independent Monte Carlo and native-attribute validation of locked residual-noise checkpoints."""
# V5 implementation preserved as a separate version to protect running protocols.
import argparse
import json
import time
import torch
from sklearn.metrics import roc_auc_score
from .common import ROOT,atomic_json,atomic_save,digest,gpu_lease,official_module
from .joint_noise import FeatureNoise
from .joint_feature_augmented_train import evaluate

@torch.no_grad()
def concept_probabilities(cbl,noise,x,context,samples,seed):
    generator=torch.Generator(device=x.device).manual_seed(seed);result=[]
    for start in range(0,len(x),128):
        rows=x[start:start+128];ctx=context[start:start+len(rows)]
        draws=noise.draw(rows,ctx,samples,generator)
        raw=cbl(draws.reshape(-1,x.shape[1])).reshape(len(draws),len(rows),-1)
        result.append(raw.sigmoid().mean(0).cpu())
    return torch.cat(result)

def attribute_metrics(probabilities,annotations,protocol):
    rows=[];undefined=[]
    for key,item in protocol['evaluated_cohort'].items():
        aid=int(key);known=annotations['certainty'][:,aid]>=protocol['certainty_minimum']
        truth=annotations['positive'][:,aid][known]
        if not truth.any() or truth.all():
            undefined.append(aid);continue
        scores=probabilities[:,item['concept_ids']].max(1).values[known]
        pred=scores>=item['threshold'];positive=truth;negative=~truth
        ba=.5*(pred[positive].float().mean().item()+(~pred[negative]).float().mean().item())
        rows.append({'attribute_id':aid,'name':item['name'],'balanced_accuracy':ba,
                     'auroc':float(roc_auc_score(truth.numpy(),scores.numpy())),
                     'known_positive':int(positive.sum()),'known_negative':int(negative.sum())})
    return {'macro_balanced_accuracy':sum(r['balanced_accuracy'] for r in rows)/len(rows),
            'macro_auroc':sum(r['auroc'] for r in rows)/len(rows),'evaluated_attributes':len(rows),
            'train_selected_attributes':len(protocol['evaluated_cohort']),'undefined_attribute_ids':undefined,
            'details':rows,'split':'val','threshold_split':'train','test_used':False}

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',default='cub',choices=['cifar10','cifar100','cub'])
    p.add_argument('--input',default=None);p.add_argument('--samples',type=int,default=256)
    args=p.parse_args();torch.set_num_threads(8)
    folder=ROOT/(args.input or f'outputs/{args.dataset}/validation_joint_feature_augmented_v13')
    paths=sorted(folder.glob('*/model.pt'))
    if len(paths)!=16:raise RuntimeError(f'Expected the complete 16-case cohort, found {len(paths)}')
    if any(json.loads((path.parent/'status.json').read_text())['phase']!='complete' for path in paths):
        raise RuntimeError('Cohort is not complete')
    output=folder.parent/(folder.name+'_audit');output.mkdir(exist_ok=True)
    directory=ROOT/'artifacts'/args.dataset
    baseline_path=ROOT/'outputs'/args.dataset/'baseline/seed42/corrected_best_validation.pt'
    pack=torch.load(baseline_path,weights_only=False)
    bank=torch.load(directory/'joint_feature_augmented_v13/directions.pt',weights_only=False)
    protocol={'dataset':args.dataset,'samples':args.samples,'independent_mc_seeds':[9071,9072],
              'attribute_mc_seed':9081,'checkpoint_selection_frozen':True,'test_used':False,
              'source_sha256':digest(__file__),'baseline_sha256':digest(baseline_path),
              'directions_sha256':digest(directory/'joint_feature_augmented_v13/directions.pt'),
              'checkpoint_sha256':{str(path.relative_to(ROOT)):digest(path) for path in paths}}
    attribute_protocol=None;annotations=None
    if args.dataset=='cub':
        attribute_path=directory/'native_attributes_v1/protocol.json'
        attribute_protocol=json.loads(attribute_path.read_text())
        annotations=torch.load(directory/'native_attributes_v1/train_val.pt',weights_only=True)['val']
        protocol['attribute_protocol_sha256']=digest(attribute_path)
        protocol['annotation_sha256']=digest(directory/'native_attributes_v1/train_val.pt')
    guard=output/'protocol.json'
    if guard.exists() and json.loads(guard.read_text())!=protocol:raise RuntimeError('Audit protocol changed')
    atomic_json(protocol,guard);rows=[]
    with gpu_lease(f'Joint independent MC audit {args.dataset}'):
        data=torch.load(directory/'val_target.pt',weights_only=True)
        if annotations is not None and not torch.equal(annotations['indices'],data['indices']):raise RuntimeError('Native annotation alignment differs')
        x=data['features'].cuda();y=data['labels'].cuda()
        pseudo=torch.load(directory/'val_concept_labels.pt',weights_only=True).cuda()
        context=torch.load(directory/'baseline_concepts'/digest(baseline_path)/'val.pt',weights_only=True).cuda()
        for key in ['train_mean','train_std','weight','bias']:pack[key]=pack[key].cuda()
        CBL=official_module('model.cbl').ConceptBottleneckLayer
        cbl=CBL(x.shape[1],len(pack['concepts']),pack['config']['cbl_layer_num'],pack['config']['cbl_bias']).cuda()
        cbl.load_state_dict(pack['cbl_state'])
        clean_noise=FeatureNoise(bank,len(pack['bias']),'clean').cuda()
        baseline,_=evaluate(cbl,clean_noise,x,context,y,pseudo,pack,args.samples)
        native_baseline=None
        if annotations is not None:
            native_baseline=attribute_metrics(concept_probabilities(cbl,clean_noise,x,context,args.samples,9081),annotations,attribute_protocol)
        atomic_json({'metrics':baseline,'native_attributes':native_baseline},output/'baseline.json')
        for index,path in enumerate(paths):
            destination=output/path.parent.name;destination.mkdir(exist_ok=True)
            metric_path=destination/'metrics.json'
            if metric_path.exists():rows.append(json.loads(metric_path.read_text()));continue
            model=torch.load(path,weights_only=False);config=model['protocol']['config']
            if model['protocol']['baseline_sha256']!=protocol['baseline_sha256']:raise RuntimeError('Audit baseline differs')
            cbl.load_state_dict(model['state']['cbl'])
            noise=FeatureNoise(bank,len(pack['bias']),config['mode'],config['hidden'],config['max_mean'],config['initial_scale'],config['min_scale'],config['max_scale']).cuda()
            noise.load_state_dict(model['state']['noise'])
            runs=[]
            for seed in protocol['independent_mc_seeds']:
                metrics,probs=evaluate(cbl,noise,x,context,y,pseudo,pack,args.samples,eval_seed=seed)
                runs.append(metrics)
                atomic_save({'probabilities':probs,'labels':y.cpu(),'indices':data['indices']},destination/f'predictions_seed{seed}.pt')
            mean,mean_probs=evaluate(cbl,noise,x,context,y,pseudo,pack,args.samples,scale_multiplier=0.)
            atomic_save({'probabilities':mean_probs,'labels':y.cpu(),'indices':data['indices']},destination/'mean_predictions.pt')
            native=None
            if annotations is not None:
                native=attribute_metrics(concept_probabilities(cbl,noise,x,context,args.samples,9081),annotations,attribute_protocol)
            result={'case':path.parent.name,'mode':config['mode'],'config':config,'best_epoch':model['best_epoch'],
                    'mc_runs':runs,'same_model_mean':mean,'native_attributes':native,
                    'delta_baseline_pp_range':[100*(min(r['accuracy'] for r in runs)-baseline['accuracy']),100*(max(r['accuracy'] for r in runs)-baseline['accuracy'])],
                    'native_ba_delta_pp':None if native is None else 100*(native['macro_balanced_accuracy']-native_baseline['macro_balanced_accuracy']),
                    'native_auc_delta_pp':None if native is None else 100*(native['macro_auroc']-native_baseline['macro_auroc']),
                    'checkpoint_sha256':digest(path),'test_used':False}
            atomic_json(result,metric_path);rows.append(result)
            atomic_json({'phase':'audit','completed':index+1,'total':len(paths),'time':time.time()},output/'status.json')
            print(result['case'],'ACC',[round(100*r['accuracy'],3) for r in runs],
                  'attribute BA delta',result['native_ba_delta_pp'],flush=True)
    atomic_json({'baseline':baseline,'native_baseline':native_baseline,'rows':rows,'protocol':protocol,
                 'limits':'Selected single-seed validation checkpoints, independently resampled without re-selection. Not independent test or five-seed evidence.'},output/'summary.json')
    atomic_json({'phase':'complete','completed':len(rows),'test_used':False,'time':time.time()},output/'status.json')

if __name__=='__main__':main()
