"""Prepare train/validation concept contexts and seed-specific native thresholds."""
import argparse
import copy
import json
import time
import torch
from .common import ROOT, atomic_json, atomic_save, digest, official_module
from .native_attributes import threshold_from_training
from .joint_observed_views_audit import attribute_metrics


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--datasets',nargs='+',default=['cub'],choices=['cub','cifar10','cifar100'])
    args=parser.parse_args()
    torch.set_num_threads(8)
    CBL=official_module('model.cbl').ConceptBottleneckLayer
    records=[];pending=[]
    for dataset in args.datasets:
        directory=ROOT/'artifacts'/dataset
        data={s:torch.load(directory/f'{s}_target.pt',weights_only=True) for s in ['train','val']}
        manifest=json.loads((directory/'manifest.json').read_text())
        for split,item in data.items():
            if not torch.equal(item['indices'],torch.tensor(manifest['indices'][split])):
                raise RuntimeError('Fixed feature/manifest order differs')
        reference=torch.load(ROOT/f'outputs/{dataset}/baseline/seed42/corrected_best_validation.pt',weights_only=False)
        native=None;annotations=None
        if dataset=='cub':
            native_path=directory/'native_attributes_v1/protocol.json'
            native=json.loads(native_path.read_text())
            annotations=torch.load(directory/'native_attributes_v1/train_val.pt',weights_only=True)
            for split in data:
                if not torch.equal(annotations[split]['indices'],data[split]['indices']):
                    raise RuntimeError('Native annotation order differs')
            if native['threshold_split']!='train' or native['test_examples_evaluated']:
                raise RuntimeError('Expected fixed training-only attribute mapping')
        for seed in range(42,47):
            folder=ROOT/f'outputs/{dataset}/baseline/seed{seed}'
            if not (folder/'status.json').exists() or json.loads((folder/'status.json').read_text())['phase']!='complete':
                pending.append({'dataset':dataset,'seed':seed});continue
            path=folder/'corrected_best_validation.pt';sha=digest(path)
            pack=torch.load(path,weights_only=False)
            if pack['concepts']!=reference['concepts'] or pack['config']['seed']!=seed or pack['protocol']['test_used']:
                raise RuntimeError('Baseline concept order, seed, or data-access protocol differs')
            if pack['protocol']['manifest_sha256']!=digest(directory/'manifest.json'):
                raise RuntimeError('Baseline split hash changed')
            model=CBL(2048,len(pack['concepts']),pack['config']['cbl_layer_num'],pack['config']['cbl_bias'])
            model.load_state_dict(pack['cbl_state']);model.eval()
            cache=directory/'baseline_concepts'/sha;cache.mkdir(exist_ok=True)
            context={}
            for split,item in data.items():
                dest=cache/f'{split}.pt'
                computed=torch.cat([(model(x)-pack['train_mean'])/pack['train_std'] for x in item['features'].split(2048)])
                if dest.exists():
                    stored=torch.load(dest,weights_only=True)
                    if stored.shape!=computed.shape or not torch.allclose(stored,computed,atol=2e-5,rtol=2e-5):
                        raise RuntimeError('Cached baseline context does not reproduce checkpoint')
                else:
                    atomic_save(computed,dest);stored=computed
                context[split]=stored
            row={'dataset':dataset,'seed':seed,'baseline_sha256':sha,
                 'context_sha256':{s:digest(cache/f'{s}.pt') for s in data},
                 'context_scope':'fixed train/val feature order; no test features', 'test_used':False}
            if native is not None:
                protocol=copy.deepcopy(native)
                for aid,item in protocol['evaluated_cohort'].items():
                    aid=int(aid)
                    probability=(context['train'][:,item['concept_ids']]*pack['train_std'][:,item['concept_ids']]+pack['train_mean'][:,item['concept_ids']]).sigmoid()
                    scores=probability.max(1).values.numpy()
                    known=(annotations['train']['certainty'][:,aid]>=protocol['certainty_minimum']).numpy()
                    positive=annotations['train']['positive'][:,aid].numpy()
                    item['threshold']=threshold_from_training(scores,positive,known)
                protocol.update(baseline_sha256=sha,baseline_seed=seed,
                                source_sha256=digest(__file__),reference_mapping_sha256=digest(native_path),
                                native_train_val_sha256=digest(directory/'native_attributes_v1/train_val.pt'),
                                threshold_selection='Independent training-only threshold fit for each complete baseline; original semantic mapping and attribute cohort fixed')
                native_out=directory/'native_attributes_replicates'/f'seed{seed}'
                native_out.mkdir(parents=True,exist_ok=True)
                guard=native_out/'protocol.json'
                if guard.exists() and json.loads(guard.read_text())!=protocol:
                    raise RuntimeError('Seed-specific native threshold protocol changed')
                atomic_json(protocol,guard)
                probabilities=(context['val']*pack['train_std']+pack['train_mean']).sigmoid()
                metrics=attribute_metrics(probabilities,annotations['val'],protocol)
                atomic_json(metrics,native_out/'baseline_validation.json')
                row.update(native_attributes=metrics['evaluated_attributes'],native_balanced_accuracy=metrics['macro_balanced_accuracy'],
                           native_auroc=metrics['macro_auroc'],native_protocol_sha256=digest(guard),
                           native_protocol_path=str(guard.relative_to(ROOT)))
            records.append(row)
            print(dataset,seed,'contexts verified',row.get('native_balanced_accuracy'),flush=True)
    output=ROOT.parent/'Essay/PS-Shared/report/baseline_contexts.json'
    atomic_json({'records':records,'pending':pending,'checked_at':time.time(),'source_sha256':digest(__file__),
                 'test_used':False,'locked_noise_candidate':False,'final_five_seed_evidence':False,
                 'limits':'Preparatory concept contexts and training-only native thresholds; not Pi/control replication or test evidence. Native cohorts use the same conservative mapping; thresholds differ per independently trained baseline.'},output)


if __name__=='__main__':main()
