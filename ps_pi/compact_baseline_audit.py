"""Compile and verify lossless concept-output reductions on complete baselines."""
import csv
import json
import math
import time
import torch
from torch.nn import functional as F
from .baseline import validation_metrics
from .common import ROOT, atomic_json, atomic_save, digest, official_module
from .compact_outputs import compact_cbl, retained_concepts, remap_attributes, load_compact_checkpoint
from .joint_observed_views_audit import attribute_metrics


@torch.no_grad()
def main():
    torch.set_num_threads(8)
    CBL=official_module('model.cbl').ConceptBottleneckLayer
    records=[];pending=[]
    native=json.loads((ROOT/'artifacts/cub/native_attributes_v1/protocol.json').read_text())
    protected={i for item in native['evaluated_cohort'].values() for i in item['concept_ids']}
    annotations=torch.load(ROOT/'artifacts/cub/native_attributes_v1/train_val.pt',weights_only=True)
    for dataset in ['cifar10','cifar100','cub']:
        directory=ROOT/'artifacts'/dataset
        val=torch.load(directory/'val_target.pt',weights_only=True)
        for seed in range(42,47):
            folder=ROOT/f'outputs/{dataset}/baseline/seed{seed}'
            if not (folder/'status.json').exists() or json.loads((folder/'status.json').read_text())['phase']!='complete':
                pending.append({'dataset':dataset,'seed':seed});continue
            path=folder/'corrected_best_validation.pt';pack=torch.load(path,weights_only=False)
            if pack['protocol']['test_used'] or pack['config']['seed']!=seed:
                raise RuntimeError('Original baseline access/seed protocol differs')
            if pack['protocol']['manifest_sha256']!=digest(directory/'manifest.json'):
                raise RuntimeError('Baseline split source differs')
            model=CBL(2048,len(pack['concepts']),pack['config']['cbl_layer_num'],pack['config']['cbl_bias'])
            model.load_state_dict(pack['cbl_state']);model.eval()
            ids=retained_concepts(pack['weight'],protected if dataset=='cub' else [])
            compact=compact_cbl(model,ids,len(pack['concepts']));compact.eval()
            original=torch.cat([model(x) for x in val['features'].split(2048)])
            reduced=torch.cat([compact(x) for x in val['features'].split(2048)])
            if not torch.allclose(reduced,original[:,ids],atol=2e-5,rtol=2e-5):
                raise RuntimeError('Retained concept responses differ')
            full_logits=F.linear((original-pack['train_mean'])/pack['train_std'],pack['weight'],pack['bias'])
            logits=F.linear((reduced-pack['train_mean'][:,ids])/pack['train_std'][:,ids],pack['weight'][:,ids],pack['bias'])
            if not torch.allclose(logits,full_logits,atol=2e-5,rtol=2e-5):
                raise RuntimeError('Compact logits violate exact zero-column algebra')
            if not torch.equal(logits.argmax(1),full_logits.argmax(1)):
                raise RuntimeError('Compact float32 predictions changed')
            metrics=validation_metrics(logits,val['labels']);before=validation_metrics(full_logits,val['labels'])
            if any(abs(metrics[k]-before[k])>1e-6 for k in ['accuracy','nll','ece15']):
                raise RuntimeError('Probability quality changed after compaction')
            n=len(ids);k=len(pack['concepts']);classes=len(pack['bias'])
            original_parameters=sum(p.numel() for p in model.parameters())+pack['weight'].numel()+pack['bias'].numel()
            compact_parameters=sum(p.numel() for p in compact.parameters())+pack['weight'][:,ids].numel()+pack['bias'].numel()
            row={'dataset':dataset,'seed':seed,'original_concepts':k,'retained_concepts':n,
                 'original_cbl_hidden_width':model.cbl[0].out_features,'compact_cbl_hidden_width':compact.cbl[0].out_features,
                 'protected_native_concepts':len(protected) if dataset=='cub' else 0,
                 'output_reduction_percent':100*(1-n/k),'original_cbl_head_parameters':original_parameters,
                 'compact_cbl_head_parameters':compact_parameters,'parameter_reduction_percent':100*(1-compact_parameters/original_parameters),
                 'validation_accuracy_percent':100*metrics['accuracy'],'nll':metrics['nll'],'ece15':metrics['ece15'],
                 'validation_images':len(val['labels']),'max_raw_difference':float((reduced-original[:,ids]).abs().max()),
                 'max_logit_difference':float((logits-full_logits).abs().max()),'prediction_changes':0,
                 'nll_delta':metrics['nll']-before['nll'],'ece_delta':metrics['ece15']-before['ece15'],
                 'original_cea':100*metrics['accuracy']/math.log(k,math.ceil(math.log2(classes)))**.25,
                 'compact_cea':100*metrics['accuracy']/math.log(n,math.ceil(math.log2(classes)))**.25,
                 'retained_original_ids':ids.tolist(),'baseline_sha256':digest(path),'test_used':False}
            protocol={'source_sha256':{name:digest(ROOT/'ps_pi'/name) for name in ['compact_outputs.py','compact_baseline_audit.py']},
                      'baseline_sha256':digest(path),'retention':'all exact nonzero head columns plus train-mapped native attribute channels for CUB',
                      'uses_validation_for_retention':False,'test_used':False,
                      'hidden_width_preserved':True,'original_scaler_preserved':True}
            compact_native=None
            if dataset=='cub':
                native_path=directory/f'native_attributes_replicates/seed{seed}/protocol.json'
                if not native_path.exists():raise RuntimeError('Prepare seed-specific native thresholds first')
                source_native=json.loads(native_path.read_text());compact_native=remap_attributes(source_native,ids)
                if not torch.equal(annotations['val']['indices'],val['indices']):
                    raise RuntimeError('Native validation order differs')
                a=attribute_metrics(original.sigmoid(),annotations['val'],source_native)
                b=attribute_metrics(reduced.sigmoid(),annotations['val'],compact_native)
                if any(abs(a[m]-b[m])>1e-7 for m in ['macro_balanced_accuracy','macro_auroc']):
                    raise RuntimeError('Compact native attribute quality changed')
                row.update(native_attributes=b['evaluated_attributes'],native_ba=b['macro_balanced_accuracy'],
                           native_auroc=b['macro_auroc'],native_ba_delta=b['macro_balanced_accuracy']-a['macro_balanced_accuracy'],
                           native_auroc_delta=b['macro_auroc']-a['macro_auroc'])
                protocol['native_protocol_sha256']=digest(native_path)
            output=ROOT/f'outputs/{dataset}/compact_baselines/seed{seed}';output.mkdir(parents=True,exist_ok=True)
            atomic_save({'cbl_state':compact.state_dict(),'weight':pack['weight'][:,ids],
                         'bias':pack['bias'],'train_mean':pack['train_mean'][:,ids],'train_std':pack['train_std'][:,ids],
                         'concepts':[pack['concepts'][i] for i in ids.tolist()], 'retained_original_ids':ids,
                         'source_config':pack['config'],'native_protocol':compact_native,'protocol':protocol,
                         'input_dimension':model.cbl[0].in_features,'original_concept_count':k,
                         'validation_metrics':metrics,
                         'architecture':'source hidden width retained; only last-linear outputs and head input columns reduced'},output/'model.pt')
            restored,restored_pack=load_compact_checkpoint(output/'model.pt')
            probe=val['features'][:128]
            if not torch.equal(restored(probe),compact(probe)):
                raise RuntimeError('Serialized compact model does not reproduce the in-memory model')
            row['reload_verified']=True
            atomic_json(row,output/'verification.json');records.append(row)
            print(dataset,seed,k,'->',n,'exact validation predictions/native responses preserved',flush=True)
    report=ROOT.parent/'Essay/PS-Shared/report'
    atomic_json({'records':records,'pending':pending,'checked_at':time.time(),'source_sha256':digest(__file__),
                 'test_used':False,'noise_gain':False,'final_five_seed_pi_evidence':False,
                 'scope':'Compile exact-zero head output compaction, preserving all train-mapped CUB native audit responses and original hidden widths. Retention never uses validation accuracy.',
                 'limits':'Clean-model efficiency result, not Pi efficacy. CEA here uses retained concept-output count; it is not matched to original-paper concept-bank construction. Native preservation covers the fixed 68 mapped attributes only. No test, runtime or GPU memory benchmark, and no pruning of hidden channels, feature encoder or a learned noise conditioner.'},report/'compact_baselines.json')
    with (report/'compact_baselines.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0])+['native_attributes','native_ba','native_auroc','native_ba_delta','native_auroc_delta'],lineterminator='\n')
        writer.writeheader();writer.writerows(records)


if __name__=='__main__':main()
