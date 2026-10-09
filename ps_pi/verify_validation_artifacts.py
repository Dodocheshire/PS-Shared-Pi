"""Recompute frozen MC metrics and verify source/state integrity on real artifacts."""
import json
import torch
from .common import ROOT,atomic_json,digest

VERSIONS=['validation_joint_native_training_v15','validation_joint_native_calibrated_v16',
          'validation_compact_concept_v17','validation_protected_concept_v18',
          'validation_protected_predictive_v19','validation_protected_fullmean_v20','validation_protected_signed_v21','validation_anchored_signed_v22']


def metrics(probabilities,labels):
    confidence,pred=probabilities.max(-1);correct=pred.eq(labels);ece=0.
    for lo,hi in zip(torch.linspace(0,1,16)[:-1],torch.linspace(0,1,16)[1:]):
        mask=(confidence>lo)&(confidence<=hi)
        if mask.any():ece+=mask.float().mean().item()*abs(correct[mask].float().mean().item()-confidence[mask].mean().item())
    return {'correct':int(correct.sum()),'nll':float(-probabilities[torch.arange(len(labels)),labels].clamp_min(1e-30).log().mean()),'ece15':ece}


def main():
    torch.set_num_threads(8)
    data=torch.load(ROOT/'artifacts/cub/val_target.pt',weights_only=True,map_location='cpu')
    compact=torch.load(ROOT/'outputs/cub/compact_baselines/seed42/model.pt',weights_only=False,map_location='cpu')
    protected=torch.tensor(sorted({i for item in compact['native_protocol']['evaluated_cohort'].values() for i in item['concept_ids']}))
    original=compact['cbl_state'];last=f'cbl.{max(int(k.split(".")[1]) for k in original)}'
    records=[];pending=[];checkpoints=0;native_arrays=0
    for version in VERSIONS:
        folder=ROOT/'outputs/cub'/version;audit=folder.parent/(version+'_audit')
        status=audit/'status.json'
        if not status.exists() or json.loads(status.read_text())['phase']!='complete':pending.append(version);continue
        summary=json.loads((audit/'summary.json').read_text());protocol=summary['protocol']
        if len(summary['rows'])!=16 or protocol['test_used']:raise RuntimeError('Cohort scope/access differs')
        for row in summary['rows']:
            path=folder/row['case']/'model.pt';model=torch.load(path,weights_only=False,map_location='cpu')
            sha=digest(path)
            if sha!=row['checkpoint_sha256'] or sha!=protocol['checkpoint_sha256'][str(path.relative_to(ROOT))]:
                raise RuntimeError('Frozen checkpoint SHA256 differs')
            for name,expected in model['protocol']['source_sha256'].items():
                if digest(ROOT/'ps_pi'/name)!=expected:raise RuntimeError(f'Training source differs: {name}')
            if model['protocol']['test_access']:raise RuntimeError('Training used test')
            if row.get('native_protection_verified'):
                if not torch.equal(model['state']['noise']['active_mask'].eq(0).nonzero().flatten(),protected):
                    raise RuntimeError('Protected coordinate IDs differ')
                for name,value in original.items():
                    if 'anchor.'+name in model['state']['cbl']:
                        candidate=model['state']['cbl']['anchor.'+name]
                    else:
                        candidate=model['state']['cbl'][name]
                        if name.startswith(last+'.'):value=value[protected];candidate=candidate[protected]
                    if not torch.equal(value,candidate):raise RuntimeError('Hidden/protected CBL state changed')
                native=torch.load(audit/row['case']/'native_probabilities.pt',weights_only=True,map_location='cpu')
                if not torch.equal(native['indices'],data['indices']):raise RuntimeError('Native probability order differs')
                if not torch.equal(native['probabilities'][:,native['protected_ids']],native['reference_protected_probabilities']):
                    raise RuntimeError('Recorded protected probabilities differ')
                if row['native_ba_delta_pp']!=0. or row['native_auc_delta_pp']!=0.:raise RuntimeError('Native quality changed')
                native_arrays+=1
            for seed,reported in zip(protocol['independent_mc_seeds'],row['mc_runs']):
                source=audit/row['case']/f'predictions_seed{seed}.pt'
                cache=torch.load(source,weights_only=True,map_location='cpu');probs=cache['probabilities']
                if not torch.equal(cache['labels'],data['labels']) or not torch.equal(cache['indices'],data['indices']):
                    raise RuntimeError('Validation prediction order/labels differ')
                if not torch.isfinite(probs).all() or (probs<0).any() or not torch.allclose(probs.sum(-1),torch.ones(len(probs)),atol=2e-5):
                    raise RuntimeError('Invalid probability array')
                recomputed=metrics(probs,cache['labels'])
                if recomputed['correct']!=reported['correct'] or abs(recomputed['correct']/len(probs)-reported['accuracy'])>1e-7:
                    raise RuntimeError('Recorded accuracy differs')
                if any(abs(recomputed[key]-reported[key])>1e-6 for key in ['nll','ece15']):
                    raise RuntimeError('Recorded probability quality differs')
                records.append({'version':version,'case':row['case'],'mc_seed':seed,**recomputed,
                                'n':len(probs),'checkpoint_sha256':sha,'predictions_sha256':digest(source)})
            checkpoints+=1
        print(version,'16 checkpoints / 32 prediction arrays verified',flush=True)
    atomic_json({'verified_checkpoints':checkpoints,'verified_prediction_arrays':len(records),
                 'verified_protected_native_arrays':native_arrays,'pending':pending,'records':records,
                 'source_sha256':digest(__file__),'test_used':False,'final_five_seed_evidence':False,
                 'limits':'Integrity/recomputed validation metrics only. Does not prove stochastic efficacy, test generalization or independent five-seed Pi gains. Protected native scope is 68 mapped attributes.'},
                ROOT.parent/'Essay/PS-Shared/report/validation_artifact_verification.json')


if __name__=='__main__':main()
