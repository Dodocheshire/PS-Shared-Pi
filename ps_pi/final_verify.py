"""Independent CPU recomputation of every final metric and provenance link."""
import json
import torch
from .common import ROOT, atomic_json, digest
from .final_lock import verify_lock, PATH, SEEDS, MODES
from .verify_validation_artifacts import metrics
from .anchored_signed_audit import attribute_metrics


def main():
    torch.set_num_threads(8);plan=verify_lock()
    lock=json.loads((ROOT/'reports/final_v1/EVALUATION_LOCK.json').read_text())
    if digest(PATH)!=lock['final_lock_sha256']:raise RuntimeError('Final lock hash changed')
    for name,sha in lock['checkpoint_sha256'].items():
        if digest(ROOT/name)!=sha:raise RuntimeError(f'Final checkpoint changed: {name}')
    for name,sha in lock['evaluation_sources'].items():
        if digest(ROOT/'ps_pi'/name)!=sha:raise RuntimeError('Evaluation source changed')
    delivery=json.loads((ROOT/'reports/final_v1/DELIVERY_PLAN.json').read_text())
    if delivery['test_used_for_selection'] or digest(ROOT/'ps_pi/final_infer.py')!=delivery['inference_source_sha256']:raise RuntimeError('Test-selected delivery or changed inference code')
    for seed in SEEDS:
        compact=torch.load(ROOT/f'outputs/cub/compact_baselines/seed{seed}/model.pt',weights_only=False,map_location='cpu')
        protected=sorted({i for item in compact['native_protocol']['evaluated_cohort'].values() for i in item['concept_ids']})
        for mode in MODES:
            checkpoint=ROOT/f'outputs/cub/final_v1/seed{seed}/{mode}_seed{seed-42}_cw5_lr3e-06/model.pt'
            saved=torch.load(checkpoint,weights_only=False,map_location='cpu');config=saved['protocol']['config']
            for key in ['epochs','concept_weight','cbl_lr','noise_lr','batch_size','hidden','train_samples','max_mean','initial_scale','min_scale','max_scale','mixup_alpha','preservation_weight','energy_weight','prior_weight']:
                if config[key]!=plan['config'][key]:raise RuntimeError('Final model deviates from locked configuration')
            if saved['protocol']['test_access'] or config['baseline_seed']!=seed or config['seed']!=seed-42:raise RuntimeError('Final independence or test isolation differs')
            for name,value in compact['cbl_state'].items():
                if not torch.equal(saved['state']['cbl']['anchor.'+name],value):raise RuntimeError('Frozen anchor differs')
            noise=saved['state']['noise'];ids=torch.tensor(protected)
            if noise['active_mask'][ids].ne(0).any() or noise['factor'][ids].ne(0).any() or noise['directions'][ids].ne(0).any():raise RuntimeError('Protected channels have nonzero perturbation support')
    verified=[];native_count=0
    for split in ['val','test']:
        summary=json.loads((ROOT/f'outputs/final_evaluation_v1/{split}/summary.json').read_text())
        if summary['models']!=65 or summary['test_used_for_selection']:raise RuntimeError('Incomplete or test-selected results')
        for row in summary['rows']:
            dataset=row['dataset'];seed=row['baseline_seed'];method=row['method']
            destination=ROOT/f'outputs/final_evaluation_v1/{split}/{dataset}/seed{seed}/{method}'
            cache=torch.load(destination/'predictions.pt',weights_only=True,map_location='cpu')
            data_path=ROOT/f'artifacts/{dataset}/val_target.pt' if split=='val' else ROOT/f'artifacts/{dataset}/final_test_v1/test_target.pt'
            data=torch.load(data_path,weights_only=True,map_location='cpu')
            if not torch.equal(cache['indices'],data['indices']) or not torch.equal(cache['labels'],data['labels']):raise RuntimeError('Final prediction alignment differs')
            p=cache['probabilities'];y=cache['labels']
            if not torch.isfinite(p).all() or (p<0).any() or not torch.allclose(p.sum(1),torch.ones(len(p)),atol=2e-5):raise RuntimeError('Invalid final probability array')
            computed=metrics(p,y)
            if computed['correct']!=row['correct'] or abs(computed['correct']/len(y)-row['accuracy'])>1e-8:raise RuntimeError('Final ACC recomputation differs')
            if any(abs(computed[k]-row[k])>1e-7 for k in ['nll','ece15']):raise RuntimeError('Final calibration recomputation differs')
            if method in MODES:
                compact=torch.load(ROOT/f'outputs/cub/compact_baselines/seed{seed}/model.pt',weights_only=False,map_location='cpu')
                annotations=(torch.load(ROOT/'artifacts/cub/native_attributes_v1/train_val.pt',weights_only=True)['val'] if split=='val' else torch.load(ROOT/'artifacts/cub/final_test_v1/native_annotations.pt',weights_only=True))
                scores=torch.zeros(len(y),len(compact['concepts']));scores[:,cache['protected_ids']]=cache['protected_probability']
                native=attribute_metrics(scores,annotations,compact['native_protocol'])
                for key in ['macro_balanced_accuracy','macro_auroc']:
                    if native[key]!=row['native_attributes'][key]:raise RuntimeError('Final native metric differs')
                mean=metrics(cache['mean_probabilities'],y)
                if mean['correct']!=row['same_model_mean']['correct'] or any(abs(mean[k]-row['same_model_mean'][k])>1e-7 for k in ['nll','ece15']):raise RuntimeError('Same-model mean inference differs')
                if not torch.equal(p,torch.stack(cache['mc_probabilities']).mean(0)):raise RuntimeError('Final MC ensemble differs')
                native_count+=1
            verified.append({'dataset':dataset,'seed':seed,'method':method,'split':split,'correct':computed['correct'],
                             'prediction_sha256':digest(destination/'predictions.pt'),'checkpoint_sha256':row['checkpoint_sha256']})
    atomic_json({'verified_checkpoints':65,'verified_prediction_arrays':len(verified),'verified_native_arrays':native_count,
                 'test_used_for_selection':False,'native_protection_scope':'68 mapped attributes,161 channels; no localization claim',
                 'source_sha256':digest(__file__),'final_lock_sha256':digest(PATH),'records':verified},ROOT/'reports/final_v1/VERIFICATION.json')
    print('Final CPU verification passed:',len(verified),'prediction arrays;',native_count,'native arrays;65 frozen checkpoints',flush=True)


if __name__=='__main__':main()
