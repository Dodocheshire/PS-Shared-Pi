"""Recompute validation metrics for complete independently trained full baselines."""
import csv
import json
import time
import torch
from torch.nn import functional as F
from .baseline import validation_metrics
from .common import ROOT, atomic_json, digest, official_module, verify_official


def main():
    torch.set_num_threads(8)
    report = ROOT.parent / 'Essay/PS-Shared/report'
    records = []
    pending = []
    sources = []
    CBL = official_module('model.cbl').ConceptBottleneckLayer
    for dataset in ['cifar10', 'cifar100', 'cub']:
        directory = ROOT / 'artifacts' / dataset
        val = torch.load(directory / 'val_target.pt', weights_only=True)
        manifest = json.loads((directory / 'manifest.json').read_text())
        if not torch.equal(val['indices'], torch.tensor(manifest['indices']['val'])):
            raise RuntimeError('Validation feature order differs from fixed split')
        for seed in range(42, 47):
            folder = ROOT / 'outputs' / dataset / 'baseline' / f'seed{seed}'
            status = folder / 'status.json'
            if not status.exists() or json.loads(status.read_text())['phase'] != 'complete':
                pending.append({'dataset': dataset, 'seed': seed})
                continue
            protocol = json.loads((folder / 'protocol.json').read_text())
            if protocol['test_used'] or protocol['config']['seed'] != seed:
                raise RuntimeError('Baseline seed or data-access protocol differs')
            if protocol['manifest_sha256'] != digest(directory / 'manifest.json'):
                raise RuntimeError('Baseline split provenance differs')
            if protocol['bank_sha256'] != digest(directory / 'concept_bank.pt'):
                raise RuntimeError('Baseline concept bank provenance differs')
            for variant in ['official_final_state', 'corrected_best_validation']:
                path = folder / f'{variant}.pt'
                pack = torch.load(path, weights_only=False)
                recorded = json.loads((folder / f'{variant}.json').read_text())
                if pack['protocol'] != protocol or pack['validation_metrics'] != recorded or recorded['smoke']:
                    raise RuntimeError('Full baseline checkpoint does not match registered record')
                if not torch.isfinite(pack['train_std']).all() or (pack['train_std'] <= 0).any():
                    raise RuntimeError('Invalid train-only concept scaler')
                if not torch.isfinite(pack['weight']).all() or not torch.isfinite(pack['bias']).all():
                    raise RuntimeError('Nonfinite original SAGA head')
                model = CBL(2048, len(pack['concepts']), pack['config']['cbl_layer_num'], pack['config']['cbl_bias'])
                model.load_state_dict(pack['cbl_state'])
                model.eval()
                with torch.no_grad():
                    raw = torch.cat([model(x) for x in val['features'].split(2048)])
                    concepts = (raw - pack['train_mean']) / pack['train_std']
                    recomputed = validation_metrics(F.linear(concepts, pack['weight'], pack['bias']), val['labels'])
                n = len(val['labels'])
                if round(recorded['accuracy'] * n) != round(recomputed['accuracy'] * n):
                    raise RuntimeError(f'Validation correct-count mismatch: {dataset} {seed} {variant}')
                if any(abs(recomputed[k] - recorded[k]) > 1e-5 for k in ['nll', 'ece15']):
                    raise RuntimeError('Recomputed probability metrics differ')
                # State hashes prove distinct full-CBL fits, rather than adaptation aliases.
                import hashlib
                state_hash = hashlib.sha256()
                for key, value in sorted(pack['cbl_state'].items()):
                    state_hash.update(key.encode())
                    state_hash.update(value.contiguous().numpy().tobytes())
                records.append({'dataset': dataset, 'seed': seed, 'variant': variant,
                                'accuracy_percent': 100 * recomputed['accuracy'], 'nll': recomputed['nll'],
                                'ece15': recomputed['ece15'], 'validation_images': n,
                                'concept_count': len(pack['concepts']), 'head_nonzero': int(pack['weight'].count_nonzero()),
                                'checkpoint_sha256': digest(path), 'cbl_state_sha256': state_hash.hexdigest(),
                                'checkpoint_step': recorded['checkpoint_step'], 'full_cbl_steps': pack['config']['cbl_steps'],
                                'original_saga_max_iterations': pack['config']['n_iters'], 'test_used': False})
                sources.append({'path': str(path.relative_to(ROOT.parent)), 'sha256': digest(path)})
    for dataset in ['cifar10', 'cifar100', 'cub']:
        rows = [r for r in records if r['dataset'] == dataset and r['variant'] == 'corrected_best_validation']
        if len({r['cbl_state_sha256'] for r in rows}) != len(rows):
            raise RuntimeError('Independent baseline seeds reuse a CBL state')
    counts = {d: len([r for r in records if r['dataset'] == d and r['variant'] == 'corrected_best_validation'])
              for d in ['cifar10', 'cifar100', 'cub']}
    evidence = {'phase': 'complete' if not pending else 'partial', 'checked_at': time.time(),
                'official_commit': verify_official(), 'auditor_sha256': digest(__file__),
                'completed_full_baselines': sum(counts.values()), 'completed_models': len(records),
                'completed_seeds_per_dataset': counts, 'records': records, 'pending': pending, 'sources': sources,
                'test_access': False, 'locked_noise_candidate': False, 'final_five_seed_evidence': False,
                'scope': 'Independently trained full CBL and original SAGA; shared training-only concept bank and fixed pretrained features. Validation metrics recomputed from each complete checkpoint.',
                'limits': 'Preparation and baseline uncertainty evidence only. No independent five-seed Pi/control comparison, no test results, no extra stochastic gain established.'}
    atomic_json(evidence, report / 'independent_baselines.json')
    with (report / 'independent_baselines.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(records)
    print('Reproduced complete validation baselines:', counts, 'models:', len(records), 'pending:', len(pending), flush=True)


if __name__ == '__main__':
    main()
