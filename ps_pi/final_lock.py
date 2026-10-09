"""Immutable final specification registered before held-out test access."""
import json
from .common import ROOT, atomic_json, digest

PATH=ROOT/'reports/final_v1/LOCK.json'
MODES=['clean','mean_only','adapter_matched','shared_no_sampling','diag_noise','shared_noise','fixed_isotropic']
SEEDS=[42,43,44,45,46]


def register():
    if PATH.exists():return verify_lock()
    baseline={d:{str(s):digest(ROOT/f'outputs/{d}/baseline/seed{s}/corrected_best_validation.pt') for s in SEEDS} for d in ['cifar10','cifar100','cub']}
    sources={str(p.relative_to(ROOT)):digest(p) for p in sorted((ROOT/'ps_pi').glob('*.py')) if p.name not in ['final_report.py','final_evaluate.py','final_test_prepare.py','evidence_snapshot.py','export_major.py']}
    plan={'version':'final_v1','registered_utc':'2026-10-09 11:10:00 UTC',
          'deadline_utc':'2026-10-09 13:00:00 UTC','baseline_seeds':SEEDS,'adaptation_seeds':[0,1,2,3,4],
          'datasets_baseline_test':['cifar10','cifar100','cub'],'datasets_final_adaptation':['cub'],
          'modes':MODES,'config':{'epochs':40,'concept_weight':5.,'cbl_lr':3e-6,'noise_lr':1e-3,
          'batch_size':128,'hidden':128,'train_samples':4,'selection_samples':32,'evaluation_samples':256,
          'max_mean':.3,'initial_scale':.1,'min_scale':.02,'max_scale':.3,'mixup_alpha':.2,
          'preservation_weight':1.,'energy_weight':.1,'prior_weight':.01},
          'mc_seeds':[9071,9072],'paired_interval':'paired t interval over five independently trained baseline/adaptation seeds; test-image paired bootstrap separately',
          'selection':'validation accuracy/NLL/ECE/pseudo BCE/native BA/native AUROC nondegradation and drift <=0.3, then correct count/NLL; never test selection',
          'frozen_choice':'V22 configuration B, chosen using completed validation studies before new independent/test results; identical configuration for all methods',
          'primary_contrast':'shared_noise minus shared_no_sampling; strong matched adapter reported independently',
          'test_usage':'only evaluation after lock; no tuning or post-test changes',
          'scope':'Time-limited finalization requested by user. Independent final noise evaluation focuses CUB; CIFAR tests evaluate reproduced clean baselines only. No additional ablations.',
          'baseline_sha256':baseline,'source_sha256':sources}
    atomic_json(plan,PATH);return plan


def verify_lock():
    plan=json.loads(PATH.read_text())
    for name,sha in plan['source_sha256'].items():
        if digest(ROOT/name)!=sha:raise RuntimeError(f'Frozen final source changed: {name}')
    for dataset,seeds in plan['baseline_sha256'].items():
        for seed,sha in seeds.items():
            if digest(ROOT/f'outputs/{dataset}/baseline/seed{seed}/corrected_best_validation.pt')!=sha:raise RuntimeError('Frozen baseline changed')
    return plan
