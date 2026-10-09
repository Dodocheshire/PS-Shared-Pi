"""Conditional adaptation-seed replication; original seed-42 backbone stays fixed."""
import json
import os
import subprocess
import sys
import time
from .common import ROOT, atomic_json, digest
from .calibrated_scale import scale_from_training_diagnostic


def main():
    output = 'outputs/cub/validation_feature_mixup_replication_v11'
    logs = ROOT / 'artifacts/logs/cub_feature_mixup_replication_v11'
    logs.mkdir(parents=True, exist_ok=True)
    diagnostic = ROOT / 'artifacts/cub/joint_feature_residual_v6/train_sensitivity.json'
    calibration = scale_from_training_diagnostic(json.loads(diagnostic.read_text()))
    modes = ['adapter_matched', 'diag_noise', 'shared_no_sampling', 'shared_noise']
    cases = [{'seed': seed, 'mode': mode} for seed in [1, 2, 3, 4] for mode in modes]
    plan = {
        'iteration': 11, 'dataset': 'cub', 'baseline_seed': 42,
        'adaptation_seeds': [1, 2, 3, 4], 'reference_adaptation_seed': 0,
        'reference_output': 'outputs/cub/validation_joint_feature_mixup_v10',
        'output': output, 'cases': cases, 'epochs': 40, 'mixup_alpha': 0.2,
        'concept_weight': 1., 'cbl_lr': 1e-5,
        'calibration': calibration, 'calibration_sha256': digest(diagnostic),
        'selection': 'same original per-run validation gates; independent MC does not reselect',
        'scope': 'conditional adaptation-seed replication with fixed seed-42 CBL/head/bank',
        'limits': 'not independent complete-baseline five-seed evidence or a final test result',
        'reason': 'preset A seed-0 shared noise exceeds same-preset deterministic controls; check reproducibility before any final selection',
        'test_access': False, 'final_evidence': False,
        'source_sha256': {name: digest(ROOT / 'ps_pi' / name) for name in [
            'feature_mixup_replication.py', 'joint_feature_mixup_train.py',
            'joint_feature_mixup_audit.py', 'expected_concept_audit.py',
            'mixup_likelihood.py', 'joint_noise.py', 'calibrated_scale.py']},
    }
    plan_path = logs / 'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise RuntimeError('Replication protocol changed')
    atomic_json(plan, plan_path)
    for index, case in enumerate(cases):
        command = [sys.executable, '-u', '-m', 'ps_pi.joint_feature_mixup_train',
                   '--dataset', 'cub', '--seed', str(case['seed']), '--mode', case['mode'],
                   '--epochs', '40', '--concept-weight', '1', '--cbl-lr', '1e-5',
                   '--output', output, '--mixup-alpha', '.2',
                   '--initial-scale', str(calibration['initial_scale']),
                   '--min-scale', str(calibration['min_scale']),
                   '--max-scale', str(calibration['max_scale'])]
        with (logs / f"{case['mode']}_seed{case['seed']}.log").open('a') as stream:
            child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            atomic_json({'phase': 'running', 'pid': os.getpid(), 'child_pid': child.pid,
                         'index': index, 'total': len(cases), 'case': case, 'time': time.time()}, logs / 'status.json')
            result = child.wait()
        if result:
            atomic_json({'phase': 'failed', 'case': case, 'returncode': result}, logs / 'status.json')
            raise SystemExit(result)
    for module in ['joint_feature_mixup_audit', 'expected_concept_audit']:
        atomic_json({'phase': 'audit', 'module': module, 'completed_training': 16,
                     'pid': os.getpid(), 'time': time.time()}, logs / 'status.json')
        with (logs / f'{module}.log').open('a') as stream:
            subprocess.run([sys.executable, '-u', '-m', f'ps_pi.{module}', '--input', output],
                           cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    atomic_json({'phase': 'complete', 'completed_training': 16, 'baseline_seed': 42,
                 'final_evidence': False, 'test_used': False, 'time': time.time()}, logs / 'status.json')


if __name__ == '__main__':
    main()
