"""Run gradient-calibrated controls after the unchanged native V15 cohort completes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .common import ROOT, atomic_json, digest
from .joint_noise import MODES


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--wait-v15-pid',type=int,required=True)
    args=parser.parse_args()
    logs=ROOT/'artifacts/logs/native_calibrated_workflow_v16';logs.mkdir(exist_ok=True)
    calibration=ROOT/'artifacts/cub/native_gradient_v16/calibration.json'
    sources={name:digest(ROOT/'ps_pi'/name) for name in ['native_calibrated_workflow.py',
             'joint_native_calibrated_train.py','joint_native_calibrated_screen.py','joint_native_calibrated_audit.py',
             'native_training.py','native_gradient_diagnostic.py','training_views.py','joint_noise.py']}
    plan={'iteration':16,'setting':'same attribute-supervised extension as V15',
          'change':'only native loss coefficient from fixed training-only gradient norm rule',
          'calibration_sha256':digest(calibration),'native_weight':json.loads(calibration.read_text())['candidate_native_weight'],
          'cases':[{'mode':mode,'concept_weight':cw,'cbl_lr':lr} for cw,lr in [(1.,1e-5),(5.,3e-6)] for mode in MODES],
          'baseline_seed':42,'adaptation_seed':0,'epochs':40,'source_sha256':sources,
          'preceding_cohort':'outputs/cub/validation_joint_native_training_v15',
          'selection_gates':'all unchanged V15 gates, including pseudo BCE, native BA/AUROC and probability gates',
          'test_access':False,'locked_final_candidate':False,'final_five_seed_evidence':False}
    path=logs/'plan.json'
    if path.exists() and json.loads(path.read_text())!=plan:
        raise RuntimeError('Registered prospective V16 plan changed')
    atomic_json(plan,path)
    handle=Path(f'/proc/{args.wait_v15_pid}/cmdline')
    while handle.exists():
        try:cmd=handle.read_bytes().replace(b'\0',b' ').decode()
        except FileNotFoundError:break
        if not cmd:break
        if 'ps_pi.native_training_workflow' not in cmd:
            raise RuntimeError('Preceding workflow PID changed identity')
        atomic_json({'phase':'waiting_for_v15','pid':os.getpid(),'verified_v15_pid':args.wait_v15_pid,
                     'planned_cases':16,'time':time.time()},logs/'status.json')
        time.sleep(5)
    previous=json.loads((ROOT/'artifacts/logs/native_training_workflow_v15/status.json').read_text())
    if previous['phase']!='complete' or not previous['native_cohort_audited']:
        raise RuntimeError('Preceding registered V15 cohort did not complete its audit')
    if digest(calibration)!=plan['calibration_sha256'] or any(digest(ROOT/'ps_pi'/n)!=sha for n,sha in sources.items()):
        raise RuntimeError('Prospective training-gradient plan source changed')
    for module in ['joint_native_calibrated_screen','baseline_replication_evidence','prepare_baseline_contexts','export_major']:
        with (logs/f'{module}.log').open('a') as stream:
            child=subprocess.Popen([sys.executable,'-u','-m',f'ps_pi.{module}'],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            atomic_json({'phase':'running','module':module,'pid':os.getpid(),'child_pid':child.pid,'time':time.time()},logs/'status.json')
            code=child.wait()
        if code:
            atomic_json({'phase':'failed','module':module,'returncode':code,'time':time.time()},logs/'status.json')
            raise SystemExit(code)
    atomic_json({'phase':'complete','native_calibrated_cohort_audited':True,'test_used':False,
                 'final_five_seed_evidence':False,'time':time.time()},logs/'status.json')


if __name__=='__main__':main()
