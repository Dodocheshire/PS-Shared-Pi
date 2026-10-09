"""Run the registered native-attribute cohort only after a verified one-epoch smoke."""
import json
import argparse
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from .common import ROOT, atomic_json, digest


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--wait-smoke-pid',type=int)
    args=parser.parse_args()
    logs=ROOT/'artifacts/logs/native_training_workflow_v15';logs.mkdir(exist_ok=True)
    if args.wait_smoke_pid:
        handle=Path(f'/proc/{args.wait_smoke_pid}/cmdline')
        while handle.exists():
            try: command=handle.read_bytes().replace(b'\0',b' ').decode()
            except FileNotFoundError: break
            if not command: break
            if 'ps_pi.joint_native_training_train' not in command or 'smoke_joint_native_training_v15' not in command:
                raise RuntimeError('Smoke PID changed identity')
            atomic_json({'phase':'waiting_for_verified_smoke','smoke_pid':args.wait_smoke_pid,
                         'pid':os.getpid(),'time':time.time()},logs/'status.json')
            time.sleep(5)
    smoke = ROOT/'outputs/cub/smoke_joint_native_training_v15/shared_noise_seed0_cw1_lr1e-05'
    status = json.loads((smoke/'status.json').read_text())
    protocol = json.loads((smoke/'protocol.json').read_text())
    results = json.loads((smoke/'metrics.json').read_text())
    if status['phase']!='complete' or protocol['test_access'] or protocol['config']['epochs']!=1:
        raise RuntimeError('Native-training smoke has not completed under the expected protocol')
    if len(results['history'])!=2:
        raise RuntimeError('Expected initialization and a completed training epoch')
    last = results['history'][-1]
    if not math.isfinite(last['training_objective']) or not all(math.isfinite(last['validation'][k]) for k in ['nll','native_balanced_accuracy','native_auroc']):
        raise RuntimeError('Native supervision smoke did not produce finite metrics')
    for name,sha in protocol['source_sha256'].items():
        if digest(ROOT/'ps_pi'/name)!=sha:
            raise RuntimeError('Smoke-verified training source changed')
    atomic_json({'smoke_protocol_sha256':digest(smoke/'protocol.json'),
                 'smoke_metrics_sha256':digest(smoke/'metrics.json'),
                 'screen_sha256':digest(ROOT/'ps_pi/joint_native_training_screen.py'),
                 'setting':'additional native train-attribute supervision, shared by all controls',
                 'test_access':False,'final_evidence':False},logs/'plan.json')
    for module in ['joint_native_training_screen','baseline_replication_evidence','export_major']:
        with (logs/f'{module}.log').open('a') as stream:
            child=subprocess.Popen([sys.executable,'-u','-m',f'ps_pi.{module}'],cwd=ROOT,
                                   stdout=stream,stderr=subprocess.STDOUT)
            atomic_json({'phase':'running','module':module,'pid':os.getpid(),'child_pid':child.pid,
                         'time':time.time()},logs/'status.json')
            result=child.wait()
        if result:
            atomic_json({'phase':'failed','module':module,'returncode':result,'time':time.time()},logs/'status.json')
            raise SystemExit(result)
    atomic_json({'phase':'complete','native_cohort_audited':True,'test_used':False,'final_evidence':False,
                 'major_report_note':'Native cohort remains separate pending explicit evidence review; prior 234 rows stay unchanged',
                 'time':time.time()},logs/'status.json')


if __name__=='__main__':main()
