"""Independent CPU validation cohort; four isolated jobs use four CPU threads each."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
import subprocess
import sys
import time
from .common import ROOT,atomic_json,digest

def main():
    logs=ROOT/'artifacts/logs/cifar10_predictive_cpu';logs.mkdir(parents=True,exist_ok=True)
    methods=['mean_only','adapter_matched','diag_noise','shared_noise']
    cases=[{'mode':mode,'mixup_alpha':alpha} for alpha in [0.,.2] for mode in methods]
    plan={'iteration':3,'cohort':'cpu','device':'cpu','threads_per_job':4,'workers':4,'dataset':'cifar10','epochs':60,
          'seed':0,'baseline_seed':42,'cases':cases,'initial_scale':.1,'max_scale':1.,'train_mc':8,'eval_mc':64,
          'concept_weight':.1,'energy_weight':.01,'test_access':False,
          'gates':{'NLL_vs_clean':0.,'ECE_vs_clean':.002,'predictive_concept_BCE_vs_clean':0.,'drift_limit':.3},
          'source_sha256':{name:digest(ROOT/'ps_pi'/name) for name in ['noise.py','noise_v2.py','adapt_predictive.py','adapt_device.py','screen_device.py']}}
    path=logs/'plan.json'
    if path.exists() and json.loads(path.read_text())!=plan:raise RuntimeError('CPU cohort plan changed')
    atomic_json(plan,path)
    def execute(case):
        name=f'{case["mode"]}_alpha{case["mixup_alpha"]:g}'
        output=f'outputs/cifar10/validation_predictive_v3_cpu/alpha{case["mixup_alpha"]:g}'
        command=[sys.executable,'-u','-m','ps_pi.adapt_device','--mode',case['mode'],'--mixup-alpha',str(case['mixup_alpha']),
                 '--device','cpu','--threads','4','--output',output]
        with (logs/f'{name}.log').open('a') as stream:
            process=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            atomic_json({'phase':'running','case':case,'pid':process.pid,'manager_pid':os.getpid(),'time':time.time()},logs/f'{name}.status.json')
            code=process.wait()
        atomic_json({'phase':'complete' if code==0 else 'failed','returncode':code,'case':case,'time':time.time()},logs/f'{name}.status.json')
        return name,code
    atomic_json({'phase':'running','pid':os.getpid(),'total':len(cases),'completed':0,'time':time.time()},logs/'status.json')
    completed=[];failed=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(execute,case) for case in cases]
        for future in as_completed(futures):
            name,code=future.result();completed.append(name)
            if code:failed.append(name)
            atomic_json({'phase':'running','pid':os.getpid(),'total':len(cases),'completed':len(completed),'completed_cases':completed,'failed':failed,'time':time.time()},logs/'status.json')
    atomic_json({'phase':'failed' if failed else 'complete','completed':len(completed),'failed':failed,'test_used':False,'time':time.time()},logs/'status.json')
    if failed:raise SystemExit(1)
if __name__=='__main__':main()
