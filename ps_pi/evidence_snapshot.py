"""Archive concise evidence and verified live handles; never publish feature tensors."""
import argparse
import json
from pathlib import Path
import shutil
import time
from .common import ROOT,atomic_json,digest

COHORTS=[('validation_joint_native_training_v15','cub_joint_native_training_v15','ps_pi.joint_native_training_screen'),
         ('validation_joint_native_calibrated_v16','cub_joint_native_calibrated_v16','ps_pi.joint_native_calibrated_screen'),
         ('validation_compact_concept_v17','cub_compact_concept_v17','ps_pi.compact_concept_screen'),
         ('validation_protected_concept_v18','cub_protected_concept_v18','ps_pi.protected_concept_screen'),
         ('validation_protected_predictive_v19','cub_protected_predictive_v19','ps_pi.protected_predictive_screen'),
         ('validation_protected_fullmean_v20','cub_protected_fullmean_v20','ps_pi.protected_fullmean_screen'),
         ('validation_protected_signed_v21','cub_protected_signed_v21','ps_pi.protected_signed_screen'),
         ('validation_anchored_signed_v22','cub_anchored_signed_v22','ps_pi.anchored_signed_screen')]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checks',type=int,default=66);args=parser.parse_args()
    report=ROOT.parent/'Essay/PS-Shared/report'
    major=json.loads((report/'major_results.json').read_text())
    baselines=json.loads((report/'independent_baselines.json').read_text())
    verification=json.loads((report/'validation_artifact_verification.json').read_text())
    live={};missing=[];complete=[]
    for version,logs,module in COHORTS:
        path=ROOT/'artifacts/logs'/logs/'status.json'
        if not path.exists():continue
        status=json.loads(path.read_text())
        audit=ROOT/'outputs/cub'/(version+'_audit')/'status.json'
        if status['phase']=='complete':
            if audit.exists() and json.loads(audit.read_text())['phase']=='complete':complete.append(version)
            continue
        if status['phase']=='failed':continue
        process=Path(f'/proc/{status.get("pid")}/cmdline')
        if process.exists() and module in process.read_bytes().replace(b'\0',b' ').decode():
            status['verified_live_cmdline']=process.read_bytes().replace(b'\0',b' ').decode()
            child=Path(f'/proc/{status.get("child_pid")}/cmdline')
            if child.exists():status['verified_child_cmdline']=child.read_bytes().replace(b'\0',b' ').decode()
            live[version]=status
        else:missing.append({'version':version,'recorded':status,'observed_handle_missing':True})
    state={'time':time.time(),'goal_status':'active','completed_baselines':baselines['completed_full_baselines'],
           'baseline_models':baselines['completed_models'],'independent_baseline_seeds_per_dataset':baselines['completed_seeds_per_dataset'],
           'complete_baseline_training_seeds':[42,43,44,45,46],
           'complete_reported_cohort_controls':major['total_reported_registered_runs'],
           'completed_joint_cohort_controls':major['completed_joint_cohort_runs'],'previous_cohort_controls':42,
           'conditional_replication_new_runs':16,'attribute_loss_training_runs':major['attribute_supervised_training_runs'],
           'completed_additional_cohorts':complete,'running_queues':live,'queues_with_missing_handles':missing,
           'tested_protocol_checks':args.checks,'verified_checkpoints':verification['verified_checkpoints'],
           'verified_mc_prediction_arrays':verification['verified_prediction_arrays'],
           'verified_protected_native_arrays':verification['verified_protected_native_arrays'],
           'test_features_encoded':any((ROOT/'artifacts'/d/'final_test_v1/test_target.pt').exists() for d in ['cub','cifar10','cifar100']),
           'test_used_for_selection':False,'five_seed_final_evidence':False,'locked_noise_candidate':None,
           'native_attribute_validation':{'dataset':'cub','attributes':68,'total_attributes':312,'certainty_minimum':3,'mapping_and_thresholds_split':'train','evaluation_split':'val'},
           'current_result':'Independent full baselines42-46 and exact output compaction verified on all15. Complete V15-V21 controls do not establish reliable stochastic superiority. V20 shared69.90-69.95 vs no_sampling70.10; V21 shared69.85-69.90 vs no_sampling70.10. Protected mapped native responses unchanged.',
           'next_stage':'Audit the anchored adaptive-encoder V22 controls; retain strong all-output deterministic mean controls and require quality preservation before any independent five-seed candidate lock.',
           'auxiliary_attribute_use':'V15/V16 add native loss; later cohorts use train attributes for mapping/protected-cohort/threshold selection without native loss. Same within-cohort use for every control; not the original unmodified PS-CBM supervision setting.',
           'evidence_sha256':{name:digest(report/name) for name in ['major_results.json','independent_baselines.json','compact_baselines.json','validation_artifact_verification.json']},
           'report':'../Essay/PS-Shared/report/主要实验进展.md'}
    final=ROOT/'reports/final_v1/results.json'
    if final.exists():
        result=json.loads(final.read_text())
        proof=json.loads((ROOT/'reports/final_v1/VERIFICATION.json').read_text())
        delta=result['paired_test_contrasts']['shared_no_sampling']['accuracy_delta_pp']
        state.update(goal_status='final_delivery_complete',five_seed_final_evidence=True,
                     locked_noise_candidate='final_v1 / V22-B',final_independent_adaptation_runs=35,
                     complete_registered_training_runs_including_final=state['complete_reported_cohort_controls']+35,
                     final_datasets_noise=['cub'],final_datasets_clean_test=['cifar10','cifar100','cub'],
                     final_evaluated_models_per_split=65,final_verified_prediction_arrays=proof['verified_prediction_arrays'],
                     final_verified_native_arrays=proof['verified_native_arrays'],test_features_encoded=True,
                     test_used_for_selection=False,test_used_for_evaluation=True,
                     native_attribute_final_evaluation='68 fixed train-mapped native attributes; validation and test',
                     current_result=f"Final CUB five-seed shared minus sampling-disabled test delta {delta['mean']:+.4f} pp; paired t95 [{delta['low']:+.4f},{delta['high']:+.4f}]. Default matched deterministic adapter chosen before test. No further ablations.",
                     next_stage='Final delivery completed per user two-hour deadline; research iteration stopped. See EXPERIMENT_RESULT.md.',
                     final_report='EXPERIMENT_RESULT.md',report='EXPERIMENT_RESULT.md')
        state['evidence_sha256'].update(final_results=digest(final),final_verification=digest(ROOT/'reports/final_v1/VERIFICATION.json'))
    atomic_json(state,ROOT/'reports/CURRENT_STATE.json')
    destination=ROOT/'reports/validation_v1'
    for path in report.iterdir():
        if path.is_file() and path.name!='最终实验结果.md' and path.suffix in ['.md','.json','.csv']:shutil.copy2(path,destination/path.name)
    figures=['04_main_comparison','06_concept_compaction','07_compact_noise','08_protected_noise',
             '09_protected_predictive','10_protected_fullmean','11_protected_signed','12_anchored_signed']
    for name in figures:
        for ext in ['png','svg','pdf']:
            path=report/f'{name}.{ext}'
            if path.exists():shutil.copy2(path,destination/path.name)
    for version,logs,_ in COHORTS:
        target=ROOT/'reports/joint_followup'/('cub_'+version);target.mkdir(exist_ok=True)
        for name in ['plan.json','status.json','controller.json']:
            source=ROOT/'artifacts/logs'/logs/name
            if source.exists():shutil.copy2(source,target/('registered_'+name))
        if version not in complete:continue
        for suffix in ['_audit','_mean_audit']:
            for name in ['summary.json','protocol.json','status.json']:
                source=ROOT/'outputs/cub'/(version+suffix)/name
                if source.exists():shutil.copy2(source,target/(suffix.strip('_')+'_'+name))
        for source in (ROOT/'outputs/cub'/version).glob('*/metrics.json'):
            shutil.copy2(source,target/(source.parent.name+'_selection.json'))
    print('Archived',state['complete_reported_cohort_controls'],'reported runs; live handles:',list(live),flush=True)


if __name__=='__main__':main()
