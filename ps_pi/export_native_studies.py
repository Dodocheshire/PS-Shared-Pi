"""Keep additional attribute-supervised cohorts separate from label-free controls."""
import csv
import json
from .common import ROOT, atomic_json, digest

COHORTS=[('validation_joint_native_training_v15','原生属性监督'),
         ('validation_joint_native_calibrated_v16','训练梯度校准属性权重')]


def main():
    report=ROOT.parent/'Essay/PS-Shared/report';cohorts=[];records=[];pending=[]
    for version,label in COHORTS:
        folder=ROOT/'outputs/cub'/version;audit=folder.parent/(version+'_audit')
        status=audit/'status.json'
        if not status.exists() or json.loads(status.read_text())['phase']!='complete':
            pending.append(version);continue
        summary=json.loads((audit/'summary.json').read_text())
        if len(summary['rows'])!=16 or summary['protocol']['test_used']:
            raise RuntimeError('Expected complete validation-only native cohort')
        eligible=0
        for row in summary['rows']:
            source=folder/row['case']/'metrics.json';d=json.loads(source.read_text());m=d['metrics'];b=d['original_baseline']
            gates={'accuracy':m['correct']>=b['correct'],'nll':m['nll']<=b['nll'],
                   'ece':m['ece15']<=b['ece15']+.002,'pseudo_concept_bce':m['concept_bce']<=b['concept_bce'],
                   'drift':m['concept_drift_rms']<=.3,'native_ba':m['native_balanced_accuracy']+1e-12>=b['native_balanced_accuracy'],
                   'native_auroc':m['native_auroc']+1e-12>=b['native_auroc']}
            if all(gates.values())!=m['selection_eligible']:
                raise RuntimeError('Stored gate result differs from registered selection rule')
            eligible+=int(m['selection_eligible'])
            acc=[x['accuracy']*100 for x in row['mc_runs']]
            records.append({'version':version,'label':label,'case':row['case'],'mode':row['mode'],
                            'selection_eligible':m['selection_eligible'],'selection_epoch':m['best_epoch'],
                            'selection_correct':m['correct'],'selection_nll':m['nll'],
                            'accuracy_min_percent':min(acc),'accuracy_max_percent':max(acc),
                            'native_ba_delta_pp':row['native_ba_delta_pp'],'native_auroc_delta_pp':row['native_auc_delta_pp'],
                            'pseudo_bce_delta':m['concept_bce']-b['concept_bce'],
                            'failed_gates':','.join(k for k,v in gates.items() if not v),
                            'selection_metrics_sha256':digest(source),'checkpoint_sha256':row['checkpoint_sha256'],
                            'audit_summary_sha256':digest(audit/'summary.json'),'test_used':False})
        cohorts.append({'version':version,'label':label,'completed_training_runs':16,
                        'selection_eligible_runs':eligible,'rejected_runs':16-eligible})
    atomic_json({'cohorts':cohorts,'records':records,'pending':pending,'completed_training_runs':16*len(cohorts),
                 'setting':'Additional native CUB training-label supervision, identical for all controls',
                 'source_sha256':digest(__file__),'test_used':False,'final_five_seed_evidence':False,
                 'limits':'Registered selection remains unchanged. Ineligible checkpoints are explicitly retained as failures and never promoted. MC ranges are resampling variation, not confidence intervals. Cannot attribute additional human labels or deterministic adaptation to noise.'},report/'native_studies.json')
    if records:
        with (report/'native_studies.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(records[0]),lineterminator='\n');writer.writeheader();writer.writerows(records)
    lines=['# 属性监督扩展：预登记质量门槛与结果','',
           '**目的：** 在所有控制中加入相同训练原生属性目标，检查分类适配能否同时改善原生属性；与原无属性监督队列分开报告。','',
           '**选模：** 原准确率/NLL/ECE/漂移/伪概念BCE门槛与原生属性BA/AUROC门槛共同生效；没有据独立MC或下面的分析修改原选择。未通过门槛的高分模型仍不晋级。','',
           '| 完整队列 | 完成运行数 | 通过预登记门槛 | 未通过 |','|---|---:|---:|---:|']
    for c in cohorts:lines.append(f"| {c['label']} | 16 | {c['selection_eligible_runs']} | {c['rejected_runs']} |")
    lines+=['','| 队列 / 共享模型配置 | 独立MC验证ACC（%） | 原生BA变化（pp） | 原生AUROC变化（pp） | 未通过门槛 |','|---|---:|---:|---:|---|']
    for row in records:
        if row['mode']=='shared_noise':
            lines.append(f"| {row['label']} / {row['case']} | {row['accuracy_min_percent']:.2f}–{row['accuracy_max_percent']:.2f} | {row['native_ba_delta_pp']:+.3f} | {row['native_auroc_delta_pp']:+.3f} | {row['failed_gates'] or '通过'} |")
    lines+=['','**合格模型的公平比较：** 以下各方法仍按原选模正确数/NLL固定配置，不据独立MC复核改选。','',
            '| 队列 | 同容量适配器ACC（%） | 关闭采样ACC（%） | 共享噪声ACC（%） |',
            '|---|---:|---:|---:|']
    for cohort in cohorts:
        values=[]
        for mode in ['adapter_matched','shared_no_sampling','shared_noise']:
            allowed=[r for r in records if r['version']==cohort['version'] and r['mode']==mode and r['selection_eligible']]
            if not allowed:values.append('无合格模型');continue
            chosen=max(allowed,key=lambda r:(r['selection_correct'],-r['selection_nll']))
            lo=chosen['accuracy_min_percent'];hi=chosen['accuracy_max_percent']
            values.append(f'{lo:.2f}' if abs(lo-hi)<1e-5 else f'{lo:.2f}–{hi:.2f}')
        lines.append(f"| {cohort['label']} | "+' | '.join(values)+' |')
    lines+=['','**结论：** 不能因原生属性改善或单个验证高分而跳过其他预登记门槛。伪标签保真与真实属性是不同指标；本表如显示伪概念BCE失败，表示原代理约束未满足，不等价于真实属性退化。仍需在新协议下保留全部确定性与随机对照，不能从失败模型倒推噪声成功。','',
            '**范围：** 68/312项保守映射的CUB属性，训练确定映射及阈值，验证排除certainty<3；这里仍是基线42、适配0的探索，不是独立五种子或测试结果。','',
            '[全部方法与门槛明细](native_studies.csv)、[来源与待完成队列](native_studies.json)、[训练梯度诊断](native_gradient_diagnostic.json)。']
    (report/'属性监督扩展.md').write_text('\n'.join(lines)+'\n')
    print('Native supervised completed cohorts:',len(cohorts),'runs:',16*len(cohorts),flush=True)


if __name__=='__main__':main()
