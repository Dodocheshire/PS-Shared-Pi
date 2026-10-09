"""Compact baseline figure and an audited summary of all completed validation cohorts."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from .common import ROOT,atomic_json,digest

def main():
    report=ROOT.parent/'Essay/PS-Shared/report';report.mkdir(parents=True,exist_ok=True)
    datasets=['cifar10','cifar100','cub'];variants=['official_final_state','corrected_best_validation']
    records={};sources=[]
    for dataset in datasets:
        folder=ROOT/'outputs'/dataset/'baseline/seed42'
        if json.loads((folder/'status.json').read_text())['phase']!='complete':raise RuntimeError('Baseline not complete')
        records[dataset]={}
        for variant in variants:
            path=folder/f'{variant}.json';records[dataset][variant]=json.loads(path.read_text())
            sources.append({'path':str(path.relative_to(ROOT.parent)),'sha256':digest(path)})
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    figure,axis=plt.subplots(figsize=(7.4,3.7),layout='constrained');x=np.arange(3);width=.34
    for offset,variant,color,label in [(-width/2,variants[0],'#0072B2','Official final checkpoint'),(width/2,variants[1],'#E69F00','Saved best validation checkpoint')]:
        values=[records[d][variant]['accuracy']*100 for d in datasets]
        axis.bar(x+offset,values,width,color=color,label=label)
        for xx,value in zip(x+offset,values):axis.text(xx,value+.9,f'{value:.2f}',ha='center',fontsize=9)
    axis.set_xticks(x,['CIFAR-10','CIFAR-100','CUB']);axis.set_ylim(0,102);axis.set_ylabel('Validation accuracy (%)')
    axis.set_title('PS-CBM complete RN50 baselines (seed 42)');axis.legend(loc='lower left',frameon=False)
    axis.grid(axis='y',alpha=.18);axis.set_axisbelow(True)
    for ext in ['png','svg','pdf']:figure.savefig(report/f'02_complete_baselines.{ext}',dpi=220)
    plt.close(figure)
    cohorts=[]
    for dataset,version,expected in [('cifar10','validation_v1',9),('cifar10','validation_mixup_v2',8),('cifar10','validation_predictive_v3',8),('cifar10','validation_predictive_v3_cpu',8),('cub','validation_crossfit_v4',9)]:
        rows=[]
        for path in sorted((ROOT/'outputs'/dataset/version).glob('**/metrics.json')):
            if json.loads((path.parent/'status.json').read_text())['phase']!='complete':continue
            item=json.loads(path.read_text());m=item['metrics']
            if m['split']!='val' or m['test_used']:raise RuntimeError('Non-validation result in cohort')
            rows.append({'path':str(path.relative_to(ROOT.parent)),'sha256':digest(path),'metrics':m})
        if len(rows)!=expected:raise RuntimeError(f'Incomplete cohort {version}: {len(rows)}/{expected}')
        positive_and_feasible=[r for r in rows if r['metrics'].get('selection_eligible',False) and r['metrics']['delta_clean_pp']>0]
        cohorts.append({'dataset':dataset,'version':version,'completed':len(rows),'rows':rows,'positive_and_feasible':positive_and_feasible})
    atomic_json({'baseline_sources':sources,'baselines':records,'cohorts':cohorts,'completed_validation_controls':sum(c['completed'] for c in cohorts),
                 'test_used':False,'five_seed_verified':False,'figures':{ext:digest(report/f'02_complete_baselines.{ext}') for ext in ['png','svg','pdf']}},report/'iteration_summary.json')
    primary=report/'主要实验进展.md';text=primary.read_text()
    text+='''\n## 完整基线与后续对照

![完整单种子基线](02_complete_baselines.png)

**目的：** 确认三个优先数据集都已建立官方配置的完整可比基线，避免将权重保存修正或不同划分的差异计为 Pi 收益。

**结论：** 最佳验证权重对应 CIFAR-10 90.72%、CIFAR-100 72.36%、CUB 68.70%；详情见 [基线表](首轮基线.md)。这些是各自完整验证集的单种子成绩。

首轮九组、第二轮八组、第三轮 GPU 八组与独立 CPU 八组、CUB 三折 CBL 交叉拟合后的九组，共完成 **42 组验证对照**。第二/第三轮没有获得同时优于干净基线并满足预先记录的校准与概念门槛的学习方案。第三轮保留初始均值为零的随机模型，NLL 略低，但准确率没有提升。CUB 交叉拟合后，对角噪声的一组模型虽降低 ECE，却使准确率降到 66.95%，因此不符合性能目标；共享噪声也未获可靠提升。

**下一步：** 冻结 CBL 后再拟合扰动的路线目前没有足够证据。将检验训练期间的结构化特征噪声，并保留相同训练预算的干净、均值、确定性适配、固定随机对照，继续隔离测试集。五种子与人工属性验证仍未完成，不能将当前探索结果视为最终结论。完整状态与逐组来源哈希见 [迭代汇总](iteration_summary.json)。
'''
    # Export is idempotent: keep the original first-screen portion and replace this section.
    head=text.split('\n## 完整基线与后续对照',1)[0]
    suffix=text.rsplit('\n## 完整基线与后续对照',1)[-1]
    primary.write_text(head+'\n## 完整基线与后续对照'+suffix)
    print('Verified 6 full baseline models and 42 validation controls; no feasible classification gain beyond baseline',flush=True)
if __name__=='__main__':main()
