"""Two primary figures and concise evidence across all complete joint cohorts."""
import csv
import json
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from .common import ROOT,atomic_json,digest

COHORTS=[('cifar10','validation_joint_v5','CIFAR-10: semantic'),
         ('cifar100','validation_joint_v5','CIFAR-100: semantic'),
         ('cub','validation_joint_v5','CUB: semantic'),
         ('cub','validation_joint_residual_v6','CUB: residual'),
         ('cub','validation_joint_residual_calibrated_v7','CUB: residual, calibrated'),
         ('cub','validation_joint_residual_predictive_v8','CUB: predictive objective'),
         ('cub','validation_joint_sparse_head_v9','CUB: sparse head adapted'),
         ('cub','validation_joint_feature_mixup_v10','CUB: feature mixup'),
         ('cub','validation_joint_robust_mixup_v12','CUB: expected task risk'),
         ('cub','validation_joint_feature_augmented_v13','CUB: observed covariance'),
         ('cub','validation_joint_observed_views_v14','CUB: actual weak views')]
METHODS=[('clean','Clean joint','#555555'),('mean_only','Mean only','#0072B2'),
         ('adapter_matched','Matched adapter','#CC79A7'),('shared_no_sampling','Sampling disabled','#009E73'),
         ('diag_noise','Diagonal noise','#56B4E9'),('shared_noise','Shared noise','#D55E00'),('fixed_structured','Fixed structured','#E69F00')]

def main():
    report=ROOT.parent/'Essay/PS-Shared/report';sources=[];records=[];selected={}
    replication_path=report/'conditional_seed_replication.json'
    replication=json.loads(replication_path.read_text()) if replication_path.exists() else None
    additional=0 if replication is None else replication['new_training_runs']
    replication_text=''
    if replication is not None:
        contrast=next(c for c in replication['contrasts'] if c['control']=='shared_no_sampling')
        lo,hi=contrast['conditional_t95_pp']
        ba=np.mean([r['native_ba_delta_pp'] for r in replication['rows'] if r['mode']=='shared_noise'])
        replication_text=f'''
![五个适配种子的配对复验](05_conditional_seed_replication.png)

**目的：** 固定基线种子 42、原配置 A 和 mixup alpha=0.2，新增适配种子 1–4，与种子 0 合成五个条件种子的同容量、对角、关闭采样和共享对照。全部种子保留，独立 MC 不重选模型。

**结论：** 共享噪声相对关闭采样的平均差值 **{contrast['mean_delta_pp']:+.2f} pp**，条件配对 95% t 区间 **[{lo:+.2f}, {hi:+.2f}] pp**，跨过零；相对同容量适配/对角噪声也没有可靠优势。首个种子的 +0.10 pp 没有稳定重复。共享属性 BA 平均改变 **{ba:+.2f} pp**，仍非解释提升。该区间仅覆盖适配种子变化，不覆盖完整基线、概念库或测试不确定性。
'''
    fig,axes=plt.subplots(1,2,figsize=(11.5,8.6))
    fig.subplots_adjust(left=.25,right=.98,top=.85,bottom=.23,wspace=.30)
    for index,(dataset,version,label) in enumerate(COHORTS):
        folder=ROOT/'outputs'/dataset/version;audit=folder.parent/(version+'_audit')
        if json.loads((audit/'status.json').read_text())['phase']!='complete':raise RuntimeError(f'Audit incomplete: {dataset} {version}')
        summary=json.loads((audit/'summary.json').read_text());baseline=summary['baseline']
        candidates=[]
        for path in sorted(folder.glob('*/metrics.json')):
            m=json.loads(path.read_text())['metrics'];candidates.append((path,m))
        if len(candidates)!=16:raise RuntimeError('Incomplete registered cohort')
        selected[label]={}
        for mi,(mode,name,color) in enumerate(METHODS):
            allowed=[(path,m) for path,m in candidates if m['mode']==mode and m['selection_eligible']]
            if not allowed:continue
            # Lock case by the original selection MC before reading independent resampling.
            path,m=max(allowed,key=lambda pair:(pair[1]['correct'],-pair[1]['nll']))
            audited=next(r for r in summary['rows'] if r['case']==path.parent.name)
            acc=[r['accuracy'] for r in audited['mc_runs']];nll=[r['nll'] for r in audited['mc_runs']]
            ece=[r['ece15'] for r in audited['mc_runs']]
            classes=10 if dataset=='cifar10' else 100 if dataset=='cifar100' else 200
            concepts=m['concept_count'];factor=math.log(concepts,math.ceil(math.log2(classes)))**.25
            row={'dataset':dataset,'version':version,'label':label,'mode':mode,'selected_case':path.parent.name,
                 'selection_epoch':m['best_epoch'],'selection_gate_eligible':True,
                 'selection_accuracy':m['accuracy'],'selection_nll':m['nll'],'baseline_accuracy':baseline['accuracy'],
                 'accuracy_min':min(acc),'accuracy_max':max(acc),'nll_min':min(nll),'nll_max':max(nll),
                 'ece_min':min(ece),'ece_max':max(ece),'concept_count':concepts,
                 'cea_min':100*min(acc)/factor,'cea_max':100*max(acc)/factor,
                 'baseline_cea':100*baseline['accuracy']/factor,
                 'native_ba_delta_pp':audited['native_ba_delta_pp'],'native_auc_delta_pp':audited['native_auc_delta_pp']}
            records.append(row);selected[label][mode]=row
            for source in [path,audit/path.parent.name/'metrics.json']:
                sources.append({'path':str(source.relative_to(ROOT.parent)),'sha256':digest(source)})
            values=100*(np.array(acc)-baseline['accuracy']);mid=values.mean()
            axes[0].errorbar(mid,index+(mi-(len(METHODS)-1)/2)*.075,xerr=np.array([[mid-values.min()],[values.max()-mid]]),
                             marker='o',color=color,linestyle='none',markersize=4.5,capsize=2)
        best_det=max((row for mode,row in selected[label].items() if mode in ['clean','mean_only','adapter_matched','shared_no_sampling']),
                     key=lambda row:(row['selection_accuracy'],-row['selection_nll']))
        shared=selected[label]['shared_noise']
        delta=[100*(shared['accuracy_min']-best_det['accuracy_max']),100*(shared['accuracy_max']-best_det['accuracy_min'])]
        axes[1].plot([delta[0],delta[1]],[index,index],color='#D55E00',lw=2)
        axes[1].plot(np.mean(delta),index,'o',color='#D55E00',ms=5)
        axes[1].text(max(delta)+.025,index,f'{min(delta):+.2f} to {max(delta):+.2f}',va='center',fontsize=8)
        selected[label]['strongest_deterministic_case']=best_det['selected_case']
        selected[label]['shared_minus_strongest_deterministic_pp_range']=delta
    for ax in axes:
        ax.axvline(0,color='#777777',ls='--',lw=1);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
        ax.set_ylim(len(COHORTS)-.5,-.5);ax.spines[['top','right']].set_visible(False)
    axes[0].set_yticks(range(len(COHORTS)),[c[2] for c in COHORTS]);axes[1].set_yticks(range(len(COHORTS)),[])
    axes[0].set_xlabel('Accuracy delta vs original baseline (pp)');axes[0].set_xlim(-.035,.80)
    axes[1].set_xlabel('Shared noise minus strongest deterministic (pp)');axes[1].set_xlim(-.40,.45)
    fig.suptitle('Complete joint-training screens: independent 256-sample validation checks',fontsize=11,y=.955)
    fig.legend(handles=[Line2D([0],[0],marker='o',ls='none',color=c,label=n) for _,n,c in METHODS],
               loc='lower center',bbox_to_anchor=(.5,.065),ncol=4,frameon=False,fontsize=8)
    fig.text(.5,.02,'Cases selected by original validation MC; independent resampling never reselects epochs. Ranges are MC variation, not confidence intervals.',ha='center',fontsize=7.5)
    for ext in ['png','svg','pdf']:
        path=report/f'04_main_comparison.{ext}';fig.savefig(path,dpi=220)
        if ext=='svg':path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    plt.close(fig)
    with (report/'major_results.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]),lineterminator='\n');writer.writeheader();writer.writerows(records)
    atomic_json({'completed_joint_cohort_runs':16*len(COHORTS),'previous_cohort_runs':42,'total_reported_registered_runs':16*len(COHORTS)+42+additional,'screen_runs':16*len(COHORTS)+42,'conditional_replication_new_runs':additional,'conditional_replication':replication,
                 'training_seeds':1,'baseline_seed':42,'adaptation_seed':0,'test_used':False,'five_seed_verified':False,
                 'selection':'original MC selection; no case/epoch re-selection from high-sample audit',
                 'limits':'Repeated deterministic controls across objective/scale versions are not independent training replicates. CEA is the paper compactness proxy, not human semantic quality.',
                 'records':records,'selected':selected,'sources':sources,
                 'figures':{ext:digest(report/f'04_main_comparison.{ext}') for ext in ['png','svg','pdf']}},report/'major_results.json')
    table=[]
    for dataset,version,label in COHORTS:
        chosen=selected[label]
        shared=chosen['shared_noise']
        deterministic=next(r for r in chosen.values() if isinstance(r,dict) and r.get('selected_case')==chosen['strongest_deterministic_case'])
        def percent(row):
            lo,hi=100*row['accuracy_min'],100*row['accuracy_max']
            return f'{lo:.2f}%' if abs(lo-hi)<1e-5 else f'{lo:.2f}–{hi:.2f}%'
        table.append(f"| {label} | {percent(deterministic)} | {percent(shared)} |")
    baseline_text=''
    baseline_path=report/'independent_baselines.json'
    if baseline_path.exists():
        evidence=json.loads(baseline_path.read_text());lines=[]
        for dataset in ['cifar10','cifar100','cub']:
            rows=[r for r in evidence['records'] if r['dataset']==dataset and r['variant']=='corrected_best_validation']
            values=[r['accuracy_percent'] for r in rows]
            sd=f'{np.std(values,ddof=1):.3f}' if len(values)>1 else '—'
            seed_text=', '.join(f"{r['seed']}: {r['accuracy_percent']:.2f}" for r in rows)
            lines.append(f"| {dataset} | {len(rows)}/5 | {np.mean(values):.3f} | {sd} | {seed_text} |")
        baseline_text='''
**独立完整基线准备：** 按原配置独立训练 CBL 与原 SAGA 头，逐检查点重新计算验证指标。共享固定训练概念库与预训练图像特征；下表仅使用已完成的最佳验证权重，未完成种子不填值。这里没有最终 Pi 配对复验或测试成绩。

| 数据集 | 已核验完整种子数 | 验证ACC均值（%） | 种子SD（pp） | 各种子验证ACC（%） |
|---|---:|---:|---:|---|
'''+ '\n'.join(lines)+'\n'
    primary=f"""# PS-CBM 主要实验进展

已完成原始三个 RN50 基线、{16*len(COHORTS)+42+additional} 条注册验证运行（不含调试运行）。**最强 CUB 确定性适配达到 69.45%，尚未证明随机分量的额外收益。** 主队列为基线种子 42、适配种子 0；另有固定基线下五个适配种子的条件复验，与独立完整基线准备分开报告。重复控制不算独立种子。测试集保持隔离，最终五种子验证尚未完成。

[完整基线图](02_complete_baselines.png)

**目的：** 按官方文件名划分和配置建立基线，区分官方末轮权重与正确保存的最佳验证权重。

**结论：** 最佳验证权重的 CIFAR-10 / CIFAR-100 / CUB 准确率为 **90.72% / 72.36% / 68.70%**，概念数为 **61 / 583 / 509**。不能与论文测试成绩直接比较。详见 [基线表](首轮基线.md)。

{baseline_text}

![主要噪声对照](04_main_comparison.png)

**共同目的：** 保留干净联合微调、仅均值、同容量非线性适配、相同生成器关闭采样、对角噪声和固定随机噪声，以相同预算区分适配与随机性作用。每组两套预登记配置、八类方法；先按原验证 MC 选权重，再以两个独立种子各 256 次采样复核。高采样复核不重新选权重或配置。

| 完整队列 | 最强确定性对照 | 共享噪声（原选择固定后复核） |
|---|---:|---:|
"""+'\n'.join(table)+f"""

| 实验 | 目的 | 结论 |
|---|---|---|
| 前四轮（42 条） | 检验冻结模型的噪声适配、概念 mixup、边际似然和交叉拟合 | 没有满足性能与质量门槛的独立随机收益。 |
| 语义方向联合训练（48 条） | 在 CBL 学习期间注入训练概念方向噪声 | CIFAR-10 共享模型选第 0 轮，90.74% 复核回到 90.72%；CIFAR-100 共享模型未超过 72.76% 的关闭采样对照。原概念门槛不合格的高分模型不晋级。 |
| 类内残差方向（16 条） | 从训练类内变化中去除概念均值方向，降低属性扰动 | CUB 相同输入噪声能量下训练子集概念漂移由 0.105 降到 0.022；确定性适配达到 69.40%，噪声未超过它。 |
| 训练敏感度校准（16 条） | 用训练漂移比值确定尺度 0.481，避免验证尺度搜索 | 加大残差噪声后仍没有额外分类收益。 |
| 联合预测似然（16 条） | 用实际平均预测概率的分类/概念似然，保持概念均值而非惩罚每次采样 | 独立复核共享模型为 69.20%–69.25%，未超过同轮 69.40% 的适配对照。 |
| 原稀疏分类头适配（16 条） | 让已有稀疏头适应新的概念分布，所有对照更新相同非零支持 | 663 个原非零位置保持不变；确定性适配达到 69.45%，共享模型复核为 69.30%。 |
| 特征层 mixup（16 条） | 改善 CBL 训练覆盖，固定 alpha=0.2，使用软目标边际似然 | 共享模型复核 69.40%，比同配置关闭采样高 0.10 pp；同配置 A 对角噪声 69.35%–69.40%，共享结构没有额外优势。 |
| 期望分类风险（16 条） | 只将分类目标改为逐次采样交叉熵，保留概念边际似然、均值保持和 mixup | 共享模型为 69.10% / 69.20%，未超过关闭采样的 69.30%。 |
| 实测弱视图协方差（16 条） | 只用训练图像的两种弱变换估计低秩协方差，并按原训练敏感度规则校准尺度 | 共享复核为 68.90%–68.95% / 69.15%，未超过历史最强确定性 69.45%；共享属性 BA 下降约 0.17–0.19 pp。 |
| 真实弱视图训练（16 条） | 所有方法均匀抽取原图/两种真实弱视图，保持原图概念目标及相同 mixup 和预算 | 共享复核为 69.05% / 69.15%–69.20%，属性 BA 仍下降约 0.15–0.23 pp，未确认随机额外收益。 |

{replication_text}
**解释检查：** CUB 原始人工属性按明确颜色/部位保守映射，覆盖 **68/312 属性**；筛选及阈值仅用训练集，验证排除 certainty < 3。基线 BA / AUROC 为 **67.43% / 72.52%**。共享方案 BA 仍略降，不能宣称解释提升；覆盖有限，且没有定位或因果干预证据。

**效率与限制：** 概念数不变，CEA 随准确率同比变化；CEA 是论文紧凑度代理，不能替代人工属性质量。对照总览（04）横线是 Monte Carlo 范围，**不是置信区间**；条件复验（05）横线是注明范围的适配种子配对 t 区间。解析高斯概念均值用于区分非线性均值变化与随机推理效果，不替代重新训练的确定性对照。

**继续迭代：** 实测协方差和真实弱视图两轮对照均已完成，属性退化仍未消除。独立完整基线种子 43–46 正按原 CBL/SAGA 预算准备；完成状态与逐检查点验证见 [独立基线证据](independent_baselines.json)。这是最终复验准备，不是锁定 Pi 方案后的五种子结论。下一轮检验加入训练原生属性目标的扩展设置，全部控制使用相同监督；与无属性监督基线的差异不归因于随机性。Goal 保持 active；仍需可靠候选、独立五种子、配对区间及最终测试。

主要数值、CEA、属性变化和来源哈希见 [CSV](major_results.csv)、[来源汇总](major_results.json)。早期详细对照保留在 [首轮 CSV](validation_v1.csv)、[联合训练复核](joint_v5_summary.json)。
"""
    (report/'主要实验进展.md').write_text(primary)
    print(f'Exported 2 primary figures, {16*len(COHORTS)} complete joint runs plus 42 earlier controls',flush=True)

if __name__=='__main__':main()
