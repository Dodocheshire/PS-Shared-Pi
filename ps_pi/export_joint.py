"""Compact CUB joint-training results, independently resampled and attribute audited."""
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from .common import ROOT,atomic_json,digest
from .joint_noise import MODES

LABELS=['Clean joint fine-tuning','Mean only','Matched adapter','Diagonal noise','Shared noise',
        'Shared: no sampling','Fixed structured','Fixed isotropic']
COLORS=['#555555','#0072B2','#CC79A7','#E69F00','#D55E00','#009E73','#7B4F28','#56B4E9']

def main():
    audit=ROOT/'outputs/cub/validation_joint_v5_audit'
    if json.loads((audit/'status.json').read_text())['phase']!='complete':raise RuntimeError('Audit incomplete')
    summary=json.loads((audit/'summary.json').read_text());rows=summary['rows'];baseline=summary['baseline']
    report=ROOT.parent/'Essay/PS-Shared/report';report.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,4,figsize=(11.5,5.4),sharey=True)
    fig.subplots_adjust(left=.21,right=.99,top=.86,bottom=.22,wspace=.12)
    specs=[('accuracy',100,'Validation accuracy (%)',100*baseline['accuracy']),
           ('nll',1,'NLL (lower is better)',baseline['nll']),
           ('ece15',100,'ECE (%)',100*baseline['ece15']),
           ('native_ba_delta_pp',1,'Native attribute BA delta (pp)',0)]
    sources=[];records=[]
    for preset,cw,marker,offset in [('A',1.,'o',-.12),('B',5.,'s',.12)]:
        for index,mode in enumerate(MODES):
            row=next(r for r in rows if r['mode']==mode and r['config']['concept_weight']==cw)
            metric_path=audit/row['case']/'metrics.json'
            sources.append({'path':str(metric_path.relative_to(ROOT.parent)),'sha256':digest(metric_path)})
            original=ROOT/'outputs/cub/validation_joint_v5'/row['case']/'metrics.json'
            original_metrics=json.loads(original.read_text())['metrics']
            record={'case':row['case'],'preset':preset,'mode':mode,'training_seed':0,'baseline_seed':42,
                    'best_epoch':row['best_epoch'],'original_gate_eligible':original_metrics['selection_eligible'],
                    'native_ba_delta_pp':row['native_ba_delta_pp'],'native_auc_delta_pp':row['native_auc_delta_pp']}
            for key,_,_,_ in specs[:-1]:
                values=[r[key] for r in row['mc_runs']]
                record[key+'_min']=min(values);record[key+'_max']=max(values)
            records.append(record)
            for ax,(key,mult,label,reference) in zip(axes,specs):
                values=[row[key]] if key=='native_ba_delta_pp' else [m[key]*mult for m in row['mc_runs']]
                value=np.mean(values);low=value-min(values);high=max(values)-value
                ax.errorbar(value,index+offset,xerr=np.array([[low],[high]]),marker=marker,markersize=5,
                            color=COLORS[index],markerfacecolor=COLORS[index] if preset=='A' else 'white',
                            capsize=2,linewidth=1.2,linestyle='none')
    for ax,(_,_,label,reference) in zip(axes,specs):
        ax.axvline(reference,color='#777777',ls='--',lw=1,alpha=.8)
        ax.set_xlabel(label);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    axes[0].set_yticks(range(8),LABELS);axes[0].invert_yaxis()
    axes[0].set_xlim(68.62,69.03);axes[0].set_xticks([68.7,68.8,68.9,69.0])
    axes[1].set_xlim(1.067,1.14);axes[2].set_xlim(2.8,10.8);axes[3].set_xlim(-.30,.10)
    axes[3].set_xticks([-.3,-.2,-.1,0,.1])
    fig.suptitle('CUB joint-training controls: one training seed, independent MC verification',fontsize=11,y=.955)
    legend=[Line2D([0],[0],marker='o',ls='none',color='#333333',label='A: concept weight 1, CBL lr 1e-5'),
            Line2D([0],[0],marker='s',ls='none',color='#333333',markerfacecolor='white',label='B: concept weight 5, CBL lr 3e-6'),
            Line2D([0],[0],ls='--',color='#777777',label='Original PS-CBM baseline')]
    fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.5,.065),ncol=3,frameon=False,fontsize=8)
    fig.text(.5,.028,'Whiskers: two independent 256-sample MC runs, not confidence intervals. Native BA: train-only mapping and thresholds, 68/312 attributes.',ha='center',fontsize=7.5)
    for ext in ['png','svg','pdf']:
        path=report/f'03_joint_training_controls.{ext}';fig.savefig(path,dpi=220)
        if ext=='svg':path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    plt.close(fig)
    with (report/'joint_v5.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]),lineterminator='\n');writer.writeheader();writer.writerows(records)
    sources.append({'path':str((audit/'summary.json').relative_to(ROOT.parent)),'sha256':digest(audit/'summary.json')})
    attribute_protocol=ROOT/'artifacts/cub/native_attributes_v1/protocol.json'
    sources.append({'path':str(attribute_protocol.relative_to(ROOT.parent)),'sha256':digest(attribute_protocol)})
    atomic_json({'dataset':'cub','training_seeds':1,'completed_controls':16,'eval_split':'val','validation_images':2000,
                 'mc_samples':256,'mc_seeds':[9071,9072],'test_used':False,'five_seed_verified':False,
                 'baseline':baseline,'native_baseline':{k:v for k,v in summary['native_baseline'].items() if k!='details'},
                 'records':records,'sources':sources,
                 'figures':{ext:digest(report/f'03_joint_training_controls.{ext}') for ext in ['png','svg','pdf']}},report/'joint_v5_summary.json')
    main_report=report/'主要实验进展.md'
    text=main_report.read_text().split('\n## 联合训练与人工属性复核',1)[0]
    text+='''
## 联合训练与人工属性复核

![联合训练完整对照](03_joint_training_controls.png)

**目的：** 将训练特征按概念方向加噪，同时微调 CBL；用相同生成器、尺度先验与训练预算的“关闭训练采样”对照区分随机训练收益，并复核人工属性。

**结论：** CUB 两套预登记配置 × 八类对照共 **16 组**完成，累计完成 **58 组验证探索**。冻结所选权重后，以两组独立随机数、各 256 次采样复核，共享噪声的准确率为 68.75% / 68.80%，没有超过最强“关闭训练采样”对照的 68.95%。干净联合微调为 68.85% / 68.80%。NLL、ECE 有改善，但均值与确定性适配也有类似改善，尚不能据此归因于噪声。

人工属性仅按文本中的明确颜色/部位短语映射，排除条纹、色斑等局部描述；**68/312 属性**满足训练集已知正负样本数要求。阈值只在训练集设定，验证集排除 certainty < 3 标签。原始基线属性宏平均 BA / AUROC 为 67.43% / 72.52%；共享噪声 BA 变化为 −0.173 / −0.082 个百分点，AUROC 变化为 −0.092 / −0.007 个百分点。这是有限属性覆盖的整图预测检查，仍不能宣称解释性能提升或因果验证完成。

图中横线仅表示两次独立 Monte Carlo 的范围，**不是置信区间**。均为同一个基线种子 42、训练种子 0 的验证结果；未使用测试集，没有五种子结论。CIFAR-10/100 的相同联合训练对照继续运行。明细与来源见 [CSV](joint_v5.csv) 和 [复核汇总](joint_v5_summary.json)。
'''
    main_report.write_text(text)
    print('Exported 16 audited CUB joint controls, native attributes and concise figure',flush=True)

if __name__=='__main__':main()
