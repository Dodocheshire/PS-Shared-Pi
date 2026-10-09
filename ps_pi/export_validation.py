"""Concise figures from verified completed validation outputs, with source provenance."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from .common import ROOT,atomic_json,digest

COLORS={'clean':'#777777','mean_only':'#0072B2','adapter_matched':'#E69F00',
        'diag_noise':'#009E73','shared_noise':'#CC79A7','fixed_noise':'#56B4E9'}
LABELS={'clean':'Clean','mean_only':'Mean only','adapter_matched':'Matched adapter',
        'diag_noise':'Diagonal Pi','shared_noise':'Shared Pi','fixed_noise':'Fixed Gaussian'}

def load_rows(directory):
    rows=[];sources=[]
    for path in sorted(directory.glob('*/metrics.json')):
        status_path=path.parent/'status.json'
        if not status_path.exists() or json.loads(status_path.read_text()).get('phase')!='complete':continue
        record=json.loads(path.read_text());row=record['metrics'];protocol=json.loads((path.parent/'protocol.json').read_text())
        if row.get('test_used') or row.get('split')!='val' or protocol['test_opened']:raise RuntimeError('Non-validation result mixed into screen')
        row=dict(row,concept_weight=protocol['config']['concept_weight'],energy_weight=protocol['config']['energy_weight'],path=str(path))
        rows.append(row);sources.append({'path':str(path.relative_to(ROOT.parent)),'sha256':digest(path),'protocol_sha256':digest(path.parent/'protocol.json')})
    return rows,sources

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--directory',default='outputs/cifar10/validation_v1')
    parser.add_argument('--report',default='../Essay/PS-Shared/report');args=parser.parse_args()
    rows,sources=load_rows(ROOT/args.directory);report=(ROOT/args.report).resolve();report.mkdir(parents=True,exist_ok=True)
    if len(rows)!=9:raise RuntimeError(f'Expected all nine v1 controls, got {len(rows)}')
    clean=json.loads((ROOT/'outputs/cifar10/baseline/seed42/corrected_best_validation.json').read_text())
    chosen=[row for row in rows if row['concept_weight']==.1 and row['energy_weight']==.01]
    order=['fixed_noise','mean_only','adapter_matched','diag_noise','shared_noise'];chosen=sorted(chosen,key=lambda r:order.index(r['mode']))
    plotted=[dict(clean,mode='clean',delta_same_model_mean_pp=0.)]+chosen
    labels=[LABELS[r['mode']] for r in plotted];colors=[COLORS[r['mode']] for r in plotted]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    figure,axes=plt.subplots(1,3,figsize=(12.6,3.8),layout='constrained')
    x=np.arange(len(plotted))
    values=[100*r['accuracy'] for r in plotted]
    axes[0].bar(x,values,color=colors);axes[0].set_ylim(90.5,91.1);axes[0].set_ylabel('Validation accuracy (%)')
    axes[0].axhline(values[0],color=COLORS['clean'],linestyle='--',lw=1)
    for i,value in enumerate(values):axes[0].text(i,value+.012,f'{value:.2f}',ha='center',fontsize=8)
    values=[r['nll'] for r in plotted]
    axes[1].bar(x,values,color=colors);axes[1].set_ylim(.27,.31);axes[1].set_ylabel('Validation NLL (lower is better)')
    axes[1].axhline(values[0],color=COLORS['clean'],linestyle='--',lw=1)
    for i,value in enumerate(values):axes[1].text(i,value+.001,f'{value:.4f}',ha='center',fontsize=8)
    values=[100*r['ece15'] for r in plotted]
    axes[2].bar(x,values,color=colors);axes[2].set_ylim(0,3.5);axes[2].set_ylabel('Validation ECE, 15 bins (%)')
    axes[2].axhline(values[0],color=COLORS['clean'],linestyle='--',lw=1)
    for i,value in enumerate(values):axes[2].text(i,value+.07,f'{value:.2f}',ha='center',fontsize=8)
    for axis in axes:axis.set_xticks(x,labels,rotation=32,ha='right');axis.grid(axis='y',alpha=.18);axis.set_axisbelow(True)
    figure.suptitle('PS-CBM CIFAR-10: first validation screen (one adaptation seed)',fontsize=12)
    for extension in ['png','svg','pdf']:figure.savefig(report/f'01_validation_controls.{extension}',dpi=220)
    plt.close(figure)
    columns=['dataset','mode','concept_weight','energy_weight','seed','accuracy','nll','ece15','concept_bce','concept_drift_rms','delta_clean_pp','delta_same_model_mean_pp','stochastic_energy_fraction','best_epoch']
    with (report/'validation_v1.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,columns);writer.writeheader();writer.writerows({k:row[k] for k in columns} for row in rows)
    pi=next(r for r in chosen if r['mode']=='shared_noise');adapter=next(r for r in chosen if r['mode']=='adapter_matched')
    text=f'''# PS-CBM 主要实验进展

当前只报告完整验证集。未打开测试特征；这是单种子的探索阶段，尚未锁定方案或完成五种子独立验证。

![首轮完整对照](01_validation_controls.png)

**目的：** 检查可学习噪声是否优于干净、仅均值、同容量确定性适配与固定随机噪声，同时观察校准。

**结论：** 在同一首轮配置（概念损失 0.1、能量损失 0.01、40 epoch）下，共享 Pi 验证准确率为 {100*pi['accuracy']:.2f}%，干净基线为 {100*clean['accuracy']:.2f}%，同容量适配为 {100*adapter['accuracy']:.2f}%。共享 Pi 关闭随机尺度后准确率为 {100*pi['same_model_mean_accuracy']:.2f}%，随机分量额外增益为 {pi['delta_same_model_mean_pp']:.2f} 个百分点。因此不能把这次提升归因于随机噪声。共享 Pi 的 NLL 和 ECE 也高于干净基线，当前配置不满足下一阶段校准要求。

共享 Pi 的随机能量比例为 {100*pi['stochastic_energy_fraction']:.2f}%；14/61 个概念允许共享因子相关噪声。所有学习方法分配相同的 24,964 个参数，同容量适配激活确定性分支，固定随机对照没有可训练参数。详细九组探索结果见 [CSV](validation_v1.csv)。同一个基线种子 42、适配种子 0、相同批次顺序；验证 Monte Carlo 使用固定种子和 64 个反向配对样本。图中纵轴范围明确标出，单种子不绘制虚假的种子区间。

**下一轮：** 保留首轮结果，测试概念空间 mixup，增加可学习尺度范围，并按预先设定的 NLL、ECE、概念伪标签 BCE 与扰动能量约束筛选。训练和选择仍只使用训练/验证数据。CIFAR-100 与 CUB 官方基线继续执行。

首轮官方末轮与正确保存最佳验证权重的独立结果见 [首轮基线](首轮基线.md)。伪标签概念 BCE 是自动监督的诊断指标，不能代替 CUB 人工属性或因果解释验证。
'''
    (report/'主要实验进展.md').write_text(text)
    atomic_json({'stage':'validation_v1','source_results':sources,'rows':rows,'test_used':False,'five_seed_verified':False,
                 'figure_sha256':{ext:digest(report/f'01_validation_controls.{ext}') for ext in ['png','svg','pdf']}},report/'validation_v1_manifest.json')
    print('Exported',report,'9 verified controls, 3 plot formats',flush=True)
if __name__=='__main__':main()
