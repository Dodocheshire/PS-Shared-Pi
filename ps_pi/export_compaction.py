"""Plot output-count and parameter reductions without claiming a noise gain."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import ROOT, atomic_json, digest


def main():
    report=ROOT.parent/'Essay/PS-Shared/report'
    path=report/'compact_baselines.json';data=json.loads(path.read_text())
    datasets=['cifar10','cifar100','cub'];labels=['CIFAR-10','CIFAR-100','CUB']
    groups=[[r for r in data['records'] if r['dataset']==d] for d in datasets]
    if any(not rows or any(r['prediction_changes'] or not r['reload_verified'] for r in rows) for rows in groups):
        raise RuntimeError('Incomplete compact artifact verification')
    fig,axes=plt.subplots(1,2,figsize=(9.4,3.8),layout='constrained')
    x=np.arange(3);width=.32
    axes[0].bar(x-width/2,[rows[0]['original_concepts'] for rows in groups],width,color='#777777',label='Original outputs')
    axes[0].bar(x+width/2,[np.mean([r['retained_concepts'] for r in rows]) for rows in groups],width,color='#009E73',label='Retained outputs')
    for i,rows in enumerate(groups):
        values=[r['retained_concepts'] for r in rows]
        axes[0].text(i+.16,max(values)+12,f'{min(values)}–{max(values)}',ha='center',fontsize=8)
    axes[0].set_ylabel('Concept output count');axes[0].legend(frameon=False,fontsize=8)
    for j,(key,color,label) in enumerate([('output_reduction_percent','#009E73','Output count'),('parameter_reduction_percent','#0072B2','CBL + dense head parameters')]):
        offset=(-.16 if j==0 else .16)
        means=[np.mean([r[key] for r in rows]) for rows in groups]
        axes[1].bar(x+offset,means,width,color=color,label=label,alpha=.85)
        for i,rows in enumerate(groups):
            values=[r[key] for r in rows]
            axes[1].scatter(i+offset+np.linspace(-.035,.035,len(values)),values,color='#222222',s=9,zorder=3)
    axes[1].set_ylabel('Reduction (%)');axes[1].legend(frameon=False,fontsize=8)
    for ax in axes:
        ax.set_xticks(x,[f'{label}\n(n={len(rows)})' for label,rows in zip(labels,groups)])
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    axes[0].set_ylim(0,650);axes[1].set_ylim(0,20)
    fig.suptitle('Exact unused-output compaction: validation predictions preserved',fontsize=11)
    for ext in ['png','svg','pdf']:
        fig.savefig(report/f'06_concept_compaction.{ext}',dpi=220)
    plt.close(fig)
    lines=['# 概念输出精简：干净基线效率证据','',
           '**目的：** 删除分类头完全不使用的概念输出，同时保留原生属性审计依赖的概念。保留规则只读取训练所得权重和训练确定的属性映射，不按验证表现挑选概念。','',
           '| 数据集 | 已完成基线种子数 | 原输出数 | 保留输出数范围 | CBL+稠密头参数减少范围 | 验证预测改变数 |',
           '|---|---:|---:|---:|---:|---:|']
    for name,rows in zip(labels,groups):
        n=[r['retained_concepts'] for r in rows];p=[r['parameter_reduction_percent'] for r in rows]
        lines.append(f'| {name} | {len(rows)} | {rows[0]["original_concepts"]} | {min(n)}–{max(n)} | {min(p):.2f}%–{max(p):.2f}% | 0 |')
    lines+=['','![概念输出精简](06_concept_compaction.png)','',
            '**结论：** 已核验基线的验证分类预测保持不变，NLL/ECE仅有浮点误差范围内差异。CUB五种子保留全部161个属性映射涉及的概念，68项已映射属性的BA/AUROC不变。compact检查点已保存并重新加载验证。','',
            '**实现口径：** 只裁掉最后一个概念线性层的输出行、对应标准化参数和分类头输入列，原隐层宽度保持不变。必须保留任意非零头权重，即使其幅值小于NEC计数阈值；以阈值近零代替精确零不能保证恒等。','',
            '**限制：** 这是干净模型的输出精简，不是Pi噪声收益。参数减少仅计算CBL和稠密存储的分类头，未包括冻结图像编码器，没有时延或显存测量。CUB属性保护仅覆盖固定映射的68/312项。尚未裁剪共享隐层或已训练生成器的概念条件输入。输出数下降不等于整网同等比例加速，也不作为与原论文构建概念数直接匹配的SOTA比较。','',
            '[逐基线核验数据](compact_baselines.json)、[CSV](compact_baselines.csv)、[独立基线属性阈值准备](baseline_contexts.json)。']
    (report/'概念输出精简.md').write_text('\n'.join(lines)+'\n')
    atomic_json({'audit_sha256':digest(path),'exporter_sha256':digest(__file__),
                 'figures':{ext:digest(report/f'06_concept_compaction.{ext}') for ext in ['png','svg','pdf']},
                 'noise_gain':False,'test_used':False,'final_evidence':False},report/'compact_plot_manifest.json')
    print('Exported compact baseline results:',len(data['records']),flush=True)


if __name__=='__main__':main()
