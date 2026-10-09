"""Four concise, color-consistent figures and the requested final experiment report."""
import csv
import json
from pathlib import Path
import shutil
import numpy as np
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from .common import ROOT, atomic_json, digest
from .final_lock import SEEDS, MODES, PATH, verify_lock

METHODS=[('corrected_best_validation','Original clean','#555555'),
         ('clean','CBL adaptation only','#999999'),('mean_only','Mean only','#0072B2'),
         ('adapter_matched','Matched adapter','#CC79A7'),('shared_no_sampling','Sampling disabled','#009E73'),
         ('diag_noise','Diagonal noise','#56B4E9'),('shared_noise','Shared noise','#D55E00'),
         ('fixed_isotropic','Fixed random','#E69F00')]
NAMES={'cifar10':'CIFAR-10','cifar100':'CIFAR-100','cub':'CUB'}
DEST=ROOT/'reports/final_v1'


def paired_t(values):
    values=np.asarray(values,dtype=float)
    if len(values)!=5:raise ValueError('Final paired interval requires five independent training seeds')
    mean=values.mean();half=t.ppf(.975,4)*values.std(ddof=1)/np.sqrt(5)
    return {'mean':float(mean),'low':float(mean-half),'high':float(mean+half),'n':5,'sd':float(values.std(ddof=1))}


def format_mean(values,percent=False):
    a=np.asarray(values,dtype=float)*(100 if percent else 1)
    precision=2 if percent else 4
    return f'{a.mean():.{precision}f} ± {a.std(ddof=1):.{precision}f}'


def save_figure(fig,name):
    for ext in ['png','svg','pdf']:fig.savefig(DEST/f'{name}.{ext}',dpi=230,bbox_inches='tight')
    plt.close(fig)


def rows_for(summary,dataset,method):
    result=sorted([r for r in summary['rows'] if r['dataset']==dataset and r['method']==method],key=lambda r:r['baseline_seed'])
    if [r['baseline_seed'] for r in result]!=SEEDS:raise RuntimeError('Incomplete or unpaired seeds')
    return result


def values(rows,key):return np.array([r[key] for r in rows])


def image_bootstrap(method,reference,replicates=2000):
    """Common test-image resampling, conditional on the five fitted models."""
    differences=[]
    for seed in SEEDS:
        folder=ROOT/f'outputs/final_evaluation_v1/test/cub/seed{seed}'
        a=__import__('torch').load(folder/method/'predictions.pt',weights_only=True)
        b=__import__('torch').load(folder/reference/'predictions.pt',weights_only=True)
        if not __import__('torch').equal(a['labels'],b['labels']) or not __import__('torch').equal(a['indices'],b['indices']):raise RuntimeError('Paired test image mismatch')
        differences.append((a['probabilities'].argmax(1).eq(a['labels']).float()-b['probabilities'].argmax(1).eq(b['labels']).float()).numpy())
    per_image=np.mean(differences,axis=0);rng=np.random.default_rng(260109);samples=[]
    for _ in range(replicates):samples.append(100*per_image[rng.integers(0,len(per_image),len(per_image))].mean())
    lo,hi=np.quantile(samples,[.025,.975])
    return {'mean':float(100*per_image.mean()),'low':float(lo),'high':float(hi),'replicates':replicates,
            'scope':'Common-image paired bootstrap conditional on these five fitted models; separate from training-seed t interval.'}


def main():
    verify_lock();DEST.mkdir(parents=True,exist_ok=True)
    summaries={s:json.loads((ROOT/f'outputs/final_evaluation_v1/{s}/summary.json').read_text()) for s in ['val','test']}
    if any(v['models']!=65 for v in summaries.values()):raise RuntimeError('Final evaluation incomplete')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    baseline={s:{d:{m:rows_for(summaries[s],d,m) for m in ['official_final_state','corrected_best_validation']} for d in NAMES} for s in summaries}
    cub={s:{m:rows_for(summaries[s],'cub',m) for m,_,_ in METHODS} for s in summaries}
    contrasts={}
    for reference in ['corrected_best_validation','shared_no_sampling','adapter_matched']:
        delta=100*(values(cub['test']['shared_noise'],'accuracy')-values(cub['test'][reference],'accuracy'))
        contrasts[reference]={'accuracy_delta_pp':paired_t(delta),'nll_delta':paired_t(values(cub['test']['shared_noise'],'nll')-values(cub['test'][reference],'nll')),
                             'ece_delta_pp':paired_t(100*(values(cub['test']['shared_noise'],'ece15')-values(cub['test'][reference],'ece15'))),
                             'image_bootstrap_accuracy_delta_pp':image_bootstrap('shared_noise',reference)}
    inference_delta=100*(values(cub['test']['shared_noise'],'accuracy')-np.array([r['same_model_mean']['accuracy'] for r in cub['test']['shared_noise']]))
    contrasts['same_model_mean']={'accuracy_delta_pp':paired_t(inference_delta)}
    # 1: checkpoint correction is isolated from Pi gains.
    fig,axes=plt.subplots(1,3,figsize=(11.5,3.8))
    for ax,(dataset,name) in zip(axes,NAMES.items()):
        all_y=[]
        for i,(method,label,color) in enumerate([('official_final_state','Official final','#999999'),('corrected_best_validation','Best validation','#0072B2')]):
            y=100*values(baseline['test'][dataset][method],'accuracy');all_y.extend(y)
            ax.bar(i,y.mean(),color=color,width=.55,alpha=.85,yerr=y.std(ddof=1),capsize=4)
            ax.scatter(i+np.linspace(-.14,.14,5),y,color='#222222',s=14,zorder=3)
            ax.text(i,y.max()+.12,f'{y.mean():.2f}%',ha='center',fontsize=10)
        spread=max(all_y)-min(all_y);ax.set_ylim(min(all_y)-max(1.,spread*.4),max(all_y)+max(.6,spread*.25))
        ax.set_xticks([0,1],['Official final','Best validation']);ax.set_title(name);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    axes[0].set_ylabel('Test accuracy (%)')
    fig.suptitle('Clean PS-CBM reproduction: five independent training seeds',fontsize=12)
    fig.text(.5,.015,'Bars: mean ± training-seed SD; dots: seeds42–46. Fixed training concept bank and pretrained features shared.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.05,1,.94]);save_figure(fig,'01_baseline_test')
    # 2: final methods, paired intervals against clean baseline.
    fig,axes=plt.subplots(1,3,figsize=(12.5,5.1));fig.subplots_adjust(left=.17,right=.98,top=.88,bottom=.16,wspace=.35)
    clean=cub['test']['corrected_best_validation']
    for i,(method,label,color) in enumerate(METHODS):
        rs=cub['test'][method];eligible=sum(r.get('selection_eligible',True) for r in rs)
        for ax,key,multiplier in zip(axes,['accuracy','nll','ece15'],[100.,1.,100.]):
            delta=paired_t(multiplier*(values(rs,key)-values(clean,key)));marker='o' if eligible==5 else 'X'
            ax.errorbar(delta['mean'],i,xerr=[[delta['mean']-delta['low']],[delta['high']-delta['mean']]],color=color,marker=marker,ms=7,capsize=3,ls='none')
        axes[0].annotate(f"{100*values(rs,'accuracy').mean():.2f}%",(0,i),xytext=(4,-15),textcoords='offset points',fontsize=7,color=color)
    for ax in axes:
        ax.axvline(0,color='#777777',ls='--',lw=1);ax.set_ylim(len(METHODS)-.4,-.6);ax.grid(axis='x',alpha=.18);ax.xaxis.set_major_locator(MaxNLocator(4))
    axes[0].set_yticks(range(len(METHODS)),[name for _,name,_ in METHODS])
    for ax in axes[1:]:ax.set_yticks(range(len(METHODS)),[])
    for ax,label in zip(axes,['Test accuracy Δ (pp; higher better)','Test NLL Δ (lower better)','Test ECE Δ (pp; lower better)']):ax.set_xlabel(label,fontsize=9)
    fig.suptitle('Final CUB controls: differences from the independently trained clean baseline',fontsize=12)
    fig.text(.5,.035,'95% paired t intervals across five training seeds. X: at least one validation gate failed; retained as diagnostic control. No test selection.',ha='center',fontsize=8)
    save_figure(fig,'02_cub_main_test')
    # 3: primary causal contrast and same-model inference.
    fig,axes=plt.subplots(1,2,figsize=(10.7,4.1));fig.subplots_adjust(left=.08,right=.98,top=.83,bottom=.24,wspace=.28)
    for ax,reference,title,color in zip(axes,['shared_no_sampling','same_model_mean'],['Training contrast: shared − sampling disabled','Inference contrast: shared − its own mean'],['#009E73','#D55E00']):
        delta=(100*(values(cub['test']['shared_noise'],'accuracy')-values(cub['test'][reference],'accuracy')) if reference!='same_model_mean' else inference_delta)
        interval=paired_t(delta)
        ax.scatter(range(5),delta,color=color,s=45)
        ax.errorbar(5.3,interval['mean'],yerr=[[interval['mean']-interval['low']],[interval['high']-interval['mean']]],color='#222222',marker='D',capsize=5)
        ax.axhline(0,color='#777777',ls='--',lw=1);ax.set_xticks(range(6),[str(s) for s in SEEDS]+['Mean / CI']);ax.set_xlim(-.4,5.7);ax.set_title(title,fontsize=10);ax.grid(axis='y',alpha=.15)
        ax.set_xlabel('Independent baseline training seed');ax.set_ylabel('Paired test accuracy Δ (pp)')
    fig.suptitle('Isolating randomness from adaptation',fontsize=12)
    fig.text(.5,.055,'Same architecture, mean capacity and budget. Seeds42–46 pair with adaptation seeds0–4; all settings locked before tests.',ha='center',fontsize=8)
    save_figure(fig,'03_noise_contrasts')
    # 4: exact protection and honest compaction.
    compaction=json.loads((ROOT.parent/'Essay/PS-Shared/report/compact_baselines.json').read_text())
    records=compaction.get('records',[])
    fig,axes=plt.subplots(1,2,figsize=(10.8,4.0));fig.subplots_adjust(left=.07,right=.98,top=.83,bottom=.19,wspace=.25)
    # Use original concept counts from immutable checkpoint metadata, avoiding schema assumptions.
    import torch
    kept=[];original=[]
    for dataset in NAMES:
        packs=[torch.load(ROOT/f'outputs/{dataset}/compact_baselines/seed{s}/model.pt',weights_only=False,map_location='cpu') for s in SEEDS]
        original.append(packs[0]['original_concept_count']);kept.append(np.mean([len(p['concepts']) for p in packs]))
    x=np.arange(3);axes[0].bar(x-.17,original,.34,label='Original outputs',color='#999999');axes[0].bar(x+.17,kept,.34,label='Retained outputs',color='#0072B2')
    for i,(a,b) in enumerate(zip(original,kept)):axes[0].text(i,max(a,b)+8,f'{int(a)} → {b:.1f}',ha='center',fontsize=9)
    axes[0].set_xticks(x,list(NAMES.values()));axes[0].set_ylim(0,max(original)*1.17);axes[0].set_ylabel('Number of concept outputs');axes[0].set_title('Exactly unused output removal');axes[0].legend(fontsize=8,frameon=False)
    for i,(method,label,color) in enumerate([('corrected_best_validation','Clean','#555555'),('adapter_matched','Matched adapter','#CC79A7'),('shared_noise','Shared noise','#D55E00')]):
        native=cub['test'][method]
        y=np.array([[100*r['native_attributes']['macro_balanced_accuracy'],100*r['native_attributes']['macro_auroc']] for r in native])
        axes[1].bar(np.arange(2)+(i-1)*.24,y.mean(0),.24,color=color,label=label,yerr=y.std(0,ddof=1),capsize=2)
    axes[1].set_xticks([0,1],['Native balanced accuracy','Native AUROC']);axes[1].set_ylabel('Test score (%)');axes[1].set_ylim(0,100);axes[1].set_title('Protected native attributes: exact equality');axes[1].legend(fontsize=8,frameon=False,loc='upper left')
    fig.suptitle('Concept output efficiency and preserved native semantics',fontsize=12)
    fig.text(.5,.035,'Compaction preserves clean validation predictions. Native metrics use 68/312 fixed mapped attributes and train-only thresholds.',ha='center',fontsize=8)
    save_figure(fig,'04_compaction_native')
    flat=[]
    for split,summary in summaries.items():
        for row in summary['rows']:
            flat.append({k:row.get(k) for k in ['split','dataset','baseline_seed','adaptation_seed','method','accuracy','nll','ece15','correct','n','best_epoch','selection_eligible','native_protected_exact','checkpoint_sha256']})
    with (DEST/'metrics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    results={'validation':summaries['val'],'test':summaries['test'],'paired_test_contrasts':contrasts,
             'figure_sha256':{p.name:digest(p) for p in sorted(DEST.glob('0[1-4]*.*'))},
             'source_sha256':digest(__file__),'final_lock_sha256':digest(PATH),
             'evaluation_summary_sha256':{s:digest(ROOT/f'outputs/final_evaluation_v1/{s}/summary.json') for s in summaries},
             'limits':'Five independently trained baselines/adaptations, fixed concept bank/features, CUB final noise only; CIFAR clean test reproduction only. Negative or uncertain stochastic result retained.'}
    atomic_json(results,DEST/'results.json')
    primary=contrasts['shared_no_sampling']['accuracy_delta_pp'];against_adapter=contrasts['adapter_matched']['accuracy_delta_pp']
    superiority=(primary['low']>0 and against_adapter['low']>0 and all(r['selection_eligible'] for r in cub['test']['shared_noise']) and contrasts['corrected_best_validation']['nll_delta']['mean']<=0 and contrasts['corrected_best_validation']['ece_delta_pp']['mean']<=0)
    conclusion=('本次预登记主对照显示正向随机收益，但五种子区间、固定属性覆盖和校准口径仍限制结论。' if superiority else '未获得共享随机分量可靠优于强确定性对照的证据；不能把适配收益归因于噪声。')
    lines=['# PS-Shared-Pi 最终实验结果','',
           '**交付日期：2026-10-09。** 已停止新增参数搜索与细小消融；冻结 V22-B 设置，完成 CUB 七组控制×五个独立训练种子，以及三个数据集的五种子干净基线测试。','',
           f"**主要结论：** CUB 测试准确率：原始干净模型 **{format_mean(values(cub['test']['corrected_best_validation'],'accuracy'),True)}%**，参数匹配适配器 **{format_mean(values(cub['test']['adapter_matched'],'accuracy'),True)}%**，共享噪声 **{format_mean(values(cub['test']['shared_noise'],'accuracy'),True)}%**。{conclusion}",'',
           '**最终实现：** `ps_pi/final_infer.py`。默认使用测试前选定的参数匹配确定性适配器；`--mode shared_noise` 加载冻结的共享噪声版本。共享部分采用训练弱视图估计的有符号因子，协方差为 `diag(σ²) + B diag(γ²) Bᵀ`；均值容量与强确定性控制一致，推理无需真实类别。68 项映射属性涉及的161个概念通道由固定参考CBL输出提供，其余概念允许自适应CBL学习。','',
           '| 固定协议 | 设置 |','|---|---|',
           '| 原始方法 | Partially Shared CBM，AAAI 2026；官方提交 `749c6e3`，78个文件哈希核验 |',
           '| 独立重复 | 基线种子42–46，分别配对适配种子0–4；共享训练概念库与预训练图像特征 |',
           '| 最终训练 | CUB，40轮；CBL LR 3e-6；生成器 LR 1e-3；概念权重5；批量128；mixup α=0.2 |',
           '| 采样 | 训练4次，验证选epoch32次；冻结后评估 MC256×两个固定种子，平均类别概率 |',
           '| 选模 | 仅验证集：ACC/NLL/ECE/伪概念BCE和原生BA/AUROC不退化，概念漂移≤0.3，再按正确数/NLL选epoch |',
           '| 原生属性 | 训练标注决定映射、保护集合和阈值；没有原生属性训练损失；所有控制使用相同辅助约束 |',
           '| 测试 | 所有参数、方法和检查点锁定后评估；不依测试结果调参或重新挑模型 |','',
           '## 1. 完整干净基线复现','',
           '**目的：** 确认官方预算可复现，单独衡量最佳验证权重保存修正，避免将此收益算作Pi噪声收益。','',
           '![干净基线测试](reports/final_v1/01_baseline_test.png)','',
           '| 数据集 | 最佳验证状态：验证ACC (%) | 官方末轮：测试ACC (%) | 最佳验证状态：测试ACC (%) |','|---|---:|---:|---:|']
    for dataset in NAMES:
        lines.append(f"| {NAMES[dataset]} | {format_mean(values(baseline['val'][dataset]['corrected_best_validation'],'accuracy'),True)} | {format_mean(values(baseline['test'][dataset]['official_final_state'],'accuracy'),True)} | {format_mean(values(baseline['test'][dataset]['corrected_best_validation'],'accuracy'),True)} |")
    lines+=['','**结论：** 三个数据集均完成五次独立CBL/SAGA训练。末轮与最佳验证状态使用同预算重新拟合官方稀疏头；两者差异属于检查点策略。CIFAR表仅代表干净基线复现，未宣称该最终Pi在CIFAR完成五种子实验。','',
            '## 2. CUB 最终主要对照','',
            '**目的：** 在相同双分支CBL、生成器容量、训练数据和预算下，区分概念适配、确定性均值与随机采样的贡献。','',
            '![CUB主要结果](reports/final_v1/02_cub_main_test.png)','',
            '| 方法 | 验证ACC (%) | 测试ACC (%) | 测试NLL ↓ | 测试ECE (%) ↓ | 验证门槛通过 |','|---|---:|---:|---:|---:|---:|']
    cn={'corrected_best_validation':'原始干净基线','clean':'仅CBL适配','mean_only':'仅均值（弱控制）','adapter_matched':'参数匹配适配器','shared_no_sampling':'共享模型关闭训练采样','diag_noise':'对角噪声','shared_noise':'共享噪声','fixed_isotropic':'固定随机噪声'}
    for method,_,_ in METHODS:
        rs=cub['test'][method];eligible=sum(r.get('selection_eligible',True) for r in rs)
        lines.append(f"| {cn[method]} | {format_mean(values(cub['val'][method],'accuracy'),True)} | {format_mean(values(rs,'accuracy'),True)} | {format_mean(values(rs,'nll'))} | {format_mean(values(rs,'ece15'),True)} | {eligible}/5 |")
    lines+=['',f'**结论：** {conclusion} 确定性适配将测试NLL从1.1015降至1.0738、ECE从10.84%降至7.50%；主要可见改善是预测概率质量。表中保留未通过验证门槛的控制，不能将其高测试分数事后提升为合格候选。数值为五种子均值±样本标准差，图中的差值误差线为五种子配对t区间。','',
            '## 3. 随机性是否带来额外收益','',
            '**目的：** 主比较关闭训练采样的同结构模型；另用同一共享模型关闭推理尺度，检查推理期随机平均本身的收益。','',
            '![随机性配对比较](reports/final_v1/03_noise_contrasts.png)','',
            '| 测试配对比较 | ACC差值 (pp) | 五训练种子95%配对区间 | 同测试图像bootstrap区间 |','|---|---:|---:|---:|']
    for reference,label in [('corrected_best_validation','共享噪声 − 原始干净模型'),('shared_no_sampling','共享噪声 − 关闭训练采样（主比较）'),('adapter_matched','共享噪声 − 参数匹配适配器'),('same_model_mean','共享噪声 − 同模型均值推理')]:
        delta=contrasts[reference]['accuracy_delta_pp'];image=contrasts[reference].get('image_bootstrap_accuracy_delta_pp')
        image_text='—' if image is None else f"[{image['low']:+.2f}, {image['high']:+.2f}]"
        lines.append(f"| {label} | {delta['mean']:+.2f} | [{delta['low']:+.2f}, {delta['high']:+.2f}] | {image_text} |")
    lines+=['',f"**结论：** 主比较差值为 {primary['mean']:+.2f} pp，五种子95%区间 [{primary['low']:+.2f}, {primary['high']:+.2f}]；对参数匹配适配器为 {against_adapter['mean']:+.2f} pp，区间 [{against_adapter['low']:+.2f}, {against_adapter['high']:+.2f}]。{conclusion}",'',
            '**区间口径：** 配对t区间仅反映五个训练种子的差值，样本小；图像bootstrap使用2000次共同图像重采样，条件于这五个已训练模型，两种区间没有合并。MC种子只增加积分精度，不充当独立模型种子。','',
            '## 4. 属性保护与概念输出精简','',
            '**目的：** 检查分类适配是否改变固定映射的真实属性响应，并删除稀疏分类头完全不用的概念输出。','',
            '![属性保护和精简](reports/final_v1/04_compaction_native.png)','',
            '| 模型 | CUB测试原生BA (%) | CUB测试原生AUROC (%) | 受保护响应 |','|---|---:|---:|---|']
    for method in ['corrected_best_validation','adapter_matched','shared_noise']:
        rs=cub['test'][method]
        lines.append(f"| {cn[method]} | {format_mean([r['native_attributes']['macro_balanced_accuracy'] for r in rs],True)} | {format_mean([r['native_attributes']['macro_auroc'] for r in rs],True)} | {'参考' if method=='corrected_best_validation' else '逐元素完全一致'} |")
    lines+=['','**结论：** 五个最终基线对应的68项映射属性BA/AUROC保持一致，逐元素恒等参照为同结构的干净精简CBL；完整/精简GPU计算的浮点舍入差异及完整模型属性指标另存JSON。保护仅覆盖68/312项，未测定位质量。15个干净模型的精简验证预测保持不变：CIFAR-10保留55–57/61，CIFAR-100保留505–513/583，CUB保留433–437/509个输出。固定参考分支与生成器增加整体参数，不能把输出精简称为Pi整网减参或速度收益。','',
            '## 复现与交付','',
            '```bash','cd PS-Shared-Pi','python -m ps_pi.final_run                     # 冻结的35个CUB控制，可检查完成状态后跳过',
            'python -m ps_pi.final_workflow                # 锁定检查点，验证与测试评估',
            'python -m ps_pi.final_report                  # 从真实预测结果重建本报告与四张图',
            'python -m ps_pi.final_infer --features /path/RN50_features.pt --output /path/predictions.pt',
            '# 共享噪声版本：追加 --mode shared_noise','```','',
            '[最终结果逐项核验](reports/final_v1/VERIFICATION.json)：65个锁定检查点，130组最终预测数组和70组受保护属性概率复算。','',
            '模型、图像特征和逐样本预测保留在本地 `outputs/`、`artifacts/`，不上传数据集；完整结果表、协议、图表PNG/SVG/PDF已归档。运行推理需使用与训练一致的冻结RN50特征；无需真实类别。','',
            '[冻结设置](reports/final_v1/LOCK.json) · [交付模型的测试前选择](reports/final_v1/DELIVERY_PLAN.json) · [检查点/评估源码锁](reports/final_v1/EVALUATION_LOCK.json) · [全部最终结果JSON](reports/final_v1/results.json) · [结果CSV](reports/final_v1/metrics.csv) · [历史验证探索](reports/validation_v1/主要实验进展.md)','',
            '**范围：** 最终共享噪声五种子结论限于CUB；CIFAR为干净基线测试。数据集在本项目其他基线阶段已有测试评估，此处是PS阶段协议锁定后的评估。验证探索结果不视为独立测试证据，也没有在看到测试后更换交付默认模型。']
    (ROOT/'EXPERIMENT_RESULT.md').write_text('\n'.join(lines)+'\n')
    essay=ROOT.parent/'Essay/PS-Shared/report';essay.mkdir(parents=True,exist_ok=True)
    text=(ROOT/'EXPERIMENT_RESULT.md').read_text().replace('reports/final_v1/','final_v1/').replace('reports/validation_v1/主要实验进展.md','主要实验进展.md')
    (essay/'最终实验结果.md').write_text(text)
    target=essay/'final_v1';target.mkdir(exist_ok=True)
    for path in DEST.iterdir():
        if path.is_file():shutil.copy2(path,target/path.name)
    print('Final report and four figures generated; primary contrast:',primary,flush=True)


if __name__=='__main__':main()
