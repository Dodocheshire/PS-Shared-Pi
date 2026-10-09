"""Show every registered V21 compact control, including gate failures."""
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from .common import ROOT, atomic_json, digest

METHODS=[('clean','Clean joint','#555555'),('mean_only','Mean only','#0072B2'),
         ('adapter_matched','Matched adapter','#CC79A7'),('shared_no_sampling','Sampling disabled','#009E73'),
         ('diag_noise','Diagonal noise','#56B4E9'),('shared_noise','Shared noise','#D55E00'),
         ('fixed_structured','Fixed structured','#E69F00'),('fixed_isotropic','Fixed isotropic','#999999')]


def main():
    report=ROOT.parent/'Essay/PS-Shared/report'
    folder=ROOT/'outputs/cub/validation_protected_signed_v21';audit=folder.parent/(folder.name+'_audit')
    status=audit/'status.json'
    if not status.exists() or json.loads(status.read_text())['phase']!='complete':
        atomic_json({'completed_training_runs':0,'pending':True,'test_used':False,
                     'scope':'Await complete registered 16-case training and independent MC/native audits'},report/'signed_study.json')
        return
    summary=json.loads((audit/'summary.json').read_text())
    if len(summary['rows'])!=16 or summary['protocol']['test_used']:
        raise RuntimeError('Expected complete validation-only compact cohort')
    records=[]
    for row in summary['rows']:
        source=folder/row['case']/'metrics.json';d=json.loads(source.read_text());m=d['metrics'];b=d['original_baseline']
        gates={'accuracy':m['correct']>=b['correct'],'nll':m['nll']<=b['nll'],
               'ece':m['ece15']<=b['ece15'],'pseudo_concept_bce':m['concept_bce']<=b['concept_bce'],
               'drift':m['concept_drift_rms']<=.3,'native_ba':m['native_balanced_accuracy']+1e-12>=b['native_balanced_accuracy'],
               'native_auroc':m['native_auroc']+1e-12>=b['native_auroc']}
        if all(gates.values())!=m['selection_eligible']:raise RuntimeError('V21 selection gate mismatch')
        if digest(folder/row['case']/'model.pt')!=row['checkpoint_sha256']:raise RuntimeError('Frozen checkpoint changed')
        if not row['native_protection_verified'] or row['protected_probability_max_abs_delta']!=0.:
            raise RuntimeError('Protected native responses changed')
        runs=row['mc_runs']
        records.append({'case':row['case'],'mode':row['mode'],'selection_eligible':m['selection_eligible'],
                        'selection_correct':m['correct'],'selection_nll':m['nll'],'selection_epoch':m['best_epoch'],
                        'accuracy_min_percent':100*min(r['accuracy'] for r in runs),
                        'accuracy_max_percent':100*max(r['accuracy'] for r in runs),
                        'delta_original_min_pp':min(row['delta_baseline_pp_range']),
                        'delta_original_max_pp':max(row['delta_baseline_pp_range']),
                        'nll_min':min(r['nll'] for r in runs),'nll_max':max(r['nll'] for r in runs),
                        'ece_min':min(r['ece15'] for r in runs),'ece_max':max(r['ece15'] for r in runs),
                        'native_ba_delta_pp':row['native_ba_delta_pp'],'native_auroc_delta_pp':row['native_auc_delta_pp'],
                        'same_model_mean_accuracy_percent':100*row['same_model_mean']['accuracy'],
                        'concept_count':m['concept_count'],'failed_gates':','.join(k for k,v in gates.items() if not v),
                        'checkpoint_sha256':row['checkpoint_sha256'],'source_metrics_sha256':digest(source),
                        'protected_probability_max_abs_delta':row['protected_probability_max_abs_delta'],
                        'native_protection_verified':row['native_protection_verified'],
                        'test_used':False})
    selected={}
    for mode,_,_ in METHODS:
        allowed=[r for r in records if r['mode']==mode and r['selection_eligible']]
        if allowed:selected[mode]=max(allowed,key=lambda r:(r['selection_correct'],-r['selection_nll']))
    fig,axes=plt.subplots(1,3,figsize=(11.8,5.7))
    fig.subplots_adjust(left=.16,right=.98,top=.85,bottom=.23,wspace=.34)
    for index,(mode,name,color) in enumerate(METHODS):
        cases=sorted([r for r in records if r['mode']==mode],key=lambda r:r['case'])
        for offset,row in zip([-.12,.12],cases):
            lo=row['delta_original_min_pp'];hi=row['delta_original_max_pp'];mid=(lo+hi)/2
            marker='o' if row['selection_eligible'] else 'x'
            axes[0].errorbar(mid,index+offset,xerr=[[mid-lo],[hi-mid]],marker=marker,color=color,ls='none',capsize=2,ms=5)
            for ax,key,base,multiplier in zip(axes[1:],['nll','ece'],[summary['baseline']['nll'],summary['baseline']['ece15']],[1.,100.]):
                lo=(row[key+'_min']-base)*multiplier;hi=(row[key+'_max']-base)*multiplier;mid=(lo+hi)/2
                ax.errorbar(mid,index+offset,xerr=[[mid-lo],[hi-mid]],marker=marker,color=color,ls='none',capsize=2,ms=5)
    for ax in axes:
        ax.axvline(0,color='#777777',ls='--',lw=1);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
        ax.set_ylim(len(METHODS)-.5,-.5);ax.spines[['top','right']].set_visible(False)
        ax.xaxis.set_major_locator(MaxNLocator(4));ax.tick_params(axis='x',labelsize=8)
    axes[0].set_yticks(range(len(METHODS)),[n for _,n,_ in METHODS])
    for ax in axes[1:]:ax.set_yticks(range(len(METHODS)),[])
    for ax,label in zip(axes,['Accuracy delta (pp)','NLL delta','ECE delta (pp)']):ax.set_xlabel(label)
    fig.suptitle('Observed signed covariance: all 16 registered validation controls',fontsize=11,y=.95)
    fig.legend(handles=[Line2D([0],[0],marker='o',color='#333333',ls='none',label='Eligible at original selection'),
                        Line2D([0],[0],marker='x',color='#333333',ls='none',label='Rejected; retained as failure')],
               loc='lower center',bbox_to_anchor=(.5,.065),ncol=2,frameon=False,fontsize=8)
    fig.text(.5,.025,'Baseline seed42 / adaptation seed0. Two configurations per method. 68 native attributes exactly preserved. MC256 ranges are not confidence intervals.',ha='center',fontsize=7.5)
    for ext in ['png','svg','pdf']:fig.savefig(report/f'11_protected_signed.{ext}',dpi=220)
    plt.close(fig)
    data={'completed_training_runs':16,'pending':False,'records':records,'selected':selected,
          'selection_eligible_runs':sum(r['selection_eligible'] for r in records),
          'setting':'classification/pseudo-concept objective without native attribute loss; train native labels determine mapping/protected cohort/thresholds; same auxiliary constraint for all controls',
          'cohort_sha256':digest(audit/'summary.json'),'source_sha256':digest(__file__),
          'figures':{ext:digest(report/f'11_protected_signed.{ext}') for ext in ['png','svg','pdf']},
          'test_used':False,'final_five_seed_evidence':False}
    atomic_json(data,report/'signed_study.json')
    with (report/'signed_study.csv').open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(records[0]),lineterminator='\n');w.writeheader();w.writerows(records)
    lines=['# 训练观测的有符号共享结构噪声','',
           '**目的：** 只替换共享协方差的方向：在原共享类别支持中估计训练弱视图变化的主方向，允许负相关；保留上一轮逐概念初始方差、原均值解码器、参数、属性保护、预算和质量门槛。全部方法使用同一精简CBL、冻结头、数据顺序和预算；本轮没有原生属性训练损失，但使用训练属性标注确定保护映射/集合和阈值；这是同条件的辅助解释约束，不能当作纯原始PS-CBM监督设置。','',
           '**协议：** 协方差因子来自训练弱视图变化，按原共享支持分组求最大特征向量，再逐行归一化保持初始方差；均值仍使用原未改符号的因子。同一生成器的first、second及group输出共同产生均值；对应second/group仍产生尺度。强适配器、共享、对角及关闭采样的均值函数完全相同，参数数量不变。仅均值保留first分支作为弱对照。分类损失对MC平均类别概率计算软目标NLL。两配置×八方法、40轮、四次概念采样；CBL每图仅计算一次。噪声条件是当前干净概念的停止梯度值；条件协方差为diag(σ²)+B diag(γ²) Bᵀ，B由原概念共享类别关系确定，推理不用真实类别。均值上限0.3，与尺度上限相同。固定CBL隐藏层及161个属性输出行，屏蔽对应均值和协方差，Adam更新后恢复受保护行；保留标准化及分类头支持。残差重构及受保护通道的解析概率保证零扰动恒等。','',
           '**选模：** ACC/NLL/ECE/伪概念BCE不退化、漂移≤0.3、原生BA/AUROC不退化，再按原验证正确数/NLL选epoch及配置。固定选择后用MC256×两种子复核，不重新选模。','',
           '![全部精简对照](11_protected_signed.png)','',
           '| 方法 | 原选择合格配置 | 独立MC验证ACC（%） | 原生BA变化（pp） | 原生AUROC变化（pp） |',
           '|---|---|---:|---:|---:|']
    for mode,name,_ in METHODS:
        row=selected.get(mode)
        if row is None:lines.append(f'| {name} | 无合格配置 | — | — | — |');continue
        lo=row['accuracy_min_percent'];hi=row['accuracy_max_percent']
        acc=f'{lo:.2f}' if abs(lo-hi)<1e-5 else f'{lo:.2f}–{hi:.2f}'
        lines.append(f"| {name} | {row['case']} | {acc} | {row['native_ba_delta_pp']:+.3f} | {row['native_auroc_delta_pp']:+.3f} |")
    shared=selected.get('shared_noise')
    deterministic=[r for m,r in selected.items() if m in ['clean','mean_only','adapter_matched','shared_no_sampling']]
    if shared is None:
        conclusion='共享噪声没有通过全部预登记门槛的配置，不能将失败模型的高分当成成功。'
    else:
        best=max(deterministic,key=lambda r:(r['selection_correct'],-r['selection_nll']))
        lo=shared['accuracy_min_percent']-best['accuracy_max_percent'];hi=shared['accuracy_max_percent']-best['accuracy_min_percent']
        conclusion=f"共享噪声相对本轮最强合格确定性对照的独立MC差值为{lo:+.2f}–{hi:+.2f} pp。这是单基线、单适配种子的探索；即使为正，也需要预先锁定后独立五种子复验。"
    lines+=['',f'**结论：** {conclusion}','',
            '**属性保护核验：** 每个检查点以MC256重算受保护概念概率，并与原始基线逐元素比对；审计要求概率响应完全相同、68项属性的BA/AUROC完全相同，否则停止。该恒等只覆盖固定映射的68/312项，不能推断其余属性或定位质量。','',
            '**参数口径：** 全部控制分配同一193,066参数生成器。原精简CBL+稠密头为1,350,571参数，连生成器为1,543,637参数，高于原509输出干净CBL+头的1,404,531；减少概念输出不等于Pi整网参数更少，也没有时延/显存测量。冻结图像编码器不计入这些数值。','',
            '**范围与限制：** 精简本身是各对照共同的干净模型效率改进，不是随机噪声收益。伪概念BCE在保留的433输出上计算，与509输出旧队列绝对值不能直接比较。直接概念采样使其期望等于同检查点均值推理，但均值logits与平均类别概率仍不同。结构零相关仅指给定图像的条件协方差。真实属性覆盖68/312项，仍没有定位证据。','',
            '[训练协方差诊断](signed_geometry_diagnostic.json)、[全部配置及失败门槛](signed_study.csv)、[来源哈希](signed_study.json)、[精简基线核验](概念输出精简.md)。']
    (report/'有符号共享噪声.md').write_text('\n'.join(lines)+'\n')
    print('Exported all 16 compact noise controls',flush=True)


if __name__=='__main__':main()
