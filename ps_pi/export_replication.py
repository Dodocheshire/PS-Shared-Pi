"""Paired adaptation-seed checks, explicitly conditional on a fixed full baseline."""
import csv
import json
import numpy as np
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import ROOT, atomic_json, digest

MODES = [('adapter_matched', 'Matched adapter', '#CC79A7'),
         ('diag_noise', 'Diagonal noise', '#56B4E9'),
         ('shared_no_sampling', 'Sampling disabled', '#009E73'),
         ('shared_noise', 'Shared noise', '#D55E00')]


def main():
    report = ROOT.parent / 'Essay/PS-Shared/report'
    rows = []
    sources = []
    for seed in range(5):
        version = 'validation_joint_feature_mixup_v10' if seed == 0 else 'validation_feature_mixup_replication_v11'
        folder = ROOT / 'outputs/cub' / version
        audit = folder.parent / (version + '_audit')
        assert json.loads((audit / 'status.json').read_text())['phase'] == 'complete'
        summary = json.loads((audit / 'summary.json').read_text())
        for mode, _, _ in MODES:
            name = f'{mode}_seed{seed}_cw1_lr1e-05'
            case = next(r for r in summary['rows'] if r['case'] == name)
            config = case['config']
            assert config['seed'] == seed and config['baseline_seed'] == 42
            assert config['concept_weight'] == 1 and config['cbl_lr'] == 1e-5 and config['mixup_alpha'] == .2
            initial_path = folder / name / 'metrics.json'
            initial = json.loads(initial_path.read_text())['metrics']
            mc = case['mc_runs']
            row = {'adaptation_seed': seed, 'baseline_seed': 42, 'mode': mode,
                   'case': name, 'original_selection_eligible': initial['selection_eligible'],
                   'best_epoch': case['best_epoch'],
                   'accuracy_mc_average': float(np.mean([m['accuracy'] for m in mc])),
                   'accuracy_mc_min': min(m['accuracy'] for m in mc),
                   'accuracy_mc_max': max(m['accuracy'] for m in mc),
                   'nll_mc_average': float(np.mean([m['nll'] for m in mc])),
                   'ece_mc_average': float(np.mean([m['ece15'] for m in mc])),
                   'same_model_sampling_off_accuracy': case['same_model_mean']['accuracy'],
                   'native_ba_delta_pp': case['native_ba_delta_pp'],
                   'native_auc_delta_pp': case['native_auc_delta_pp'],
                   'checkpoint_sha256': case['checkpoint_sha256']}
            rows.append(row)
            for path in [initial_path, audit / name / 'metrics.json']:
                sources.append({'path': str(path.relative_to(ROOT.parent)), 'sha256': digest(path)})
    grouped = {mode: [next(r for r in rows if r['mode'] == mode and r['adaptation_seed'] == seed)
                      for seed in range(5)] for mode, _, _ in MODES}
    shared = np.array([r['accuracy_mc_average'] for r in grouped['shared_noise']])
    contrasts = []
    for mode, name, _ in MODES[:-1]:
        control = np.array([r['accuracy_mc_average'] for r in grouped[mode]])
        delta = 100 * (shared - control)
        mean = float(delta.mean())
        half = float(t.ppf(.975, 4) * delta.std(ddof=1) / np.sqrt(5))
        contrasts.append({'control': mode, 'name': name, 'paired_delta_pp': delta.tolist(),
                          'mean_delta_pp': mean, 'conditional_t95_pp': [mean-half, mean+half],
                          'positive_pairs': int((delta > 1e-5).sum()),
                          'all_original_selection_eligible': all(r['original_selection_eligible'] for r in grouped[mode]+grouped['shared_noise'])})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={'width_ratios': [1.2, 1]})
    fig.subplots_adjust(left=.08, right=.98, bottom=.29, top=.82, wspace=.56)
    for mode, name, color in MODES:
        values = 100*np.array([r['accuracy_mc_average'] for r in grouped[mode]])
        axes[0].plot(range(5), values, marker='o', color=color, label=name, linewidth=1.2, markersize=4)
    axes[0].set_xticks(range(5));axes[0].set_xlabel('Adaptation seed (full baseline seed = 42)')
    axes[0].set_ylabel('Validation accuracy (%)')
    for i, contrast in enumerate(contrasts):
        mean = contrast['mean_delta_pp'];lo, hi = contrast['conditional_t95_pp']
        axes[1].errorbar(mean, i, xerr=[[mean-lo], [hi-mean]], fmt='o', color='#D55E00', capsize=4)
    axes[1].set_yticks(range(3), [f"vs {c['name']}" for c in contrasts]);axes[1].invert_yaxis()
    axes[1].axvline(0, color='#777777', linestyle='--', linewidth=1)
    axes[1].set_xlabel('Paired accuracy delta (pp)\nConditional 95% t interval', fontsize=9)
    for ax in axes:
        ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Feature mixup: five adaptation seeds, one fixed complete baseline', fontsize=11)
    axes[0].legend(ncol=2, fontsize=7, loc='upper center', bbox_to_anchor=(.5,-.27), frameon=False)
    fig.text(.5,.015,'Exploratory validation evidence; two independent 256-draw checks averaged per seed. Intervals do not cover full-baseline or test uncertainty.',ha='center',fontsize=7)
    for ext in ['png','svg','pdf']:
        path=report/f'05_conditional_seed_replication.{ext}';fig.savefig(path,dpi=220)
        if ext=='svg':path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    plt.close(fig)
    with (report/'conditional_seed_replication.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    atomic_json({'baseline_seed':42,'adaptation_seeds':list(range(5)),
                 'new_training_runs':16,'reference_runs':4,'test_used':False,'final_evidence':False,
                 'score':'mean of two independent MC accuracy checks; not a probability ensemble',
                 'interval':'paired Student t, df4; exploratory and conditional on shared baseline, bank and validation set',
                 'selection':'no seed/case omission, no reselection from independent MC',
                 'rows':rows,'contrasts':contrasts,'sources':sources},report/'conditional_seed_replication.json')
    print(json.dumps(contrasts,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
