"""Separate exact mean-concept effects from Monte Carlo class-probability effects."""
import argparse
import json
import torch
from torch.nn import functional as F
from .common import ROOT,atomic_json,atomic_save,digest,official_module
from .joint_noise import FeatureNoise
from .expected_concepts import expected_concepts

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',default='cub')
    p.add_argument('--input',default='outputs/cub/validation_joint_residual_v6');args=p.parse_args()
    torch.set_num_threads(8);folder=ROOT/args.input;paths=sorted(folder.glob('*/model.pt'))
    if len(paths)!=16:raise RuntimeError('Expected complete 16-case cohort')
    if any(json.loads((path.parent/'status.json').read_text())['phase']!='complete' for path in paths):raise RuntimeError('Cohort incomplete')
    output=folder.parent/(folder.name+'_mean_audit');output.mkdir(exist_ok=True)
    directory=ROOT/'artifacts'/args.dataset;baseline_path=ROOT/f'outputs/{args.dataset}/baseline/seed42/corrected_best_validation.pt'
    pack=torch.load(baseline_path,weights_only=False,map_location='cpu')
    reference_support=pack['weight'].ne(0)
    data=torch.load(directory/'val_target.pt',weights_only=True,map_location='cpu');x=data['features'];y=data['labels']
    context=torch.load(directory/'baseline_concepts'/digest(baseline_path)/'val.pt',weights_only=True)
    bank=torch.load(directory/'joint_feature_residual_v6/directions.pt',weights_only=False)
    CBL=official_module('model.cbl').ConceptBottleneckLayer
    cbl=CBL(x.shape[1],len(pack['concepts']),pack['config']['cbl_layer_num'],pack['config']['cbl_bias'])
    rows=[]
    for path in paths:
        model=torch.load(path,weights_only=False,map_location='cpu');config=model['protocol']['config']
        assert model['protocol']['baseline_sha256']==digest(baseline_path)
        assert model['protocol']['directions_sha256']==digest(directory/'joint_feature_residual_v6/directions.pt')
        cbl.load_state_dict(model['state']['cbl'])
        if 'head' in model['state']:
            head=model['state']['head']
            assert torch.equal(head['support'],reference_support)
            pack['weight']=head['weight']*head['support'];pack['bias']=head['bias']
        noise=FeatureNoise(bank,len(pack['bias']),config['mode'],config['hidden'],config['max_mean'],config['initial_scale'],config['min_scale'],config['max_scale'])
        noise.load_state_dict(model['state']['noise'])
        exact=expected_concepts(cbl,noise,x,context)
        no_sampling=expected_concepts(cbl,noise,x,context,scale_multiplier=0.)
        probabilities=F.linear((exact-pack['train_mean'])/pack['train_std'],pack['weight'],pack['bias']).softmax(-1)
        mean_probabilities=F.linear((no_sampling-pack['train_mean'])/pack['train_std'],pack['weight'],pack['bias']).softmax(-1)
        row={'case':path.parent.name,'mode':config['mode'],'device':'cpu','split':'val','test_used':False,
             'expected_concept_accuracy':float(probabilities.argmax(1).eq(y).float().mean()),
             'expected_concept_nll':float(-probabilities[torch.arange(len(y)),y].clamp_min(1e-30).log().mean()),
             'same_cbl_without_sampling_accuracy':float(mean_probabilities.argmax(1).eq(y).float().mean()),
             'expected_concept_mean_shift_rms':float(((exact-no_sampling)/pack['train_std']).square().mean().sqrt()),
             'class_changes_from_expected_concept_shift':int((probabilities.argmax(1)!=mean_probabilities.argmax(1)).sum()),
             'checkpoint_sha256':digest(path)}
        independent_path=folder.parent/(folder.name+'_audit')/path.parent.name/'metrics.json'
        if independent_path.exists():
            mc=json.loads(independent_path.read_text())['mc_runs']
            row['independent_mc_accuracy']=[m['accuracy'] for m in mc]
            row['mc_minus_exact_concept_mean_pp']=[100*(m['accuracy']-row['expected_concept_accuracy']) for m in mc]
        initial=json.loads((path.parent/'metrics.json').read_text())['metrics']
        row['selection_mc_minus_exact_concept_mean_pp']=100*(initial['accuracy']-row['expected_concept_accuracy'])
        destination=output/path.parent.name;destination.mkdir(exist_ok=True)
        atomic_save({'probabilities':probabilities,'labels':y,'indices':data['indices']},destination/'predictions.pt')
        atomic_json(row,destination/'metrics.json');rows.append(row)
        print(row['case'],round(100*row['expected_concept_accuracy'],3),'MC delta',row.get('mc_minus_exact_concept_mean_pp',row['selection_mc_minus_exact_concept_mean_pp']),flush=True)
    atomic_json({'rows':rows,'source_sha256':digest(__file__),'analytic_source_sha256':digest(ROOT/'ps_pi/expected_concepts.py'),
                 'test_used':False,'concept_expectation_exact_for':'affine or Linear-ReLU-Linear CBL with conditional Gaussian feature noise',
                 'limits':'Frozen-checkpoint validation counterfactual computed on CPU. Mean logits and marginal class probabilities are different predictions. Does not test the causal effect of training sampling or replace a retrained capacity control.'},output/'summary.json')
    atomic_json({'phase':'complete','completed':len(rows),'test_used':False},output/'status.json')

if __name__=='__main__':main()
