"""Conservative text matching and train-only thresholds for native CUB attributes."""
import argparse
import json
import re
from pathlib import Path
import numpy as np
import torch
from torchvision.datasets import ImageFolder
from .common import ROOT,atomic_json,atomic_save,digest

PARTS={'wing':['wing','wings'],'back':['back'],'belly':['belly'],'bill':['bill','beak'],
       'breast':['breast','chest'],'crown':['crown','cap'],'eye':['eye','eyes'],
       'forehead':['forehead'],'leg':['leg','legs'],'nape':['nape'],'throat':['throat'],
       'under_tail':['undertail','under tail'],'upper_tail':['uppertail','upper tail'],
       'underparts':['underparts'],'upperparts':['upperparts'],'primary':['plumage','body']}

def text_mapping(concepts,attribute_names):
    """Map literal single-color body-part phrases, excluding patterns and local marks."""
    mapping={}
    for aid,name in enumerate(attribute_names):
        kind,color=name.split('::')
        if not kind.endswith('_color'):continue
        part=kind[len('has_'):-len('_color')]
        if part not in PARTS:continue
        colors=['gray','grey'] if color=='grey' else [color]
        ids=[]
        for cid,concept in enumerate(concepts):
            text=concept.lower().replace('-',' ')
            if re.search(r'\b(and|bars?|stripes?|streak\w*|spott\w*|patch\w*|tips?|edges?|edged|bands?)\b',text):continue
            if any(re.search(r'\b'+re.escape(c)+r'\s+'+re.escape(p)+r'\b',text)
                   for c in colors for p in PARTS[part]):ids.append(cid)
        if ids:mapping[aid]=ids
    return mapping

def threshold_from_training(scores,positive,known):
    """Exact empirical balanced-accuracy threshold using only known training labels."""
    x=np.asarray(scores)[known];y=np.asarray(positive,dtype=bool)[known]
    if not y.any() or y.all():raise ValueError('Both known training classes required')
    order=np.argsort(-x,kind='stable');x=x[order];y=y[order]
    last=np.flatnonzero(np.r_[x[:-1]!=x[1:],True])
    tp=np.cumsum(y)[last];fp=np.cumsum(~y)[last]
    ba=.5*(tp/y.sum()+1-fp/(~y).sum())
    # Predict all negatives as an explicit tied candidate. Highest threshold wins ties.
    # Preserve the score dtype: a float64 successor can round back to the same
    # float32 value when the threshold is applied to PyTorch model scores.
    above_max=np.nextafter(x[0],np.array(np.inf,dtype=x.dtype))
    candidates=np.r_[above_max,x[last]]
    values=np.r_[.5,ba]
    return float(candidates[np.argmax(values)])

def main():
    p=argparse.ArgumentParser();p.add_argument('--annotations',type=Path,default=ROOT.parent/'D-Pi/artifacts/annotations')
    args=p.parse_args();torch.set_num_threads(8)
    directory=ROOT/'artifacts/cub';manifest=json.loads((directory/'manifest.json').read_text())
    source=ImageFolder(Path(manifest['data_root'])/'CUB/train')
    root=args.annotations/'CUB_200_2011'
    name_to_id={line.split()[1]:int(line.split()[0]) for line in (root/'images.txt').read_text().splitlines()}
    split_truth={int(a):int(b) for a,b in (line.split() for line in (root/'train_test_split.txt').read_text().splitlines())}
    names=[line.split(' ',1)[1] for line in (args.annotations/'attributes.txt').read_text().splitlines()]
    ids={s:[name_to_id['/'.join(Path(source.samples[i][0]).parts[-2:])] for i in manifest['indices'][s]] for s in ['train','val']}
    assert all(split_truth[i]==1 for rows in ids.values() for i in rows)
    assert not set(ids['train'])&set(ids['val'])
    lookup={image_id:(s,j) for s,rows in ids.items() for j,image_id in enumerate(rows)}
    output={s:{'image_ids':torch.tensor(rows),'indices':torch.tensor(manifest['indices'][s]),
               'positive':torch.zeros(len(rows),len(names),dtype=torch.bool),
               'certainty':torch.zeros(len(rows),len(names),dtype=torch.uint8)} for s,rows in ids.items()}
    # Skip non-train/val image IDs before interpreting the attribute values.
    with (root/'attributes/image_attribute_labels.txt').open() as stream:
        for line in stream:
            fields=line.split();image_id=int(fields[0])
            if image_id not in lookup:continue
            s,j=lookup[image_id];aid=int(fields[1])-1
            output[s]['positive'][j,aid]=bool(int(fields[2]));output[s]['certainty'][j,aid]=int(fields[3])
    baseline_path=ROOT/'outputs/cub/baseline/seed42/corrected_best_validation.pt'
    pack=torch.load(baseline_path,weights_only=False)
    context=torch.load(directory/'baseline_concepts'/digest(baseline_path)/'train.pt',weights_only=True)
    raw=context*pack['train_std']+pack['train_mean'];probabilities=raw.sigmoid().numpy()
    assignments=text_mapping(pack['concepts'],names);cohort={}
    for aid,cids in assignments.items():
        positive=output['train']['positive'][:,aid].numpy();known=(output['train']['certainty'][:,aid]>=3).numpy()
        if (positive&known).sum()<10 or (~positive&known).sum()<10:continue
        scores=probabilities[:,cids].max(1)
        cohort[str(aid)]={'name':names[aid],'concept_ids':cids,'concepts':[pack['concepts'][c] for c in cids],
                          'threshold':threshold_from_training(scores,positive,known),
                          'known_training_positive':int((positive&known).sum()),'known_training_negative':int((~positive&known).sum())}
    record={'mapping_rule':'literal color and part; exclude patterns, patches, tips and multiple-part conjunctions',
            'cohort_selection':'text match and >=10 known positives/negatives in train; no validation selection',
            'threshold_split':'train','certainty_minimum':3,'total_attributes':len(names),'mapped_attributes':len(assignments),
            'evaluated_cohort':cohort,'test_examples_evaluated':False,'baseline_sha256':digest(baseline_path),
            'source_sha256':digest(__file__),'attribute_names_sha256':digest(args.annotations/'attributes.txt'),
            'native_annotation_sha256':digest(root/'attributes/image_attribute_labels.txt'),
            'manifest_sha256':digest(directory/'manifest.json'),
            'limits':'Partial literal mapping, whole-image attribute prediction only; not localization or causal explanation.'}
    atomic_save(output,directory/'native_attributes_v1/train_val.pt')
    atomic_json(record,directory/'native_attributes_v1/protocol.json')
    print(f'Native CUB train/val prepared: {len(cohort)}/{len(names)} attributes, train-only thresholds; no test evaluation',flush=True)

if __name__=='__main__':main()
