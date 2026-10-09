"""Encode official held-out tests only after final protocol registration."""
import argparse
import json
from pathlib import Path
import torch
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
from .common import ROOT, atomic_json, atomic_save, digest, gpu_lease
from .final_lock import verify_lock, PATH
from .prepare import clip, encode
from .splits import official_indices


def native_annotations(source,indices):
    annotations=ROOT.parent/'D-Pi/artifacts/annotations'
    root=annotations/'CUB_200_2011'
    names={line.split()[1]:int(line.split()[0]) for line in (root/'images.txt').read_text().splitlines()}
    split={int(a):int(b) for a,b in (line.split() for line in (root/'train_test_split.txt').read_text().splitlines())}
    image_ids=[names['/'.join(Path(source.samples[i][0]).parts[-2:])] for i in indices]
    if any(split[i]!=0 for i in image_ids):raise RuntimeError('Native test annotations include training images')
    lookup={value:i for i,value in enumerate(image_ids)}
    count=len((annotations/'attributes.txt').read_text().splitlines())
    positive=torch.zeros(len(indices),count,dtype=torch.bool)
    certainty=torch.zeros(len(indices),count,dtype=torch.uint8)
    seen=torch.zeros(len(indices),count,dtype=torch.bool)
    annotation_file=root/'attributes/image_attribute_labels.txt'
    for line in annotation_file.read_text().splitlines():
        fields=line.split();image_id=int(fields[0])
        if image_id not in lookup:continue
        row=lookup[image_id];aid=int(fields[1])-1
        if seen[row,aid]:raise RuntimeError('Duplicate native test annotation')
        positive[row,aid]=bool(int(fields[2]));certainty[row,aid]=int(fields[3]);seen[row,aid]=True
    if not seen.all():raise RuntimeError('Missing native test annotations')
    return {'positive':positive,'certainty':certainty,'indices':torch.tensor(indices),
            'image_ids':torch.tensor(image_ids),'annotation_sha256':digest(annotation_file),
            'split':'test','thresholds_fitted':False}


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['cifar10','cifar100','cub'],required=True)
    args=p.parse_args();verify_lock();torch.set_num_threads(8)
    directory=ROOT/'artifacts'/args.dataset
    manifest=json.loads((directory/'manifest.json').read_text());root=Path(manifest['data_root'])
    if args.dataset=='cub':train,test=ImageFolder(root/'CUB/train'),ImageFolder(root/'CUB/test')
    else:
        cls=CIFAR10 if args.dataset=='cifar10' else CIFAR100
        train,test=cls(root,train=True,download=False),cls(root,train=False,download=False)
    indices,split_hashes=official_indices(args.dataset,train,test)
    if any(indices[s]!=manifest['indices'][s] for s in ['train','val']):raise RuntimeError('Training split mismatch')
    output=directory/'final_test_v1';output.mkdir(exist_ok=True)
    protocol={'dataset':args.dataset,'final_lock_sha256':digest(PATH),'manifest_sha256':digest(directory/'manifest.json'),
              'source_sha256':digest(__file__),'split_file_sha256':split_hashes,'split':'test',
              'n':len(indices['test']),'test_used_for_selection':False,'test_used_for_evaluation':True,
              'encoder':'same frozen CLIP RN50 visual.float with identity attnpool projection as training',
              'batch_size':256,'workers':8,'data_root':str(root)}
    guard=output/'protocol.json'
    if guard.exists() and json.loads(guard.read_text())!=protocol:raise RuntimeError('Test preparation protocol changed')
    atomic_json(protocol,guard)
    if not (output/'test_target.pt').exists():
        with gpu_lease(f'Locked held-out test encoding {args.dataset}'):
            model,transform=clip.load('RN50',device='cuda');visual=model.visual
            dim=visual.attnpool.c_proj.in_features
            if dim!=2048:raise RuntimeError('Unexpected RN50 feature width')
            identity=torch.nn.Linear(dim,dim,device='cuda',dtype=torch.float16)
            with torch.no_grad():identity.weight.copy_(torch.eye(dim,device='cuda'));identity.bias.zero_()
            visual.attnpool.c_proj=identity;visual.float().eval().requires_grad_(False)
            atomic_save(encode(test,indices['test'],transform,visual,256,8,f'{args.dataset} locked test'),output/'test_target.pt')
    if args.dataset=='cub' and not (output/'native_annotations.pt').exists():atomic_save(native_annotations(test,indices['test']),output/'native_annotations.pt')
    data=torch.load(output/'test_target.pt',weights_only=True)
    if not torch.equal(data['indices'],torch.tensor(indices['test'])) or not torch.equal(data['labels'],torch.tensor([test.targets[i] for i in indices['test']])):
        raise RuntimeError('Test order or labels mismatch')
    atomic_json({'phase':'complete','n':len(data['labels']),'test_target_sha256':digest(output/'test_target.pt'),**protocol},output/'status.json')


if __name__=='__main__':main()
