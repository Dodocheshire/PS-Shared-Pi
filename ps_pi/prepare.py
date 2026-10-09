"""Full-data PS-CBM preparation; concept selection sees training images only."""
import argparse
import gc
from importlib.metadata import version
import json
from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
from .common import ROOT, atomic_json, atomic_save, digest, gpu_lease, official_module
from .splits import official_indices
clip = official_module("clip")

class Transformed(Dataset):
    def __init__(self, source, transform): self.source, self.transform = source, transform
    def __len__(self): return len(self.source)
    def __getitem__(self, i):
        image, label = self.source[i]
        return self.transform(image), label

def encode(source, indices, transform, encoder, batch_size, workers, name):
    loader = DataLoader(Subset(Transformed(source, transform), indices), batch_size=batch_size,
                        shuffle=False, num_workers=workers, pin_memory=True)
    values, labels = [], []; started = time.monotonic()
    with torch.no_grad():
        for step, (images, y) in enumerate(loader):
            values.append(encoder(images.cuda()).float().cpu()); labels.append(y)
            if step % 50 == 0:
                print(f'{name}: {min((step+1)*batch_size,len(indices))}/{len(indices)}, {time.monotonic()-started:.1f}s', flush=True)
    return {'features': torch.cat(values), 'labels': torch.cat(labels), 'indices': torch.tensor(indices)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['cifar10','cifar100','cub'], default='cifar10')
    parser.add_argument('--data-root', default='/home/pengcheng/VLG-CBM/datasets')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args(); torch.set_num_threads(8)
    config = json.loads((ROOT/'configs'/f'{args.dataset}_rn50.json').read_text())
    directory = ROOT/'artifacts'/args.dataset; directory.mkdir(parents=True,exist_ok=True)
    if args.dataset in ['cifar10','cifar100']:
        cls = CIFAR10 if args.dataset=='cifar10' else CIFAR100
        train, test = cls(args.data_root,train=True,download=False), cls(args.data_root,train=False,download=False)
    else:
        train, test = ImageFolder(Path(args.data_root)/'CUB/train'), ImageFolder(Path(args.data_root)/'CUB/test')
    assert train.class_to_idx == test.class_to_idx
    registered, split_hashes = official_indices(args.dataset, train, test)
    indices={split:registered[split] for split in ['train','val']}
    train_ids, val_ids = indices['train'], indices['val']
    assert not set(indices['train']) & set(indices['val'])
    assert set(indices['train']) | set(indices['val']) == set(range(len(train)))
    manifest={'dataset':args.dataset,'data_root':str(Path(args.data_root).resolve()),'classes':train.classes,
              'indices':indices,'split_protocol':'official filename mapping','split_file_sha256':split_hashes,'feature_dim':2048,'target':'CLIP RN50 visual.float(), identity attnpool.c_proj',
              'teacher':'CLIP ViT-B/16','encoding_batch_size':args.batch_size,'test_used_for_selection':False,
              'software':{name:version(name) for name in ['torch','torchvision','openai-clip','numpy','scikit-learn']}}
    old = directory/'manifest.json'
    if old.exists() and json.loads(old.read_text()) != manifest: raise RuntimeError('Preparation manifest changed; use a fresh artifact directory')
    atomic_json(manifest, old)
    official=official_module('utils_my')
    concept_path=ROOT/'official/data/generate_concept/concept'/f'{args.dataset.upper() if args.dataset!="cub" else "CUB200"}_concepts_gpt-4o_final.json'
    concepts=json.loads(concept_path.read_text())
    native=[name.replace('_',' ') for name in train.classes]
    classes_path=ROOT/'official/data/classes_name'/f'{args.dataset.upper() if args.dataset!="cub" else "CUB200"}_classes.txt'
    classes=classes_path.read_text().splitlines()
    if args.dataset.startswith('cifar') and native != classes: raise RuntimeError('Class order differs from official concept labels')
    with gpu_lease(f'PS-CBM prepare {args.dataset}'):
        status=directory/'status.json'
        if not all((directory/f'{split}_teacher.pt').exists() for split in indices) or not (directory/'text.pt').exists():
            teacher, transform=clip.load('ViT-B/16',device='cuda'); teacher.eval().requires_grad_(False)
            for split, ids in indices.items():
                path=directory/f'{split}_teacher.pt'
                if not path.exists():
                    atomic_json({'phase':'teacher','split':split,'pid':__import__('os').getpid()},status)
                    atomic_save(encode(train,ids,transform,teacher.encode_image,args.batch_size,args.workers,f'teacher {split}'),path)
            if not (directory/'text.pt').exists():
                with torch.no_grad(): text=teacher.encode_text(clip.tokenize(concepts['concepts']).cuda()).float().cpu()
                atomic_save(text,directory/'text.pt')
            del teacher; gc.collect(); torch.cuda.empty_cache()
        if not all((directory/f'{split}_target.pt').exists() for split in indices):
            model, transform=clip.load('RN50',device='cuda'); visual=model.visual
            dim=visual.attnpool.c_proj.in_features
            assert dim==2048
            identity=torch.nn.Linear(dim,dim,device='cuda',dtype=torch.float16)
            with torch.no_grad(): identity.weight.copy_(torch.eye(dim,device='cuda')); identity.bias.zero_()
            visual.attnpool.c_proj=identity; visual.float().eval().requires_grad_(False)
            for split, ids in indices.items():
                path=directory/f'{split}_target.pt'
                if not path.exists():
                    atomic_json({'phase':'penultimate','split':split,'pid':__import__('os').getpid()},status)
                    atomic_save(encode(train,ids,transform,visual,args.batch_size,args.workers,f'RN50 {split}'),path)
            del model,visual;gc.collect();torch.cuda.empty_cache()
        if not (directory/'concept_bank.pt').exists():
            t=torch.load(directory/'train_teacher.pt',weights_only=True);text=torch.load(directory/'text.pt',weights_only=True)
            image=torch.nn.functional.normalize(t['features'],dim=1);text=torch.nn.functional.normalize(text,dim=1)
            result=official.filter_and_merge_concepts(image@text.T,text,concepts['concepts'],concepts['concepts_to_class'],
                 t['labels'].tolist(),Tconf=config['Tconf'],Tmerge=config['Tmerge'],K=4,strategy='max',K_indep=config['K_indep'])
            names,text,groups,redirect=result
            if not names: raise RuntimeError('Empty concept bank')
            atomic_save({'concepts':names,'text':text,'class_sets':groups,'redirect':redirect,
                        'source_sha256':digest(concept_path),'selection_split':'train'},directory/'concept_bank.pt')
        bank=torch.load(directory/'concept_bank.pt',weights_only=False)
        for split in indices:
            path=directory/f'{split}_concept_labels.pt'
            if not path.exists():
                t=torch.load(directory/f'{split}_teacher.pt',weights_only=True)
                labels=official.generate_concept_labels(torch.nn.functional.normalize(t['features'],dim=1),
                            bank['text'],t['labels'].tolist(),bank['class_sets'],Tconf=config['Tconf'])
                atomic_save(labels,path)
        atomic_json({'phase':'complete','concept_count':len(bank['concepts']),'train_n':len(train_ids),'val_n':len(val_ids),
                     'test_loaded':False,'time':time.time()},status)
        print('Preparation complete',len(bank['concepts']),'concepts; no test features opened',flush=True)
if __name__=='__main__': main()
