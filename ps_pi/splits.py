"""Map the pinned paper split filenames onto native datasets, without copying images."""
from pathlib import Path
import pickle
import torch
from .common import ROOT, digest

def source_names(source):
    if hasattr(source, 'samples'):
        root=Path(source.root)
        return [str(Path(p).relative_to(root)) for p,_ in source.samples]
    names=[]
    listing=source.train_list if source.train else source.test_list
    for filename,_ in listing:
        path=Path(source.root)/source.base_folder/filename
        with path.open('rb') as stream: data=pickle.load(stream,encoding='latin1')
        names.extend(data['filenames'])
    assert len(names)==len(source)
    assert len(set(names))==len(names)
    return names

def official_indices(dataset,train,test):
    identifier={'cifar10':'CIFAR10','cifar100':'CIFAR100','cub':'CUB200'}[dataset]
    directory=ROOT/'official/dataset'/identifier/'splits'
    maps={split:{name:i for i,name in enumerate(source_names(source))}
          for split,source in [('train',train),('test',test)]}
    indices={}; hashes={}
    class_names=(ROOT/'official/data/classes_name'/f'{identifier}_classes.txt').read_text().splitlines()
    class_map={name:i for i,name in enumerate(class_names)}
    for split in ['train','val','test']:
        path=directory/f'class2images_{split}.p'
        with path.open('rb') as stream: groups=pickle.load(stream)
        source=test if split=='test' else train; lookup=maps['test' if split=='test' else 'train']
        selected=[]
        for label,names in groups.items():
            if label not in class_map: raise RuntimeError(f'Unknown official class: {label}')
            for name in names:
                if name not in lookup: raise RuntimeError(f'Cannot map official filename: {name}')
                idx=lookup[name]
                if source.targets[idx]!=class_map[label]: raise RuntimeError(f'Official/native label mismatch: {name}')
                selected.append(idx)
        if len(set(selected))!=len(selected): raise RuntimeError(f'Duplicate image in {split}')
        indices[split]=sorted(selected);hashes[split]=digest(path)
    assert not set(indices['train']) & set(indices['val'])
    assert set(indices['train']) | set(indices['val']) == set(range(len(train)))
    assert set(indices['test'])==set(range(len(test)))
    return indices,hashes
