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
    return names

def align_class_order(source, official_classes):
    """ImageFolder CUB uses numeric taxonomy order; official code uses alphabetical names."""
    def normalize(name):
        if len(name)>4 and name[:3].isdigit() and name[3]=='.': name=name[4:]
        return name.replace('_',' ').lower()
    normalized=[normalize(name) for name in source.classes]
    if len(set(normalized))!=len(normalized) or set(normalized)!=set(official_classes):
        raise RuntimeError('Native/official classes are not a bijection')
    destination={name:i for i,name in enumerate(official_classes)}
    mapping={i:destination[name] for i,name in enumerate(normalized)}
    source.targets=[mapping[int(label)] for label in source.targets]
    if hasattr(source,'samples'):
        source.samples=[(path,mapping[label]) for path,label in source.samples]
        source.imgs=source.samples
    source.classes=list(official_classes)
    source.class_to_idx=destination

def official_indices(dataset,train,test):
    identifier={'cifar10':'CIFAR10','cifar100':'CIFAR100','cub':'CUB200'}[dataset]
    directory=ROOT/'official/dataset'/identifier/'splits'
    class_names=(ROOT/'official/data/classes_name'/f'{identifier}_classes.txt').read_text().splitlines()
    for source in [train,test]: align_class_order(source,class_names)
    maps={}
    for split,source in [('train',train),('test',test)]:
        keys=list(zip(source.targets,source_names(source)))
        if len(set(keys))!=len(keys): raise RuntimeError('Ambiguous filename within the same class')
        maps[split]={key:i for i,key in enumerate(keys)}
    indices={}; hashes={}
    class_map={name:i for i,name in enumerate(class_names)}
    for split in ['train','val','test']:
        path=directory/f'class2images_{split}.p'
        with path.open('rb') as stream: groups=pickle.load(stream)
        source=test if split=='test' else train; lookup=maps['test' if split=='test' else 'train']
        selected=[]
        for label,names in groups.items():
            if label not in class_map: raise RuntimeError(f'Unknown official class: {label}')
            for name in names:
                key=(class_map[label],name)
                if key not in lookup: raise RuntimeError(f'Cannot map official filename in class {label}: {name}')
                idx=lookup[key]
                if source.targets[idx]!=class_map[label]: raise RuntimeError(f'Official/native label mismatch: {name}')
                selected.append(idx)
        if len(set(selected))!=len(selected): raise RuntimeError(f'Duplicate image in {split}')
        indices[split]=sorted(selected);hashes[split]=digest(path)
    assert not set(indices['train']) & set(indices['val'])
    assert set(indices['train']) | set(indices['val']) == set(range(len(train)))
    assert set(indices['test'])==set(range(len(test)))
    return indices,hashes
