"""Cache measured nuisance feature changes from registered training images only."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from .common import ROOT, atomic_json, atomic_save, digest, gpu_lease, official_module
from .splits import align_class_order


class AugmentedTraining(Dataset):
    def __init__(self, source, indices, transform, view, seed=13001):
        self.source, self.indices, self.transform = source, list(indices), transform
        self.view, self.seed = view, seed

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, row):
        index = int(self.indices[row])
        image, label = self.source[index]
        # Reproducible across worker counts, independent of loader/global RNG state.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.seed + 100000 * self.view + index)
            image = self.transform(image)
        return image, label


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--views', type=int, default=2)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--limit', type=int, default=0)
    args = p.parse_args()
    if args.views < 2 or args.limit < 0:
        raise ValueError('At least two views and a nonnegative debug limit required')
    torch.set_num_threads(8)
    directory = ROOT / 'artifacts/cub'
    manifest = json.loads((directory / 'manifest.json').read_text())
    if args.batch_size != manifest['encoding_batch_size']:
        raise ValueError('Use the original feature encoding batch size to match CUDA numerics')
    data = torch.load(directory / 'train_target.pt', weights_only=True)
    assert data['indices'].tolist() == manifest['indices']['train']
    indices = data['indices'][:args.limit] if args.limit else data['indices']
    source = ImageFolder(Path(manifest['data_root']) / 'CUB/train')
    align_class_order(source, manifest['classes'])
    assert torch.equal(torch.tensor([source.targets[int(i)] for i in indices]), data['labels'][:len(indices)])
    image_hash = hashlib.sha256()
    for index in indices:
        path = Path(source.samples[int(index)][0])
        image_hash.update(str(path.relative_to(source.root)).encode() + b'\0' + digest(path).encode())
    output = directory / ('debug_augmentation_pairs_v13' if args.limit else 'augmentation_pairs_v13')
    output.mkdir(exist_ok=True)
    protocol = {'dataset': 'cub', 'split': 'train', 'rows': len(indices), 'views': args.views,
                'debug': bool(args.limit), 'seed': 13001, 'batch_size': args.batch_size,
                'source_sha256': digest(__file__), 'splits_source_sha256': digest(ROOT/'ps_pi/splits.py'),
                'manifest_sha256': digest(directory/'manifest.json'),
                'training_features_sha256': digest(directory/'train_target.pt'),
                'training_image_content_sha256': image_hash.hexdigest(),
                'clip_weight_sha256': digest(Path.home()/'.cache/clip/RN50.pt'),
                'transform': 'original CLIP resize+center crop; horizontal flip p0.5; 224 crop with 8px reflection padding; original RGB/tensor/normalization',
                'encoder': 'frozen RN50 float penultimate2048, identity attnpool output projection',
                'val_images_opened': False, 'test_images_opened': False,
                'limits': 'Observed weak-view differences; no assumption of exact native-attribute or Gaussian manifold invariance.'}
    guard = output / 'protocol.json'
    if guard.exists() and json.loads(guard.read_text()) != protocol:
        raise RuntimeError('Augmentation cache protocol changed')
    atomic_json(protocol, guard)
    with gpu_lease('Training-only CUB weak-view feature pairs'):
        clip = official_module('clip')
        model, base_transform = clip.load('RN50', device='cuda')
        visual = model.visual
        dim = visual.attnpool.c_proj.in_features
        assert dim == 2048
        identity = torch.nn.Linear(dim, dim, device='cuda', dtype=torch.float32)
        with torch.no_grad():
            identity.weight.copy_(torch.eye(dim, device='cuda')); identity.bias.zero_()
        visual.attnpool.c_proj = identity
        visual.float().eval().requires_grad_(False)
        transform = transforms.Compose(base_transform.transforms[:2] + [
            transforms.RandomHorizontalFlip(.5),
            transforms.RandomCrop(224, padding=8, padding_mode='reflect')
        ] + base_transform.transforms[2:])
        with torch.no_grad():
            alignment_indices=data['indices'][:manifest['encoding_batch_size']]
            original = torch.stack([base_transform(source[int(i)][0]) for i in alignment_indices]).cuda()
            measured = visual(original).float().cpu()
            reference = data['features'][:len(measured)]
            relative_error = float((measured-reference).square().mean().sqrt()/reference.square().mean().sqrt())
            if relative_error > 3e-4:
                raise RuntimeError(f'Original RN50 features fail cache alignment: relative RMS {relative_error}')
        atomic_json({'rows_checked':len(measured),'relative_rms_error':relative_error,
                     'original_encoding_batch_size':manifest['encoding_batch_size'],
                     'alignment_uses_original_preparation_batch':True,
                     'split':'train','val_used':False,'test_used':False}, output/'original_alignment.json')
        for view in range(args.views):
            path = output / f'view{view}.pt'
            if path.exists():
                continue
            values, labels = [], []
            loader = DataLoader(AugmentedTraining(source, indices, transform, view), batch_size=args.batch_size,
                                shuffle=False, num_workers=args.workers, pin_memory=True)
            for step, (images, y) in enumerate(loader):
                with torch.no_grad():
                    values.append(visual(images.cuda()).float().cpu())
                labels.append(y)
                atomic_json({'phase':'encode','view':view,'views':args.views,
                             'rows':min((step+1)*args.batch_size,len(indices)), 'total_rows':len(indices),
                             'pid':os.getpid(),'time':time.time()},output/'status.json')
            assert torch.equal(torch.cat(labels), data['labels'][:len(indices)])
            atomic_save({'features':torch.cat(values),'labels':torch.cat(labels),'indices':indices,
                         'protocol':protocol,'view':view}, path)
        atomic_json({'phase':'complete','rows':len(indices),'views':args.views,'debug':bool(args.limit),
                     'view_sha256':[digest(output/f'view{v}.pt') for v in range(args.views)],
                     'val_used':False,'test_used':False,'time':time.time()},output/'status.json')
        print('Prepared training-only measured feature pairs',len(indices),args.views,flush=True)


if __name__ == '__main__':
    main()
