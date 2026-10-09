"""Independent train-only signed geometry for each locked baseline seed."""
import json
import torch
from .common import ROOT, atomic_json, atomic_save, digest
from .compact_outputs import load_compact_checkpoint
from .protected_concept_forward import remapped_bank as positive_bank
from .signed_shared_geometry import estimate_factor


def folder(seed):
    return ROOT/f'artifacts/cub/final_geometry_v1/seed{seed}'


@torch.no_grad()
def prepare(seed):
    torch.set_num_threads(8)
    baseline=ROOT/f'outputs/cub/baseline/seed{seed}/corrected_best_validation.pt'
    compact=ROOT/f'outputs/cub/compact_baselines/seed{seed}/model.pt'
    cbl,pack=load_compact_checkpoint(compact)
    original=torch.load(baseline,map_location='cpu',weights_only=False)
    bank=positive_bank(original,pack)
    protocol={'baseline_seed':seed,'baseline_sha256':digest(baseline),
              'compact_checkpoint_sha256':digest(compact),'split':'train',
              'test_used':False,'validation_used':False,'source_sha256':digest(__file__)}
    path=folder(seed);path.mkdir(parents=True,exist_ok=True)
    if (path/'bank.pt').exists():
        if json.loads((path/'protocol.json').read_text())!=protocol:raise RuntimeError('Frozen final geometry changed')
        return
    train=torch.load(ROOT/'artifacts/cub/train_target.pt',weights_only=True)
    def concepts(x):
        return torch.cat([(cbl(b)-pack['train_mean'])/pack['train_std'] for b in x.split(2048)])
    clean=concepts(train['features']);residual=[]
    for view in range(2):
        item=torch.load(ROOT/f'artifacts/cub/augmentation_pairs_v13/view{view}.pt',weights_only=False)
        if not torch.equal(item['indices'],train['indices']) or item['protocol']['test_images_opened'] or item['protocol']['val_images_opened']:
            raise RuntimeError('Geometry requires aligned training views only')
        residual.append(concepts(item['features'])-clean)
    residual=torch.cat(residual).double();residual-=residual.mean(0);residual*=bank['active_mask']
    positive=bank['factor'].clone()
    bank['mean_factor']=positive
    bank['factor']=estimate_factor(residual.T@residual/len(residual),positive)
    bank['group_directions']=bank['factor'].T.contiguous()
    atomic_json(protocol,path/'protocol.json');atomic_save({'bank':bank,'protocol':protocol},path/'bank.pt')


def remapped_bank(original,compact):
    seed=compact['source_config']['seed']
    expected=positive_bank(original,compact)
    frozen=torch.load(folder(seed)/'bank.pt',weights_only=False,map_location='cpu')
    if frozen['protocol']['baseline_sha256']!=compact['protocol']['baseline_sha256']:raise RuntimeError('Geometry baseline mismatch')
    bank=frozen['bank']
    for key in ['directions','active_mask','feature_std']:
        if not torch.equal(bank[key],expected[key]):raise RuntimeError('Geometry protected support mismatch')
    if not torch.equal(bank['mean_factor'],expected['factor']):raise RuntimeError('Mean decoder mismatch')
    if bank['factor'][expected['factor']==0].ne(0).any():raise RuntimeError('Covariance sharing support mismatch')
    return bank
