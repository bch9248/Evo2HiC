# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# script to predict HC matrices with diffusion-based approaches.
# --------------------------------------------------------

import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from config import *
from utils import *
import json
import random
from torch.utils.data import DataLoader
import argparse

import pandas as pd

from dataset.hic_dna_dataset import HiC_DNA_Dataset
from dataset.hic_loader import HiC_Loader, create_loaders_with_index
from dataset.DNA_loader import DNA_Loader
from dataset.mappability_loader import Mappability_Loader
from dataset.normalizer import Normalizer
from model.create_CDNA2d import create_model
from model.CDNA2d import CDNA2d
from train.train_utils import cut2d, cut1d
from hic_utils import *
from inference.infer_utils import *
from inference.inference_CDNA2d import CDNA2d_data_predict_parser, load_model

def infer_embeddings(dataset : HiC_DNA_Dataset, model:CDNA2d, batch_size, remove_boundary=0):
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    device = 'cuda'
    model = model.to(device)

    HiC_valuess = []
    HiC_embedss = []
    DNA_embedss = []
    positionss = []

    model.eval()
    with torch.no_grad():
        pbar = tqdm(loader)
        for data in pbar:            
            for k in data:
                data[k] = data[k].to(device)

            mappability_row = data['mappability_row']
            mappability_col = data['mappability_col']

            if torch.numel(mappability_col) == 0:
                mappability_col = mappability_row[:, :1, ...].expand_as(mappability_row)

            mappability_row = mappability_row.flatten(0,2)
            mappability_col = mappability_col.flatten(0,2)

            HiC_embeds, DNA_embeds, _ = model(**data, return_emb_directly = True)

            assert HiC_embeds.isfinite().all()

            HiC_values = cut2d(data['input_matrix'], remove_boundary)
            HiC_embeds = cut2d(HiC_embeds, remove_boundary)
            DNA_embeds = cut2d(DNA_embeds, remove_boundary)

            HiC_values = HiC_values.flatten(0, 2)
            HiC_embeds = HiC_embeds.flatten(0, 2)
            DNA_embeds = DNA_embeds.flatten(0, 2)

            HiC_values = HiC_values.permute((0,2,3,1))
            HiC_embeds = HiC_embeds.permute((0,2,3,1))
            DNA_embeds = DNA_embeds.permute((0,2,3,1))

            b, n, m, _ = HiC_embeds.shape

            positions = []
            mask = torch.ones((b,n,m), dtype=torch.bool, device=device)
            for i in range(b):
                patch_pos = data['positions'].flatten(0,2)[i]

                ch, x0, _, st, _, y0, _, _ = patch_pos

                map_row = mappability_row[i].view(-1, dataset.resolution).mean(dim=-1)
                map_col = mappability_col[i].view(-1, dataset.resolution).mean(dim=-1)
                if st == 1:
                    map_row = map_row.flip()
                    map_col = map_col.flip()

                for d in range(-5, 5+1):
                    mask[i, :, map_row[remove_boundary+d:-remove_boundary+d]<0.5] = 0
                    mask[i, map_col[remove_boundary+d:-remove_boundary+d]<0.5, :] = 0

                r = dataset.resolution

                x0 += remove_boundary * r
                y0 += remove_boundary * r

                # Vectorized positions calculation using torch operations
                x_idx = torch.arange(n, device=device)
                y_idx = torch.arange(m, device=device)
                x_grid, y_grid = torch.meshgrid(x_idx, y_idx, indexing='ij')

                if st == 0:
                    x_pos = x0 + x_grid * r
                    y_pos = y0 + y_grid * r
                else:
                    x_pos = x0 + (n - 1 - x_grid) * r
                    y_pos = y0 + (m - 1 - y_grid) * r

                mask[i] = mask[i] & (y_pos>=x_pos) #upper diag

                ch_tensor = torch.full_like(x_pos, ch)
                poses = torch.stack([ch_tensor, x_pos, y_pos], dim=-1)
                positions.append(poses)

            positions = torch.stack(positions, dim=0)

            HiC_values = HiC_values[mask]
            HiC_embeds = HiC_embeds[mask]
            DNA_embeds = DNA_embeds[mask]
            positions = positions[mask]

            HiC_values = dataset.normalizer.unnormalize(HiC_values, tensor=True)

            HiC_valuess.append(HiC_values.cpu())
            HiC_embedss.append(HiC_embeds.cpu())
            DNA_embedss.append(DNA_embeds.cpu())
            positionss.append(positions.cpu())

    HiC_valuess = torch.concat(HiC_valuess)
    HiC_embedss = torch.concat(HiC_embedss)
    DNA_embedss = torch.concat(DNA_embedss)
    positionss = torch.concat(positionss)

    return HiC_valuess, HiC_embedss, DNA_embedss, positionss

if __name__ == '__main__':
    parser = CDNA2d_data_predict_parser()
    parser.set_defaults(
        input_option = 'HC',
        target_option = 'hic',
        avg_hic = False,

        max_separation = 500000,

        chunk = 100,
        stride = 80,

        whole_row = True,

        batch_size = 16
    )

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    try:
        ckpt = os.readlink(args.checkpoint)
    except:
        ckpt = args.checkpoint

    save_dir = args.save_dir
    print(f'Inferencing {save_dir}')
    out_dir = os.path.join('.'.join(ckpt.split('.')[:-1]), save_dir)
    mkdir(out_dir)

    with open(os.path.join(out_dir, "args.json"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    model, modelargs, normalizer = load_model(ckpt)

    species = args.species
    resolution = args.resolution
    read_count = modelargs['read_count']
    max_separation = args.max_separation
    DNA_option = args.DNA_option

    boundary = (args.chunk - args.stride)//2

    if species in splits.keys():
        target_specified = args.target_specified

        if target_specified is not None:
            if not os.path.isfile(target_specified):
                target_specified = os.path.join(hic_data_dir, target_specified)
            loader = HiC_Loader(target_specified, resolution=resolution, read_count=read_count)
            accession = '.'.join(os.path.basename(target_specified).split('.')[:-1])
            hic_loaders = [(accession, loader)]
        else:
            hic_loaders = create_loaders_with_index(args.split, species, return_name=True, resolution=resolution, read_count=read_count)

        DNA_loader = DNA_Loader(DNA_map[species], DNA_option)
        mappability_loader = Mappability_Loader(mappability_map[species], DNA_option)

        recall_file = os.path.join(out_dir, 'recall.tsv')

        results = {'Name' : [], 'Chr': [], 'Recall@1': [], 'Recall@5': [], 'Recall@10': [], 'Avg Rank': []}

        for name, hic_loader in hic_loaders:
            cellline_dir = os.path.join(out_dir, name)
            mkdir(cellline_dir)

            print(f'Saving the data to {cellline_dir}')

            for ch in splits[args.species][args.split]:
                split = (ch, )

                hic_value_file = os.path.join(cellline_dir, f'hic_v_{ch}.npy')
                pos_file = os.path.join(cellline_dir, f'pos_{ch}.npy')

                test_set = HiC_DNA_Dataset(
                    chromosome_split = split,
                    HiC_loaders = [hic_loader],
                    DNA_loader = DNA_loader,
                    Mappability_loader = mappability_loader,
                    normalizer = normalizer,
                    **args.__dict__
                )

                HiC_values, HiC_embeds, DNA_embeds, positions = infer_embeddings(test_set, model, args.batch_size, boundary)

                np.save(hic_value_file, HiC_values.numpy())
                np.save(pos_file, positions.numpy())

                with torch.no_grad():

                    HiC_embeds = HiC_embeds.to('cuda')
                    DNA_embeds = DNA_embeds.to('cuda')

                    HiC_embeds = nn.functional.normalize(HiC_embeds, dim=-1)
                    DNA_embeds = nn.functional.normalize(DNA_embeds, dim=-1)

                    positions = positions.to('cuda')
                    dis = (positions[:, 2]-positions[:, 1])//2000
                    if max_separation > 0:
                        MXD = int(max_separation // 2000 - boundary)
                    else:
                        MXD = max(dis)+1

                    mask = (dis >= 0) & (dis < MXD)
                    ranks = -torch.ones(positions.shape[0], dtype=int, device='cuda')
                    bests = -torch.ones(positions.shape[0], dtype=int, device='cuda')
                    sims_pos = torch.zeros(positions.shape[0], device='cuda')
                    sims_neg = torch.zeros(positions.shape[0], device='cuda')

                    for d in tqdm(range(MXD)):
                        inds = torch.argwhere(dis == d).squeeze()
                        B = 64
                        for i in range(0, len(inds), B):
                            sims = DNA_embeds[inds[i:i+B], :] @ HiC_embeds[inds, :].t()

                            rank = (sims > torch.diag(sims[:, i:i+B])[:, None]).sum(dim=-1)
                            best = inds[torch.argmax(sims, dim=-1)]

                            ranks[inds[i:i+B]] = rank
                            bests[inds[i:i+B]] = best

                            sims_pos[inds[i:i+B]] = torch.diag(sims[:, i:i+B])
                            sims_neg[inds[i:i+B]] = sims.mean(dim=-1)

                    print((sims_pos - sims_neg)[mask].mean().cpu().item())

                    sim_pos_file = os.path.join(cellline_dir, f'sim_pos_{ch}.npy')
                    sim_neg_file = os.path.join(cellline_dir, f'sim_neg_{ch}.npy')

                    np.save(sim_pos_file, sims_pos.cpu().numpy())
                    np.save(sim_neg_file, sims_neg.cpu().numpy())

                    rank_file = os.path.join(cellline_dir, f'rank_{ch}.npy')
                    best_file = os.path.join(cellline_dir, f'best_{ch}.npy')

                    np.save(rank_file, ranks.cpu().numpy())
                    np.save(best_file, bests.cpu().numpy())

                    r1 = (ranks[mask]<1).to(float).mean().cpu().item()
                    r5 = (ranks[mask]<5).to(float).mean().cpu().item()
                    r10 = (ranks[mask]<10).to(float).mean().cpu().item()
                    avg = ranks[mask].to(float).mean().cpu().item()
                    print(r1, r5, r10, avg)

                    results['Name'].append(name)
                    results['Chr'].append(ch)
                    results['Recall@1'].append(r1)
                    results['Recall@5'].append(r5)
                    results['Recall@10'].append(r10)
                    results['Avg Rank'].append(avg)

                    r = pd.DataFrame.from_dict(
                        results
                    )

                    r.to_csv(recall_file, sep='\t', index=False)