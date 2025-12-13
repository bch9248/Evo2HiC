# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
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
from dataset.hic_loader import HiC_Loader
from dataset.DNA_loader import DNA_Loader
from dataset.mappability_loader import Mappability_Loader
from dataset.evo2_embedding_loader import evo2_Embedding_Loader
from dataset.normalizer import Normalizer
from model.create_CDNA2d import create_model
from model.CDNA2d import CDNA2d
from train.train_utils import cut1d
from hic_utils import *
from inference.infer_utils import *
from inference.inference_CDNA2d import CDNA2d_data_predict_parser

def load_model(checkpoint):
    args_file = os.path.join(os.path.dirname(checkpoint), 'args.json')
    with open(args_file) as f:
        args = json.load(f)

    normalizer = Normalizer(args['normalization'], max_reads=args['max_reads'], denominator = args['denominator'], step=args['step'])
    
    model = create_model(**{**args, 'normalizer' : normalizer, 'diffusion_steps' : 0})
    
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    state_unified = {k.replace('unet', 'decoder'):v for k,v in state['model'].items()}
    model.load_state_dict(state_unified)

    projection0 = nn.Linear(args['emb_dim'], args['evo2_projection_dim'])
    projection1 = nn.Linear(evo2_hidden_size * 2, args['evo2_projection_dim'])
    if 'projection0' in state:
        projection0.load_state_dict(state['projection0'])
    if 'projection1' in state:
        projection1.load_state_dict(state['projection1'])

    return model, args, normalizer, projection0, projection1

def infer_dna_embeddings(dataset : HiC_DNA_Dataset, model:CDNA2d, projection0:nn.Module, projection1:nn.Module, batch_size, remove_boundary=0):
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    device = 'cuda'
    model = model.to(device)
    projection0 = projection0.to(device)
    projection1 = projection1.to(device)

    embeds = []
    projected_embeds = []
    evo2_embeds = []
    projected_evo2_embeds = []
    poses = []

    model.eval()
    with torch.no_grad():
        pbar = tqdm(loader)
        for data in pbar:
            DNA_row = data['DNA_row'].to(device)
            mappability_row = data['mappability_row'].to(device)
            DNA_embed1s = model.DNA_encoder.encoder(DNA_row.flatten(0, 2), mappability_row.flatten(0, 2))
            DNA_embed1s = DNA_embed1s.transpose(-1, -2)
            DNA_embed1s = cut1d(DNA_embed1s, remove_boundary)

            embeds.append(DNA_embed1s.transpose(-1, -2))
            projected_embeds.append(projection0(DNA_embed1s).transpose(-1, -2))

            if 'embedding_row' in data and data['embedding_row'] is not None and torch.numel(data['embedding_row']) > 0:
                evo2_embed1s = data['embedding_row'].to(device)
                evo2_embed1s = evo2_embed1s.flatten(0,2)
                evo2_embed1s = cut1d(evo2_embed1s, remove_boundary)

                evo2_embeds.append(evo2_embed1s.transpose(-1, -2))
                projected_evo2_embeds.append(projection1(evo2_embed1s).transpose(-1, -2))            

            pos = data['positions'].flatten(0,2)
            r = dataset.resolution
            pos[:, 1] += remove_boundary*r
            pos[:, 2] -= remove_boundary*r
            poses.append(pos)

        embeds = torch.concat(embeds, dim=0).cpu().numpy()
        projected_embeds = torch.concat(projected_embeds, dim=0).cpu().numpy()

        if len(evo2_embeds)>0:
            evo2_embeds = torch.concat(evo2_embeds, dim=0).cpu().numpy()
            projected_evo2_embeds = torch.concat(projected_evo2_embeds, dim=0).cpu().numpy()
        else:
            evo2_embeds = None
            projected_evo2_embeds = None

        poses = torch.concat(poses, dim=0).cpu().numpy()

    return embeds, projected_embeds, evo2_embeds, projected_evo2_embeds, poses

if __name__ == '__main__':
    parser = CDNA2d_data_predict_parser()
    parser.set_defaults(
        input_option = 'HC',
        target_option = 'hic',
        avg_hic = False,

        max_separation = 0,

        chunk = 100,
        stride = 80,

        whole_row = False,

        batch_size = 64
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

    model, modelargs, normalizer, projection0, projection1 = load_model(ckpt)

    species = args.species
    resolution = args.resolution
    read_count = modelargs['read_count']
    max_separation = args.max_separation
    DNA_option = args.DNA_option
    evo2_option = args.evo2_option

    boundary = (args.chunk - args.stride)//2

    if species in splits.keys():
        index = pd.read_csv(hic_index, sep='\t')
        index = index[index['Organism'] == species]
        loaders = []
        hic_file = os.path.join(hic_data_dir, index['Hi-C Accession'].iloc[0] + '.hic')
        hic_loader = HiC_Loader(hic_file, resolution=resolution, read_count=read_count)

        DNA_loader = DNA_Loader(DNA_map[species], DNA_option)
        mappability_loader = Mappability_Loader(mappability_map[species], DNA_option)
        embedding_loader = evo2_Embedding_Loader(evo2_embedding_map[species], evo2_option)

        names = []
        chrs = []

        for ch in splits[args.species][args.split]:
            split = (ch, )

            test_set = HiC_DNA_Dataset(
                chromosome_split = split,
                HiC_loaders = [hic_loader],
                DNA_loader = DNA_loader,
                Mappability_loader = mappability_loader,
                Embedding_loader = embedding_loader,
                normalizer = normalizer,
                **args.__dict__
            )

            dna_embeddings, dna_projected_embeddings, evo2_embeddings, evo2_projected_embeddings, positions = infer_dna_embeddings(test_set, model, projection0, projection1, args.batch_size, boundary)

            chrom_size = DNA_loader.get_size(ch)
         
            dna_embedding = construct_1d_tracks(dna_embeddings, positions, ch, chrom_size, resolution, modelargs['emb_dim'])
            dna_projected_embedding = construct_1d_tracks(dna_projected_embeddings, positions, ch, chrom_size, resolution, modelargs['evo2_projection_dim'])

            dna_embedding_file = os.path.join(out_dir, f'dna_emb_{ch}.npy')
            np.save(dna_embedding_file, dna_embedding)

            dna_proj_embedding_file = os.path.join(out_dir, f'dna_proj_emb_{ch}.npy')
            np.save(dna_proj_embedding_file, dna_projected_embedding)

            if evo2_embeddings is not None:
                evo2_embedding = construct_1d_tracks(evo2_embeddings, positions, ch, chrom_size, resolution, evo2_hidden_size * 2)
                evo2_projected_embedding = construct_1d_tracks(evo2_projected_embeddings, positions, ch, chrom_size, resolution, modelargs['evo2_projection_dim'])

                evo2_embedding_file = os.path.join(out_dir, f'evo2_emb_{ch}.npy')
                np.save(evo2_embedding_file, evo2_embedding)

                evo2_proj_embedding_file = os.path.join(out_dir, f'evo2_proj_emb_{ch}.npy')
                np.save(evo2_proj_embedding_file, evo2_projected_embedding)