# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# --------------------------------------------------------

import numpy as np
from tqdm import tqdm
import torch
from config import *
from utils import *
import json
import random
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
import pandas as pd

from dataset.hic_dna_track_dataset import HiC_DNA_track_Dataset, create_hic2track_loaders
from dataset.hic_loader import HiC_Loader
from dataset.DNA_loader import DNA_Loader
from dataset.mappability_loader import Mappability_Loader
from dataset.normalizer import Normalizer
from model.create_CDNA1d import create_model
from model.CDNA1d import CDNA1d
from hic_utils import *
from inference.infer_utils import data_predict_parser, construct_1d_tracks

def CDNA1d_data_predict_parser():
    
    parser = data_predict_parser()

    Model_arguments = parser.add_argument_group('Model Arguments')
    Model_arguments.add_argument('-ckpt', '--checkpoint', type=str, required = True, help='the path to model checkpoint')

    return parser

def load_model(checkpoint):
    args_file = os.path.join(os.path.dirname(checkpoint), 'args.json')
    with open(args_file) as f:
        args = json.load(f)

    normalizer = Normalizer(args['normalization'], max_reads=args['max_reads'], denominator = args['denominator'], step=args['step'])
    
    model = create_model(**args)
    
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(state['model'])

    return model, args, normalizer

def inference(dataset : HiC_DNA_track_Dataset, model:CDNA1d, batch_size):
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    pccs = [[] for _ in tracks]
    preds = []
    pred_poses = []

    device = 'cuda'
    model = model.to(device)

    model.eval()
    with torch.no_grad():
        pbar = tqdm(loader)
        for data in pbar:            
            for k in data:
                data[k] = data[k].to(device)

            pred = model(**data).flatten(0,2)

            preds.extend(list(pred.cpu().numpy()))

            pred_poses.extend(list(data['positions'].flatten(0,2).cpu().numpy()))

            target = data['track'].flatten(0,2)

            s = ""

            for j in range(pred.shape[0]):
                for i, _ in enumerate(tracks):
                    pccs[i].append(np.nan_to_num(pearsonr(pred[j, i].cpu().numpy(), target[j, i].cpu().numpy())[0]))
            
            for i, _ in enumerate(tracks):
                s += f"{np.mean(pccs[i]):.3f} "
            
            pbar.set_description(s)
        
    return preds, pred_poses

if __name__ == '__main__':
    parser = CDNA1d_data_predict_parser()
    parser.set_defaults(
        batch_size = 8
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

    if species in splits.keys():
        assert species == 'human'
        names, hic_loaders, track_loaders = create_hic2track_loaders(None, resolution=resolution, return_name=True, read_count=read_count)

        DNA_loader = DNA_Loader(DNA_map[species], DNA_option)
        mappability_loader = Mappability_Loader(mappability_map[species], DNA_option)

        result_file = os.path.join(out_dir, 'result.tsv')


        results = {'Name' : [], 'Chr': []}
        for track_name in tracks.keys():
            results[track_name] = []

        for name, hic_loader, track_loader in zip(names, hic_loaders, track_loaders):

            cellline_dir = os.path.join(out_dir, name)
            mkdir(cellline_dir)

            print(f'Saving the data to {cellline_dir}')

            first = True
            for ch in splits[args.species][args.split]:
                track_file = os.path.join(cellline_dir, f'{ch}.npy')

                if not os.path.exists(track_file):

                    split = (ch, )

                    test_set = HiC_DNA_track_Dataset(
                        chromosome_split=split,
                        HiC_loaders = [hic_loader],
                        Track_loaders = [track_loader],
                        DNA_loader = DNA_loader,
                        Mappability_loader= mappability_loader,
                        normalizer=normalizer,
                        training=False,
                        hic_per_pos=1,
                        **args.__dict__
                    )
                        
                    preds, pred_poses = inference(test_set, model, args.batch_size)

                    chrom_size = DNA_loader.get_size(ch)
                    t_tracks = construct_1d_tracks(preds, pred_poses, ch, chrom_size, resolution, len(tracks))

                    np.save(track_file, t_tracks)
                else:
                    chrom_size = DNA_loader.get_size(ch)
                    t_tracks = np.load(track_file)

                target = track_loader.get(ch, 0, int(np.ceil(chrom_size/resolution))*resolution, 0)

                results['Name'].append(name)
                results['Chr'].append(ch)
                for i, n in enumerate(tracks):
                    results[n].append(pearsonr(t_tracks[i], target[i])[0])
                
                r = pd.DataFrame.from_dict(
                    results
                )

                r.to_csv(result_file, sep='\t', index=False)