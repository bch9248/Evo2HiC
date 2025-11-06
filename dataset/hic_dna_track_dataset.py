from torch.utils.data import Dataset, Subset
from dataset.hic_dna_dataset import get_with_drop
from dataset.DNA_loader import DNA_Loader
from dataset.hic_loader import HiC_Loader
from dataset.mappability_loader import Mappability_Loader
from dataset.track_loader import Track_Loader
from dataset.normalizer import Normalizer

import utils
from utils import *
from hic_utils import *
from config import *
import numpy as np
import pandas as pd


class HiC_DNA_track_Dataset(Dataset):
    def __init__(
            self,

            resolution,
            chromosome_split,

            HiC_loaders : list[HiC_Loader],
            Track_loaders : list[Track_Loader],

            DNA_loader : DNA_Loader,
            Mappability_loader : Mappability_Loader,

            normalizer : Normalizer,

            max_separation,
            chunk,
            stride,

            sample = False,

            dataset_size = -1,

            map_thres = 0.5,

            DNA_drop_prob = 0,
            map_drop_prob = 0,

            hic_per_pos = -1,

            training = False,

            seed = 0,
            **kwargs
        ) -> None:
        super().__init__()

        self.HiC_loaders = HiC_loaders
        self.Track_loaders = Track_loaders

        self.DNA_loader = DNA_loader
        self.Mappability_loader = Mappability_loader

        self.normalizer = normalizer

        self.chromosome_split = chromosome_split

        self.resolution = resolution
        self.max_separation = max_separation
        self.chunk = chunk
        self.stride = stride

        self.sample = sample

        self.map_thres = map_thres

        self.DNA_drop_prob = DNA_drop_prob
        self.map_drop_prob = map_drop_prob

        self.hic_per_pos = hic_per_pos

        self.training = training
        self.seed = seed if training else 0

        if self.sample and self.training:
            self.len_p = dataset_size
            self.len_h = len(self.HiC_loaders)
        else:
            self.prepare_position()

            self.len_p = len(self.positions)
            self.len_h = len(self.HiC_loaders)

        print_info(f'Dataset of {self.len_p} genome positions and {self.len_h} HiC datas.')

    def prepare_position(self):
        self.positions = []
        for chid in self.chromosome_split:
            size = self.DNA_loader.get_size(chid)

            stride = self.stride * self.resolution
            chunk = self.chunk * self.resolution

            assert chunk>=stride

            for i in range(0, size, stride):
                if self.training:
                    map = np.mean(self.Mappability_loader.get(chid, i, i+chunk, 0))
                    if map < self.map_thres:
                        continue

                self.positions.append([(chid, i, i+chunk, 0, chid, i-self.max_separation, i+self.max_separation, 0)])

    def __getitem__(self, index):
        data = {
            'positions' : [],

            'DNA0': [],
            'DNA1': [],
            'mappability0': [],
            'mappability1': [],

            'input_matrix': [],
            'input_read_count': [],

            'track': []
        }

        if self.sample and self.training:
            ch, s, e, strand = 0, 0, 0, 0
            Find = False
            while not Find:
                ch = np.random.choice(self.chromosome_split)
                size = self.DNA_loader.get_size(ch)
                i = np.random.choice(size//self.resolution-self.chunk)
                ii = i + self.chunk

                ch, s, e, strand = ch, i * self.resolution, ii * self.resolution, 0

                if e>=size:
                    continue

                map = self.Mappability_loader.get(ch, s, e, strand).mean()
                if map < self.map_thres:
                    continue

                positions = [(ch, s, e, strand, ch, s-self.max_separation, s+self.max_separation, strand)]
                Find = True
        else:
            positions = self.positions[index]

        for position in positions:
            chr0, s0, e0, strand0, chr1, s1, e1, strand1  = position

            data['positions'].append(((chr0, s0, e0, strand0, chr1, s1, e1, strand1),))

            if self.DNA_loader is not None and self.DNA_loader.option != 'No':
                drop = self.training and np.random.rand() < self.DNA_drop_prob
                data['DNA0'].append(get_with_drop(self.DNA_loader, drop, chr0, s0, e0, strand0))

            if self.Mappability_loader is not None and self.DNA_loader.option != 'No':
                drop = self.training and np.random.rand() < self.map_drop_prob
                data['mappability0'].append(get_with_drop(self.Mappability_loader, drop, chr0, s0, e0, strand0))

            if self.hic_per_pos == -1 or self.hic_per_pos > self.len_h:
                indices = [i for i in range(self.len_h)]
            else:
                if self.training:
                    indices = np.random.choice(self.len_h, self.hic_per_pos, replace=False)
                else:
                    rng = np.random.default_rng(seed = self.seed + index)
                    indices = rng.choice(self.len_h, self.hic_per_pos, replace=False)

            def read_hic(HiC_loader : HiC_Loader, position):
                input_read_count = HiC_loader.read_count
                input_submatrix = HiC_loader.get(*position)

                return self.normalizer.normalize(input_submatrix).clip(max=1), np.log(input_read_count)

            input_matrix, input_read_count = \
                zip(*[read_hic(self.HiC_loaders[hi], (chr0, s0, e0, strand0, chr1, s1, e1, strand1)) for hi in indices])

            input_matrix = np.stack(input_matrix, axis=0).astype(np.float32)
            input_matrix = np.nan_to_num(input_matrix)
            input_read_count = np.stack(input_read_count, axis=0).astype(np.float32)

            data['input_matrix'].append(input_matrix)
            data['input_read_count'].append(input_read_count)

            def read_tracks(Track_loader : Track_Loader, position):
                track = Track_loader.get(*position)

                return track

            track = [read_tracks(self.Track_loaders[hi], (chr0, s0, e0, strand0)) for hi in indices]

            track = np.stack(track, axis=0).astype(np.float32)

            data['track'].append(track)

        for k in data.keys():
            if len(data[k]) > 0:
                data[k] = np.stack(data[k], axis=0)
            else:
                data[k] = np.array([])
        return data

    def __len__(self):
        return self.len_p

def create_hic2track_loaders(split, return_name = False, **kwargs):
    index = pd.read_csv(hic2track_index, sep='\t')
    if split is not None:
        index = index[index['split'] == split]

    names = []
    hic_loaders = []
    track_loaders = []

    for _, row in tqdm(index.iterrows(), desc='creating paired loaders', disable=not utils.VERBOSE):
        hic_file = os.path.join(hic_data_dir, row['Hi-C'] + '.hic')
        track_dir = os.path.join(hic2tarck_dir, row['Cell Type'])
        hic_loader = HiC_Loader(hic_file,**kwargs)
        track_loader = Track_Loader(track_dir, **kwargs)

        names.append(row['Cell Type'])
        hic_loaders.append(hic_loader)
        track_loaders.append(track_loader)
    if return_name:
        return names, hic_loaders, track_loaders
    return hic_loaders, track_loaders

def prepare_datasets(
    resolution,

    DNA_option,

    train_hic_per_pos,
    valid_hic_per_pos,
    
    read_count,

    species = 'human',

    **kwargs
):
    assert species == 'human'
    hic_loaders, track_loaders = create_hic2track_loaders('train', resolution=resolution, read_count=read_count)

    DNA_loader = DNA_Loader(DNA_map[species], DNA_option)
    mappability_loader = Mappability_Loader(mappability_map[species], DNA_option)        

    train_set = HiC_DNA_track_Dataset(
        resolution,
        splits[species]['train'],
        hic_loaders,
        track_loaders,
        DNA_loader,
        mappability_loader,
        training=True,
        hic_per_pos=train_hic_per_pos,
        **kwargs
    )

    valid_set = HiC_DNA_track_Dataset(
        resolution,
        splits[species]['valid'],
        hic_loaders,
        track_loaders,
        DNA_loader,
        mappability_loader,
        training=False,
        hic_per_pos=valid_hic_per_pos,
        **kwargs
    )

    return train_set, valid_set