import pyBigWig
import numpy as np
import warnings
from tqdm import tqdm
from hic_utils import *
from itertools import accumulate 
from config import tracks
from utils import print_info
class Track_Loader:
    def __init__(
            self,
            track_file_dir,
            resolution,
            option = 'Yes',
            *args, **kwargs
        ) -> None:

        self.option = option

        self.track_file_dir = track_file_dir
        self.resolution = resolution

        if track_file_dir is None:
            self.get = self.get_dummy
            return

        bws = []
        for name in tracks.values():
            path = os.path.join(track_file_dir, name)
            bws.append(pyBigWig.open(path))
        
        self.chr_lens = bws[0].chroms()
        for bw in bws:
            chr_lens = bw.chroms()
            for c in list(self.chr_lens):
                if c not in chr_lens:
                    del self.chr_lens[c]

        for c in list(self.chr_lens):
            if '_' in c:
                del self.chr_lens[c]

        lst = 0
        self.chr_pos = {}
        for k,v in self.chr_lens.items():
            r = v//resolution + 1
            self.chr_pos[k] = (lst, lst+r)
            lst += r

        tracks_values = os.path.join(track_file_dir, f'tracks.r{resolution}.values')
        
        shape = (len(tracks), lst)

        if not os.path.exists(tracks_values):
            print(f'Creating {tracks_values}...')

            values = np.zeros(shape, np.float16)

            cutoffs = np.zeros(len(tracks), float)

            for i, bw in enumerate(bws):
                val = np.zeros(lst)
                for c, (lo, hi) in self.chr_pos.items():
                    seq = np.nan_to_num(bw.values(c, 0, self.chr_lens[c]))
                    if len(seq) % resolution > 0:
                        pad_len = resolution - len(seq) % resolution
                        seq = np.concatenate([seq, np.zeros(pad_len)], axis=0)
                    seq = seq.reshape(-1, resolution).mean(axis=1)
                    val[lo:hi] = seq
                
                p = int(0.9*len(val))
                cutoff = np.partition(val.copy(), p)[p]*20

                print(i, cutoff, np.count_nonzero(val>cutoff)/len(val))

                val = np.minimum(val, cutoff)/cutoff

                values[i] = val
                cutoffs[i] = cutoff

            np.save(tracks_values+'.cutoff', cutoffs)

            mmap = np.memmap(
                tracks_values, dtype=np.float16, mode="w+", shape=shape
            )

            mmap[:] = values

            print(f'Track memmap in {tracks_values} created')

        self.values = np.memmap(
            tracks_values, dtype=np.float16, mode="r", shape = shape
        )

        print_info(track_file_dir)
        for i, _ in enumerate(tracks):
            print_info(np.mean(self.values[i]), np.std(self.values[i], dtype=float))
        
    def get_dummy(self, chr, start, end, strand):
        assert start%self.resolution == 0 and end % self.resolution == 0
        return np.zeros((len(self.names), end//self.resolution-start//self.resolution)).astype(np.float32)

    def get(self, chr, start, end, strand):
        
        c = find_chr_in_set(chr, self.chr_pos)
        
        assert start%self.resolution == 0 and end % self.resolution == 0
        start, end = start//self.resolution, end//self.resolution

        lo, hi = self.chr_pos[c]
        val = self.values[:, max(lo+start, lo):min(lo+end, hi)]
        
        if start < 0:
            val = np.concatenate([np.zeros((len(tracks), abs(start))), val], axis=1)

        if val.shape[1] < end-start:
            pad_len = (end-start - val.shape[1])
            val = np.concatenate([val, np.zeros((len(tracks), pad_len))], axis=1)

        if strand == 1:
            val = np.flip(val, axis=1)

        return val.astype(np.float32).copy()








