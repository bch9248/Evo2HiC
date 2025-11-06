import pyBigWig
import numpy as np
import warnings
from tqdm import tqdm
from hic_utils import *
from itertools import accumulate 

class Mappability_Loader:
    def __init__(
            self,
            mappability_file_path,
            option = 'Yes',
            *args, **kwargs
        ) -> None:

        self.option = option

        self.mappability_file_path = mappability_file_path

        if mappability_file_path is None:
            self.get = self.get_dummy
            return

        bw = pyBigWig.open(mappability_file_path)
        
        self.chr_lens = bw.chroms()

        lst = 0
        self.chr_pos = {}
        for k,v in self.chr_lens.items():
            self.chr_pos[k] = (lst, lst+v)
            lst +=v

        mappability_values = '.'.join((mappability_file_path, 'values'))
        shape = lst

        if not os.path.exists(mappability_values):
            values = np.zeros(shape, np.float16)
            
            for c in self.chr_pos.keys():
                seq = np.nan_to_num(bw.values(c, 0, self.chr_lens[c]))
                values[self.chr_pos[c][0]:self.chr_pos[c][1]] = seq

            mmap = np.memmap(
                mappability_values, dtype=np.float16, mode="w+", shape=shape
            )
            mmap[:] = values

            print(f'Mappability memmap in {mappability_values} created')

        self.values = np.memmap(
            mappability_values, dtype=np.float16, mode="r", shape = shape
        )

        
    def get_dummy(self, chr, start, end, strand):
        return np.ones((end-start)).astype(np.float32)

    def get(self, chr, start, end, strand):
        
        c = find_chr_in_set(chr, self.chr_pos)
        
        lo, hi = self.chr_pos[c]

        val = self.values[max(lo+start, lo):min(lo+end, hi)]
        
        if start < 0:
            val = np.concatenate((np.zeros(abs(start)), val))

        if len(val) < end-start:
            pad_len = (end-start - len(val))
            val = np.concatenate((val, np.zeros(pad_len)))

        if strand == 1:
            val = np.flip(val)

        return val.astype(np.float32).copy()








