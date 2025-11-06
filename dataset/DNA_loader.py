import pysam
from hic_utils import *
import numpy as np
import warnings
from tqdm import tqdm
from itertools import accumulate 

class DNA_Loader:
    def __init__(
            self,
            DNA_file_path,
            option = 'Yes',
            encoding_dir = None,
            store = True
        ) -> None:

        self.DNA_file_path = DNA_file_path
        self.option = option

        fasta = pysam.FastaFile(DNA_file_path)

        self.chr_lens = dict(zip(fasta.references, fasta.lengths))

        lst = 0
        self.chr_pos = {}
        for k,v in self.chr_lens.items():
            self.chr_pos[k] = (lst, lst+v)
            lst += v

        base_code = ['A', 'C', 'G', 'T']
        unknown_base = 'N'
        
        self.base_code = base_code
        self.code_size = len(base_code)
        self.unknown_base = unknown_base

        #prepare encode function
        base2onehot = {}
        for id, base in enumerate(self.base_code):
            enc = np.zeros((1, self.code_size))
            enc[0, id] = 1
            base2onehot[base] = enc
        
        base2onehot[self.unknown_base] = np.zeros((1, self.code_size))

        def encode_base(c):
            return base2onehot.get(c, base2onehot[self.unknown_base])
        
        def encode(seq):
            return np.concatenate(list(map(encode_base, seq)), dtype=np.float16)

        self.encode = encode

        self.dummy = base2onehot[self.unknown_base]
        self.padding = np.zeros((1, self.code_size))

        if option != 'Yes':
            return

        if encoding_dir is None:
            encoding_dir = os.path.dirname(DNA_file_path)

        DNA_encoding = '.'.join((os.path.basename(DNA_file_path), 'encoding'))
        DNA_encoding = os.path.join(encoding_dir, DNA_encoding)

        shape = (lst, 4)

        # Create memmap
        if not os.path.exists(DNA_encoding):
            encoding = np.zeros(shape, dtype=np.float16)
            
            for c in tqdm(self.chr_pos, desc='convert DNA to encoding'):
                seq = fasta.fetch(c).upper()

                encoding[self.chr_pos[c][0]:self.chr_pos[c][1]] = self.encode(seq)

            if store:
                mmap = np.memmap(
                    DNA_encoding, dtype=np.float16, mode="w+", shape=shape
                )
                mmap[:] = encoding

                print(f'DNA memmap in {DNA_encoding} created')

                self.encoding = np.memmap(
                    DNA_encoding, dtype=np.float16, mode="r", shape = shape
                )

            else:
                self.encoding = encoding
        else:
            self.encoding = np.memmap(
                DNA_encoding, dtype=np.float16, mode="r", shape = shape
            )

    def get_dummy(self, chr, start, end, strand):
        return np.concatenate([self.dummy]*(end-start)).astype(np.float32)
    
    def get(self, chr, start, end, strand):
        c = find_chr_in_set(chr, self.chr_pos)
        lo, hi = self.chr_pos[c]

        enc = self.encoding[max(lo+start, lo):min(lo+end, hi)] 

        if start < 0:
            enc = np.concatenate([self.padding] * abs(start)+ [enc])

        if end - start > enc.shape[0]:
            pad_len = (end - start - enc.shape[0])
            enc = np.concatenate([enc] + [self.padding] * pad_len)

        if strand == 1:
            enc = np.flip(enc)

        return enc.astype(np.float32).copy()
    
    def get_size(self, chr):
        c = find_chr_in_set(chr, self.chr_pos)
        return self.chr_lens[c]






