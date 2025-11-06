import pysam
from hic_utils import *
import numpy as np
from tqdm import tqdm

class evo2_Embedding_Loader:
    def __init__(
            self,
            evo2_embedding_dir,
            option,
            resolution=2000
        ) -> None:

        self.evo2_embedding_dir = evo2_embedding_dir
        self.option = option
        self.resolution = resolution

        if self.option == 'No':
            return

        shape_file = os.path.join(evo2_embedding_dir, 'shape.json')
        with open(shape_file) as f:
            self.shapes = json.load(f)

        self.hidden = None

        self.embeddings = {}
        self.embeddings_rev = {}
        for c, shape in self.shapes.items():
            assert evo2_hidden_size == shape[-1]
            shape = tuple(shape)

            DNA_encoding     = os.path.join(evo2_embedding_dir, '.'.join((c, 'embedding')))
            DNA_encoding_rev = os.path.join(evo2_embedding_dir, '.'.join((c, 'rev', 'embedding')))
            
            mmap = np.memmap(DNA_encoding, dtype=np.float16, mode="r", shape=shape)
            mmap_r = np.memmap(DNA_encoding_rev, dtype=np.float16, mode="r", shape=shape)

            self.embeddings[c] = mmap
            self.embeddings_rev[c] = mmap_r

        self.padding = np.zeros((1, evo2_hidden_size*2))
        self.dummy = np.zeros((1, evo2_hidden_size*2))
    
    def get(self, chr, start, end, strand):
        c = find_chr_in_set(chr, self.shapes)

        assert start%self.resolution == 0 and end % self.resolution == 0

        start, end = start//self.resolution, end//self.resolution

        emb, embr = self.embeddings[c][max(start, 0):end], self.embeddings_rev[c][max(start, 0):end]
        if strand == 1:
            embr, emb = emb, embr
        
        emb = np.concatenate([emb, embr], axis=-1)

        if start < 0:
            emb = np.concatenate([self.padding] * abs(start)+ [emb])

        if end - start > emb.shape[0]:
            pad_len = (end - start - emb.shape[0])
            emb = np.concatenate([emb] + [self.padding] * pad_len)

        if strand == 1:
            emb = np.flip(emb, axis=0)

        return emb.astype(np.float32).copy()
    
    def get_shape(self, chr):
        c = find_chr_in_set(chr, self.shapes)
        return self.shapes[c]