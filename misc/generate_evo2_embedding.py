import torch
import pysam
from tqdm import tqdm
import numpy as np
from Bio.Seq import Seq
import os
import argparse
from hic_utils import *
from utils import *

model_name = 'evo2_7b'
hidden = 4096
layer_name = 'blocks.27.pre_norm'

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--chromosome', type=int, required=True)
parser.add_argument('-r', '--reference', type=str, default='hg38')
parser.add_argument('--DNA-file-path', type=str, default='data/dna/human/hg38.fa')
args = parser.parse_args()

ch = args.chromosome
reference = args.reference
DNA_file_path = args.DNA_file_path

save_dir = os.path.join(os.path.dirname(DNA_file_path), '_'.join((reference, model_name)))
mkdir(save_dir)

res = 2000
stride = 50000
chunk = 60000

assert chunk % res == 0
assert stride % res == 0

fasta = pysam.FastaFile(DNA_file_path)

shapes = {}
for c, L in zip(fasta.references, fasta.lengths):
    if '_' in c: continue
    shapes[c] = ((L-1)//res+1, hidden)

shape_file = os.path.join(save_dir, 'shape.json')
with open(shape_file, 'w') as f:
    json.dump(shapes, f, indent=2)

ch = id2chr(ch)

shape = shapes[ch]

DNA_encoding     = os.path.join(save_dir, '.'.join((ch, 'embedding')))
DNA_encoding_rev = os.path.join(save_dir, '.'.join((ch, 'rev', 'embedding')))

mmap = np.memmap(DNA_encoding, dtype=np.float16, mode="w+", shape=shape)
mmap_r = np.memmap(DNA_encoding_rev, dtype=np.float16, mode="w+", shape=shape)

from evo2 import Evo2
evo2_model = Evo2(model_name)

print(f'processing {ch}')
p = 0
sequence = fasta.fetch(ch)

for i in tqdm(range(0, L, stride)):
    pi = p + i//res
    b = 0 if i == 0 else (chunk-stride)//res
    r = chunk//res if L-i > chunk  else (L-i-1)//res+1

    seq = sequence[i:i+chunk]
    reverse_complement = str(Seq(seq).reverse_complement())

    input_ids = torch.tensor(
        evo2_model.tokenizer.tokenize_batch([seq, reverse_complement]),
        dtype=torch.int,
    ).to('cuda:0')
    input_ids = torch.nn.functional.pad(input_ids, (0, chunk-input_ids.shape[-1]), 'constant', evo2_model.tokenizer.pad_id)

    _, embeddings = evo2_model(input_ids, return_embeddings=True, layer_names=[layer_name])
    
    embeddings = embeddings[layer_name].squeeze(0)

    embeddings[1] = embeddings[1].flip(-2)
    embeddings = torch.nn.functional.avg_pool1d(embeddings.transpose(-1, -2), res, ceil_mode=True).transpose(-1,-2)

    mmap[pi+b:pi+r] = embeddings[0, b:r].cpu().half().numpy()
    mmap_r[pi:pi+r] = embeddings[1,  :r].cpu().half().numpy()
