import os
from config import *
from dataset.hic_loader import HiC_Loader, Downsample_HiC_Loader
from dataset.DNA_loader import DNA_Loader
from dataset.mappability_loader import Mappability_Loader
from tqdm import tqdm

def create_multispecies_loaders_with_index(data_index = None):
    index = open(dnazoo_index, 'r')

    lines = index.readlines()

    pbar = tqdm(lines)

    for i, line in enumerate(pbar):
        species = line.strip()
        pbar.set_description(f'creating loaders for {i}. {species}')
        if data_index is not None and i < data_index: continue
        
        dna_file = os.path.join(dnazoo_fasta_dir, species+'.fasta')
        if not os.path.isfile(dna_file):
            print(f'{dna_file} does not exist')
            continue

        hic_file = os.path.join(dnazoo_hic_dir, species+'.hic')
        if not os.path.isfile(hic_file):
            print(f'{hic_file} does not exist')
            continue
        else:
            print(hic_file)

        yield species, hic_file, dna_file
