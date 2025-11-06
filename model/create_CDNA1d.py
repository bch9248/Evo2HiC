import torch
from torch import nn
from functools import partial

from model.DNAencoder import DNAEncoder
from model.HiCencoder import HiCEncoder
from model.track_decoder import Track_Decoder
from model.CDNA1d import CDNA1d
from utils import *
from config import *

def create_model(
    resolution,

    input_channels,
    track_input_dim,

    use_multiresolution_block,
    relative_resolutions,

    emb_dim,
    normalize_emb,

    force_final_conv,
    use_mrcrossembed,
    encoder_version,

    **kwargs
) -> CDNA1d:    
    print_info('Creating model')
    HiC_encoder = HiCEncoder(
        input_channels, 
        emb_dim, 
        relative_resolutions=relative_resolutions if use_multiresolution_block else [1],
        force_final_conv = force_final_conv,
        use_mrcrossembed=use_mrcrossembed
    )

    DNA_encoder = DNAEncoder(
        resolution=resolution,
        encoder_version = encoder_version
    )

    track_decoder = Track_Decoder(
        dim=track_input_dim,
        num_tracks=len(tracks)
    )
    
    model = CDNA1d(
        track_decoder, 
        DNA_encoder = DNA_encoder,
        HiC_encoder = HiC_encoder,
        normalize_emb = normalize_emb
    )

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model) 

    return model