import torch
from torch import nn
from functools import partial

from model.DNAencoder import DualDNAEncoder
from model.HiCencoder import HiCEncoder
from model.CUnet import CUnet
from model.CDNA2d import CDNA2d
from utils import *
from config import *

from model.loss import multiresolution_loss
from torch.nn.functional import mse_loss

def create_model(
    resolution,

    input_channels,
    output_channels,
    dim,
    diffusion_steps,

    use_multiresolution_block,
    relative_resolutions,

    emb_dim,
    normalize_emb,

    force_final_conv,
    use_mrcrossembed,
    encoder_version,

    **kwargs
) -> CDNA2d:    
    print_info('Creating model')
    HiC_encoder = HiCEncoder(
        input_channels, 
        emb_dim, 
        relative_resolutions=relative_resolutions if use_multiresolution_block else [1],
        force_final_conv = force_final_conv,
        use_mrcrossembed=use_mrcrossembed
    )

    DNA_encoder = DualDNAEncoder(
        resolution=resolution,
        dim = emb_dim,
        relative_resolutions=relative_resolutions if use_multiresolution_block else [1],
        encoder_version = encoder_version,
        force_final_conv = force_final_conv
    )

    has_diffusion = (diffusion_steps > 1)
    assert not has_diffusion, 'diffusion-based model is deprecated'

    decoder = CUnet(
        dim = dim,
        num_resnet_blocks = (2, 2, 3),
        dim_mults = (1, 2, 4),
        channels = input_channels,
        channels_out = output_channels,
        layer_attns = False,
        layer_cross_attns = False,
        memory_efficient = True,
        init_conv_to_final_conv_residual = True,
        relative_resolutions=relative_resolutions if use_multiresolution_block else [1],

        cond_on_input_matrix=True,
        input_dim = emb_dim,

        cond_on_DNA=True,
        DNA_dim = emb_dim,

        cond_on_read_count=False,
        cond_on_seperation=False,

        has_diffusion = has_diffusion,
        force_final_conv=force_final_conv
    )

    model = CDNA2d(
        decoder = decoder, 
        DNA_encoder = DNA_encoder,
        HiC_encoder = HiC_encoder,
        normalize_emb = normalize_emb
    )

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model) 

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print_info(f'Model created. Number of parameters: {count_parameters(model)}')

    return model

def create_loss(
    use_multiresolution_loss,
    relative_resolutions,
    normalizer,
    **kwargs
):
    if use_multiresolution_loss:
        return partial(
            multiresolution_loss,
            relative_resolutions=relative_resolutions, 
            normalizer = normalizer, 
            img_normalize=False,
            loss_fn = mse_loss
        )
    else:
        return mse_loss
