import torch
from torch import nn
from config import *
from utils import *
from model.DNAencoder import DNAEncoder
from model.HiCencoder import HiCEncoder
from model.track_decoder import Track_Decoder

class CDNA1d(nn.Module):
    def __init__(self, decoder:Track_Decoder, DNA_encoder:DNAEncoder, HiC_encoder:HiCEncoder, normalize_emb = False):
        super().__init__()
        self.DNA_encoder = DNA_encoder
        self.HiC_encoder = HiC_encoder

        self.decoder = decoder

        self.normalize_emb = normalize_emb

    def forward(self, return_emb=False, **data):
        # shape: (B, S, H, *)
        b, s, h, *_ = data['input_matrix'].shape

        HiC_embeds = self.HiC_encoder(data['input_matrix'].flatten(0, 2)).unflatten(0, (b,s,h))

        data['input_matrix_embeds'] = HiC_embeds
        if self.normalize_emb:
            data['input_matrix_embeds'] = nn.functional.normalize(data['input_matrix_embeds'], dim=-3)
        data['input_matrix_embeds'] = data['input_matrix_embeds'].mean(dim=-1)

        if 'DNA0' in data and torch.numel(data['DNA0']) > 0:
            # have DNA input

            b1, s1, h1, *_ = data['DNA0'].shape

            DNA_embeds = self.DNA_encoder(data['DNA0'].flatten(0,2), data['mappability0'].flatten(0,2)).unflatten(0, (b1, s1, h1))

            data['DNA_embeds'] = DNA_embeds
            if self.normalize_emb:
                data['DNA_embeds'] = nn.functional.normalize(data['DNA_embeds'], dim=-2)

            data['DNA_embeds']=data['DNA_embeds'].expand_as(data['input_matrix_embeds'])
        else:
            # don't have DNA input
            data['DNA_embeds'] = torch.zeros_like(data['input_matrix_embeds'])

        if return_emb:
            return HiC_embeds, DNA_embeds

        embeds = torch.cat([data['input_matrix_embeds'], data['DNA_embeds']], dim=-2).transpose(-1, -2)

        pred = self.decoder(embeds.flatten(0,2)).unflatten(0, (b, s, h)).transpose(-1, -2)

        return pred