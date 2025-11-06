import torch
from torch import nn
from model.multiresolution_block import ResnetBlock

class EvoEmbeddingEncoder(nn.Module):
    def __init__(self, resolution, input_dim):
        super().__init__()
        dim = 128

        self.dim = dim

        self.projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, dim)
        )

        encoder_resolution = 2000
        
        assert resolution%encoder_resolution == 0
        self.pooler = nn.MaxPool1d(kernel_size=resolution//encoder_resolution)

        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim, dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim, dim, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.output_layer = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim)
        )

    def forward(self, embedding):
        embedding = self.projection(embedding)
        embedding = self.pooler(embedding.transpose(-1, -2))
        embedding = self.conv(embedding)
        return self.output_layer(embedding.transpose(-1,-2))

class DualEvoEncoder(nn.Module):
    def __init__(
            self, 
            resolution,
            input_dim,
            dim,
            num_resnet = 1,
            relative_resolutions = [1],
            **kwargs
        ):
        super(DualEvoEncoder, self).__init__()

        self.encoder = EvoEmbeddingEncoder(resolution=resolution, input_dim = input_dim)
        self.map_pooler = torch.nn.AvgPool1d(resolution)

        if dim != self.encoder.dim:
            self.projection = nn.Linear(self.encoder.dim, dim)
        else:
            self.projection = None

        self.resnets = nn.ModuleList([ResnetBlock(dim+2 if i==0 else dim, dim , use_gca = True, relative_resolutions=relative_resolutions) for i in range(num_resnet)])
        self.final_conv = nn.Conv2d(dim, dim, 1)

    def forward(self, embedding0, embedding1, mappability0, mappability1, return_emb = False):
        b, s1, h, *_ = embedding1.shape
        s0 = embedding0.shape[1] if embedding0 is not None else 0

        init_emb1 = self.encoder(embedding1.flatten(0, 2)).unflatten(0, (b, s1, h)).transpose(-1, -2)
        map1 = self.map_pooler(mappability1.flatten(0,2)).unflatten(0, (b, s1, h))

        if self.projection is not None:
            emb1 = self.projection(init_emb1.transpose(-1,-2)).transpose(-1,-2)
        else:
            emb1 = init_emb1

        if s0 > 0:
            init_emb0 = self.encoder(embedding0.flatten(0, 2)).unflatten(0, (b, s0, h)).transpose(-1, -2)
            map0 = self.map_pooler(mappability0.flatten(0,2)).unflatten(0, (b, s0, h))
            if self.projection is not None:
                emb0 = self.projection(init_emb0.transpose(-1,-2)).transpose(-1,-2)
            else:
                emb0 = init_emb0
        else:
            s0 = 1
            
            emb0 = emb1[:, :1, ...]
            map0 = map1[:, :1, ...]

        map0 = map0.unsqueeze(3)
        map1 = map1.unsqueeze(3)

        if s0 == 1:
            emb0 = emb0.expand(-1, s1, -1, -1, -1)
            map0 = map0.expand(-1, s1, -1, -1, -1)

        if s1 == 1:
            emb1 = emb1.expand(-1, s0, -1, -1, -1)
            map1 = map1.expand(-1, s0, -1, -1, -1)

        DNA_embeds = emb0[:, :, :, :, :, None] + emb1[:, :, :, :, None, :]

        n = DNA_embeds.shape[-1]

        map0 = map0[:, :, :, :, :, None].expand(-1, -1, -1, -1, -1, n)
        map1 = map1[:, :, :, :, None, :].expand(-1, -1, -1, -1, n, -1)

        x = torch.concatenate([DNA_embeds, map0, map1], dim=-3).flatten(0, 2)

        for resnet in self.resnets:
            x = resnet(x)

        final_emb = self.final_conv(x).unflatten(0, (b, -1, h))

        if return_emb:
            return final_emb, init_emb1
        
        return final_emb
