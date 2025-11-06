import torch
from torch import nn
from model.multiresolution_block import ResnetBlock

class DNAResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dropout=0.1):
        super().__init__()

        self.linear_conv = nn.Sequential(
            nn.Conv1d(in_channels , out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(out_channels),
        )

        self.conv = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        hidden = self.linear_conv(x)
        out = self.dropout(self.conv(hidden)) + hidden
        return out

class DNAResBlock_v2(nn.Module):
    def __init__(self, pool, in_channels, out_channels, kernel_size, stride=1, padding=0, dropout=0):
        super().__init__()

        self.linear_conv = nn.Sequential(
            nn.Conv1d(in_channels , out_channels, kernel_size=17, stride=pool, padding=8),
            nn.BatchNorm1d(out_channels),
        )

        self.conv = nn.Sequential(
            nn.Conv1d(out_channels, out_channels * 2, 9, 1, padding=4, bias=False),
            nn.BatchNorm1d(out_channels * 2),
            nn.SiLU(inplace=False),
            nn.Conv1d(out_channels * 2, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm1d(out_channels),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        hidden = self.linear_conv(x)
        out = self.dropout(self.conv(hidden)) + hidden
        return out


settings_base = {
    'encoder_resolution' : 2000,    
    'pools'       : [1,   4,   4,   5,   5,   5,   1],
    'kernel_sizes': [9,   9,   9,   9,   9,   9,   9],
    'paddings'    : [4,   4,   4,   4,   4,   4,   4],
    'hidden_dims' : [64,  96,  128, 128, 128, 128, 128],
}

class DNAEncoder(nn.Module):
    def __init__(
            self, 
            resolution,
            init_dim = 4,
            encoder_version = 'v1'
        ):
        super(DNAEncoder, self).__init__()

        self.resolution = resolution

        encoder_resolution = settings_base['encoder_resolution']
        pools = settings_base['pools']
        kernel_sizes = settings_base['kernel_sizes']
        paddings = settings_base['paddings']
        hidden_dims = settings_base['hidden_dims']

        hidden_dims = [init_dim] + hidden_dims

        self.dim = hidden_dims[-1]

        assert resolution%encoder_resolution == 0
        pools[-1] = resolution//encoder_resolution

        blocks = []
        for p, hi, ho, k, pad in zip(pools, hidden_dims[:-1], hidden_dims[1:], kernel_sizes, paddings):
            if encoder_version == 'v1':
                blocks.append(nn.MaxPool1d(kernel_size=p))
                blocks.append(DNAResBlock(hi, ho, kernel_size=k, padding=pad))
            elif encoder_version == 'v2':
                blocks.append(DNAResBlock_v2(p, hi, ho, kernel_size=k, padding=pad))

        self.encoder = nn.Sequential(*blocks)

        self.output_dim = hidden_dims[-1]

    def forward(self, x, map):
        x = x * (0.3 + map.unsqueeze(-1)*0.7)
        x = x.transpose(-1, -2).contiguous()
        out=self.encoder(x)
        return out

class DualDNAEncoder(nn.Module):
    def __init__(
            self, 
            resolution,
            dim,
            num_resnet = 1,
            relative_resolutions = [1],
            force_final_conv = True,
            **kwargs    
        ):
        super(DualDNAEncoder, self).__init__()

        self.encoder = DNAEncoder(resolution, **kwargs)
        self.map_pooler = torch.nn.AvgPool1d(resolution)

        if dim != self.encoder.dim:
            self.projection = nn.Linear(self.encoder.dim, dim)
        else:
            self.projection = None

        self.resnets = nn.ModuleList([ResnetBlock(dim+2 if i==0 else dim, dim , use_gca = True, relative_resolutions=relative_resolutions, force_final_conv=force_final_conv) for i in range(num_resnet)])
        self.final_conv = nn.Conv2d(dim, dim, 1)

    def forward(self, DNA0, DNA1, mappability0, mappability1, return_emb = False):
        b, s1, h, *_ = DNA1.shape
        s0 = DNA0.shape[1] if DNA0 is not None else 0

        init_emb1 = self.encoder(DNA1.flatten(0, 2), mappability1.flatten(0, 2)).unflatten(0, (b, s1, h))
        map1 = self.map_pooler(mappability1.flatten(0,2)).unflatten(0, (b, s1, h))

        if self.projection is not None:
            emb1 = self.projection(init_emb1.transpose(-1,-2)).transpose(-1,-2)
        else:
            emb1 = init_emb1

        if s0 > 0:
            init_emb0 = self.encoder(DNA0.flatten(0, 2), mappability0.flatten(0, 2)).unflatten(0, (b, s0, h))
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


def _params_1d(m: nn.Module):
    if isinstance(m, (nn.Conv1d, nn.MaxPool1d, nn.AvgPool1d)):
        k = m.kernel_size if isinstance(m.kernel_size, tuple) else (m.kernel_size,)
        s = m.stride if isinstance(m.stride, tuple) else (m.stride,)
        if hasattr(m, 'dilation'):
            d = m.dilation if isinstance(m.dilation, tuple) else (m.dilation,)
        else:
            d = (1,)
        p = m.padding if isinstance(m.padding, tuple) else (m.padding,)
        return k[0], s[0], d[0], p[0]
    return None

def receptive_field_1d_theoretical(model: nn.Module, until_module: str | None = None):
    """
      rf_out   = rf_in + (k - 1) * d * jump_in
      jump_out = jump_in * s
      start_out= start_in + ((k - 1)/2 - p) * d * jump_in
    """
    rf = 1
    jump = 1
    start = 0.5

    for name, m in model.named_modules():
        if m is model:
            continue
        params = _params_1d(m)
        if params is None:
            continue
        k, s, d, p = params

        rf   = rf + (k - 1) * d * jump
        start= start + ((k - 1) / 2 - p) * d * jump
        jump = jump * s

        if until_module is not None and name == until_module:
            break

    return dict(rf=int(rf), jump=int(jump), start=float(start))

if __name__ == '__main__':
    model = DNAEncoder(2000)
    result = receptive_field_1d_theoretical(model)
    print(result)
    device='cuda:0'
    from tqdm import tqdm
    import time
    start = time.time()
    with torch.no_grad():
        model=model.to(device)
        for i in tqdm(range(100)):
            dna = torch.randint(4, (1, 146032)).to(device)
            dna = torch.nn.functional.one_hot(dna, 4)
            map = torch.ones((1, 146032)).to(device)
            res = model(dna, map)
    end = time.time()
    print(f"run time: {end - start:.4f}s")

