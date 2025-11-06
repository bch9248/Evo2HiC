import torch
from torch import nn
from torch.nn import BatchNorm2d
from imagen_pytorch.imagen_pytorch import default, ChanRMSNorm, Identity, exists, CrossAttention, LinearCrossAttention, GlobalContext, Always, CrossEmbedLayer
from imagen_pytorch.imagen_pytorch import CrossAttention, LinearCrossAttention, GlobalContext, Always
from einops import rearrange, pack, unpack
from torch.nn.functional import avg_pool2d
import numpy as np
def restore_size(x:torch.Tensor, r, original:torch.Tensor):
    h0,w0 = x.shape[-2], x.shape[-1]
    h,w = original.shape[-2], original.shape[-1]
    rh = torch.tensor([r]*(h0-1) + [h-r*(h0-1)]).to(x.device)
    rw = torch.tensor([r]*(w0-1) + [w-r*(w0-1)]).to(x.device)
    return x.repeat_interleave(rh, dim=-2, output_size=h).repeat_interleave(rw, dim=-1, output_size=w)

class Block(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        norm = 'ChanRMSNorm',
        relative_resolutions = [1],
        force_final_conv = True
    ):
        super().__init__()
        if norm == 'BatchNorm2d':
            self.norm = BatchNorm2d(dim)
        elif norm == 'ChanRMSNorm':
            self.norm = ChanRMSNorm(dim)            
        else:
            self.norm = Identity()

        self.activation = nn.SiLU()
        self.relative_resolutions = relative_resolutions
        dim_outs = [int(dim_out//(2**np.floor(np.log2(r)+1))) for r in relative_resolutions]
        self.projects = nn.ModuleList([nn.Conv2d(dim, dout, 3, padding = 1) for dout in dim_outs])
        sum_dim = sum(dim_outs)
        self.conv = nn.Conv2d(sum_dim, dim_out, 1) if sum_dim != dim_out or force_final_conv else Identity()

    def forward(self, x, scale_shift = None):
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.activation(x)
        projections = []
        for r, project in zip(self.relative_resolutions, self.projects):
            xr = avg_pool2d(x, r, r, ceil_mode=True)
            xr = project(xr)
            projections.append(restore_size(xr, r, x))
        x = torch.concat(projections, dim=1)
        return self.conv(x)

class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        *,
        cond_dim = None,
        time_cond_dim = None,
        linear_attn = False,
        use_gca = False,
        squeeze_excite = False,
        relative_resolutions = [1],
        force_final_conv = True,
        **attn_kwargs
    ):
        super().__init__()

        self.time_mlp = None

        if exists(time_cond_dim):
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_cond_dim, dim_out * 2)
            )

        self.cross_attn = None

        if exists(cond_dim):
            attn_klass = CrossAttention if not linear_attn else LinearCrossAttention

            self.cross_attn = attn_klass(
                dim = dim_out,
                context_dim = cond_dim,
                **attn_kwargs
            )

        self.block1 = Block(dim, dim_out, relative_resolutions=relative_resolutions, force_final_conv=force_final_conv)
        self.block2 = Block(dim_out, dim_out, relative_resolutions=relative_resolutions, force_final_conv=force_final_conv)

        self.gca = GlobalContext(dim_in = dim_out, dim_out = dim_out) if use_gca else Always(1)

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out or force_final_conv else Identity()

    def forward(self, x, time_emb = None, cond = None):

        scale_shift = None
        if exists(self.time_mlp) and exists(time_emb):
            time_emb = self.time_mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x)

        if exists(self.cross_attn):
            assert exists(cond)
            h = rearrange(h, 'b c h w -> b h w c')
            h, ps = pack([h], 'b * c')
            h = self.cross_attn(h, context = cond) + h
            h, = unpack(h, ps, 'b * c')
            h = rearrange(h, 'b h w c -> b c h w')

        h = self.block2(h, scale_shift = scale_shift)

        h = h * self.gca(h)

        return h + self.res_conv(x)

class MRCrossEmbedLayer(nn.Module):
    def __init__(
        self,
        dim_in,
        kernel_sizes,
        dim_out = None,
        stride = 2,
        relative_resolutions = [1],
        force_final_conv = True
    ):
        super().__init__()
        assert all([*map(lambda t: (t % 2) == (stride % 2), kernel_sizes)])
        dim_out = default(dim_out, dim_in)

        kernel_sizes = sorted(kernel_sizes)
        num_scales = len(kernel_sizes)

        self.relative_resolutions = relative_resolutions

        self.convs = nn.ModuleList([])
        for r in relative_resolutions:
            # calculate the dimension at each scale
            dim_scales = [int(dim_out / (2 ** i)) for i in range(1, num_scales)]
            dim_scales = [*dim_scales, dim_out - sum(dim_scales)]

            convrs = nn.ModuleList([])
            for kernel, dim_scale in zip(kernel_sizes, dim_scales):
                convrs.append(nn.Conv2d(dim_in, dim_scale, kernel, stride = stride, padding = (kernel-1) // 2))
            
            self.convs.append(convrs)

        sum_dim = dim_out * len(relative_resolutions)
        self.conv = nn.Conv2d(sum_dim, dim_out, 1) if sum_dim != dim_out or force_final_conv else Identity()

    def forward(self, x):
        xs = []
        for r, convrs in zip(self.relative_resolutions, self.convs):
            xr = avg_pool2d(x, r, r)
            fmaps = tuple(map(lambda conv: conv(xr), convrs))
            xr = torch.cat(fmaps, dim = 1)
            xs.append(restore_size(xr, r, x))

        return self.conv(torch.concat(xs, dim=1))