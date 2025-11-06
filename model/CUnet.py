import torch
import torch.nn as nn
from imagen_pytorch.imagen_pytorch import Unet, LearnedSinusoidalPosEmb, CrossEmbedLayer
from imagen_pytorch.imagen_pytorch import LinearAttentionTransformerBlock, TransformerBlock, Parallel, PixelShuffleUpsample, UpsampleCombiner, Identity
from imagen_pytorch.imagen_pytorch import print_once, default, cast_tuple, Downsample, Upsample, zero_init_, resize_image_to
from einops import rearrange
from einops.layers.torch import Rearrange
from functools import partial
from utils import exists

from model.multiresolution_block import ResnetBlock

class CUnet(Unet):
    def __init__(
        self,
        *,
        dim,
        num_resnet_blocks = 1,
        cond_dim = None,
        num_time_tokens = 2,
        learned_sinu_pos_emb_dim = 16,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        channels_out = None,
        attn_dim_head = 64,
        attn_heads = 8,
        ff_mult = 2.,
        layer_attns = True,
        layer_attns_depth = 1,
        layer_mid_attns_depth = 1,
        attend_at_middle = True,            # whether to have a layer of attention at the bottleneck (can turn off for higher resolution in cascading DDPM, before bringing in efficient attention)
        layer_cross_attns = True,
        use_linear_attn = False,
        use_linear_cross_attn = False,
        init_dim = None,
        init_conv_kernel_size = 7,          # kernel size of initial conv, if not using cross embed
        init_cross_embed = True,
        init_cross_embed_kernel_sizes = (3, 7, 15),
        cross_embed_downsample = False,
        cross_embed_downsample_kernel_sizes = (2, 4),
        memory_efficient = False,
        init_conv_to_final_conv_residual = False,
        use_global_context_attn = True,
        scale_skip_connection = True,
        final_resnet_block = True,
        final_conv_kernel_size = 3,
        resize_mode = 'nearest',
        combine_upsample_fmaps = False,      # combine feature maps from all upsample blocks, used in unet squared successfully
        pixel_shuffle_upsample = True,       # may address checkboard artifacts

        relative_resolutions = [1],

        cond_on_input_matrix = False,
        input_dim = 0,
        input_matrix_dropout = 0,

        cond_on_DNA = False,
        DNA_dim = 0,
        DNA_dropout = 0,

        cond_on_read_count = False,
        cond_on_seperation = False,
                
        has_diffusion = False,
        force_final_conv = True
    ):
        nn.Module.__init__(self)

        self.has_diffusion = has_diffusion

        # guide researchers

        assert attn_heads > 1, 'you need to have more than 1 attention head, ideally at least 4 or 8'

        if dim < 128:
            print_once('The base dimension of your u-net should ideally be no smaller than 128, as recommended by a professional DDPM trainer https://nonint.com/2022/05/04/friends-dont-let-friends-train-small-diffusion-models/')

        # save locals to take care of some hyperparameters for cascading DDPM

        self._locals = locals()
        self._locals.pop('self', None)
        self._locals.pop('__class__', None)

        # determine dimensions

        self.channels = channels
        self.channels_out = default(channels_out, channels)

        # (1) in cascading diffusion, one concats the low resolution image, blurred, for conditioning the higher resolution synthesis
        # (2) in self conditioning, one appends the predict x0 (x_start)
        init_channels = channels * (int(self.has_diffusion))
        init_dim = default(init_dim, dim)

        # initial convolution
        if self.has_diffusion:
            self.init_conv = CrossEmbedLayer(init_channels, dim_out = init_dim, kernel_sizes = init_cross_embed_kernel_sizes, stride = 1, relative_resolutions=relative_resolutions) if init_cross_embed else nn.Conv2d(init_channels, init_dim, init_conv_kernel_size, padding = init_conv_kernel_size // 2)

        # (C-mat) optional C-mat conditioning

        self.cond_on_input_matrix = cond_on_input_matrix

        if cond_on_input_matrix:
            self.input_matrix_dropout = nn.Dropout2d(p=input_matrix_dropout)

        # (DNA) conditioning on DNA
         
        self.cond_on_DNA = cond_on_DNA
                
        if cond_on_DNA:
            self.DNA_dropout = nn.Dropout2d(p=DNA_dropout)

        # dims for Unet layers
             
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time conditioning

        cond_dim = default(cond_dim, dim)
        time_cond_dim = dim * 4 * (int(has_diffusion) + int(cond_on_read_count) + int(cond_on_seperation))

        # embedding time for log(snr) noise from continuous version
        if self.has_diffusion:
            sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinu_pos_emb_dim)
            sinu_pos_emb_input_dim = learned_sinu_pos_emb_dim + 1

            self.to_time_hiddens = nn.Sequential(
                sinu_pos_emb,
                nn.Linear(sinu_pos_emb_input_dim, time_cond_dim),
                nn.SiLU()
            )

            self.to_time_cond = nn.Sequential(
                nn.Linear(time_cond_dim, time_cond_dim)
            )

            # project to time tokens as well as time hiddens

            self.to_time_tokens = nn.Sequential(
                nn.Linear(time_cond_dim, cond_dim * num_time_tokens),
                Rearrange('b (r d) -> b r d', r = num_time_tokens)
            )

        # (C-mat) conditioning on number of reads
            
        self.cond_on_read_count = cond_on_read_count

        if cond_on_read_count:
            self.to_read_count_hiddens = nn.Sequential(
                LearnedSinusoidalPosEmb(learned_sinu_pos_emb_dim),
                nn.Linear(learned_sinu_pos_emb_dim + 1, time_cond_dim),
                nn.SiLU()
            )

            self.to_HC_read_count_cond = nn.Sequential(
                nn.Linear(time_cond_dim, time_cond_dim)
            )

            self.to_HC_read_count_tokens = nn.Sequential(
                nn.Linear(time_cond_dim, cond_dim * num_time_tokens),
                Rearrange('b (r d) -> b r d', r = num_time_tokens)
            )

        # (C-mat) conditioning on seperation
            
        self.cond_on_seperation = cond_on_seperation

        if cond_on_seperation:
            self.to_seperation_hiddens = nn.Sequential(
                LearnedSinusoidalPosEmb(learned_sinu_pos_emb_dim),
                nn.Linear(learned_sinu_pos_emb_dim + 1, time_cond_dim),
                nn.SiLU()
            )

            self.to_seperation_cond = nn.Sequential(
                nn.Linear(time_cond_dim, time_cond_dim)
            )

            self.to_seperation_tokens = nn.Sequential(
                nn.Linear(time_cond_dim, cond_dim * num_time_tokens),
                Rearrange('b (r d) -> b r d', r = num_time_tokens)
            )

        # normalizations

        self.norm_cond = nn.LayerNorm(cond_dim)

        if time_cond_dim == 0:
            cond_dim = None
            time_cond_dim = None

        # attention related params

        attn_kwargs = dict(heads = attn_heads, dim_head = attn_dim_head)

        num_layers = len(in_out)

        # resnet block klass

        num_resnet_blocks = cast_tuple(num_resnet_blocks, num_layers)

        resnet_klass = partial(ResnetBlock, relative_resolutions=relative_resolutions, force_final_conv=force_final_conv, **attn_kwargs)

        layer_attns = cast_tuple(layer_attns, num_layers)
        layer_attns_depth = cast_tuple(layer_attns_depth, num_layers)
        layer_cross_attns = cast_tuple(layer_cross_attns, num_layers)

        use_linear_attn = cast_tuple(use_linear_attn, num_layers)
        use_linear_cross_attn = cast_tuple(use_linear_cross_attn, num_layers)

        assert all([layers == num_layers for layers in list(map(len, (layer_attns, layer_cross_attns)))])

        # downsample klass

        downsample_klass = Downsample

        if cross_embed_downsample:
            downsample_klass = partial(CrossEmbedLayer, kernel_sizes = cross_embed_downsample_kernel_sizes)
        
        merge_dim = (init_dim if self.has_diffusion else 0) + input_dim + DNA_dim + int(cond_on_seperation)

        self.init_merge_resnet = resnet_klass(merge_dim, init_dim, cond_dim = cond_dim, linear_attn = use_linear_cross_attn[0], time_cond_dim = time_cond_dim, use_gca = use_global_context_attn)

        # initial resnet block (for memory efficient unet)

        self.init_resnet_block = resnet_klass(init_dim, init_dim, time_cond_dim = time_cond_dim, use_gca = use_global_context_attn) if memory_efficient else None

        # scale for resnet skip connections

        self.skip_connect_scale = 1. if not scale_skip_connection else (2 ** -0.5)

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        layer_params = [num_resnet_blocks, layer_attns, layer_attns_depth, layer_cross_attns, use_linear_attn, use_linear_cross_attn]
        reversed_layer_params = list(map(reversed, layer_params))

        # downsampling layers

        skip_connect_dims = [] # keep track of skip connection dimensions

        for ind, ((dim_in, dim_out), layer_num_resnet_blocks, layer_attn, layer_attn_depth, layer_cross_attn, layer_use_linear_attn, layer_use_linear_cross_attn) in enumerate(zip(in_out, *layer_params)):
            is_last = ind >= (num_resolutions - 1)

            layer_cond_dim = cond_dim if layer_cross_attn or layer_use_linear_cross_attn else None

            if layer_attn:
                transformer_block_klass = TransformerBlock
            elif layer_use_linear_attn:
                transformer_block_klass = LinearAttentionTransformerBlock
            else:
                transformer_block_klass = Identity

            current_dim = dim_in

            # whether to pre-downsample, from memory efficient unet

            pre_downsample = None

            if memory_efficient:
                pre_downsample = downsample_klass(dim_in, dim_out)
                current_dim = dim_out

            skip_connect_dims.append(current_dim)

            # whether to do post-downsample, for non-memory efficient unet

            post_downsample = None
            if not memory_efficient:
                post_downsample = downsample_klass(current_dim, dim_out) if not is_last else Parallel(nn.Conv2d(dim_in, dim_out, 3, padding = 1), nn.Conv2d(dim_in, dim_out, 1))

            self.downs.append(nn.ModuleList([
                pre_downsample,
                resnet_klass(current_dim, current_dim, cond_dim = layer_cond_dim, linear_attn = layer_use_linear_cross_attn, time_cond_dim = time_cond_dim),
                nn.ModuleList([ResnetBlock(current_dim, current_dim, time_cond_dim = time_cond_dim, use_gca = use_global_context_attn, relative_resolutions=relative_resolutions, force_final_conv=force_final_conv) for _ in range(layer_num_resnet_blocks)]),
                transformer_block_klass(dim = current_dim, depth = layer_attn_depth, ff_mult = ff_mult, context_dim = cond_dim, **attn_kwargs),
                post_downsample
            ]))

        # middle layers

        mid_dim = dims[-1]

        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, cond_dim = cond_dim, time_cond_dim = time_cond_dim, relative_resolutions=relative_resolutions, force_final_conv=force_final_conv)
        self.mid_attn = TransformerBlock(mid_dim, depth = layer_mid_attns_depth, **attn_kwargs) if attend_at_middle else None
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, cond_dim = cond_dim, time_cond_dim = time_cond_dim, relative_resolutions=relative_resolutions, force_final_conv=force_final_conv)

        # upsample klass

        upsample_klass = Upsample if not pixel_shuffle_upsample else PixelShuffleUpsample

        # upsampling layers

        upsample_fmap_dims = []

        for ind, ((dim_in, dim_out), layer_num_resnet_blocks, layer_attn, layer_attn_depth, layer_cross_attn, layer_use_linear_attn, layer_use_linear_cross_attn) in enumerate(zip(reversed(in_out), *reversed_layer_params)):
            is_last = ind == (len(in_out) - 1)

            layer_cond_dim = cond_dim if layer_cross_attn or layer_use_linear_cross_attn else None

            if layer_attn:
                transformer_block_klass = TransformerBlock
            elif layer_use_linear_attn:
                transformer_block_klass = LinearAttentionTransformerBlock
            else:
                transformer_block_klass = Identity

            skip_connect_dim = skip_connect_dims.pop()

            upsample_fmap_dims.append(dim_out)

            self.ups.append(nn.ModuleList([
                resnet_klass(dim_out + skip_connect_dim, dim_out, cond_dim = layer_cond_dim, linear_attn = layer_use_linear_cross_attn, time_cond_dim = time_cond_dim),
                nn.ModuleList([ResnetBlock(dim_out + skip_connect_dim, dim_out, time_cond_dim = time_cond_dim, use_gca = use_global_context_attn, relative_resolutions=relative_resolutions, force_final_conv=force_final_conv) for _ in range(layer_num_resnet_blocks)]),
                transformer_block_klass(dim = dim_out, depth = layer_attn_depth, ff_mult = ff_mult, context_dim = cond_dim, **attn_kwargs),
                upsample_klass(dim_out, dim_in) if not is_last or memory_efficient else Identity()
            ]))

        # whether to combine feature maps from all upsample blocks before final resnet block out

        self.upsample_combiner = UpsampleCombiner(
            dim = dim,
            enabled = combine_upsample_fmaps,
            dim_ins = upsample_fmap_dims,
            dim_outs = dim
        )

        # whether to do a final residual from initial conv to the final resnet block out

        self.init_conv_to_final_conv_residual = init_conv_to_final_conv_residual
        final_conv_dim = self.upsample_combiner.dim_out + (dim if init_conv_to_final_conv_residual else 0)

        # final optional resnet block and convolution out

        self.final_res_block = ResnetBlock(final_conv_dim, dim, time_cond_dim = time_cond_dim, use_gca = True, relative_resolutions=relative_resolutions, force_final_conv=force_final_conv) if final_resnet_block else None

        final_conv_dim_in = dim if final_resnet_block else final_conv_dim

        self.final_conv = nn.Conv2d(final_conv_dim_in, self.channels_out, final_conv_kernel_size, padding = final_conv_kernel_size // 2)

        zero_init_(self.final_conv)

        # resize mode

        self.resize_mode = resize_mode

    def forward(
        self,
        x = None,
        time = None,

        input_matrix_embeds = None,
        DNA_embeds = None,

        input_read_count = None,
        HC_read_count = None,

        seperation = None,
        seperation_matrix = None,

        **kwargs
    ):
        assert not (self.has_diffusion ^ exists(x))
                
        # initial convolution
        if self.has_diffusion:
            x = self.init_conv(x)

        if self.cond_on_input_matrix:
            assert input_matrix_embeds is not None
            input_matrix_embeds = self.input_matrix_dropout(input_matrix_embeds)

            if x is not None:
                x = torch.cat([input_matrix_embeds, x], dim=1)
            else:
                x = input_matrix_embeds

        if self.cond_on_DNA:
            assert DNA_embeds is not None
            DNA_embeds = self.DNA_dropout(DNA_embeds)
        
            if x is not None:
                x = torch.cat([DNA_embeds, x], dim=1)
            else:
                x = DNA_embeds

        if self.cond_on_seperation:
            if x is not None:
                x = torch.cat([seperation_matrix, x], dim=1)
            else:
                x = seperation_matrix

        # init conv residual

        if self.has_diffusion:
            # time conditioning

            time_hiddens = self.to_time_hiddens(time)

            # derive time tokens

            time_tokens = self.to_time_tokens(time_hiddens)
            t = self.to_time_cond(time_hiddens)
        else:
            time_tokens = None
            t = None

        # (C-mat) add read counts conditioning to time hiddens
        if self.cond_on_read_count:
            HC_read_count_hiddens = self.to_read_count_hiddens(HC_read_count)
            HC_read_count_tokens = self.to_HC_read_count_tokens(HC_read_count_hiddens)
            HC_read_count_t = self.to_HC_read_count_cond(HC_read_count_hiddens)

            if t is not None:
                t = t + HC_read_count_t
            else:
                t = HC_read_count_t

            if time_tokens is not None:
                time_tokens = torch.cat((time_tokens, HC_read_count_tokens), dim = -2)
            else:
                time_tokens = HC_read_count_tokens

        # (C-mat) add seperation conditioning to time hiddens

        if self.cond_on_seperation:
            seperation_hiddens = self.to_seperation_hiddens(seperation)
            seperation_tokens = self.to_seperation_tokens(seperation_hiddens)
            seperation_t = self.to_seperation_cond(seperation_hiddens)

            if t is not None:
                t = t + seperation_t
            else:
                t = seperation_t

            if time_tokens is not None:
                time_tokens = torch.cat((time_tokens, seperation_tokens), dim = -2)
            else:
                time_tokens = seperation_tokens

        # main conditioning tokens (c)
        # normalize conditioning tokens
        if time_tokens is not None:
            c = self.norm_cond(time_tokens)
        else:
            c = None
        
        x = self.init_merge_resnet(x, t, c)

        if self.init_conv_to_final_conv_residual:
            init_conv_residual = x.clone()

        # initial resnet block (for memory efficient unet)

        if exists(self.init_resnet_block):
            x = self.init_resnet_block(x, t)

        # go through the layers of the unet, down and up

        hiddens = []

        for pre_downsample, init_block, resnet_blocks, attn_block, post_downsample in self.downs:
            if exists(pre_downsample):
                x = pre_downsample(x)

            x = init_block(x, t, c)

            for resnet_block in resnet_blocks:
                x = resnet_block(x, t)
                hiddens.append(x)

            x = attn_block(x, c)
            hiddens.append(x)

            if exists(post_downsample):
                x = post_downsample(x)

        x = self.mid_block1(x, t, c)

        if exists(self.mid_attn):
            x = self.mid_attn(x)

        x = self.mid_block2(x, t, c)

        add_skip_connection = lambda x: torch.cat((x, hiddens.pop() * self.skip_connect_scale), dim = 1)

        up_hiddens = []

        for init_block, resnet_blocks, attn_block, upsample in self.ups:
            x = add_skip_connection(x)
            x = init_block(x, t, c)

            for resnet_block in resnet_blocks:
                x = add_skip_connection(x)
                x = resnet_block(x, t)

            x = attn_block(x, c)
            up_hiddens.append(x.contiguous())
            x = upsample(x)

        # whether to combine all feature maps from upsample blocks

        x = self.upsample_combiner(x, up_hiddens)

        # final top-most residual if needed

        if self.init_conv_to_final_conv_residual:
            x = torch.cat((x, init_conv_residual), dim = 1)

        if exists(self.final_res_block):
            x = self.final_res_block(x, t)

        return self.final_conv(x)
