from abc import abstractmethod

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops.stochastic_depth import StochasticDepth

from models.qna import FusedQnA1d, FusedUpQnA1d

GATE_SCALE = False


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, ):
        super().__init__()

    def forward(self, x, temb=None):
        x = F.interpolate(x, (1, x.shape[3] * 2), mode="bilinear")
        return x


class QnADownBlock(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self,
                 channels,
                 out_channels,
                 timesteps_features,
                 head_dim,
                 use_timestep_query,
                 ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels
        self.use_timestep_query = use_timestep_query
        self.norm = LayerNorm2d(channels, eps=1e-6, elementwise_affine=True)
        self.shiftscale = ShiftAndScale(channels, timesteps_features)
        self.op = FusedQnA1d(
            in_features=channels,
            hidden_features=out_channels,
            timesteps_features=timesteps_features if self.use_timestep_query else None,
            heads=out_channels // head_dim,
            kernel_size=2,
            stride=2,
            padding=0,
        )

    def forward(self, x, temb):
        x = self.norm(x)
        x = self.shiftscale(x, temb)
        x = self.op(x, temb if self.use_timestep_query else None)
        return x


class QnAUpBlock(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self,
                 channels,
                 out_channels,
                 timesteps_features,
                 head_dim,
                 use_timestep_query,
                 ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels
        self.use_timestep_query = use_timestep_query
        self.norm = LayerNorm2d(channels, eps=1e-6, elementwise_affine=True)
        self.shiftscale = ShiftAndScale(channels, timesteps_features)
        self.gs = nn.Sequential(
            nn.SiLU(),
            nn.Linear(timesteps_features, self.out_channels)

        )
        self.op = FusedUpQnA1d(
            in_features=channels,
            hidden_features=channels,
            output_features=out_channels,
            scale_factor=4,
            timesteps_features=timesteps_features if self.use_timestep_query else None,
            heads=out_channels // head_dim,
            kernel_size=2,
            stride=2,
            padding=0,
        )

    def forward(self, x, temb):
        x = self.norm(x)
        x = self.shiftscale(x, temb)
        x = self.op(x, temb if self.use_timestep_query else None)
        if GATE_SCALE:
            x = x * (1.0 + self.gs(temb)[:, :, None, None])
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, out_channels):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels
        self.op = nn.Conv2d(
            in_channels=self.channels,
            out_channels=self.out_channels,
            kernel_size=(1, 2),
            stride=(1, 2),
            padding=0
        )

    def forward(self, x, temb=None):
        return self.op(x)


class MLP2d(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, drouput=0.0):
        super(MLP2d, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.conv1 = nn.Conv2d(self.in_features, self.hidden_features, kernel_size=(1, 1), stride=1, padding=0)
        self.conv2 = nn.Conv2d(self.hidden_features, self.out_features, kernel_size=(1, 1), stride=1, padding=0)
        self.dropout = nn.Dropout(drouput) if drouput > 0.0 else nn.Identity()
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.conv2(self.dropout(self.activation(self.conv1(x))))
        return x


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class ShiftAndScale(nn.Module):
    def __init__(self, features, cond_features):
        super(ShiftAndScale, self).__init__()
        self.features = features
        self.cond_features = cond_features
        self.cond_map = nn.Sequential(nn.SiLU(),
                                      nn.Linear(cond_features, 2 * features)
                                      )

    def forward(self, x, cond):
        shift, scale = self.cond_map(cond).chunk(2, dim=1)
        x = x * (1.0 + scale[..., None, None]) + shift[..., None, None]
        return x


class QnABlock2d(nn.Module):
    def __init__(self,
                 features,
                 head_dim,
                 timesteps_features,
                 kernel_size=7,
                 dropout=0.0,
                 drop_path_prob=0.0,
                 use_timestep_query=False,
                 in_features=None,
                 axial_dim=1,
                 ):
        super(QnABlock2d, self).__init__()
        in_features = features if in_features is None else in_features
        self.drop_path_prob = drop_path_prob
        self.features = features
        self.head_dim = head_dim
        self.timesteps_features = timesteps_features
        self.heads = self.features // self.head_dim
        self.norm1 = LayerNorm2d(in_features, eps=1e-6, elementwise_affine=True)
        self.shiftscale1 = ShiftAndScale(in_features, timesteps_features)
        self.qna_layer = FusedQnA1d(
            in_features=in_features,
            timesteps_features=timesteps_features if use_timestep_query else None,
            hidden_features=features,
            heads=self.heads,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            axial_dim=axial_dim
        )
        if in_features != features:
            self.skip_process = nn.Conv2d(in_channels=in_features,
                                          out_channels=features,
                                          kernel_size=(1, 1),
                                          stride=(1, 1),
                                          padding=(0, 0)
                                          )
        else:
            self.skip_process = nn.Identity()
        self.drop_path1 = (
            StochasticDepth(self.drop_path_prob, mode="row") if self.drop_path_prob > 0.0 else nn.Identity())
        self.norm2 = LayerNorm2d(features, eps=1e-6, elementwise_affine=True)
        self.shiftscale2 = ShiftAndScale(features, timesteps_features)
        self.mlp_layer = MLP2d(in_features=features,
                               hidden_features=4 * features,
                               out_features=features,
                               drouput=dropout)
        self.drop_path2 = (
            StochasticDepth(self.drop_path_prob, mode="row") if self.drop_path_prob > 0.0 else nn.Identity())
        self.skip_gate = nn.Sequential(
            nn.SiLU(),
            nn.Linear(timesteps_features, 2 * features)
        )

    def forward(self, x, timesteps, global_pe=None):
        skip = x
        gs1, gs2 = self.skip_gate(timesteps).chunk(2, dim=1)
        x = self.norm1(x)
        x = self.shiftscale1(x, timesteps)
        x = self.qna_layer(x, timesteps, global_pe)
        if GATE_SCALE:
            x = (1.0 + gs1[:, :, None, None]) * x
        x = skip = self.skip_process(skip) + self.drop_path1(x)
        x = self.norm2(x)
        x = self.shiftscale2(x, timesteps)
        x = self.mlp_layer(x)
        if GATE_SCALE:
            x = (1.0 + gs2[:, :, None, None]) * x
        x = skip + self.drop_path2(x)
        return x


class CustomIdentity(nn.Module):
    def __init__(self, ret_idx=0):
        super(CustomIdentity, self).__init__()
        self.ret_idx = ret_idx

    def forward(self, *args):
        return args[self.ret_idx]


class QnAMDM(nn.Module):
    def __init__(
        self,
        njoints,
        nfeats,
        data_rep,
        latent_dim=256,
        num_layers=4,
        dataset='amass',
        arch='trans_enc',
        # QnA specific
        num_downsample=0,
        drop_path=0.0,
        diffusion_query=False,
        head_dim=16,
        kernel_size=3,
        use_global_pe=False,
        **kwargs
    ):
        super(QnAMDM, self).__init__()
        # ================================================
        # #################### LEGACY ####################
        # ================================================
        self.nfeats = nfeats
        self.njoints = njoints
        self.dataset = dataset
        self.data_rep = data_rep
        # self.rot2xyz = Rotation2xyz(device='cpu', dataset=self.dataset)
        self.input_feats = self.njoints * self.nfeats
        self.arch = arch
        self.cond_mode = "None"
        # ================================================
        # ################################################
        # ================================================
        self.head_dim = head_dim
        time_embed_dim = latent_dim * 4
        self.time_embed = nn.Sequential(
            nn.Linear(latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.time_embed_dim = time_embed_dim
        self.latent_dim = latent_dim
        self.input_proj = nn.Conv2d(njoints * nfeats, self.latent_dim, kernel_size=(1, 1))
        self.output_proj = nn.Conv2d(self.latent_dim, njoints * nfeats, kernel_size=(1, 1))
        self.output_norm = LayerNorm2d(self.latent_dim, eps=1e-6, elementwise_affine=True)
        self.output_shift_and_scale = ShiftAndScale(self.latent_dim, time_embed_dim)
        self.use_global_pe = use_global_pe
        if use_global_pe:
            self.global_pe = nn.Parameter(torch.zeros(1, self.latent_dim, 1, 392))
        total_num_of_layers_for_drop_path = num_layers * (num_downsample + 1)
        if drop_path > 0.0:
            assert num_downsample == 0, "We don't support drop_path for now"
            drop_path_list = list(np.linspace(0.0, drop_path, total_num_of_layers_for_drop_path))
        else:
            drop_path_list = [0.0] * total_num_of_layers_for_drop_path
        self.hierarchy_blocks = nn.ModuleDict()
        self.hierarchy_downsample = nn.ModuleDict()
        self.hierarchy_upsample = nn.ModuleDict()
        self.num_downsample = num_downsample
        self.num_layers = num_layers
        assert (
            num_layers % 2 == 0 or num_downsample == 0
        ), F"When downsample is set, use even number of layers per hierarchy"
        for i in range(num_downsample + 1):
            qna_blocks = nn.ModuleList()
            for j in range(num_layers):
                drop_path_prob = drop_path_list.pop(0)
                qna_blocks.append(
                    QnABlock2d(
                        features=self.latent_dim * (2 ** i),
                        timesteps_features=time_embed_dim,
                        head_dim=self.head_dim,
                        kernel_size=kernel_size,
                        drop_path_prob=drop_path_prob,
                        use_timestep_query=diffusion_query,
                        in_features=(
                            self.latent_dim * (3 * 2 ** i) if (j == num_layers // 2 and i != num_downsample) else None
                        ),
                        axial_dim=1,
                    )
                )
            self.hierarchy_blocks[f"h{i}"] = qna_blocks
            self.hierarchy_downsample[f"h{i}"] = (
                Downsample(self.latent_dim * (2 ** i), self.latent_dim * (2 ** (i + 1)))
                if i != num_downsample else
                CustomIdentity()
            )
            self.hierarchy_upsample[f"h{i}"] = Upsample() if i != 0 else CustomIdentity()

        # self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

            if isinstance(module, nn.Conv2d):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.input_proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.time_embed[0].weight, std=0.02)
        nn.init.normal_(self.time_embed[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for h in self.hierarchy_blocks.keys():
            for block in self.hierarchy_blocks[h]:
                nn.init.constant_(block.shiftscale1.cond_map[-1].weight, 0)
                nn.init.constant_(block.shiftscale1.cond_map[-1].bias, 0)
                nn.init.constant_(block.shiftscale2.cond_map[-1].weight, 0)
                nn.init.constant_(block.shiftscale2.cond_map[-1].bias, 0)
                nn.init.constant_(block.skip_gate[-1].weight, 0)
                nn.init.constant_(block.skip_gate[-1].bias, 0)
            if isinstance(self.hierarchy_downsample[h], QnADownBlock):
                nn.init.constant_(self.hierarchy_downsample[h].shiftscale.cond_map[-1].weight, 0)
                nn.init.constant_(self.hierarchy_downsample[h].shiftscale.cond_map[-1].bias, 0)
            if isinstance(self.hierarchy_upsample[h], QnAUpBlock):
                nn.init.constant_(self.hierarchy_upsample[h].shiftscale.cond_map[-1].weight, 0)
                nn.init.constant_(self.hierarchy_upsample[h].shiftscale.cond_map[-1].bias, 0)
                nn.init.normal_(self.hierarchy_upsample[h].gs[-1].weight, 0.0, std=0.02)
                nn.init.constant_(self.hierarchy_upsample[h].gs[-1].bias, 0.0)

        # Zero-out output layers:
        nn.init.constant_(self.output_shift_and_scale.cond_map[-1].weight, 0)
        nn.init.constant_(self.output_shift_and_scale.cond_map[-1].bias, 0)
        nn.init.constant_(self.output_proj.weight, 0)
        nn.init.constant_(self.output_proj.bias, 0)

    def _apply_hierarchy(self, x, temb, level, global_pe):
        if level > self.num_downsample:
            return None
        level_key = f"h{level}"
        # Process current half of hierarchy
        blocks = self.hierarchy_blocks[level_key]
        hs = []
        for i in range(self.num_layers // 2):
            x = blocks[i](x, temb, global_pe)
            if level != self.num_downsample:
                hs.append(x)

        # Call next hierarchy:
        x_next = self.hierarchy_downsample[level_key](x, temb)
        x_next = self._apply_hierarchy(x_next, temb, level + 1, global_pe)
        if x_next is not None:
            if x_next.shape[2:] != x.shape[2:]:
                x_next = F.interpolate(x_next, x.shape[2:], mode='bilinear')
            x = torch.cat([x, x_next], dim=1)

        # Continue processing merged hierarchies and upsample for upper hierarchy
        for i in range(self.num_layers // 2, self.num_layers, 1):
            x = blocks[i](x, temb, global_pe)
        x = self.hierarchy_upsample[level_key](x, temb)
        return x

    def forward(self, x, timesteps, y=None):
        b, njoints, nfeats, nframes = x.shape
        temb = self.time_embed(timestep_embedding(timesteps, self.latent_dim))
        x = x.view(b, njoints * nfeats, 1, nframes)
        x = self.input_proj(x)
        if self.use_global_pe:
            if self.global_pe.shape[-1] < nframes:
                global_pe = F.interpolate(self.global_pe, (1, nframes), mode='bilinear')
            else:
                global_pe = self.global_pe[..., :nframes]
            global_pe = torch.broadcast_to(global_pe, (b, x.shape[1], 1, nframes))
        else:
            global_pe = None
        # x = x + global_pe
        # for block in self.qna_blocks:
        #     x = block(x, temb)
        x = self._apply_hierarchy(x, temb, 0, global_pe)
        x = self.output_norm(x)
        x = self.output_shift_and_scale(x, temb)
        x = self.output_proj(x)
        x = x.view(b, njoints, nfeats, nframes)
        return x
