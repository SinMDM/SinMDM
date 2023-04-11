from typing import Optional, Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

Array = Any
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any

class FusedQnA(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        heads: int,
        kernel_size: int,
        stride: int,
        padding: int,
        output_features: Optional[int] = None,
        normalize_q: bool = True,
        use_relative_pos: bool = True,
        num_queries: int = 1,
        use_bias: bool = True,
    ):
        super(FusedQnA, self).__init__()
        # Init params:
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.heads = heads
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_features = output_features
        self.normalize_q = normalize_q
        self.use_relative_pos = use_relative_pos
        self.num_queries = num_queries
        self.use_bias = use_bias
        # Init modules:
        self.to_v = nn.Conv2d(self.in_features, self.hidden_features, (1, 1), (1, 1), 0, bias=self.use_bias)
        self.to_out = nn.Conv2d(
            self.hidden_features,
            self.hidden_features if self.output_features is None else self.output_features,
            (1, 1),
            (1, 1),
            0,
            bias=self.use_bias
        )
        stddev = np.sqrt(1.0 / (self.hidden_features // self.heads))
        self.query = nn.Parameter(torch.normal(0.0, stddev, (self.hidden_features,)))
        self.Wk = nn.Parameter(torch.normal(0.0, stddev, (self.in_features, self.hidden_features,)))
        self.Wk_bias = nn.Parameter(torch.zeros((self.hidden_features,)))
        if self.use_relative_pos:
            self.rpe = nn.Parameter(torch.zeros((self.heads, 1, self.kernel_size, self.kernel_size)))
        else:
            self.rep = None

    def _compute_QK_scores(self, q, x):
        """Computes the QK dot product in fused manner.
        Since the queries are shared across windows, we compute (Q*W_k^T)X^T for better memory utilization.

        :param q: The learned queries of shape [B, N_Queries, Heads, Head_dim]
        :param x: The input features [B, C, H, W]
        :return: The query-key dot product for each query head [B, N_Queries, Heads, H, W]
        """
        # WK = [D_in, h, D]
        Wk = self.Wk.view([self.in_features, self.heads, self.hidden_features // self.heads])
        qWk = torch.einsum('Bhd,Dhd->BDh', q, Wk)
        qWkx = torch.einsum('BDHW,BDh->BhHW', x, qWk)
        if self.use_bias:
            Wk_b = self.Wk_bias.view(self.heads, self.hidden_features // self.heads)
            qWk_b = torch.einsum('hd,Bhd->Bh', Wk_b, q)[..., None, None]
            qWkx = qWkx + qWk_b
        return qWkx

    def _compute_attention(self, cost, v, rpe=None):
        """Compute the attention in memory efficent manner (see paper: https://arxiv.org/abs/2112.11435)"""
        B, _, H, W = v.shape
        cost_exp = torch.exp(cost)  # [B, heads, H, W]
        h_dim = self.hidden_features // self.heads
        v_cost_exp = cost_exp[:, :, None, ...] * v.view(B, self.heads, h_dim, H, W)  # [B, , heads, head_dim, H, W]
        v_cost_exp = v_cost_exp.view(B, self.hidden_features, H, W)
        if rpe is not None:
            rpe_exp = torch.exp(rpe)  # [k, 1, h,w]
            summation_kernel = rpe_exp
            summation_kernel = torch.repeat_interleave(summation_kernel, repeats=h_dim, dim=0)
        else:
            summation_kernel = torch.ones(self.hidden_features, 1, self.kernel_size, self.kernel_size).to(v_cost_exp)
        I = v_cost_exp
        sum_v_cost_exp = F.conv2d(I, summation_kernel, stride=self.stride, padding=self.padding,
                                  groups=self.hidden_features)

        h_out, w_out = sum_v_cost_exp.shape[2], sum_v_cost_exp.shape[3]
        sum_v_cost_exp = sum_v_cost_exp.view(B, self.heads, h_dim, h_out, w_out)
        I = cost_exp.reshape([B, -1, H, W])
        summation_kernel = rpe_exp if rpe is not None else torch.ones(self.heads, 1, self.kernel_size,
                                                                      self.kernel_size).to(
            v_cost_exp)
        sum_cost_exp = F.conv2d(I,
                                summation_kernel,
                                stride=self.stride,
                                padding=self.padding,
                                groups=self.heads,
                                ).view(B, self.heads, 1, h_out, w_out)
        out = sum_v_cost_exp / sum_cost_exp
        out = out.reshape([B, self.hidden_features, h_out, w_out])
        return out

    def forward(
        self,
        x,
        global_pe: Optional[torch.Tensor] = None,
    ):
        B, Din, H, W = x.shape
        # Prepare query
        q = torch.broadcast_to(self.query, [x.shape[0], self.hidden_features]).view(B, self.heads,
                                                                                    self.hidden_features // self.heads)
        q = q / (np.sqrt(q.shape[-1]) + 1e-6)
        if self.normalize_q:
            q = q / (torch.linalg.norm(q, dim=-1, keepdim=True) + 1e-6)

        qk_score = self._compute_QK_scores(q, x)

        v = self.to_v(x)

        out_per_head_concat = self._compute_attention(qk_score, v, self.rpe)
        out = self.to_out(out_per_head_concat)
        return out


class FusedQnA1d(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        heads: int,
        kernel_size: int,
        stride: int,
        padding: int,
        output_features: Optional[int] = None,
        normalize_q: bool = True,
        use_relative_pos: bool = True,
        num_queries: int = 1,
        use_bias: bool = True,
        axial_dim=1,
        timesteps_features: Optional[int] = None,
    ):
        super(FusedQnA1d, self).__init__()
        # Init params:
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.heads = heads
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_features = output_features
        self.normalize_q = normalize_q
        self.use_relative_pos = use_relative_pos
        self.num_queries = num_queries
        self.use_bias = use_bias
        self.axial_dim = axial_dim
        # Init modules:
        self.to_v = nn.Conv2d(self.in_features, self.hidden_features, (1, 1), (1, 1), 0, bias=self.use_bias)
        self.to_out = nn.Conv2d(
            self.hidden_features,
            self.hidden_features if self.output_features is None else self.output_features,
            (1, 1),
            (1, 1),
            0,
            bias=self.use_bias
        )
        if timesteps_features is not None:
            self.timesteps_embed = nn.Sequential(
                nn.SiLU(),
                nn.Linear(timesteps_features, self.hidden_features)
            )
        else:
            self.timesteps_embed = None
        stddev = np.sqrt(1.0 / (self.hidden_features // self.heads))
        self.query = nn.Parameter(torch.normal(0.0, stddev, (self.hidden_features,)))
        self.Wk = nn.Parameter(torch.normal(0.0, stddev, (self.in_features, self.hidden_features,)))
        self.Wk_bias = nn.Parameter(torch.zeros((self.hidden_features,)))
        if self.use_relative_pos:
            self.rpe = nn.Parameter(torch.zeros((self.heads, 1, self.axial_dim, self.kernel_size)))
        else:
            self.rep = None

    def _compute_QK_scores(self, q, x):
        """Computes the QK dot product in fused manner.
        Since the queries are shared across windows, we compute (Q*W_k^T)X^T for better memory utilization.

        :param q: The learned queries of shape [B, N_Queries, Heads, Head_dim]
        :param x: The input features [B, C, H, W]
        :return: The query-key dot product for each query head [B, N_Queries, Heads, H, W]
        """
        # WK = [D_in, h, D]
        Wk = self.Wk.view([self.in_features, self.heads, self.hidden_features // self.heads])
        qWk = torch.einsum('Bhd,Dhd->BDh', q, Wk)
        qWkx = torch.einsum('BDHW,BDh->BhHW', x, qWk)
        if self.use_bias:
            Wk_b = self.Wk_bias.view(self.heads, self.hidden_features // self.heads)
            qWk_b = torch.einsum('hd,Bhd->Bh', Wk_b, q)[..., None, None]
            qWkx = qWkx + qWk_b
        return qWkx

    def _compute_attention(self, cost, v, rpe=None):
        """Compute the attention in memory efficent manner (see paper: https://arxiv.org/abs/2112.11435)"""
        B, _, H, W = v.shape
        cost_exp = torch.exp(cost)  # [B, heads, H, W]
        h_dim = self.hidden_features // self.heads
        v_cost_exp = cost_exp[:, :, None, ...] * v.view(B, self.heads, h_dim, H, W)  # [B, , heads, head_dim, H, W]
        v_cost_exp = v_cost_exp.view(B, self.hidden_features, H, W)
        if rpe is not None:
            rpe_exp = torch.exp(rpe)  # [k, 1, h,w]
            summation_kernel = rpe_exp
            summation_kernel = torch.repeat_interleave(summation_kernel, repeats=h_dim, dim=0)
        else:
            summation_kernel = torch.ones(self.hidden_features, 1, self.axial_dim, self.kernel_size).to(v_cost_exp)
        I = v_cost_exp
        sum_v_cost_exp = F.conv2d(I, summation_kernel, stride=(1, self.stride), padding=(0, self.padding),
                                  groups=self.hidden_features)

        h_out, w_out = sum_v_cost_exp.shape[2], sum_v_cost_exp.shape[3]
        sum_v_cost_exp = sum_v_cost_exp.view(B, self.heads, h_dim, h_out, w_out)
        I = cost_exp.reshape([B, -1, H, W])
        summation_kernel = rpe_exp if rpe is not None else torch.ones(self.heads, 1, self.axial_dim,
                                                                      self.kernel_size).to(
            v_cost_exp)
        sum_cost_exp = F.conv2d(I,
                                summation_kernel,
                                stride=(1, self.stride),
                                padding=(0, self.padding),
                                groups=self.heads,
                                ).view(B, self.heads, 1, h_out, w_out)
        out = sum_v_cost_exp / sum_cost_exp
        out = out.reshape([B, self.hidden_features, h_out, w_out])
        return out

    def forward(
        self,
        x,
        timesteps: Optional[torch.Tensor] = None,
        global_pe: Optional[torch.Tensor] = None,
    ):
        assert x.shape[2] == self.axial_dim
        B, Din, H, W = x.shape
        if self.timesteps_embed is not None and timesteps is not None:
            q = self.timesteps_embed(timesteps)
            q = q.view(B, self.heads, self.hidden_features // self.heads)
        else:
            # Prepare query
            q = torch.broadcast_to(self.query, [x.shape[0], self.hidden_features]).view(B, self.heads,
                                                                                        self.hidden_features // self.heads)
        if self.normalize_q:
            q = q / (torch.linalg.norm(q, dim=-1, keepdim=True) + 1e-6)

        q = q / np.sqrt(q.shape[-1])

        x_with_pos = x if global_pe is None else x + global_pe
        qk_score = self._compute_QK_scores(q, x_with_pos)

        v = self.to_v(x)

        out_per_head_concat = self._compute_attention(qk_score, v, self.rpe)
        out = self.to_out(out_per_head_concat)
        return out


class FusedUpQnA1d(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        heads: int,
        kernel_size: int,
        stride: int,
        padding: int,
        scale_factor: int = 2,
        output_features: Optional[int] = None,
        normalize_q: bool = True,
        use_relative_pos: bool = True,
        use_bias: bool = True,
        timesteps_features: Optional[int] = None,
    ):
        super(FusedUpQnA1d, self).__init__()
        # Init params:
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.heads = heads
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_features = output_features
        self.normalize_q = normalize_q
        self.use_relative_pos = use_relative_pos
        self.use_bias = use_bias
        self.scale_factor = scale_factor
        # Init modules:
        self.to_v = nn.Conv2d(self.in_features, self.hidden_features, (1, 1), (1, 1), 0, bias=self.use_bias)
        self.to_out = nn.Conv2d(
            self.hidden_features,
            self.hidden_features if self.output_features is None else self.output_features,
            (1, 1),
            (1, 1),
            0,
            bias=self.use_bias
        )
        if timesteps_features is not None:
            self.timesteps_embed = nn.Sequential(
                nn.SiLU(),
                nn.Linear(timesteps_features, self.scale_factor * self.hidden_features)
            )
        else:
            self.timesteps_embed = None
        stddev = np.sqrt(1.0 / (self.hidden_features // self.heads))
        self.query = nn.Parameter(torch.normal(0.0, stddev, (self.scale_factor * self.hidden_features,)))
        self.Wk = nn.Parameter(torch.normal(0.0, stddev, (self.in_features, self.hidden_features,)))
        self.Wk_bias = nn.Parameter(torch.zeros((self.hidden_features,)))
        if self.use_relative_pos:
            self.rpe = nn.Parameter(torch.zeros((self.scale_factor, self.heads, 1, self.axial_dim, self.kernel_size)))
        else:
            self.rep = None

    def _compute_QK_scores(self, q, x):
        """Computes the QK dot product in fused manner.
        Since the queries are shared across windows, we compute (Q*W_k^T)X^T for better memory utilization.

        :param q: The learned queries of shape [B, N_Queries, Heads, Head_dim]
        :param x: The input features [B, C, H, W]
        :return: The query-key dot product for each query head [B, N_Queries, Heads, H, W]
        """
        # WK = [D_in, h, D]
        Wk = self.Wk.view([self.in_features, self.heads, self.hidden_features // self.heads])
        qWk = torch.einsum('Bqhd,Dhd->BDqh', q, Wk)
        qWkx = torch.einsum('BDHW,BDqh->BqhHW', x, qWk)
        if self.use_bias:
            Wk_b = self.Wk_bias.view(self.heads, self.hidden_features // self.heads)
            qWk_b = torch.einsum('Bqhd,hd->Bqh', q, Wk_b)[..., None, None]
            qWkx = qWkx + qWk_b
        return qWkx

    def _compute_attention(self, cost, v, rpe=None):
        """Compute the attention in memory efficent manner (see paper: https://arxiv.org/abs/2112.11435)"""
        B, _, H, W = v.shape
        cost_exp = torch.exp(cost)  # [B, q, heads, H, W]
        h_dim = self.hidden_features // self.heads
        v_cost_exp = cost_exp[:, :, :, None, ...] * v.view(B, 1, self.heads, h_dim, H,
                                                           W)  # [B, q, heads, head_dim, H, W]
        v_cost_exp = v_cost_exp.view(B, self.scale_factor * self.hidden_features, H, W)
        if rpe is not None:
            rpe_exp = torch.exp(rpe)  # [k, 1, h,w]
            summation_kernel = rpe_exp
            summation_kernel = torch.repeat_interleave(
                summation_kernel, repeats=h_dim, dim=1
            ).view(self.scale_factor * self.hidden_features, 1, self.axial_dim, self.kernel_size)
        else:
            summation_kernel = torch.ones(self.scale_factor * self.hidden_features, 1, self.axial_dim, self.kernel_size).to(
                v_cost_exp)
        I = v_cost_exp
        sum_v_cost_exp = F.conv2d(I,
                                  summation_kernel,
                                  stride=(1, self.stride),
                                  padding=(0, self.padding),
                                  groups=self.scale_factor * self.hidden_features
                                  )

        h_out, w_out = sum_v_cost_exp.shape[2], sum_v_cost_exp.shape[3]
        sum_v_cost_exp = sum_v_cost_exp.view(B, self.scale_factor, self.heads, h_dim, h_out, w_out)
        I = cost_exp.reshape([B, -1, H, W])
        summation_kernel = (
            rpe_exp.view(self.scale_factor * self.heads, 1, self.axial_dim, self.kernel_size)
            if rpe is not None else
            torch.ones(self.heads, 1, self.axial_dim, self.kernel_size).to(v_cost_exp)
        )
        sum_cost_exp = F.conv2d(I,
                                summation_kernel,
                                stride=(1, self.stride),
                                padding=(0, self.padding),
                                groups=self.scale_factor * self.heads,
                                ).view(B, self.scale_factor, self.heads, 1, h_out, w_out)
        out = sum_v_cost_exp / sum_cost_exp
        out = out.view(B, self.scale_factor, self.hidden_features, h_out, w_out)
        out = out.permute(0, 2, 3, 1, 4)
        out = out.reshape([B, self.hidden_features, h_out, self.scale_factor * w_out])
        return out

    def forward(
        self,
        x,
        timesteps: Optional[torch.Tensor] = None,
        global_pe: Optional[torch.Tensor] = None,
    ):
        assert x.shape[2] == self.axial_dim
        B, Din, H, W = x.shape
        if self.timesteps_embed is not None and timesteps is not None:
            q = self.timesteps_embed(timesteps)
            q = q.view(
                B, self.scale_factor, self.heads, self.hidden_features // self.heads
            )
        else:
            # Prepare query
            # assert False, "Don't unset timesteps unless explicitly done for testing"
            q = torch.broadcast_to(
                self.query, [x.shape[0], self.scale_factor * self.hidden_features]
            ).view(
                B, self.scale_factor, self.heads, self.hidden_features // self.heads
            )
        if self.normalize_q:
            q = q / (torch.linalg.norm(q, dim=-1, keepdim=True) + 1e-6)
        q = q / np.sqrt(q.shape[-1])

        qk_score = self._compute_QK_scores(q, x)

        v = self.to_v(x)

        out_per_head_concat = self._compute_attention(qk_score, v, self.rpe)
        out = self.to_out(out_per_head_concat)
        return out
