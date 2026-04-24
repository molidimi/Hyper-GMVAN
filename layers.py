# -*- coding: utf-8 -*-
from DiffDGMN_main import gol
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import add_self_loops, softmax
from torch_geometric.nn.aggr import MultiAggregation
from torchsde import sdeint
import math

import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, num_hidden_layers=1, dropout=0.0, act='relu'):
        super(MLP, self).__init__()
        layers = []
        in_dim = input_dim
        activation = nn.ReLU() if act == 'relu' else nn.GELU()
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dims))
            layers.append(activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dims
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class KANLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        result = solution.permute(2, 0, 1)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output
        
        output = output.reshape(*original_shape[:-1], self.out_features)
        return output


class KAN(nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

class GTConv(MessagePassing):
    def __init__(
        self,
        node_in_dim: int,
        hidden_dim: int,
        edge_in_dim: int = None,
        num_heads: int = 8,
        gate: bool = False,
        qkv_bias: bool = False,
        dropout: float = 0.0,
        norm: str = "ln",
        act: str = "relu",
        aggregators = ["sum"],
    ):
        super().__init__(node_dim=0, aggr=MultiAggregation(aggregators, mode="cat"))

        assert "sum" in aggregators
        assert hidden_dim % num_heads == 0
        assert (edge_in_dim is None) or (edge_in_dim > 0)

        self.aggregators = aggregators
        self.num_aggrs = len(aggregators)

        self.WQ = nn.Linear(node_in_dim, hidden_dim, bias=qkv_bias)
        self.WK = nn.Linear(node_in_dim, hidden_dim, bias=qkv_bias)
        self.WV = nn.Linear(node_in_dim, hidden_dim, bias=qkv_bias)
        self.WO = nn.Linear(hidden_dim * self.num_aggrs, node_in_dim, bias=True)

        if edge_in_dim is not None:
            self.WE = nn.Linear(edge_in_dim, hidden_dim, bias=True)
            self.WOe = nn.Linear(hidden_dim, edge_in_dim, bias=True)
            self.ffn_e = MLP(
                input_dim=edge_in_dim,
                output_dim=edge_in_dim,
                hidden_dims=hidden_dim,
                num_hidden_layers=1,
                dropout=dropout,
                act=act,
            )
            if norm.lower() in ["bn", "batchnorm", "batch_norm"]:
                self.norm1e = nn.BatchNorm1d(edge_in_dim)
                self.norm2e = nn.BatchNorm1d(edge_in_dim)
            elif norm.lower() in ["ln", "layernorm", "layer_norm"]:
                self.norm1e = nn.LayerNorm(edge_in_dim)
                self.norm2e = nn.LayerNorm(edge_in_dim)
            else:
                raise ValueError
        else:
            self.WE = self.register_parameter("WE", None)
            self.WOe = self.register_parameter("WOe", None)
            self.ffn_e = self.register_parameter("ffn_e", None)
            self.norm1e = self.register_parameter("norm1e", None)
            self.norm2e = self.register_parameter("norm2e", None)

        if norm.lower() in ["bn", "batchnorm", "batch_norm"]:
            self.norm1 = nn.BatchNorm1d(node_in_dim)
            self.norm2 = nn.BatchNorm1d(node_in_dim)
        elif norm.lower() in ["ln", "layernorm", "layer_norm"]:
            self.norm1 = nn.LayerNorm(node_in_dim)
            self.norm2 = nn.LayerNorm(node_in_dim)
        else:
            raise ValueError

        if gate:
            assert edge_in_dim is not None
            self.n_gate = nn.Linear(node_in_dim, hidden_dim, bias=True)
            self.e_gate = nn.Linear(edge_in_dim, hidden_dim, bias=True)
        else:
            self.n_gate = self.register_parameter("n_gate", None)
            self.e_gate = self.register_parameter("e_gate", None)

        self.dropout_layer = nn.Dropout(p=dropout)

        self.ffn = MLP(
            input_dim=node_in_dim,
            output_dim=node_in_dim,
            hidden_dims=hidden_dim,
            num_hidden_layers=1,
            dropout=dropout,
            act=act,
        )

        self.num_heads = num_heads
        self.node_in_dim = node_in_dim
        self.edge_in_dim = edge_in_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.norm = norm.lower()
        self.gate = gate
        self.qkv_bias = qkv_bias

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.WQ.weight)
        nn.init.xavier_uniform_(self.WK.weight)
        nn.init.xavier_uniform_(self.WV.weight)
        nn.init.xavier_uniform_(self.WO.weight)
        if self.edge_in_dim is not None:
            nn.init.xavier_uniform_(self.WE.weight)
            nn.init.xavier_uniform_(self.WOe.weight)

    def forward(self, x, edge_index, edge_attr=None):
        x_ = x
        edge_attr_ = edge_attr

        Q = self.WQ(x).view(-1, self.num_heads, self.hidden_dim // self.num_heads)
        K = self.WK(x).view(-1, self.num_heads, self.hidden_dim // self.num_heads)
        V = self.WV(x).view(-1, self.num_heads, self.hidden_dim // self.num_heads)
        if self.gate:
            G = self.n_gate(x).view(-1, self.num_heads, self.hidden_dim // self.num_heads)
        else:
            G = torch.ones_like(V)

        out = self.propagate(edge_index, Q=Q, K=K, V=V, G=G, edge_attr=edge_attr, size=None)
        out = out.view(-1, self.hidden_dim * self.num_aggrs)

        out = self.dropout_layer(out)
        out = self.WO(out) + x_
        out = self.norm1(out)
        ffn_in = out
        out = self.ffn(out)
        out = self.norm2(ffn_in + out)

        if self.edge_in_dim is None:
            out_eij = None
        else:
            out_eij = self._eij
            self._eij = None
            out_eij = out_eij.view(-1, self.hidden_dim)

            out_eij = self.dropout_layer(out_eij)
            out_eij = self.WOe(out_eij) + edge_attr_
            out_eij = self.norm1e(out_eij)
            ffn_eij_in = out_eij
            out_eij = self.ffn_e(out_eij)
            out_eij = self.norm2e(ffn_eij_in + out_eij)

        return out, out_eij

    def message(self, Q_i, K_j, V_j, G_j, index, edge_attr=None):
        d_k = Q_i.size(-1)
        qijk = (Q_i * K_j) / math.sqrt(d_k)
        if self.edge_in_dim is not None:
            assert edge_attr is not None
            E = self.WE(edge_attr).view(-1, self.num_heads, self.hidden_dim // self.num_heads)
            qijk = E * qijk
            self._eij = qijk
        else:
            self._eij = None

        if self.gate:
            assert edge_attr is not None
            e_gate = self.e_gate(edge_attr).view(-1, self.num_heads, self.hidden_dim // self.num_heads)
            qijk = torch.mul(qijk, torch.sigmoid(e_gate))

        qijk = (Q_i * K_j).sum(dim=-1) / math.sqrt(d_k)

        alpha = softmax(qijk, index)

        if self.gate:
            V_j_g = torch.mul(V_j, torch.sigmoid(G_j))
        else:
            V_j_g = V_j

        return alpha.view(-1, self.num_heads, 1) * V_j_g

    def __repr__(self) -> str:
        aggrs = ",".join(self.aggregators)
        return (
            f"{self.__class__.__name__}({self.node_in_dim}, "
            f"{self.hidden_dim}, heads={self.num_heads}, "
            f"aggrs: {aggrs}, "
            f"qkv_bias: {self.qkv_bias}, "
            f"gate: {self.gate})"
        )

'''
BiSeqGCN: Bi-directional sequence graph convolution (part of Module A: SeqGraphRep)
    Input: user-oriented POI transition graph G_u
    Output: node representation H_u on the sequence graph
'''


class BiSeqGCN(MessagePassing):
    def __init__(self, hid_dim, flow="source_to_target"):
        super(BiSeqGCN, self).__init__(aggr='add', flow=flow)
        self.hid_dim = hid_dim
        self.alpha_src = nn.Linear(hid_dim, 1, bias=False)
        self.alpha_dst = nn.Linear(hid_dim, 1, bias=False)

        # learnable attention weight matrix
        self.attention_weight = nn.Parameter(torch.Tensor(hid_dim, hid_dim))
        nn.init.xavier_uniform_(self.attention_weight.data)

        nn.init.xavier_uniform_(self.alpha_src.weight)
        nn.init.xavier_uniform_(self.alpha_dst.weight)
        self.act = nn.LeakyReLU()

    def forward(self, embs, G_u):
        POI_embs, delta_dis_embs, delta_time_embs = embs
        sess_idx = G_u.x.squeeze()
        edge_index = G_u.edge_index
        edge_time = G_u.edge_time
        edge_dist = G_u.edge_dist

        x = POI_embs[sess_idx]
        edge_l = delta_dis_embs[edge_dist]
        edge_t = delta_time_embs[edge_time]
        all_edges = torch.cat((edge_index, edge_index[[1, 0]]), dim=-1)

        H_u = self.propagate(all_edges, x=x, edge_l=edge_l, edge_t=edge_t, edge_size=edge_index.size(1))
        return H_u

    def message(self, x_j, x_i, edge_index_j, edge_index_i, edge_l, edge_t, edge_size):
        attention_coefficients = torch.matmul(x_i[edge_size:] + edge_l + edge_t, self.attention_weight.t())

        src_attention = self.alpha_src(attention_coefficients[:edge_size]).squeeze(-1)
        dst_attention = self.alpha_dst(attention_coefficients[:edge_size]).squeeze(-1)

        # softmax normalization over concatenated attentions
        tot_attention = torch.cat((src_attention, dst_attention), dim=0)
        attn_weight = softmax(tot_attention, edge_index_i)

        # aggregate neighbor features with attention weights
        updated_rep = x_j * attn_weight.unsqueeze(-1)
        return updated_rep


'''
SeqGraphEncoder: wrapper encoder for BiSeqGCN (Module A component)
'''


class SeqGraphEncoder(nn.Module):
    def __init__(self, hid_dim):
        super(SeqGraphEncoder, self).__init__()
        self.hid_dim = hid_dim
        self.encoder = BiSeqGCN(hid_dim)

    def encode(self, embs, G_u):
        return self.encoder(embs, G_u)


'''
DisDyGCN: Distance-based dynamic graph convolution (Module B component)
    Input: global POI distance graph G_D
    Return: updated node features h
'''


class DisDyGCN(MessagePassing):
    def __init__(self, in_channels, out_channels, dist_embed_dim=64):
        super(DisDyGCN, self).__init__(aggr='add')
        # Deprecated in favor of GTConv inside DisGraphRep
        self.linear = nn.Linear(in_channels, out_channels)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x, G_D: Data):
        return self.linear(x)


'''
DisGraphRep: (B) Global-based Distance Graph Geographical Representation Module in DiffDGMN
    Input: Global-based POI Distance Graph G_D
    Return: Node Geographical Representation R_V
'''


class DisGraphRep(nn.Module):
    def __init__(self, n_poi, hid_dim, G_D: Data):
        super(DisGraphRep, self).__init__()
        self.n_poi, self.hid_dim = n_poi, hid_dim
        self.num_layers = gol.conf['num_layer']

        # add self-loops and build edge attributes tensor
        edge_index, _ = add_self_loops(G_D.edge_index)
        # distance weights as features; keep original edge_attr shape [E]
        dist_vec = torch.cat([G_D.edge_attr, torch.zeros((n_poi,)).to(gol.device)])
        edge_attr = dist_vec.unsqueeze(-1)  # [E_total, 1]
        self.G_D = Data(edge_index=edge_index, edge_attr=edge_attr)

        # Stack GTConv layers
        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.layers.append(GTConv(
                node_in_dim=hid_dim,
                hidden_dim=hid_dim,
                edge_in_dim=1,
                num_heads=max(1, gol.conf.get('num_heads', 2)),
                gate=False,
                qkv_bias=False,
                dropout=0.0,
                norm='ln',
                act='relu',
                aggregators=["sum"]
            ))

        self.act = nn.LeakyReLU()

    def encode(self, poi_embs):
        x = poi_embs
        geo_embs = [x]
        edge_index = self.G_D.edge_index
        edge_attr = self.G_D.edge_attr
        for layer in self.layers:
            x, edge_attr = layer(x, edge_index, edge_attr)
            x = self.act(x)
            geo_embs.append(x)
        R_V = torch.stack(geo_embs, dim=1).mean(1)
        return R_V


'''
SDEsolver: wrapper for stochastic differential equation solver
'''


class SDEsolver(nn.Module):
    sde_type = 'stratonovich'  # available: {'ito', 'stratonovich'}
    noise_type = 'scalar'  # available: {'general', 'additive', 'diagonal', 'scalar'}

    def __init__(self, f, g):
        super(SDEsolver).__init__()
        self.f, self.g = f, g

    def f(self, t, y):
        return self.f(t, y)

    def g(self, t, y):
        return self.g(t, y)


'''
SDE_Diffusion: SDE-based diffusion unit (Module D)
    Input: location archetype hat_L_u, conditional embedding
    Output: purified location archetype L_u; marginal estimation support
'''


class SDE_Diffusion(nn.Module):
    def __init__(self, hid_dim, beta_min, beta_max, dt):
        super(SDE_Diffusion, self).__init__()
        self.hid_dim = hid_dim
        self.beta_min, self.beta_max = beta_min, beta_max
        self.dt = dt

        # score-based neural network stacked multiple fully connected layers
        self.score_FC = nn.Sequential(
            nn.Linear(2 * hid_dim, 2 * hid_dim),
            nn.LayerNorm(2 * hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(2 * hid_dim, hid_dim),
            nn.LayerNorm(hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hid_dim, hid_dim)
        )

        for w in self.score_FC:
            if isinstance(w, nn.Linear):
                nn.init.xavier_uniform_(w.weight)

    # time-dependent score network for estimating marginal probability gradient
    def Est_score(self, x, condition):
        # this score is used in SDE solving to guide the evolution of stochastic processes
        return self.score_FC(torch.cat((x, condition), dim=-1))

    # define drift term f and diffusion term g for the forward SDE
    def ForwardSDE_diff(self, x, t):
        def f(_t, y):
            beta_t = self.beta_min + _t * (self.beta_max - self.beta_min)
            return -0.5 * beta_t * y

        def g(_t, y):
            beta_t = self.beta_min + _t * (self.beta_max - self.beta_min)
            bs = y.size(0)
            noise = torch.ones((bs, self.hid_dim, 1), device=y.device)
            return (beta_t ** 0.5) * noise

        ts = torch.Tensor([0, t]).to(gol.device)
        output = sdeint(SDEsolver(f, g), y0=x, ts=ts, dt=self.dt)[-1]
        return output

    def ReverseSDE_gener(self, x, condition, T):
        def get_beta_t(_t):
            beta_t_1 = self.beta_min + _t * (self.beta_max - self.beta_min)
            beta_t_2 = self.beta_min + (self.beta_max - self.beta_min) * torch.sin(torch.pi / 2 * _t) ** 2
            beta_t_3 = self.beta_min * torch.exp(torch.log(torch.tensor(self.beta_max / self.beta_min)) * _t)
            beta_t_4 = self.beta_min + _t * (self.beta_max - self.beta_min) ** 2
            beta_t_5 = 0.1 * torch.exp(6 * _t)
            return beta_t_2

        # drift term f(): _t time, y state, return drift value
        def f(_t, y):
            beta_t = get_beta_t(_t)
            score = self.score_FC(torch.cat((x, condition), dim=-1))
            ## score = self.score_FC(y)
            drift = -0.5 * beta_t * y - beta_t * score
            return drift

        # diffusion term g(): _t time, y state, return diffusion value
        def g(_t, y):
            beta_t = get_beta_t(_t)
            bs = y.size(0)
            # noise Tensors [bs, self.hid_dim, 1] = [1024, 64, 1]  with all elements of 1
            noise = torch.ones((bs, self.hid_dim, 1), device=y.device)
            diffusion = (beta_t ** 0.5) * noise
            return diffusion

        def g_diagonal_noise(_t, y):
            beta_t = self.beta_min + _t * (self.beta_max - self.beta_min)
            bs, dim = y.size(0), y.size(1)
            noise = torch.randn((bs, dim), device=y.device)
            diagonal_beta = torch.diag(beta_t * torch.ones(dim, device=y.device))
            diffusion = (diagonal_beta ** 0.5).mm(noise.t()).t()
            return diffusion + y

        def g_vector_noise(_t, y):
            beta_t = self.beta_min + _t * (self.beta_max - self.beta_min)
            bs, dim, brownian_size = y.size(0), y.size(1), y.size(1)
            noise = torch.randn((bs, dim, brownian_size), device=y.device)
            diffusion = (beta_t ** 0.5) * noise
            return diffusion

        def g_full_cov_noise_3d(_t, y):
            beta_t = self.beta_min + _t * (self.beta_max - self.beta_min)
            bs, dim = y.size(0), y.size(1)
            noise = torch.randn((bs, dim, dim), device=y.device)
            covariance_matrix = torch.eye(dim, device=y.device)
            covariance_matrix = covariance_matrix * beta_t
            cholesky_matrix = torch.linalg.cholesky(covariance_matrix)
            diffusion = torch.einsum('bij,jk->bik', noise, cholesky_matrix)
            return diffusion

        ts = torch.Tensor([0, T]).to(gol.device)

        # output is a pure (noise-free) location archetype vector L_u
        output = sdeint(SDEsolver(f, g), y0=x, ts=ts, dt=self.dt)[-1]

        return output

    def marginal_prob(self, x, t):
        """
        x: [B, D]
        t: scalar or [B]
        returns: mean [B, D], std [B]
        """
        device = x.device
        dtype = x.dtype
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=device, dtype=dtype)
        else:
            t = t.to(device=device, dtype=dtype)

        log_mean_coeff = -0.25 * (t ** 2) * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        # broadcast to [B, 1] if needed
        if log_mean_coeff.dim() == 0:
            log_mean_coeff = log_mean_coeff.unsqueeze(0).expand(x.size(0))
        mean = torch.exp(log_mean_coeff).unsqueeze(-1) * x
        std = torch.sqrt((1.0 - torch.exp(2.0 * log_mean_coeff)).clamp_min(1e-12))
        return mean, std


# HawkesHyperGraphLayer module

class HawkesHyperGraphLayer(nn.Module):
    def __init__(self, n_poi, hidden_dim, gamma=1.0):
        """
        A Hawkes-attention hypergraph layer.

        Args:
            n_poi: Number of POIs (kept for interface compatibility).
            hidden_dim: Embedding dimension.
            gamma: Temporal decay factor in the Hawkes-style kernel.
        """
        super(HawkesHyperGraphLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        # Learnable projections and attention vector.
        self.W_node = nn.Linear(hidden_dim, hidden_dim, bias=False)  # node projection
        self.W_edge = nn.Linear(hidden_dim, hidden_dim, bias=False)  # hyperedge projection
        self.att_vec = nn.Parameter(torch.empty(hidden_dim * 2))  # attention vector a
        nn.init.xavier_uniform_(self.att_vec.unsqueeze(0))  # Xavier initialization

    def forward(self, poi_emb_matrix, batch_seqs, batch_edge_times):
        """
        Build a short-term user interest representation for each sequence in the batch.

        Args:
            poi_emb_matrix: Global POI embedding matrix of shape (n_poi, hidden_dim).
            batch_seqs: A list (length batch_size) of POI index tensors; each tensor is a user sequence.
            batch_edge_times: A list (length batch_size) of time-interval tensors with length len(seq)-1,
                or None when temporal information is unavailable.

        Returns:
            Tensor of shape (batch_size, hidden_dim): short-term user interest embeddings.
        """
        batch_size = len(batch_seqs)
        device = poi_emb_matrix.device

        user_embeddings = []  # collect per-user short-term representations
        for u in range(batch_size):
            seq = batch_seqs[u]  # current user POI sequence (Tensor of POI indices)
            if seq.dim() == 0:
                # Single POI case: ensure 1D tensor.
                seq = seq.unsqueeze(0)
            poi_ids = seq.tolist()  # kept for potential debugging/inspection
            # Node features for the POIs in this hyperedge/sequence.
            V_feat = poi_emb_matrix[seq]  # shape: (num_nodes_in_edge, hidden_dim)
            # Hawkes-style temporal weights: w = exp(-gamma * delta_t)
            w = None
            if batch_edge_times[u] is not None:
                # Convert intervals to timestamps starting from 0.
                time_intervals = batch_edge_times[u].to(device)
                if time_intervals.numel() > 0:
                    timestamps = torch.zeros(len(seq), device=device)
                    timestamps[1:] = torch.cumsum(time_intervals, dim=0)
                else:
                    timestamps = torch.zeros(len(seq), device=device)
                # Use the last timestamp as "current time".
                t_last = timestamps[-1]
                # Time since each event.
                delta_t = t_last - timestamps  # shape: [num_nodes_in_edge]
                # Exponential decay.
                w = torch.exp(-self.gamma * delta_t)
            else:
                # No temporal information: uniform weights.
                w = torch.ones(len(seq), device=device)

            # Apply temporal weights to node features.
            V_feat_time = V_feat * w.unsqueeze(-1)  # (num_nodes_in_edge, hidden_dim)

            # Initialize hyperedge feature as the mean of weighted node features.
            if V_feat_time.size(0) > 0:
                E_feat = V_feat_time.mean(dim=0, keepdim=True)  # (1, hidden_dim)
            else:
                E_feat = torch.zeros(1, self.hidden_dim, device=device)

            # Project into attention space.
            V_proj = self.W_node(V_feat_time)  # (num_nodes, hidden_dim)
            E_proj = self.W_edge(E_feat)  # (1, hidden_dim)

            # Compute attention weights.
            num_nodes = V_proj.size(0)
            E_rep = E_proj.repeat(num_nodes, 1)  # (num_nodes, hidden_dim)
            concat = torch.cat([V_proj, E_rep], dim=1)  # (num_nodes, 2*hidden_dim)
            att_scores = F.leaky_relu(torch.matmul(concat, self.att_vec.unsqueeze(-1)).squeeze(-1), negative_slope=0.2)
            att_weights = F.softmax(att_scores, dim=0)  # (num_nodes,)
            # Weighted aggregation to obtain the hyperedge representation.
            if num_nodes > 0:
                edge_rep = torch.sum(att_weights.unsqueeze(-1) * V_feat_time, dim=0, keepdim=True)  # (1, hidden_dim)
            else:
                edge_rep = torch.zeros_like(E_feat)

            # Treat the hyperedge representation as the user's short-term interest embedding.
            user_embeddings.append(edge_rep.squeeze(0))
        user_embeddings = torch.stack(user_embeddings, dim=0)  # (batch_size, hidden_dim)
        return user_embeddings

