# coding=utf-8
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from DiffDGMN_main import gol
from DiffDGMN_main.layers import (
    DisGraphRep,
    SDE_Diffusion,
    GTConv,
    KAN,
)
from DiffDGMN_main.hypergraph import HyperGraphRep  # hypergraph encoder


class HyperGMVAN(nn.Module):
    """
    Variant design: replace module A with a hypergraph encoder -> (sv, si).
      - si: used as Query for module C, and as Su (condition) for module D (diffusion)
      - sv: used as Value for module C
    Other modules (B: GTConv, C: attention+KAN, D: diffusion) keep their internal logic;
    only the wiring is adapted to use hypergraph outputs.
    """
    def __init__(self, num_users, num_poi, G_D, hid_dim=None):
        super().__init__()
        self.num_users = num_users
        self.num_poi = num_poi
        self.hid_dim = hid_dim or gol.conf['hidden']
        self.dis_rep = DisGraphRep(self.num_poi, self.hid_dim, G_D)


        self.R_V_all = None  # cache candidate-side geographic representations (full POI set)


        # Embeddings (keep original dimensions)
        self.user_emb = nn.Embedding(num_users, self.hid_dim)
        self.poi_emb = nn.Embedding(num_poi, self.hid_dim)


        # A: hypergraph representations
        self.hyper_rep = HyperGraphRep(self.hid_dim, he_attr_dim=3)

        # B: geography/structure representations (distance graph + GTConv)


        self.R_V_all = None  # external training loop may inject precomputed candidate geographic vectors

        # C: attention + KAN (wiring adapted)
        self.geo_attn = nn.MultiheadAttention(self.hid_dim, gol.conf['num_heads'], batch_first=True,
                                              dropout=gol.conf.get('attn_dp', 0.1))
        self.q_proj = nn.Linear(self.hid_dim, self.hid_dim)
        self.k_proj = nn.Linear(self.hid_dim, self.hid_dim)
        self.v_proj = nn.Linear(self.hid_dim, self.hid_dim)
        self.kan = KAN(layers_hidden=[self.hid_dim, self.hid_dim], grid_size=3)

        # Learnable scaling factors for Q/K/V (used in forward)
        self.q_scale = nn.Parameter(torch.tensor(1.0))
        self.k_scale = nn.Parameter(torch.tensor(1.0))
        self.v_scale = nn.Parameter(torch.tensor(1.0))

        # Residual gating
        self.attn_norm = nn.LayerNorm(self.hid_dim)
        self.attn_drop = nn.Dropout(p=gol.conf.get('dp', 0.1))
        self.attn_gate = nn.Linear(self.hid_dim, self.hid_dim, bias=True)
        nn.init.zeros_(self.attn_gate.weight)
        nn.init.zeros_(self.attn_gate.bias)

        # Initialization (must run after attn_gate is defined)
        nn.init.zeros_(self.attn_gate.weight)
        nn.init.zeros_(self.attn_gate.bias)
        # If you want to start from "almost no correction", you may use:
        # self.attn_gate.bias.data.fill_(-10.0)  # gate ~= sigmoid(-10) ~= 0
        # self.attn_gate.bias.data.fill_(-10.0)  # gate ~= sigmoid(-10) ~= 0

        # C post-processing: LayerNorm + L2 norm + gated residual with Su
        self.c_ln = nn.LayerNorm(self.hid_dim)
        self.c_gate = nn.Linear(self.hid_dim * 2, self.hid_dim)

        # D: diffusion
        self.diffusion = SDE_Diffusion(
            self.hid_dim,
            gol.conf['beta_min'],
            gol.conf['beta_max'],
            gol.conf['dt'],

        )


        # FFN / normalization / dropout
        self.dropout = nn.Dropout(p=gol.conf['dp'])
        self.out_norm = nn.LayerNorm(self.hid_dim)

        # Learnable fusion scalars (small initial values to avoid large early logits)
        self.alpha = nn.Parameter(torch.tensor(0.2))  # for S_u · R_V^T
        self.beta  = nn.Parameter(torch.tensor(0.2))  # for L_u · R_V^T
        self.gamma = nn.Parameter(torch.tensor(0.1))  # for User · E_poi^T

        # Learnable temperature for logits scaling (stabilizes early training)
        self.tau = nn.Parameter(torch.tensor(3.0))

        # Pre-LayerNorm for an optional scorer head
        self.pre_ln = nn.LayerNorm(self.hid_dim)
        # Scorer head (currently unused; reserved for future use)
        self.scorer = nn.Sequential(
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hid_dim, 1)
        )


    def getTrainLoss(self, batch_tuple):
        # Input: collate_edge returns (u, p, n, s, s_graph, t)
        u, p, n, s, s_graph, t = batch_tuple
        device = next(self.parameters()).device

        # Attach required fields to the graph to match forward() inputs
        s_graph = s_graph.to(device, non_blocking=True)
        s_graph.user = u.to(device, non_blocking=True)
        if isinstance(s, (list, tuple)):
            s_graph.seq = torch.cat(s, dim=0).to(device, non_blocking=True)
        else:
            s_graph.seq = s.to(device, non_blocking=True)

        # Forward
        logits = self(s_graph)  # [B, num_poi]

        # ===== Baseline in-batch negatives + dedup + margin (BPR-style) =====
        pos_items = p.to(device)
        pos_score = logits.gather(1, pos_items.view(-1, 1)).squeeze(1)  # [B]

        B, I = logits.size(0), logits.size(1)
        K_target = int(min(B - 1, 16)) if B > 1 else 0
        margin_m = 0.1

        # Build per-sample sequence sets (to exclude items from the current sequence)
        if isinstance(s, (list, tuple)):
            seq_lists = [si.detach().cpu().tolist() for si in s]
        else:
            # Fallback: split s_graph.seq by seq_lens
            if hasattr(s_graph, 'seq_lens'):
                lengths = s_graph.seq_lens.detach().cpu().tolist()
                starts = [0]
                for l in lengths: starts.append(starts[-1] + int(l))
                seq_cpu = s_graph.seq.detach().cpu()
                seq_lists = [seq_cpu[starts[i]:starts[i+1]].tolist() for i in range(len(lengths))]
            else:
                seq_lists = [[] for _ in range(B)]

        bpr_list = []
        k_list = []  # number of usable in-batch negatives per sample
        arange_B = torch.arange(B, device=device)
        for i in range(B):
            if K_target <= 0:
                continue
            # Candidate negatives: positives from other samples in the same batch
            other_mask = (arange_B != i)
            cands = pos_items[other_mask]  # [B-1]
            # Deduplicate + exclude the current positive and the current sequence items
            seq_set = set(seq_lists[i]) if len(seq_lists[i]) > 0 else set()
            seq_set.add(int(pos_items[i].item()))
            cands_list = list({int(x.item()) for x in cands if int(x.item()) not in seq_set})

            # If candidates < K, use all; if empty, sample random negatives (excluding seq_set)
            if len(cands_list) == 0:
                # Random sampling up to K
                max_trials = max(4 * K_target, 64)
                sampled = []
                trials = 0
                while len(sampled) < K_target and trials < max_trials:
                    trial = int(torch.randint(0, I, (1,), device=device).item())
                    if trial not in seq_set:
                        sampled.append(trial)
                    trials += 1
                neg_items = torch.tensor(sampled if sampled else [int((pos_items[i].item()+1)%I)], device=device, dtype=torch.long)
            else:
                if len(cands_list) > K_target:
                    perm = torch.randperm(len(cands_list))[:K_target]
                    neg_items = torch.tensor([cands_list[j.item()] for j in perm], device=device, dtype=torch.long)
                else:
                    neg_items = torch.tensor(cands_list, device=device, dtype=torch.long)

            # Record number of negatives for this sample
            k_list.append(int(neg_items.numel()))

            neg_scores = logits[i, neg_items]
            pos_s = pos_score[i].expand_as(neg_scores)
            bpr_i = F.softplus(-(pos_s - neg_scores - margin_m)).mean()
            bpr_list.append(bpr_i)

        loss_rec = torch.stack(bpr_list).mean() if bpr_list else torch.tensor(0.0, device=device)

        # Optional: log min/mean K for monitoring degenerate batches (K≈0/1)
        try:
            if gol.conf.get('log_neg_k', False) and k_list:
                import statistics
                k_min = min(k_list)
                k_mean = statistics.fmean(k_list)
                gol.pLog(f"In-batch negatives K per-sample: min={k_min}, mean={k_mean:.2f}")
        except Exception:
            pass

        # ===== Diffusion branch: divergence / score matching =====
        Su_mid = getattr(self, '_last_Su', None)
        L_hat_mid = getattr(self, '_last_L_hat_u', None)
        if Su_mid is None or L_hat_mid is None:
            _ = self(s_graph)
            Su_mid = getattr(self, '_last_Su')
            L_hat_mid = getattr(self, '_last_L_hat_u')

        # Key 1: stop-gradient on the target (avoid leakage to the target branch)
        with torch.no_grad():
            target = L_hat_mid.detach()

        # Key 2: sample time t ∈ (0, 1)
        t_rand = torch.rand(Su_mid.size(0), device=device)

        # Key 3: marginal distribution and noisy input construction x_t
        mean, std = self.diffusion.marginal_prob(target, t_rand)
        # Expand std to match target dimensions
        while std.dim() < target.dim():
            std = std.unsqueeze(-1)
        z = torch.randn_like(target)
        x_t = mean + std * z

        # Key 4: conditional score; explicitly pass Su and t (backward compatible if t is unsupported)
        try:
            score_hat = self.diffusion.Est_score(x_t, Su_mid, t_rand)
        except TypeError:
            score_hat = self.diffusion.Est_score(x_t, Su_mid)

        # Fisher/DSM: predict unscaled score and weight by σ(t)^2
        # True score s*(x_t) = -z / std
        loss_div = ((std ** 2) * (score_hat + z / (std + 1e-8)) ** 2).mean()

        return loss_rec, loss_div

    def set_R_V_all(self, R_V_all: torch.Tensor):
        """Inject a cached candidate-side geographic representation ([num_poi, d])."""
        # Keep dtype/device consistent
        self.R_V_all = R_V_all.to(self.poi_emb.weight.device).detach()

    @torch.no_grad()
    # model.py

    @torch.no_grad()
    def compute_R_V_all(self):
        """
        Compute candidate-side geographic representations using module B: encode(poi_embs).
        Falling back to poi_emb.weight is only a last resort and should rarely happen.
        """
        try:
            # poi_embs: [num_poi, d], consistent with DisGraphRep.encode signature
            poi_embs = self.poi_emb.weight.to(next(self.parameters()).device)
            R_V_all = self.dis_rep.encode(poi_embs)  # <-- only change here
            return R_V_all.to(poi_embs.device)
        except Exception as e:
            # Fallback: training can still run, but without geographic priors
            return self.poi_emb.weight.clone()

    def get_R_V_all(self):
        """In forward(), prefer the cached R_V_all; otherwise compute it on the fly."""
        if getattr(self, "R_V_all", None) is not None:
            return self.R_V_all
        # Compute on the fly (not cached by default)
        R_V_all = self.compute_R_V_all()
        return R_V_all


    def forward(self, batch):
        """
        Batch fields (from dataset GraphData / PyG Batch):
          - batch.x: [N_local, 1]       local nodes storing global POI ids (for embedding lookup)
          - batch.seq: [sum(L)]         flattened original sequence, used to build module C keys/values
          - batch.hyperedge_index: [2, M]
          - batch.hyperedge_attr : [E, d_e]
          - batch.num_nodes      : [B]
          - batch.user           : [B]
          - batch.target         : [B]
        """
        B = batch.num_graphs

        # ========= Embeddings =========
        E_poi = self.poi_emb.weight  # [num_poi, d]
        E_user = self.user_emb(batch.user.view(-1))  # [B, d]

        # ========= Module A: hypergraph representations (local -> global) =========
        # 1) Slice global ids in local order and feed to HyperGraphRep
        global_ids_local = batch.x.view(-1).long()  # [N_local]
        node_embs_local = E_poi.index_select(0, global_ids_local)  # [N_local, d]

        sv_local, si = self.hyper_rep(
            batch.hyperedge_index,  # [2, M] (row is local node id: 0..N_local-1)
            node_embs_local,  # [N_local, d]
            batch.hyperedge_attr  # [E, 3]
        )

        # 2) Broadcast sv_local back to global POI space (accumulate / average)
        num_poi, d = E_poi.size(0), E_poi.size(1)
        sv_global = E_poi.new_zeros(num_poi, d)
        cnt_global = E_poi.new_zeros(num_poi, 1)
        sv_global.index_add_(0, global_ids_local, sv_local)
        cnt_global.index_add_(0, global_ids_local,
                              torch.ones_like(global_ids_local, dtype=sv_global.dtype).unsqueeze(1))
        sv_global = sv_global / cnt_global.clamp_min(1.0)  # [num_poi, d]

        # ========= Aggregate si per graph -> si_b ([B, d]) =========
        if hasattr(batch, "hyperedge_batch") and batch.hyperedge_batch.numel() > 0:
            B = batch.num_graphs
            hb = batch.hyperedge_batch
            E_si = si.size(0)
            # If hyperedge_batch length mismatches si, fix it robustly
            if hb.numel() != E_si:
                if hasattr(batch, 'num_hyperedges'):
                    # Rebuild hb from true per-graph hyperedge counts; then truncate/pad to E_si
                    counts = torch.as_tensor(batch.num_hyperedges, device=si.device).long().view(-1)
                    parts = [torch.full((int(c),), i, dtype=torch.long, device=si.device) for i, c in enumerate(counts.tolist()) if c > 0]
                    hb = torch.cat(parts, dim=0) if parts else torch.empty(0, dtype=torch.long, device=si.device)
                    if hb.numel() > E_si:
                        hb = hb[:E_si]
                    elif hb.numel() < E_si:
                        pad = torch.zeros(E_si - hb.numel(), dtype=torch.long, device=si.device)
                        hb = torch.cat([hb, pad], dim=0)
                else:
                    # Fallback: approximate allocation to E_si by per-graph node counts
                    lengths = (batch.ptr[1:] - batch.ptr[:-1]).tolist()
                    total_len = max(sum(lengths), 1)
                    e_counts = [int(round(E_si * (l / total_len))) for l in lengths]
                    diff = E_si - sum(e_counts)
                    if diff != 0:
                        e_counts[0] += diff
                    parts = [torch.full((c,), i, dtype=torch.long, device=si.device) for i, c in enumerate(e_counts) if c > 0]
                    hb = torch.cat(parts, dim=0) if parts else torch.empty(0, dtype=torch.long, device=si.device)

            si_b = si.new_zeros(B, d)
            si_b.index_add_(0, hb, si)
            he_count = torch.bincount(hb, minlength=B).clamp_min(1).view(-1, 1).to(si.device)
            si_b = si_b / he_count
        else:
            edge_ids = batch.hyperedge_index[1]
            E_total = int(edge_ids.max().item() + 1) if edge_ids.numel() else 0
            lengths = (batch.ptr[1:] - batch.ptr[:-1]).tolist()
            total_len = sum(lengths) + 1e-6
            e_counts = [int(round(E_total * (l / total_len))) for l in lengths]
            diff = E_total - sum(e_counts)
            if diff != 0: e_counts[0] += diff
            chunks, start = [], 0
            for c in e_counts:
                end = start + c
                if end > start:
                    chunks.append(si[start:end].mean(dim=0, keepdim=True))
                else:
                    chunks.append(si.new_zeros(1, si.size(-1)))
                start = end
            si_b = torch.cat(chunks, dim=0)  # [B,d]

        # ========= Module B: candidate-side geographic representations R_V_all =========
        if getattr(self, "R_V_all", None) is not None:
            R_V_all = self.R_V_all
        else:
            poi_embs = self.poi_emb.weight.to(next(self.parameters()).device)
            R_V_all = self.dis_rep.encode(poi_embs)

        # ========= Build sequence views for K/V (index by global ids in seq) =========
        seq = batch.seq.long()  # [sum(L)]
        sv_seq = sv_global.index_select(0, seq)  # V: semantics (from A)
        rv_seq = R_V_all.index_select(0, seq)  # K: geography (from B)

        # Use per-sample true sequence lengths from collate to avoid split mismatch with batch.ptr
        lengths = batch.seq_lens.tolist() if hasattr(batch, 'seq_lens') else (batch.ptr[1:] - batch.ptr[:-1]).tolist()
        sv_pad = nn.utils.rnn.pad_sequence(torch.split(sv_seq, lengths, dim=0), batch_first=True,
                                           padding_value=0.0)  # [B,L,d]
        rv_pad = nn.utils.rnn.pad_sequence(torch.split(rv_seq, lengths, dim=0), batch_first=True,
                                           padding_value=0.0)  # [B,L,d]

        # ========= Module C: Q=si_b, K=rv_pad, V=sv_pad + gated residual correction =========
        Q = self.q_proj(si_b).unsqueeze(1) * self.q_scale  # [B,1,d]
        K = self.k_proj(rv_pad) * self.k_scale  # [B,L,d]
        V = self.v_proj(sv_pad) * self.v_scale  # [B,L,d]

        attn_out, _ = self.geo_attn(Q, K, V, key_padding_mask=None)  # [B,1,d]
        hat_L_u = self.kan(attn_out.squeeze(1))  # [B,d]

        # Post-process KAN output: LN -> L2 norm -> gated residual with Su
        Su = si_b  # [B,d]
        L_hat_u = self.c_ln(hat_L_u)
        L_hat_u = F.normalize(L_hat_u, p=2, dim=-1)
        c_gate = torch.sigmoid(self.c_gate(torch.cat([L_hat_u, Su], dim=-1)))  # [B,d]
        L_hat_u = L_hat_u + c_gate * Su

        # ========= Module D: diffusion (L_hat_u as noisy state, conditioned on Su) =========
        Lu = self.diffusion.ReverseSDE_gener(L_hat_u, Su, gol.conf['T'])  # [B,d]

        # Cache intermediate tensors for training loss (does not change forward outputs)
        self._last_Su = Su
        self._last_L_hat_u = L_hat_u

        # ========= Fused scoring =========
        User = E_user  # [B,d]
        scores_su_rv = Su @ R_V_all.t()
        scores_lu_rv = Lu @ R_V_all.t()
        scores_user = User @ E_poi.t()

        logits = self.alpha * scores_su_rv + \
                 self.beta * scores_lu_rv + \
                 self.gamma * scores_user

        # Per-user z-score normalization (stability; monotonic transform preserves ranking)
        mu  = logits.mean(dim=-1, keepdim=True)
        std = logits.std(dim=-1, keepdim=True).clamp_min(1e-6)
        logits = (logits - mu) / std

        # Temperature scaling
        logits = logits / self.tau.clamp_min(0.5)

        return logits
