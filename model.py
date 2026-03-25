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
from DiffDGMN_main.hypergraph import HyperGraphRep  # ← 新增：超图表达器


class HyperGMVAN(nn.Module):
    """
    方案二：a模块改为 超图 → (sv, si)
      - si: 作为 C 的 Query、作为 D 扩散的 Su
      - sv: 作为 C 的 Value
    其它模块（B:GTConv/C:注意力+KAN/D:扩散）只“改接线”，不改内部实现
    """
    def __init__(self, num_users, num_poi, G_D, hid_dim=None):
        super().__init__()
        self.num_users = num_users
        self.num_poi = num_poi
        self.hid_dim = hid_dim or gol.conf['hidden']
        self.dis_rep = DisGraphRep(self.num_poi, self.hid_dim, G_D)


        self.R_V_all = None  # 用于缓存“候选侧地理表示”（B模块全量输出）


        # Embeddings（保持你原有尺寸）
        self.user_emb = nn.Embedding(num_users, self.hid_dim)
        self.poi_emb = nn.Embedding(num_poi, self.hid_dim)


        # A: 超图表示
        self.hyper_rep = HyperGraphRep(self.hid_dim, he_attr_dim=3)

        # B: 地理/结构（保持你的 GTConv/距离图表示）
        # ★ 按 layers.py 的签名修正（典型是 (num_poi, hid_dim, G_D)）
        #   如果你的 DisGraphRep 真的只要 hid_dim，那就按原样；否则传入 G_D（从 main.py 注入或 gol 里读取）


        self.R_V_all = None  # 外部训练循环可注入 B 模块预编码的候选地理向量（保留这一处即可）

        # C: 注意力 + KAN（仅接线改变）
        self.geo_attn = nn.MultiheadAttention(self.hid_dim, gol.conf['num_heads'], batch_first=True,
                                              dropout=gol.conf.get('attn_dp', 0.1))
        self.q_proj = nn.Linear(self.hid_dim, self.hid_dim)
        self.k_proj = nn.Linear(self.hid_dim, self.hid_dim)
        self.v_proj = nn.Linear(self.hid_dim, self.hid_dim)
        self.kan = KAN(layers_hidden=[self.hid_dim, self.hid_dim], grid_size=3)

        # ★ 补上 Q/K/V 的可学习缩放（你 forward 里在用）
        self.q_scale = nn.Parameter(torch.tensor(1.0))
        self.k_scale = nn.Parameter(torch.tensor(1.0))
        self.v_scale = nn.Parameter(torch.tensor(1.0))

        # 残差门控
        self.attn_norm = nn.LayerNorm(self.hid_dim)
        self.attn_drop = nn.Dropout(p=gol.conf.get('dp', 0.1))
        self.attn_gate = nn.Linear(self.hid_dim, self.hid_dim, bias=True)
        nn.init.zeros_(self.attn_gate.weight)
        nn.init.zeros_(self.attn_gate.bias)

        # ★ 初始化（必须在 attn_gate 定义之后）
        nn.init.zeros_(self.attn_gate.weight)
        nn.init.zeros_(self.attn_gate.bias)
        # 若想从“几乎不校正”开始，可用：
        # self.attn_gate.bias.data.fill_(-10.0)  # 让 gate≈sigmoid(-10)≈0

        # C 后处理：LayerNorm + L2-norm + 与 Su 的门控残差
        self.c_ln = nn.LayerNorm(self.hid_dim)
        self.c_gate = nn.Linear(self.hid_dim * 2, self.hid_dim)

        # D: 扩散
        self.diffusion = SDE_Diffusion(
            self.hid_dim,
            gol.conf['beta_min'],
            gol.conf['beta_max'],
            gol.conf['dt'],

        )


        # FFN / 归一化 / Dropout
        self.dropout = nn.Dropout(p=gol.conf['dp'])
        self.out_norm = nn.LayerNorm(self.hid_dim)

        # learnable fusion scalars（缩小初值，避免初期logits过大）
        self.alpha = nn.Parameter(torch.tensor(0.2))  # for S_u · R_V^T
        self.beta  = nn.Parameter(torch.tensor(0.2))  # for L_u · R_V^T
        self.gamma = nn.Parameter(torch.tensor(0.1))  # for User · E_poi^T

        # 可学习温度（用于 logits 温度缩放，稳定初期训练）
        self.tau = nn.Parameter(torch.tensor(3.0))

        # scorer 前置 LayerNorm（若未来启用 scorer，这里可用于稳态输入）
        self.pre_ln = nn.LayerNorm(self.hid_dim)
        # 评分头（当前未参与融合；预留接口）
        self.scorer = nn.Sequential(
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hid_dim, 1)
        )


    def getTrainLoss(self, batch_tuple):
        # 输入：collate_edge 返回的 (u, p, n, s, s_graph, t)
        u, p, n, s, s_graph, t = batch_tuple
        device = next(self.parameters()).device

        # 将必要字段挂到图上，保持与 forward 的输入一致
        s_graph = s_graph.to(device, non_blocking=True)
        s_graph.user = u.to(device, non_blocking=True)
        if isinstance(s, (list, tuple)):
            s_graph.seq = torch.cat(s, dim=0).to(device, non_blocking=True)
        else:
            s_graph.seq = s.to(device, non_blocking=True)

        # 前向
        logits = self(s_graph)  # [B, num_poi]

        # ===== 起步版 in-batch negatives + 去重 + margin(BPR) =====
        pos_items = p.to(device)
        pos_score = logits.gather(1, pos_items.view(-1, 1)).squeeze(1)  # [B]

        B, I = logits.size(0), logits.size(1)
        K_target = int(min(B - 1, 16)) if B > 1 else 0
        margin_m = 0.1

        # 准备每样本序列集合（用于排除当前序列内项目）
        if isinstance(s, (list, tuple)):
            seq_lists = [si.detach().cpu().tolist() for si in s]
        else:
            # 退化兜底：按 seq_lens 切分 s_graph.seq
            if hasattr(s_graph, 'seq_lens'):
                lengths = s_graph.seq_lens.detach().cpu().tolist()
                starts = [0]
                for l in lengths: starts.append(starts[-1] + int(l))
                seq_cpu = s_graph.seq.detach().cpu()
                seq_lists = [seq_cpu[starts[i]:starts[i+1]].tolist() for i in range(len(lengths))]
            else:
                seq_lists = [[] for _ in range(B)]

        bpr_list = []
        k_list = []  # 统计每个样本实际可用的 in-batch 负样个数
        arange_B = torch.arange(B, device=device)
        for i in range(B):
            if K_target <= 0:
                continue
            # 候选负样：batch 内其他样本的正样
            other_mask = (arange_B != i)
            cands = pos_items[other_mask]  # [B-1]
            # 去重 + 排除自身 pos 与 当前序列
            seq_set = set(seq_lists[i]) if len(seq_lists[i]) > 0 else set()
            seq_set.add(int(pos_items[i].item()))
            cands_list = list({int(x.item()) for x in cands if int(x.item()) not in seq_set})

            # 若不足 K，直接用全部；若为空，随机采样补足（排除 seq_set）
            if len(cands_list) == 0:
                # 随机采样最多 K 个
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

            # 记录本样本负样个数
            k_list.append(int(neg_items.numel()))

            neg_scores = logits[i, neg_items]
            pos_s = pos_score[i].expand_as(neg_scores)
            bpr_i = F.softplus(-(pos_s - neg_scores - margin_m)).mean()
            bpr_list.append(bpr_i)

        loss_rec = torch.stack(bpr_list).mean() if bpr_list else torch.tensor(0.0, device=device)

        # —— 可选：打印本批实际 K 的 min/mean，便于监控是否常出现 K≈0/1 ——
        try:
            if gol.conf.get('log_neg_k', False) and k_list:
                import statistics
                k_min = min(k_list)
                k_mean = statistics.fmean(k_list)
                gol.pLog(f"In-batch negatives K per-sample: min={k_min}, mean={k_mean:.2f}")
        except Exception:
            pass

        # ===== 扩散分支的散度 / score matching =====
        Su_mid = getattr(self, '_last_Su', None)
        L_hat_mid = getattr(self, '_last_L_hat_u', None)
        if Su_mid is None or L_hat_mid is None:
            _ = self(s_graph)
            Su_mid = getattr(self, '_last_Su')
            L_hat_mid = getattr(self, '_last_L_hat_u')

        # 关键1：target 停梯度（防止梯度泄漏到目标支路）
        with torch.no_grad():
            target = L_hat_mid.detach()

        # 关键2：采样时间步 t ∈ (0,1)
        t_rand = torch.rand(Su_mid.size(0), device=device)

        # 关键3：边际分布 & 构造带噪输入 x_t
        mean, std = self.diffusion.marginal_prob(target, t_rand)
        # 将 std 扩展到与 target 维度对齐
        while std.dim() < target.dim():
            std = std.unsqueeze(-1)
        z = torch.randn_like(target)
        x_t = mean + std * z

        # 关键4：条件化 score，显式喂入 Su 与 t（若 Est_score 无 t 参数则自动兼容）
        try:
            score_hat = self.diffusion.Est_score(x_t, Su_mid, t_rand)
        except TypeError:
            score_hat = self.diffusion.Est_score(x_t, Su_mid)

        # Fisher/DSM（VE-SDE 推荐写法：预测未缩放 score，并加 σ(t)^2 权重）
        # 真分数 s*(x_t) = - z / std
        loss_div = ((std ** 2) * (score_hat + z / (std + 1e-8)) ** 2).mean()

        return loss_rec, loss_div

    def set_R_V_all(self, R_V_all: torch.Tensor):
        """外部注入候选侧地理表示缓存（[num_poi, d]）。"""
        # 保证 dtype/device 一致
        self.R_V_all = R_V_all.to(self.poi_emb.weight.device).detach()

    @torch.no_grad()
    # model.py

    @torch.no_grad()
    def compute_R_V_all(self):
        """
        用 B 模块的 encode(poi_embs) 计算候选侧地理表示。
        回退到 poi_emb.weight 只是兜底，不应走到。
        """
        try:
            # poi_embs: [num_poi, d]，和 DisGraphRep.encode 的签名一致
            poi_embs = self.poi_emb.weight.to(next(self.parameters()).device)
            R_V_all = self.dis_rep.encode(poi_embs)  # <-- 唯一改动
            return R_V_all.to(poi_embs.device)
        except Exception as e:
            # 兜底：不报错也能跑通训练，但没有地理先验
            return self.poi_emb.weight.clone()

    def get_R_V_all(self):
        """forward 内部优先取缓存；没有就即时计算（然后也可缓存）。"""
        if getattr(self, "R_V_all", None) is not None:
            return self.R_V_all
        # 即时算一个（不缓存也能跑；若想缓存，可以顺便 set 一下）
        R_V_all = self.compute_R_V_all()
        return R_V_all


    def forward(self, batch):
        """
        batch fields （来自 dataset.TrajDataset）:
          - batch.x: [sum(L),]          轨迹中的 POI id（供取embedding）
          - batch.seq: [sum(L),]        原始序列，给C模块 Key 的构造
          - batch.hyperedge_index: [2, M]
          - batch.hyperedge_attr : [E, d_e]
          - batch.num_nodes      : [B]
          - batch.user           : [B]
          - batch.target         : [B]
        """
        B = batch.num_graphs

        # ========= 取嵌入 =========
        E_poi = self.poi_emb.weight  # [num_poi, d]
        E_user = self.user_emb(batch.user.view(-1))  # [B, d]

        # ========= A 模块：超图表征（局部→全局）=========
        # 1) 先按“局部顺序”的全局 id 切子表，输入 HyperGraphRep
        global_ids_local = batch.x.view(-1).long()  # [N_local]
        node_embs_local = E_poi.index_select(0, global_ids_local)  # [N_local, d]

        sv_local, si = self.hyper_rep(
            batch.hyperedge_index,  # [2, M]（row 是 0..N_local-1 的局部 id）
            node_embs_local,  # [N_local, d]
            batch.hyperedge_attr  # [E, 3]
        )

        # 2) 将 sv_local 广播回全局空间（累加 / 取平均）
        num_poi, d = E_poi.size(0), E_poi.size(1)
        sv_global = E_poi.new_zeros(num_poi, d)
        cnt_global = E_poi.new_zeros(num_poi, 1)
        sv_global.index_add_(0, global_ids_local, sv_local)
        cnt_global.index_add_(0, global_ids_local,
                              torch.ones_like(global_ids_local, dtype=sv_global.dtype).unsqueeze(1))
        sv_global = sv_global / cnt_global.clamp_min(1.0)  # [num_poi, d]

        # ========= si 按图聚合 → si_b（[B,d]）=========
        if hasattr(batch, "hyperedge_batch") and batch.hyperedge_batch.numel() > 0:
            B = batch.num_graphs
            hb = batch.hyperedge_batch
            E_si = si.size(0)
            # 若 hyperedge_batch 长度与 si 不一致，做稳健修正
            if hb.numel() != E_si:
                if hasattr(batch, 'num_hyperedges'):
                    # 用真实每图超边数重建 hb，并截断/填补到 E_si
                    counts = torch.as_tensor(batch.num_hyperedges, device=si.device).long().view(-1)
                    parts = [torch.full((int(c),), i, dtype=torch.long, device=si.device) for i, c in enumerate(counts.tolist()) if c > 0]
                    hb = torch.cat(parts, dim=0) if parts else torch.empty(0, dtype=torch.long, device=si.device)
                    if hb.numel() > E_si:
                        hb = hb[:E_si]
                    elif hb.numel() < E_si:
                        pad = torch.zeros(E_si - hb.numel(), dtype=torch.long, device=si.device)
                        hb = torch.cat([hb, pad], dim=0)
                else:
                    # 兜底：按节点数比例近似分配到 E_si
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

        # ========= B 模块：候选侧地理表示 R_V_all =========
        if getattr(self, "R_V_all", None) is not None:
            R_V_all = self.R_V_all
        else:
            poi_embs = self.poi_emb.weight.to(next(self.parameters()).device)
            R_V_all = self.dis_rep.encode(poi_embs)

        # ========= 构造 K/V 的序列视图（按全局 id 的 seq 切片）=========
        seq = batch.seq.long()  # [sum(L)]
        sv_seq = sv_global.index_select(0, seq)  # V：语义（来自 A）
        rv_seq = R_V_all.index_select(0, seq)  # K：地理（来自 B）

        # 使用 collate 中提供的每样本真实序列长度，避免与 batch.ptr 不一致导致 split 报错
        lengths = batch.seq_lens.tolist() if hasattr(batch, 'seq_lens') else (batch.ptr[1:] - batch.ptr[:-1]).tolist()
        sv_pad = nn.utils.rnn.pad_sequence(torch.split(sv_seq, lengths, dim=0), batch_first=True,
                                           padding_value=0.0)  # [B,L,d]
        rv_pad = nn.utils.rnn.pad_sequence(torch.split(rv_seq, lengths, dim=0), batch_first=True,
                                           padding_value=0.0)  # [B,L,d]

        # ========= C 模块：Q=si_b, K=rv_pad, V=sv_pad + 残差门控校正 Su =========
        Q = self.q_proj(si_b).unsqueeze(1) * self.q_scale  # [B,1,d]
        K = self.k_proj(rv_pad) * self.k_scale  # [B,L,d]
        V = self.v_proj(sv_pad) * self.v_scale  # [B,L,d]

        attn_out, _ = self.geo_attn(Q, K, V, key_padding_mask=None)  # [B,1,d]
        hat_L_u = self.kan(attn_out.squeeze(1))  # [B,d]

        # —— C 后处理（针对 KAN 输出）：LN → L2-norm → 门控 Su 残差 ——
        Su = si_b  # [B,d]
        L_hat_u = self.c_ln(hat_L_u)
        L_hat_u = F.normalize(L_hat_u, p=2, dim=-1)
        c_gate = torch.sigmoid(self.c_gate(torch.cat([L_hat_u, Su], dim=-1)))  # [B,d]
        L_hat_u = L_hat_u + c_gate * Su

        # ========= D 模块：扩散（以 L_hat_u 为噪声态，Su 为条件） =========
        Lu = self.diffusion.ReverseSDE_gener(L_hat_u, Su, gol.conf['T'])  # [B,d]

        # —— 缓存中间变量供训练损失使用（不改变前向输出）——
        self._last_Su = Su
        self._last_L_hat_u = L_hat_u

        # ========= 融合打分=========
        User = E_user  # [B,d]
        scores_su_rv = Su @ R_V_all.t()
        scores_lu_rv = Lu @ R_V_all.t()
        scores_user = User @ E_poi.t()

        logits = self.alpha * scores_su_rv + \
                 self.beta * scores_lu_rv + \
                 self.gamma * scores_user

        # —— per-user z-score（可提升稳定性，单调变换不改排序）——
        mu  = logits.mean(dim=-1, keepdim=True)
        std = logits.std(dim=-1, keepdim=True).clamp_min(1e-6)
        logits = (logits - mu) / std

        # —— 温度缩放 ——
        logits = logits / self.tau.clamp_min(0.5)

        return logits
