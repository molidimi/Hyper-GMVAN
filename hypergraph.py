import torch
from torch_geometric.data import Data
from DiffDGMN_main import gol

import torch.nn as nn


def getHyperGraph(seq, time_list):
    """
    生成基于超边的序列图
    
    Args:
        seq: 用户访问的POI序列
        time_list: 对应的时间戳序列
        
    Returns:
        Data: 包含超边结构的图数据对象
    """
    # 构建节点映射和特征
    i, x, nodes = 0, [], {}
    node_indices = []
    for node in seq:
        if node not in nodes:
            nodes[node] = i
            x.append([node])
            i += 1
        node_indices.append(nodes[node])
    x = torch.LongTensor(x)
    
    # 构建超边 - 每个超边连接连续的k个POI (这里使用k=3，可以根据需要调整)
    k = min(3, len(seq))
    hyperedges = []
    hyperedge_attr = []
    
    for i in range(len(seq) - k + 1):
        # 创建一个超边连接连续的k个POI
        edge = [node_indices[i+j] for j in range(k)]
        hyperedges.append(edge)
        
        # 计算这个超边的属性（例如，时间跨度和平均距离）
        time_span = time_list[i+k-1] - time_list[i] if i+k-1 < len(time_list) else 0
        distances = []
        for j in range(i, i+k-1):
            if j+1 < len(seq):
                distances.append(gol.dist_mat[seq[j], seq[j+1]].item())
        avg_distance = sum(distances) / len(distances) if distances else 0
        hyperedge_attr.append([time_span, avg_distance])
    
    # 如果序列太短，无法形成足够的超边，则创建至少一个超边
    if not hyperedges and len(seq) > 0:
        hyperedges.append(node_indices)
        time_span = time_list[-1] - time_list[0] if len(time_list) > 1 else 0
        avg_distance = 0
        if len(seq) > 1:
            distances = [gol.dist_mat[seq[j], seq[j+1]].item() for j in range(len(seq)-1)]
            avg_distance = sum(distances) / len(distances)
        hyperedge_attr.append([time_span, avg_distance])
    
    # 构建超边索引 (节点到超边的映射)
    if hyperedges:
        # 展平所有超边中的节点
        node_to_hyperedge = []
        hyperedge_to_node = []
        
        for edge_idx, edge in enumerate(hyperedges):
            for node_idx in edge:
                node_to_hyperedge.append(node_idx)
                hyperedge_to_node.append(edge_idx)
        
        hyperedge_index = torch.LongTensor([node_to_hyperedge, hyperedge_to_node])
        hyperedge_attr = torch.FloatTensor(hyperedge_attr)
    else:
        # 处理边缘情况：没有足够的数据形成超边
        hyperedge_index = torch.LongTensor(2, 0)
        hyperedge_attr = torch.FloatTensor(0, 2)
    
    # 计算时间和距离间隔
    def get_min(interv):
        interv_min = interv.clone()
        interv_min[interv_min == 0] = 2 ** 31
        return interv_min.min()
    
    time_interv = (time_list[1:] - time_list[:-1]).long() if len(time_list) > 1 else torch.LongTensor([])
    dist_interv = gol.dist_mat[seq[:-1], seq[1:]].long() if len(seq) > 1 else torch.LongTensor([])
    mean_interv = dist_interv.float().mean() if len(dist_interv) > 0 else torch.tensor(0.0)
    
    if time_interv.size(0) > 0:
        time_interv = torch.clamp((time_interv / get_min(time_interv)).long(), 0, gol.conf['interval'] - 1)
        dist_interv = torch.clamp((dist_interv / get_min(dist_interv)).long(), 0, gol.conf['interval'] - 1)
    
    # 创建包含超边结构的图数据对象
    return Data(
        x=x, 
        hyperedge_index=hyperedge_index, 
        num_nodes=len(nodes),
        mean_interv=mean_interv, 
        edge_time=time_interv, 
        edge_dist=dist_interv,
        hyperedge_attr=hyperedge_attr
    )

def generateHyperGraph(hyperedges, node_features):
    """
    根据超边生成超图
    
    Args:
        hyperedges: 超边列表
        node_features: 节点特征
        
    Returns:
        sv: 超图节点表示
        si: 超边表示
    """
    # 初始化超图节点表示和超边表示
    num_nodes = node_features.size(0)
    num_hyperedges = hyperedges.size(1) if isinstance(hyperedges, torch.Tensor) else len(hyperedges)
    
    # 创建超图节点表示 sv
    sv = torch.zeros(num_nodes, gol.conf['hid_dim'], device=node_features.device)
    
    # 创建超边表示 si
    si = torch.zeros(num_hyperedges, gol.conf['hid_dim'], device=node_features.device)
    
    # 对于每个超边，聚合其中节点的特征来创建超边表示
    if isinstance(hyperedges, torch.Tensor):
        # 如果hyperedges是张量形式 (如hyperedge_index)
        for i in range(num_hyperedges):
            # 找出属于当前超边的所有节点
            mask = hyperedges[1] == i
            node_indices = hyperedges[0][mask]
            
            # 聚合这些节点的特征
            if len(node_indices) > 0:
                edge_features = node_features[node_indices]
                si[i] = torch.mean(edge_features, dim=0)
    else:
        # 如果hyperedges是列表形式
        for i, edge in enumerate(hyperedges):
            if len(edge) > 0:
                edge_features = node_features[edge]
                si[i] = torch.mean(edge_features, dim=0)
    
    # 对于每个节点，聚合其所属的所有超边的特征来更新节点表示
    if isinstance(hyperedges, torch.Tensor):
        for i in range(num_nodes):
            # 找出当前节点所属的所有超边
            mask = hyperedges[0] == i
            edge_indices = hyperedges[1][mask]
            
            # 聚合这些超边的特征
            if len(edge_indices) > 0:
                node_hyperedge_features = si[edge_indices]
                sv[i] = torch.mean(node_hyperedge_features, dim=0)
    else:
        node_to_edges = [[] for _ in range(num_nodes)]
        for i, edge in enumerate(hyperedges):
            for node in edge:
                node_to_edges[node].append(i)
        
        for i in range(num_nodes):
            if node_to_edges[i]:
                sv[i] = torch.mean(si[node_to_edges[i]], dim=0)
    
    return sv, si


class HyperGraphRep(nn.Module):
    """
    输入：
      - hyperedge_index: LongTensor [2, M]  (node_id, edge_id)
      - node_embs     : FloatTensor [N, d]  (全局 POI embedding table)
      - hyperedge_attr: FloatTensor [E, 3]  (长度/时间跨度/平均邻距)  —— 可选
    输出：
      - sv: [N, d]  节点（POI）在超图上下文中的表征（供 C 模块作 Value）
      - si: [E, d]  超边表征（供 C 的 Query，亦作为 D 模块 Su）
    """
    def __init__(self, dim: int, he_attr_dim: int = 3):
        super().__init__()
        self.dim = dim
        self.he_mlp = nn.Sequential(
            nn.Linear(dim + he_attr_dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim)
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim)
        )

    def forward(self, hyperedge_index, node_embs, hyperedge_attr=None):
        device = node_embs.device
        row, col = hyperedge_index  # row=node_id, col=edge_id
        E = int(col.max().item() + 1) if col.numel() else 0
        N, d = node_embs.size(0), node_embs.size(1)

        if E == 0:
            sv = node_embs
            si = node_embs.new_zeros((1, d))
            return sv, si

        # 1) 超边内部节点平均 → edge_mean: [E, d]
        edge_sum = node_embs.new_zeros((E, d))
        edge_cnt = node_embs.new_zeros((E, 1))
        edge_sum.index_add_(0, col, node_embs[row])
        edge_cnt.index_add_(0, col, torch.ones_like(col, dtype=torch.float32, device=device).unsqueeze(1))
        edge_mean = edge_sum / edge_cnt.clamp_min(1.0)

        # 2) 超边属性融合 → si
        if hyperedge_attr is None or hyperedge_attr.numel() == 0:
            si = self.node_mlp(edge_mean)  # [E, d]
        else:
            hea = hyperedge_attr
            if hea.dim() == 1:
                hea = hea.unsqueeze(-1)
            # 若传入的不是“每超边一行”，则将“每入射边属性”按 col 聚合为“每超边属性”
            if hea.size(0) != E:
                if hea.size(0) == col.numel():
                    # segment mean 到每超边
                    # 构造 [E, k]
                    k = hea.size(1)
                    hea_mean = edge_mean.new_zeros((E, k))
                    cnt = edge_mean.new_zeros((E, 1))
                    hea = hea.to(device=edge_mean.device, dtype=edge_mean.dtype)
                    hea_mean.index_add_(0, col, hea)
                    cnt.index_add_(0, col, torch.ones(col.size(0), 1, device=edge_mean.device, dtype=edge_mean.dtype))
                    hea = hea_mean / cnt.clamp_min(1.0)
                else:
                    # 兜底对齐到 E 行（pad/截断）
                    hea = hea.to(device=edge_mean.device, dtype=edge_mean.dtype)
                    if hea.size(0) < E:
                        pad = torch.zeros((E - hea.size(0), hea.size(1)), device=edge_mean.device, dtype=edge_mean.dtype)
                        hea = torch.cat([hea, pad], dim=0)
                    else:
                        hea = hea[:E]
            else:
                hea = hea.to(device=edge_mean.device, dtype=edge_mean.dtype)

            he = torch.cat([edge_mean, hea], dim=-1)  # [E, d+k]
            si = self.he_mlp(he)  # [E, d]

        # 3) 反向聚合回节点 → sv
        node_sum = node_embs.new_zeros((N, d))
        node_cnt = node_embs.new_zeros((N, 1))
        # AMP 场景下确保 dtype 一致，避免 Float/Half 混用导致 index_add_ 报错
        si = si.to(node_embs.dtype)
        node_sum.index_add_(0, row, si[col])
        node_cnt.index_add_(0, row, torch.ones_like(row, dtype=torch.float32, device=device).unsqueeze(1))
        sv = node_sum / node_cnt.clamp_min(1.0)           # [N, d]

        return sv, si
