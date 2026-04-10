import torch
from torch_geometric.data import Data
from DiffDGMN_main import gol

import torch.nn as nn


def getHyperGraph(seq, time_list):
    """
    Build a hyperedge-based sequence hypergraph.

    Args:
        seq: A user's visited POI sequence (global POI ids).
        time_list: The corresponding timestamps for the POI sequence.

    Returns:
        Data: A PyG Data object that contains hyperedge incidence structure and
              auxiliary temporal/geographical interval features.
    """
    # Build local node mapping and node features (store global POI ids in x)
    i, x, nodes = 0, [], {}
    node_indices = []
    for node in seq:
        if node not in nodes:
            nodes[node] = i
            x.append([node])
            i += 1
        node_indices.append(nodes[node])
    x = torch.LongTensor(x)
    
    # Build hyperedges: each hyperedge connects a window of consecutive POIs.
    # Here we use k=3 by default (can be adjusted if needed).
    k = min(3, len(seq))
    hyperedges = []
    hyperedge_attr = []
    
    for i in range(len(seq) - k + 1):
        # Create one hyperedge that connects k consecutive POIs (in local ids)
        edge = [node_indices[i+j] for j in range(k)]
        hyperedges.append(edge)
        
        # Hyperedge attributes: time span and average geographic distance
        time_span = time_list[i+k-1] - time_list[i] if i+k-1 < len(time_list) else 0
        distances = []
        for j in range(i, i+k-1):
            if j+1 < len(seq):
                distances.append(gol.dist_mat[seq[j], seq[j+1]].item())
        avg_distance = sum(distances) / len(distances) if distances else 0
        hyperedge_attr.append([time_span, avg_distance])
    
    # If the sequence is too short to form hyperedges, create a single hyperedge
    # that contains all nodes.
    if not hyperedges and len(seq) > 0:
        hyperedges.append(node_indices)
        time_span = time_list[-1] - time_list[0] if len(time_list) > 1 else 0
        avg_distance = 0
        if len(seq) > 1:
            distances = [gol.dist_mat[seq[j], seq[j+1]].item() for j in range(len(seq)-1)]
            avg_distance = sum(distances) / len(distances)
        hyperedge_attr.append([time_span, avg_distance])
    
    # Build hyperedge incidence index (node_id -> hyperedge_id)
    if hyperedges:
        # Flatten all (node, hyperedge) pairs
        node_to_hyperedge = []
        hyperedge_to_node = []
        
        for edge_idx, edge in enumerate(hyperedges):
            for node_idx in edge:
                node_to_hyperedge.append(node_idx)
                hyperedge_to_node.append(edge_idx)
        
        hyperedge_index = torch.LongTensor([node_to_hyperedge, hyperedge_to_node])
        hyperedge_attr = torch.FloatTensor(hyperedge_attr)
    else:
        # Edge case: no hyperedges (empty sequence)
        hyperedge_index = torch.LongTensor(2, 0)
        hyperedge_attr = torch.FloatTensor(0, 2)
    
    # Compute discrete time/distance intervals (optional auxiliary features)
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
    
    # Create the PyG Data object
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
    Generate hypergraph representations from hyperedges.

    Args:
        hyperedges: Hyperedges in either tensor form (e.g., hyperedge_index) or list form.
        node_features: Node feature matrix.

    Returns:
        sv: Node representations aggregated from incident hyperedges.
        si: Hyperedge representations aggregated from member nodes.
    """
    # Initialize node-level and hyperedge-level representations
    num_nodes = node_features.size(0)
    num_hyperedges = hyperedges.size(1) if isinstance(hyperedges, torch.Tensor) else len(hyperedges)
    
    # Node representations (sv)
    sv = torch.zeros(num_nodes, gol.conf['hid_dim'], device=node_features.device)
    
    # Hyperedge representations (si)
    si = torch.zeros(num_hyperedges, gol.conf['hid_dim'], device=node_features.device)
    
    # For each hyperedge, aggregate member node features to obtain hyperedge features
    if isinstance(hyperedges, torch.Tensor):
        # Tensor form (e.g., hyperedge_index)
        for i in range(num_hyperedges):
            # Nodes that belong to hyperedge i
            mask = hyperedges[1] == i
            node_indices = hyperedges[0][mask]
            
            # Aggregate node features within the hyperedge
            if len(node_indices) > 0:
                edge_features = node_features[node_indices]
                si[i] = torch.mean(edge_features, dim=0)
    else:
        # List form
        for i, edge in enumerate(hyperedges):
            if len(edge) > 0:
                edge_features = node_features[edge]
                si[i] = torch.mean(edge_features, dim=0)
    
    # For each node, aggregate incident hyperedge features to update node representations
    if isinstance(hyperedges, torch.Tensor):
        for i in range(num_nodes):
            # Hyperedges incident to node i
            mask = hyperedges[0] == i
            edge_indices = hyperedges[1][mask]
            
            # Aggregate incident hyperedge features
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
    Inputs:
      - hyperedge_index: LongTensor [2, M] incidence pairs (node_id, edge_id)
      - node_embs: FloatTensor [N, d] node embeddings (local nodes for one trajectory)
      - hyperedge_attr: FloatTensor [E, 3] hyperedge attributes (e.g., length / time span / avg distance), optional

    Outputs:
      - sv: FloatTensor [N, d] node representations in the hypergraph context (used as Value in module C)
      - si: FloatTensor [E, d] hyperedge representations (used as Query in module C and as Su in module D)
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

        # 1) Mean pooling over nodes within each hyperedge -> edge_mean: [E, d]
        edge_sum = node_embs.new_zeros((E, d))
        edge_cnt = node_embs.new_zeros((E, 1))
        edge_sum.index_add_(0, col, node_embs[row])
        edge_cnt.index_add_(0, col, torch.ones_like(col, dtype=torch.float32, device=device).unsqueeze(1))
        edge_mean = edge_sum / edge_cnt.clamp_min(1.0)

        # 2) Fuse hyperedge attributes -> si
        if hyperedge_attr is None or hyperedge_attr.numel() == 0:
            si = self.node_mlp(edge_mean)  # [E, d]
        else:
            hea = hyperedge_attr
            if hea.dim() == 1:
                hea = hea.unsqueeze(-1)
            # If attributes are not provided as one row per hyperedge, aggregate
            # per-incidence attributes to per-hyperedge attributes using `col`.
            if hea.size(0) != E:
                if hea.size(0) == col.numel():
                    # Segment-mean aggregation to obtain [E, k]
                    k = hea.size(1)
                    hea_mean = edge_mean.new_zeros((E, k))
                    cnt = edge_mean.new_zeros((E, 1))
                    hea = hea.to(device=edge_mean.device, dtype=edge_mean.dtype)
                    hea_mean.index_add_(0, col, hea)
                    cnt.index_add_(0, col, torch.ones(col.size(0), 1, device=edge_mean.device, dtype=edge_mean.dtype))
                    hea = hea_mean / cnt.clamp_min(1.0)
                else:
                    # Fallback alignment to E rows (pad/truncate)
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

        # 3) Aggregate hyperedge representations back to nodes -> sv
        node_sum = node_embs.new_zeros((N, d))
        node_cnt = node_embs.new_zeros((N, 1))
        # Keep dtype consistent (e.g., under AMP) to avoid index_add_ dtype mismatch.
        si = si.to(node_embs.dtype)
        node_sum.index_add_(0, row, si[col])
        node_cnt.index_add_(0, row, torch.ones_like(row, dtype=torch.float32, device=device).unsqueeze(1))
        sv = node_sum / node_cnt.clamp_min(1.0)           # [N, d]

        return sv, si
