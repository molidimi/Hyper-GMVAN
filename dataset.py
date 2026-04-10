import torch
import numpy as np
from torch.utils.data import Dataset
from DiffDGMN_main import gol
from DiffDGMN_main.gol import  pLog
from os.path import join
import pickle as pkl
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_undirected

def NDCG_at_k(r, k, method=1):
    # Use np.asarray (np.asfarray is deprecated); enforce float32 dtype.
    r = np.asarray(r, dtype=np.float32)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.

def ACC_at_k(r, k, all_pos_num):
    # Ensure r is a numpy array and keep only top-k elements.
    r_array = np.asarray(r, dtype=np.float32)[:k]
    return np.sum(r_array) / all_pos_num

def MRR(r):
    # Ensure r is a numpy array.
    r = np.asarray(r)
    if np.sum(r) == 0:
        return 0.
    return np.reciprocal(np.where(r==1)[0]+1, dtype=float)[0]

def getSeqGraph(seq, seq_time):
    """
    Build a trajectory hypergraph using sliding-window hyperedges.

    Notes:
    - Hyperedge incidence `row` uses local node ids 0..N_local-1.
    - Data.x is aligned with local node ids and stores the original global POI ids.
    - Hyperedge attributes are 3-dim: [length, time span, avg geographic distance].
    """
    import torch
    from torch_geometric.data import Data
    from DiffDGMN_main import gol

    # 1) Build local node mapping: global_id -> local_id
    loc = {}
    x_list = []               # Data.x: aligned with local ids, stores original global ids
    loc_seq = []              # Local id for each position in the sequence
    for gid in seq:
        if gid not in loc:
            loc[gid] = len(loc)
            x_list.append([gid])  # shape [N_local, 1]
        loc_seq.append(loc[gid])

    x = torch.LongTensor(x_list) if x_list else torch.LongTensor([])     # [N_local, 1]
    L = len(seq)
    w = int(gol.conf.get('hyper_edge_window', 4))
    E = max(1, L - w + 1)

    # 2) Build hyperedge incidence index (using local node ids)
    row_node, col_edge = [], []
    for e in range(E):
        start, end = e, min(e + w, L)
        for j in range(start, end):
            row_node.append(loc_seq[j])   # local id
            col_edge.append(e)
    hyperedge_index = torch.tensor([row_node, col_edge], dtype=torch.long)  # [2, M]

    # 3) Hyperedge attributes: length/span/gdist (3 dims)
    if len(seq_time) > 0:
        # Avoid wrapping a tensor again via torch.tensor(...)
        t = torch.as_tensor(seq_time, dtype=torch.float32)
    else:
        t = torch.arange(L, dtype=torch.float32)

    he_attr = []
    for e in range(E):
        start, end = e, min(e + w, L)
        length = end - start
        span = float(t[end - 1] - t[start]) if length >= 2 else 0.0
        if length >= 2:
            sub = seq[start:end]
            dsum, cnt = 0.0, 0
            for i in range(len(sub) - 1):
                dsum += float(gol.dist_mat[sub[i], sub[i + 1]])
                cnt += 1
            gdist = dsum / max(1, cnt)
        else:
            gdist = 0.0
        he_attr.append([length, span, gdist])
    hyperedge_attr = torch.tensor(he_attr, dtype=torch.float32)  # [E, 3]

    G = Data()
    G.x = x                               # [N_local, 1], values are global POI ids
    G.hyperedge_index = hyperedge_index   # [2, M], row uses local node ids
    G.hyperedge_attr  = hyperedge_attr    # [E, 3]
    # Store per-graph hyperedge count for constructing hyperedge_batch after batching
    G.num_hyperedges = torch.tensor(int(E), dtype=torch.long)
    return G



def getHyperGraph(seq, time_list):
    """Build a hyperedge-based sequence hypergraph.

    Args:
        seq: A user's visited POI sequence (global POI ids).
        time_list: The corresponding timestamps.

    Returns:
        Data: A PyG Data object containing:
            - x: node features (POI ids)
            - hyperedge_index: hyperedge incidence pairs
            - num_nodes: number of nodes
            - mean_interv: mean distance interval (auxiliary)
            - edge_time: discretized time intervals (auxiliary)
            - edge_dist: discretized distance intervals (auxiliary)
            - hyperedge_attr: hyperedge attributes
    """
    # Build local node mapping and node features
    i, x, nodes = 0, [], {}
    node_indices = []
    for node in seq:
        if node not in nodes:
            nodes[node] = i
            x.append([node])
            i += 1
        node_indices.append(nodes[node])
    x = torch.LongTensor(x)

    # Build hyperedges: each hyperedge connects k consecutive POIs (default k=3).
    k = min(3, len(seq))
    hyperedges = []
    hyperedge_attr = []

    for i in range(len(seq) - k + 1):
        # Create one hyperedge connecting k consecutive POIs
        edge = [node_indices[i+j] for j in range(k)]
        hyperedges.append(edge)

        # Hyperedge attributes (e.g., time span and average distance)
        time_span = time_list[i+k-1] - time_list[i] if i+k-1 < len(time_list) else 0
        distances = []
        for j in range(i, i+k-1):
            if j+1 < len(seq):
                distances.append(gol.dist_mat[seq[j], seq[j+1]].item())
        avg_distance = sum(distances) / len(distances) if distances else 0
        hyperedge_attr.append([time_span, avg_distance])

    # If the sequence is too short, create at least one hyperedge
    if not hyperedges and len(seq) > 0:
        hyperedges.append(node_indices)
        time_span = time_list[-1] - time_list[0] if len(time_list) > 1 else 0
        avg_distance = 0
        if len(seq) > 1:
            distances = [gol.dist_mat[seq[j], seq[j+1]].item() for j in range(len(seq)-1)]
            avg_distance = sum(distances) / len(distances)
        hyperedge_attr.append([time_span, avg_distance])

    # Build hyperedge incidence index (node -> hyperedge mapping)
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
        # Edge case: no hyperedges
        hyperedge_index = torch.LongTensor(2, 0)
        hyperedge_attr = torch.FloatTensor(0, 2)

    # Compute discretized time and distance intervals (auxiliary)
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

class GraphData(Dataset):
    def __init__(self, n_user, n_poi, seq_data, pos_dict, is_eval=False, tr_dict=None):
        self.n_user, self.n_poi = n_user, n_poi
        self.seq_data = seq_data
        self.is_eval = is_eval

        self.tr_dict = tr_dict
        self.pos_dict = pos_dict
        self.userSet = list(self.pos_dict.keys())
        self.len = len(self.seq_data)
        self.max_len = gol.conf['max_len']



    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if not self.is_eval:
            uid, poi, seq, seq_time, cur_time, _ = self.seq_data[index]
            pos_set = set(self.pos_dict[uid])
            if len(seq) > self.max_len:
                seq = seq[-self.max_len:]
                seq_time = seq_time[-self.max_len:]
            seq_time = torch.LongTensor(seq_time)
            seq_graph = getSeqGraph(seq, seq_time)      #��������ͼ���󣺽������е�POI�ڵ㼰���Ⱥ��ϵ���������ͼ�����б�����������ÿ���߶�Ӧ��ʱ���;������ɢֵ������Ϊ������

            seq = torch.LongTensor(seq)
            neg = np.random.randint(0, self.n_poi)
            while neg in pos_set:
                neg = np.random.randint(0, self.n_poi)
            return uid, poi, neg, seq, seq_graph, (cur_time // 60) % 168
        else:
            uid, poi, seq, seq_time, cur_time, _ = self.seq_data[index]
            labels = torch.zeros((self.n_poi, )).long()
            labels[poi] = 1

            exclude_set = torch.LongTensor(list(set(self.tr_dict[uid])))
            exclude_mask = torch.zeros((self.n_poi, )).bool()
            exclude_mask[exclude_set] = 1
            exclude_mask[poi] = 0

            if len(seq) > self.max_len:
                seq = seq[-self.max_len:]
                seq_time = seq_time[-self.max_len:]
            seq_time = torch.LongTensor(seq_time)
            seq_graph = getSeqGraph(seq, seq_time)
            seq = torch.LongTensor(seq)

            return uid, labels.unsqueeze(0), exclude_mask.bool().unsqueeze(0), \
                seq, seq_graph, (cur_time // 60) % 168


def collate_edge(batch):
    u, p, n, s, s_graph, t = tuple(zip(*batch))
    u = torch.LongTensor(u).to(gol.device)
    p = torch.LongTensor(p).to(gol.device)
    n = torch.LongTensor(n).to(gol.device)
    s_graph = Batch.from_data_list(s_graph).to(gol.device)
    t = torch.LongTensor(t).to(gol.device)

    # Build `hyperedge_batch`: prefer exact per-graph hyperedge counts; otherwise fall back safely.
    if not hasattr(s_graph, "hyperedge_batch"):
        # Choose a safe device source in case s_graph.x is absent
        _dev = s_graph.hyperedge_index.device

        # No hyperedges
        if s_graph.hyperedge_index.numel() == 0:
            s_graph.hyperedge_batch = torch.empty(0, dtype=torch.long, device=_dev)

        # Preferred path: use real per-graph hyperedge counts (recommended)
        elif hasattr(s_graph, "num_hyperedges"):
            # num_hyperedges should contain per-graph hyperedge counts (len == num_graphs)
            e_counts = torch.as_tensor(s_graph.num_hyperedges, device=_dev).long().view(-1)
            counts_list = e_counts.tolist()
            if any(c > 0 for c in counts_list):
                parts = [torch.full((int(c),), i, dtype=torch.long, device=_dev)
                         for i, c in enumerate(counts_list) if c > 0]
                s_graph.hyperedge_batch = torch.cat(parts, dim=0)
            else:
                s_graph.hyperedge_batch = torch.empty(0, dtype=torch.long, device=_dev)

        # Fallback: approximately allocate hyperedges across graphs (may be imprecise; last resort)
        else:
            # Assume hyperedge ids are contiguous (0..E_total-1); otherwise use unique().
            edge_ids = s_graph.hyperedge_index[1]
            if edge_ids.numel() == 0:
                s_graph.hyperedge_batch = torch.empty(0, dtype=torch.long, device=_dev)
            else:
                E_total = int(edge_ids.max().item()) + 1

                # Use per-graph node counts (from ptr) as weights; still an approximation.
                lengths = (s_graph.ptr[1:] - s_graph.ptr[:-1]).to(torch.long).tolist()
                total_len = max(sum(lengths), 1)
                # Round then correct the sum to avoid off-by-one
                e_counts = [int(round(E_total * (l / total_len))) for l in lengths]
                diff = E_total - sum(e_counts)
                if diff != 0:
                    e_counts[0] += diff

                parts = [torch.full((c,), i, dtype=torch.long, device=_dev)
                         for i, c in enumerate(e_counts) if c > 0]
                s_graph.hyperedge_batch = torch.cat(parts, dim=0) if parts else torch.empty(0, dtype=torch.long, device=_dev)

    # Store per-sample sequence lengths for splitting padded sequences in module C
    seq_lens = [int(x.numel()) for x in s]
    s_graph.seq_lens = torch.LongTensor(seq_lens).to(gol.device)

    return u, p, n, s, s_graph, t

def collate_eval(batch):
    u, label, exclude_mask, seq, s_graph, t = tuple(zip(*batch))
    u = torch.LongTensor(u).to(gol.device)
    s_graph = Batch.from_data_list(s_graph).to(gol.device)
    t = torch.LongTensor(t).to(gol.device)

    # Build hyperedge_batch for validation/test (same idea as training)
    if not hasattr(s_graph, 'hyperedge_batch'):
        if s_graph.hyperedge_index.numel() == 0:
            s_graph.hyperedge_batch = torch.empty(0, dtype=torch.long, device=s_graph.x.device)
        else:
            edge_ids = s_graph.hyperedge_index[1]
            E_total = int(edge_ids.max().item() + 1)
            lengths = (s_graph.ptr[1:] - s_graph.ptr[:-1]).tolist()
            total_len = sum(lengths) + 1e-6
            e_counts = [int(round(E_total * (l / total_len))) for l in lengths]
            diff = E_total - sum(e_counts)
            if diff != 0:
                e_counts[0] += diff
            hb = []
            for i, c in enumerate(e_counts):
                if c > 0:
                    hb.append(torch.full((c,), i, dtype=torch.long, device=s_graph.x.device))
            s_graph.hyperedge_batch = torch.cat(hb, dim=0) if hb else torch.empty(0, dtype=torch.long, device=s_graph.x.device)

    # Store per-sample sequence lengths
    seq_lens = [int(x.numel()) for x in seq]
    s_graph.seq_lens = torch.LongTensor(seq_lens).to(gol.device)

    return u, torch.cat(label, dim=0), torch.cat(exclude_mask, dim=0), seq, s_graph, t


def getDatasets(path='../data/processed', dataset='IST'):
    dist_pth = join(path, dataset.upper())
    gol.pLog(f'Loading from {dist_pth}')
    with open(join(dist_pth, 'all_data.pkl'), 'rb') as f:
        n_user, n_poi = pkl.load(f)
        df = pkl.load(f)
        trn_set, val_set, tst_set = pkl.load(f)
        trn_df, val_df, tst_df = pkl.load(f)

    trn_dict, val_dict, tst_dict = {}, {}, {}
    for uid, line in trn_df.groupby('uid'):
        trn_dict[uid] = line['poi'].tolist()
    for uid, line in val_df.groupby('uid'):
        val_dict[uid] = line['poi'].tolist()
    for uid, line in tst_df.groupby('uid'):
        tst_dict[uid] = line['poi'].tolist()

    # Build GraphData datasets
    trn_ds = GraphData(n_user, n_poi, trn_set, trn_dict)
    val_ds = GraphData(n_user, n_poi, val_set, val_dict, is_eval=True, tr_dict=trn_dict)
    tst_ds = GraphData(n_user, n_poi, tst_set, tst_dict, is_eval=True, tr_dict=trn_dict)

    with open(join(dist_pth, 'dist_graph.pkl'), 'rb') as f:
        geo_edges = torch.LongTensor(pkl.load(f))
    edge_weights = torch.Tensor(np.load(join(dist_pth, 'dist_on_graph.npy')))
    edge_weights /= edge_weights.max()
    geo_edges, edge_weights = to_undirected(geo_edges, edge_weights, num_nodes=n_poi)

    assert geo_edges.size(1) == edge_weights.size(0)
    assert len(trn_df) == len(trn_set) + n_user

    geo_graph = Data(edge_index=geo_edges, edge_attr=edge_weights).to(gol.device)
    pLog(f'#Users: {n_user}, #POIs: {n_poi}, #Check-ins: {len(df)}')
    pLog(f'#Train: {len(trn_set)}, #Valid: {len(val_set)}, #Test: {len(tst_set)}')
    pLog(f'Distance Graph: #Edges: {geo_edges.size(1) // 2}, Avg. degree: {geo_edges.size(1) / n_poi / 2:.2f}')
    pLog(f'Avg. visit: {len(df) / n_user:.2f}, Density: {len(df) / n_user / n_poi:.3f}, Sparsity: {1 - len(df) / n_user / n_poi:.3f}')
    return n_user, n_poi, (trn_ds, val_ds, tst_ds), geo_graph
