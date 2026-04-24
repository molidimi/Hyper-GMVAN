[README.md](https://github.com/user-attachments/files/27051967/README.md)
# Hyper-GMVAN

Hyper-GMVAN is a **hypergraph-enhanced, diffusion-based POI recommendation** implementation built on top of the Diff-DGMN codebase structure. It models a user’s next-POI preference from historical check-in sequences by combining:

- **A (Hypergraph)**: per-trajectory hypergraph construction + hypergraph representation learning
- **B (GTConv / Distance Graph)**: global POI distance-graph encoding with GTConv layers
- **C (KAN Fusion)**: attention fusion followed by a KAN nonlinearity to produce a noisy archetype
- **D (Diffusion)**: VP-SDE reverse-time denoising to correct the archetype

This repository keeps the original folder naming (e.g., `DiffDGMN_main`, `DiffDGMN_data`) to remain compatible with the code imports.

---

## Description

Given a user check-in trajectory \((uid, [p_1..p_L], [t_1..t_L])\), Hyper-GMVAN:

1. Builds a **sequence hypergraph** from the POI sequence using sliding-window hyperedges.
2. Encodes the hypergraph to obtain:
   - `sv_global`: POI-level hypergraph representations (used as **Value** in fusion)
   - `Su`: user-level hyperedge/semantic representation (used as **Query** and diffusion **condition**)
3. Encodes a **global POI distance graph** to obtain `R_V_all` (used as **Key** and candidate representations).
4. Fuses `Su` with sequence-level `K/V` and applies **KAN** to produce a noisy location archetype `L_hat_u`.
5. Uses **VP-SDE reverse diffusion** conditioned on `Su` to produce a denoised archetype `Lu`.
6. Computes final logits over all POIs via multi-branch dot-product fusion.

---

## Dataset Information

### Raw data format

The original raw Foursquare check-in data typically contains 7 columns:

`['uid', 'poi', 'time', 'offset', 'lat', 'lon', 'catname']`

### Processed data (required by this repo)

Training expects **processed** datasets under:

`DiffDGMN_data/data/processed/<CITY>/`

Each city folder should contain (at minimum):

- `all_data.pkl` (user/POI counts, dataframe, and train/valid/test splits)
- `dist_mat.npy` (POI distance matrix, shape `[n_poi, n_poi]`)
- `dist_graph.pkl` (distance-graph edges)
- `dist_on_graph.npy` (edge weights for the distance graph)

The included `DiffDGMN_data/data/raw/README.txt` provides background about the original dataset source.

---

## Event data → model inputs (data pipeline details)

### 1) How event data becomes model inputs

**English**

This repo trains from the preprocessed `all_data.pkl`. Each training sample in `trn_set/val_set/tst_set` is a tuple shaped like:

- `uid`: user id (integer, **0..n_user-1**)
- `poi`: the next POI label (integer, **0..n_poi-1**)
- `seq`: the historical POI sequence before `poi` (list/array of POI ids)
- `seq_time`: timestamps aligned with `seq` (list/array; integer-like)
- `cur_time`: current timestamp for this prediction (integer-like)
- an extra field (unused by the current dataloader)

At runtime (`DiffDGMN_main/dataset.py`):

- **Sequence graph input**: `seq` + `seq_time` are converted to a PyG `Data` graph by `getSeqGraph()`:
  - Local node ids are built per sequence; `Data.x` stores the **original global POI ids**.
  - A sliding window (`hyper_edge_window`, default 4) creates hyperedges.
  - Hyperedge attributes are `hyperedge_attr = [length, time_span, avg_geo_distance]`.
- **Flattened sequence input**: the raw `seq` is also passed as a flattened tensor `batch.seq` and later padded inside the model to build attention K/V.
- **Time feature**: `(cur_time // 60) % 168` produces an **hour-of-week** bucket (0..167). (Currently it is returned by the dataloader; whether it is used depends on the model wiring.)

### 2) Resize / padding behavior

- **Sequence “resize”**: if `len(seq) > max_len` (CLI arg `--length`, default 200), the dataloader **truncates to the most recent** `max_len` events:
  - `seq = seq[-max_len:]`
  - `seq_time = seq_time[-max_len:]`
- **Padding**: sequences are **not padded in the dataset**. In `DiffDGMN_main/model.py`, the flattened sequences are split back by per-sample lengths and padded with zeros via `pad_sequence(..., padding_value=0.0)` to form `[B, L, d]` tensors for attention.

### 3) Normalization (where it happens)

- **Distance-graph edge weights**: in `getDatasets()`, `dist_on_graph.npy` is normalized by max:
  - `edge_weights /= edge_weights.max()`
  These weights become `G_D.edge_attr` used by module B.
- **Hyperedge attributes**: `getSeqGraph()` uses raw `dist_mat` values for `avg_geo_distance` and raw timestamp differences for `time_span` (no extra scaling inside this repo).
- **Model-side normalization**:
  - `L_hat_u` is LayerNorm’ed then L2-normalized (`F.normalize(..., p=2)`).
  - Final logits are **z-score normalized per user**: `(logits - mean) / std`, then temperature-scaled.

### 4) Label mapping (how `poi` becomes a training target)

- The code assumes that processed POI ids are already **contiguous indices** `0..n_poi-1`.
- **Training**: the positive label is the scalar `poi` id. The model produces logits `[B, n_poi]`, and the positive score is selected by `gather`.
- **Validation/Test**: a one-hot label vector `labels` of shape `[n_poi]` is created with `labels[poi] = 1`. During evaluation, previously seen training POIs for that user are masked out (see below).

### 5) Ignore index / masking (what is ignored and how)

- There is **no `ignore_index`-style loss** in this repo (no token-level cross entropy over padded positions).
- What is “ignored” happens via **masking candidates** at evaluation time:
  - For each user, `exclude_mask` is built from that user’s **training** visited POIs (`tr_dict[uid]`).
  - In `eval_model()`, scores where `exclude_mask == True` are set to a large negative value (`-1e10`) so they are not ranked.
- Padding positions inside attention are padded with zeros; the current implementation does **not** pass a `key_padding_mask` into `nn.MultiheadAttention`, so padded steps contribute as zero vectors.

### 6) Train/valid/test split (how it is determined)

- Splits are **precomputed in the processed dataset** and stored in `all_data.pkl`:
  - `trn_set, val_set, tst_set` (sample tuples)
  - `trn_df, val_df, tst_df` (dataframes used to build per-user visited POI lists)
- This repo does **not** re-split raw events on the fly; to change the split, regenerate `all_data.pkl`.

---

## Code Information

### Entry points

- **Training / evaluation**: `DiffDGMN_main/main.py`
- **Global config + paths**: `DiffDGMN_main/gol.py`

### Main modules

- **A (Hypergraph)**:
  - Hypergraph construction and batching: `DiffDGMN_main/dataset.py`
  - Hypergraph encoder: `DiffDGMN_main/hypergraph.py` (`HyperGraphRep`)
- **B (GTConv distance graph)**:
  - `DiffDGMN_main/layers.py` (`DisGraphRep`, `GTConv`)
- **C (KAN fusion)**:
  - `DiffDGMN_main/model.py` (Multi-Head Attention + `KAN` + gated residual)
- **D (Diffusion)**:
  - `DiffDGMN_main/layers.py` (`SDE_Diffusion`, torchsde-based solver)

---

## Usage Instructions

### 1) Create environment

Tested with **Python 3.8** (recommended: 3.8.13).

### 2) Install requirements

Core packages (as used in the original setup):

- Python == 3.8.13
- torch == 1.12.1
- torchsde == 0.2.6
- torch_geometric == 2.3.1
- pandas == 2.0.3
- numpy == 1.23.3

Notes:

- `torch_geometric` requires additional wheels (`torch-scatter`, `torch-sparse`, etc.) matching your PyTorch/CUDA build. Please install PyG following the official PyG instructions for your platform.

### 3) Prepare dataset path

By default, the code resolves processed data under:

`<repo_root>/DiffDGMN_data/data/processed`

You can override it with an environment variable.

**Windows PowerShell example**:

```powershell
$env:DIFFDGMN_DATA_PATH="E:\25_Literature_materials\DiffDGMN\Hyper-GMVAN\DiffDGMN_data\data\processed"
```

You can also override checkpoint output directory:

```powershell
$env:DIFFDGMN_CKPT_PATH="E:\25_Literature_materials\DiffDGMN\Hyper-GMVAN\DiffDGMN_main\checkpoints"
```

### 4) Run training

From the project root:

```powershell
cd "E:\25_Literature_materials\DiffDGMN\Hyper-GMVAN\DiffDGMN_main"
python main.py --dataset LA --gpu 0 --dp 0.2
```

### 5) Resume training (checkpoint)

```powershell
python main.py --dataset LA --gpu 0 --resume
```

Or specify a checkpoint explicitly:

```powershell
python main.py --dataset LA --gpu 0 --resume --resume_path "path\to\ckpt_run_XXX_latest.pt"
```

### 6) Evaluate

Evaluation runs automatically during training in `main.py`:

- Validation metrics: Recall@K / NDCG@K / MRR / ACC@K
- Test metrics: computed each epoch for monitoring; best checkpoint is selected by valid NDCG@5.

---

## Requirements

See “Install requirements” above. Minimum runtime dependencies are:

- PyTorch
- torchsde
- torch_geometric
- numpy / pandas

GPU is optional but recommended for performance.

---

## Methodology (Implementation Logic)

### A) Sequence → Hypergraph construction

Implemented in `DiffDGMN_main/dataset.py`:

- Build a per-trajectory hypergraph via sliding windows over the POI sequence.
- Create incidence pairs `hyperedge_index` (local node id, hyperedge id).
- Build hyperedge attributes `hyperedge_attr = [length, time_span, avg_geo_distance]`.

### B) Global POI distance graph encoding (GTConv)

Implemented in `DiffDGMN_main/layers.py`:

- Construct a global POI distance graph `G_D` with edge weights as distance features.
- Apply stacked `GTConv` layers to produce `R_V_all` (candidate POI geographic representations).

### C) KAN fusion

Implemented in `DiffDGMN_main/model.py`:

- Build attention inputs:
  - `Q = Su` (hypergraph semantic vector)
  - `K = R_V_all[seq]` (distance-graph features indexed by the sequence)
  - `V = sv_global[seq]` (hypergraph node features indexed by the sequence)
- Multi-Head Attention → KAN → gated residual → noisy archetype `L_hat_u`.

### D) Diffusion correction (VP-SDE)

Implemented in `DiffDGMN_main/layers.py`:

- Conditioned reverse-time VP-SDE generation produces `Lu` from `L_hat_u` with condition `Su`.
- A score network is trained via Fisher/DSM-style objective.

### Final scoring

Three branches are fused:

- `Su · R_V_allᵀ`
- `Lu · R_V_allᵀ`
- `User_emb · E_poiᵀ`

Then logits are normalized per user and temperature-scaled before ranking.

---

## Citations

This implementation inherits the overall Diff-DGMN research line. If you use this code in academic work, please cite the original Diff-DGMN paper:

- J. Zuo and Y. Zhang, “Diff-DGMN: A Diffusion-Based Dual Graph Multiattention Network for POI Recommendation,” *IEEE Internet of Things Journal*, vol. 11, no. 23, pp. 38393–38409, Dec. 2024. DOI: `https://doi.org/10.1109/JIOT.2024.3446048`

If you additionally introduce Hyper-GMVAN as a new method in your work, please cite your corresponding manuscript/report accordingly.

---

## License

This repository includes an MIT license (see `DiffDGMN_main/LICENSE`).

---

## Contribution Guidelines

Issues and pull requests are welcome. For reproducible research contributions, please include:

- the dataset name (LA/NYC/IST/JK/SP),
- command-line arguments used,
- and the resulting metrics (e.g., Recall@5, NDCG@5, MRR).

