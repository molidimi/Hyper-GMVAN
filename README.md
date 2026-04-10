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

