#!/usr/bin/env python
"""Train an efficient HMH molecular classifier on TDC ADMET datasets."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import Parallel, delayed
from rdkit import Chem
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from tdc.single_pred import ADME, Tox
from tqdm.auto import tqdm


TDC_TASKS = {
    "AMES": ("Tox", "AMES"),
    "hERG": ("Tox", "hERG"),
    "hERG_Karim": ("Tox", "hERG_Karim"),
    "DILI": ("Tox", "DILI"),
    "HIA": ("ADME", "HIA_Hou"),
    "BBB": ("ADME", "BBB_Martins"),
    "CYP2C9_inhibition": ("ADME", "CYP2C9_Veith"),
    "CYP2D6_inhibition": ("ADME", "CYP2D6_Veith"),
    "CYP3A4_inhibition": ("ADME", "CYP3A4_Veith"),
}

ATOM_TYPES = ["C", "N", "O", "S", "F", "P", "Cl", "Br", "I", "B", "Si", "Se", "other"]
BOND_TYPES = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=sorted(TDC_TASKS), default="AMES")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--data-dir", default="data/TDC")
    parser.add_argument("--cache-dir", default="data/hmh_tdc_cache")
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even when CUDA/MPS is available.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_tdc_task(task_key: str, data_dir: str) -> tuple[str, pd.DataFrame]:
    loader_name, dataset_name = TDC_TASKS[task_key]
    loader_cls = Tox if loader_name == "Tox" else ADME
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    dataset = loader_cls(name=dataset_name, path=data_dir)
    df = dataset.get_data().copy()
    df = df.dropna(subset=["Drug", "Y"]).reset_index(drop=True)
    df["Y"] = df["Y"].astype(float)
    return dataset_name, df


def one_hot(value, choices: list) -> list[float]:
    out = [0.0] * len(choices)
    try:
        idx = choices.index(value)
    except ValueError:
        idx = len(choices) - 1
    out[idx] = 1.0
    return out


def atom_features(atom: Chem.Atom) -> np.ndarray:
    symbol = atom.GetSymbol()
    symbol = symbol if symbol in ATOM_TYPES[:-1] else "other"
    feats = (
        one_hot(symbol, ATOM_TYPES)
        + [
            float(atom.GetDegree()),
            float(atom.GetFormalCharge()),
            float(atom.GetTotalNumHs()),
            float(atom.GetIsAromatic()),
            float(atom.IsInRing()),
        ]
    )
    return np.asarray(feats, dtype=np.float32)


def smiles_to_arrays(smiles: str) -> tuple[np.ndarray, np.ndarray] | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    x = np.stack([atom_features(atom) for atom in mol.GetAtoms()], axis=0)
    n = mol.GetNumAtoms()
    adj = np.zeros((n, n), dtype=np.float32)

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        adj[i, j] = 1.0
        adj[j, i] = 1.0

    return x, adj


def normalized_laplacian_from_adjacency(adj: np.ndarray) -> tuple[sp.csr_matrix, np.ndarray]:
    a = sp.csr_matrix(adj)
    degree = np.asarray(a.sum(axis=1)).ravel()
    degree_safe = np.maximum(degree, 1e-12)
    d_inv = sp.diags(1.0 / np.sqrt(degree_safe))
    lap = sp.eye(a.shape[0], format="csr") - d_inv @ a @ d_inv
    return lap, degree


def bottom_eigenvectors(l_aug: sp.csr_matrix, k_latent: int) -> np.ndarray:
    n = l_aug.shape[0]
    try:
        _, u = spla.eigsh(l_aug, k=k_latent, which="SM", maxiter=max(1000, 20 * n), tol=1e-6)
    except Exception:
        dense = l_aug.toarray()
        dense = 0.5 * (dense + dense.T)
        vals, vecs = np.linalg.eigh(dense)
        u = vecs[:, np.argsort(vals)[:k_latent]]
    return u


def hmh_vector_from_smiles(
    smiles: str,
    k_feat: int = 8,
    k_diff: int = 8,
    t: float = 0.6,
    lam: float = 0.08,
    alpha: float = 0.5,
    r_probe: int = 12,
    k_spec: int = 16,
) -> np.ndarray | None:
    arrays = smiles_to_arrays(smiles)
    if arrays is None:
        return None
    x, adj = arrays
    n = x.shape[0]
    if n == 1:
        return np.concatenate(
            [
                x.mean(axis=0),
                np.zeros(x.shape[1], dtype=np.float32),
                np.zeros(2 * k_spec, dtype=np.float32),
                np.zeros(2 * r_probe, dtype=np.float32),
            ]
        ).astype(np.float32)

    l_top, degree_top = normalized_laplacian_from_adjacency(adj)

    x_norm = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    sim = x_norm @ x_norm.T
    np.fill_diagonal(sim, -np.inf)
    k_local = min(k_feat, n - 1)
    feat_adj = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        nbrs = np.argpartition(-sim[i], kth=k_local - 1)[:k_local]
        vals = np.maximum(sim[i, nbrs], 0.0)
        feat_adj[i, nbrs] = vals
    feat_adj = np.maximum(feat_adj, feat_adj.T)
    l_feat, _ = normalized_laplacian_from_adjacency(feat_adj)

    l_mix = alpha * l_top + (1.0 - alpha) * l_feat
    heat = spla.expm_multiply((-t) * l_mix, np.eye(n))
    np.fill_diagonal(heat, 0.0)
    k_local = min(k_diff, n - 1)
    diff_adj = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        nbrs = np.argpartition(-heat[i], kth=k_local - 1)[:k_local]
        diff_adj[i, nbrs] = np.maximum(heat[i, nbrs], 0.0)
    diff_adj = np.maximum(diff_adj, diff_adj.T)
    l_diff, _ = normalized_laplacian_from_adjacency(diff_adj)

    feature_dist = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)
    np.fill_diagonal(feature_dist, 0.0)
    incompat = feature_dist / (feature_dist.max() + 1e-12)
    incompat = incompat * (1.0 - adj)
    l_incompat, _ = normalized_laplacian_from_adjacency(incompat)

    l_aug = (l_top + l_feat + l_diff + lam * l_incompat).tocsr()
    k_latent = min(k_spec, max(1, n - 1))
    u = bottom_eigenvectors(l_aug, k_latent)

    probes = []
    max_power = min(r_probe, n)
    signal = degree_top / (degree_top.max() + 1e-12)
    for power in range(1, max_power + 1):
        signal = l_aug @ signal
        probes.append(np.asarray(signal).reshape(-1))
    if probes:
        probe_stats = np.stack(probes, axis=1)
        probe_pool = np.concatenate([probe_stats.mean(axis=0), probe_stats.std(axis=0)])
    else:
        probe_pool = np.zeros(2, dtype=np.float32)
    if probe_pool.shape[0] < 2 * r_probe:
        probe_pool = np.pad(probe_pool, (0, 2 * r_probe - probe_pool.shape[0]))

    spectral_pool = np.concatenate([u.mean(axis=0), u.std(axis=0)])
    if spectral_pool.shape[0] < 2 * k_spec:
        spectral_pool = np.pad(spectral_pool, (0, 2 * k_spec - spectral_pool.shape[0]))

    graph_pool = np.concatenate([x.mean(axis=0), x.std(axis=0), spectral_pool, probe_pool])
    return graph_pool.astype(np.float32)


def cache_path_for(cache_dir: Path, task_key: str, max_rows: int | None, params: dict) -> Path:
    param_tag = "_".join(f"{key}-{value}" for key, value in params.items())
    row_tag = "all" if max_rows is None else str(max_rows)
    return cache_dir / f"{task_key}_{row_tag}_{param_tag}.pt"


def compute_or_load_features(
    df: pd.DataFrame,
    task_key: str,
    max_rows: int | None,
    cache_dir: Path,
    params: dict,
    n_jobs: int,
    force_recompute: bool,
) -> tuple[torch.Tensor, torch.Tensor, list[str], list[int]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_path_for(cache_dir, task_key, max_rows, params)
    if path.exists() and not force_recompute:
        print(f"Loading cached HMH features: {path}")
        obj = torch.load(path, map_location="cpu", weights_only=False)
        return obj["X"], obj["y"], obj["smiles"], obj["valid_index"]

    print("Computing HMH features. This is the expensive step and will be cached.")
    smiles = df["Drug"].astype(str).tolist()
    y_raw = df["Y"].astype(float).to_numpy()

    iterator = tqdm(smiles, desc=f"HMH {task_key}")
    features = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(hmh_vector_from_smiles)(smile, **params) for smile in iterator
    )

    xs, ys, valid_smiles, valid_index = [], [], [], []
    for idx, feat in enumerate(features):
        if feat is None:
            continue
        xs.append(feat)
        ys.append(y_raw[idx])
        valid_smiles.append(smiles[idx])
        valid_index.append(idx)

    if not xs:
        raise RuntimeError("No valid molecules were featurized.")

    x = torch.tensor(np.stack(xs), dtype=torch.float32)
    y = torch.tensor(np.asarray(ys, dtype=np.float32)).view(-1, 1)
    obj = {"X": x, "y": y, "smiles": valid_smiles, "valid_index": valid_index, "params": params}
    torch.save(obj, path)
    print(f"Saved cache: {path}")
    return x, y, valid_smiles, valid_index


class HMHGraphClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, dropout: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def metrics_from_logits(y_true: torch.Tensor, logits: torch.Tensor) -> dict[str, float]:
    yt = y_true.detach().cpu().view(-1).numpy()
    prob = torch.sigmoid(logits.detach().cpu()).view(-1).numpy()
    pred = (prob >= 0.5).astype(int)
    out = {
        "avg_precision": float(average_precision_score(yt, prob)),
        "accuracy@0.5": float(accuracy_score(yt, pred)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(yt, prob))
    except ValueError:
        out["roc_auc"] = float("nan")
    return out


def make_batches(indices: np.ndarray, batch_size: int, shuffle: bool, seed: int):
    idx = np.asarray(indices).copy()
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    for start in range(0, len(idx), batch_size):
        yield idx[start : start + batch_size]


def run_epoch(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    indices: np.ndarray,
    optimizer: torch.optim.Optimizer | None,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)
    logits_all, y_all = [], []
    total_loss = 0.0
    total_n = 0

    for batch_idx in make_batches(indices, batch_size=batch_size, shuffle=train, seed=seed):
        xb = x[batch_idx].to(device)
        yb = y[batch_idx].to(device)
        logits = model(xb)
        loss = F.binary_cross_entropy_with_logits(logits, yb)

        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        total_loss += loss.item() * len(batch_idx)
        total_n += len(batch_idx)
        logits_all.append(logits.detach().cpu())
        y_all.append(yb.detach().cpu())

    metrics = metrics_from_logits(torch.cat(y_all), torch.cat(logits_all))
    metrics["loss"] = total_loss / max(total_n, 1)
    return metrics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.cpu)
    print(f"Device: {device}")
    print(json.dumps(vars(args), indent=2))

    dataset_name, df = load_tdc_task(args.task, args.data_dir)
    if args.max_rows is not None and len(df) > args.max_rows:
        df = df.sample(n=args.max_rows, random_state=args.seed).reset_index(drop=True)
    print(f"Loaded {args.task} ({dataset_name}) with shape {df.shape}")
    print("Label counts:", df["Y"].value_counts().sort_index().to_dict())

    hmh_params = {
        "k_feat": 8,
        "k_diff": 8,
        "t": 0.6,
        "lam": 0.08,
        "alpha": 0.5,
        "r_probe": 12,
        "k_spec": 16,
    }
    x, y, smiles, valid_index = compute_or_load_features(
        df=df,
        task_key=args.task,
        max_rows=args.max_rows,
        cache_dir=Path(args.cache_dir),
        params=hmh_params,
        n_jobs=args.n_jobs,
        force_recompute=args.force_recompute,
    )
    print(f"Valid molecules: {len(smiles)} / {len(df)}")
    print(f"Feature matrix: {tuple(x.shape)}")

    indices = np.arange(len(y))
    y_np = y.view(-1).numpy()
    train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=args.seed, stratify=y_np)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=args.seed, stratify=y_np[temp_idx]
    )

    mu = x[train_idx].mean(dim=0, keepdim=True)
    sd = x[train_idx].std(dim=0, keepdim=True)
    sd[sd < 1e-8] = 1.0
    x_norm = (x - mu) / sd

    model = HMHGraphClassifier(x_norm.size(1), hidden=args.hidden, dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = -float("inf")
    best_state = None
    best_test = None
    wait = 0

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model, x_norm, y, train_idx, optimizer, args.batch_size, device, seed=args.seed + epoch
        )
        val_metrics = run_epoch(model, x_norm, y, val_idx, None, args.batch_size, device, seed=args.seed)
        test_metrics = run_epoch(model, x_norm, y, test_idx, None, args.batch_size, device, seed=args.seed)

        score = val_metrics["roc_auc"]
        if math.isfinite(score) and score > best_val:
            best_val = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_test = test_metrics.copy()
            wait = 0
        else:
            wait += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"train AUC {train_metrics['roc_auc']:.4f} | "
                f"val AUC {val_metrics['roc_auc']:.4f} | "
                f"test AUC {test_metrics['roc_auc']:.4f}"
            )

        if wait >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    output_dir = Path(args.output_dir) / args.task / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "hmh_params": hmh_params,
            "feature_mean": mu,
            "feature_std": sd,
            "splits": {
                "train_idx": train_idx.tolist(),
                "val_idx": val_idx.tolist(),
                "test_idx": test_idx.tolist(),
            },
            "best_test": best_test,
        },
        output_dir / "checkpoint.pt",
    )

    metrics = {
        "task": args.task,
        "tdc_dataset": dataset_name,
        "num_rows": int(len(df)),
        "num_valid_molecules": int(len(smiles)),
        "best_val_roc_auc": best_val,
        "best_test": best_test,
        "args": vars(args),
        "hmh_params": hmh_params,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("Best-by-validation test metrics:", best_test)
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
