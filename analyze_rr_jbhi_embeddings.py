#!/usr/bin/env python3
"""Embedding-space and adaptation diagnostics for JBHI RR baseline runs.

Produces:
  - t-SNE plots of learned embeddings by subject/condition/error
  - subject prototype distance table
  - alpha075 adaptation shift diagnostics

NOTE: This is analysis-only. It uses labels to colour errors and evaluate shifts,
not to compute deployable no-label adaptation inputs.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def load_npz_files(root: Path, pattern: str) -> pd.DataFrame:
    rows = []
    for path in sorted(root.glob(pattern)):
        z = np.load(path, allow_pickle=True)
        emb = z["emb"]
        subject = str(z["subject"]) if "subject" in z else path.parent.name
        mode = str(z["mode"]) if "mode" in z else "none"
        rr_true = z["rr_true"] if "rr_true" in z else np.full(emb.shape[0], np.nan)
        rr_pred = z["rr_pred"] if "rr_pred" in z else np.full(emb.shape[0], np.nan)
        for i in range(emb.shape[0]):
            rows.append({
                "path": str(path), "subject": subject, "mode": mode,
                "rr_true": float(rr_true[i]), "rr_pred": float(rr_pred[i]),
                "abs_error": float(abs(rr_pred[i] - rr_true[i])),
                "emb": emb[i].astype(np.float32),
            })
    if not rows:
        raise SystemExit(f"No embedding files found under {root} with pattern {pattern}")
    return pd.DataFrame(rows)


def embed_matrix(df: pd.DataFrame) -> np.ndarray:
    return np.stack(df["emb"].to_numpy(), axis=0).astype(np.float32)


def run_tsne(x: np.ndarray, seed: int, max_points: int) -> Tuple[np.ndarray, np.ndarray]:
    n = x.shape[0]
    idx = np.arange(n)
    if n > max_points:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(idx, size=max_points, replace=False))
        x = x[idx]
    # PCA pre-reduction improves stability and speed.
    n_comp = min(30, x.shape[1], x.shape[0] - 1)
    xp = PCA(n_components=n_comp, random_state=seed).fit_transform(x) if n_comp >= 2 else x
    perplexity = max(5, min(30, (xp.shape[0] - 1) // 3))
    y = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=seed).fit_transform(xp)
    return idx, y


def plot_tsne(df: pd.DataFrame, coords: np.ndarray, idx: np.ndarray, out_dir: Path, title: str):
    sub = df.iloc[idx].reset_index(drop=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    def scatter_colour(values, name, cmap="viridis"):
        plt.figure(figsize=(7.0, 5.8))
        sc = plt.scatter(coords[:, 0], coords[:, 1], c=values, s=9, cmap=cmap, alpha=0.75)
        plt.title(title + f" by {name}")
        plt.xticks([]); plt.yticks([])
        plt.colorbar(sc, label=name)
        plt.tight_layout()
        plt.savefig(out_dir / f"tsne_{name}.png", dpi=220)
        plt.close()

    # Subject categorical plot.
    subjects = sorted(sub["subject"].unique().tolist())
    sid = {s: i for i, s in enumerate(subjects)}
    scatter_colour(sub["subject"].map(sid).to_numpy(), "subject_index", cmap="tab20")
    scatter_colour(sub["abs_error"].to_numpy(), "abs_error", cmap="magma")
    scatter_colour(sub["rr_true"].to_numpy(), "rr_true", cmap="viridis")
    sub[["subject", "mode", "rr_true", "rr_pred", "abs_error"]].assign(tsne_x=coords[:,0], tsne_y=coords[:,1]).to_csv(out_dir / "tsne_points.csv", index=False)


def subject_prototypes(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (subject, mode), sub in df.groupby(["subject", "mode"]):
        e = embed_matrix(sub)
        rows.append({
            "subject": subject, "mode": mode,
            "n": int(e.shape[0]),
            "mae": float(sub["abs_error"].mean()),
            "rr_true_mean": float(sub["rr_true"].mean()),
            "emb_norm_mean": float(np.linalg.norm(e, axis=1).mean()),
            "prototype": e.mean(axis=0),
        })
    proto = pd.DataFrame(rows)
    all_rows = []
    for mode, sub in proto.groupby("mode"):
        mat = np.stack(sub["prototype"].to_numpy(), axis=0)
        subjects = sub["subject"].to_numpy()
        d = np.sqrt(((mat[:, None, :] - mat[None, :, :]) ** 2).mean(axis=2))
        for i, s in enumerate(subjects):
            order = np.argsort(d[i])
            nearest = [j for j in order if j != i][0] if len(order) > 1 else i
            all_rows.append({
                "subject": s, "mode": mode,
                "mae": float(sub.iloc[i]["mae"]),
                "prototype_norm": float(np.linalg.norm(mat[i])),
                "nearest_subject": str(subjects[nearest]),
                "nearest_dist": float(d[i, nearest]) if nearest != i else np.nan,
                "mean_dist_to_others": float(np.mean([d[i, j] for j in range(len(subjects)) if j != i])) if len(subjects) > 1 else np.nan,
            })
    return pd.DataFrame(all_rows)


def adaptation_shift_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base = df[df["mode"] == "none"].groupby("subject")
    adapt = df[df["mode"].isin(["alpha075", "adapt_mean_alpha_075"])] .groupby("subject")
    for subject in sorted(set(base.groups.keys()) & set(adapt.groups.keys())):
        b = base.get_group(subject)
        a = adapt.get_group(subject)
        n = min(len(b), len(a))
        if n <= 0:
            continue
        eb = embed_matrix(b.iloc[:n])
        ea = embed_matrix(a.iloc[:n])
        shift = ea - eb
        base_err = np.abs(b["rr_pred"].iloc[:n].to_numpy() - b["rr_true"].iloc[:n].to_numpy())
        adapt_err = np.abs(a["rr_pred"].iloc[:n].to_numpy() - a["rr_true"].iloc[:n].to_numpy())
        rows.append({
            "subject": subject,
            "base_mae": float(base_err.mean()),
            "adapt_mae": float(adapt_err.mean()),
            "delta_mae": float(adapt_err.mean() - base_err.mean()),
            "shift_norm_mean": float(np.linalg.norm(shift, axis=1).mean()),
            "shift_norm_p95": float(np.percentile(np.linalg.norm(shift, axis=1), 95)),
            "improved_frac": float((adapt_err < base_err).mean()),
        })
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="Baseline suite output root or one model output directory")
    p.add_argument("--pattern", default="**/embeddings_*_none.npz")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-points", type=int, default=5000)
    args = p.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else root / "embedding_diagnostics"
    df = load_npz_files(root, args.pattern)
    x = embed_matrix(df)
    idx, y = run_tsne(x, args.seed, args.max_points)
    plot_tsne(df, y, idx, out_dir, title=root.name)
    proto = subject_prototypes(df)
    proto.to_csv(out_dir / "subject_prototypes.csv", index=False)
    shift = adaptation_shift_table(df)
    if not shift.empty:
        shift.to_csv(out_dir / "adaptation_shift_diagnostics.csv", index=False)
    print(f"[DONE] wrote diagnostics to {out_dir}")


if __name__ == "__main__":
    main()
