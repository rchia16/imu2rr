#!/usr/bin/env python3
"""Post-process MWL prediction rows into non-overlapping one-minute chunks."""
from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from aggregate_mwl_rest_low_high_benchmark import discover_prediction_files, load_all_predictions, validate_predictions
from mwl_rest_low_high import mapping_manifest, parse_list


SAMPLE_RE = re.compile(r"^(?P<subject>[^:]+):(?P<group>[^:]+):(?P<index>\d+)$")


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def parse_sample_id(value: object) -> Tuple[str, int]:
    text = str(value)
    match = SAMPLE_RE.match(text)
    if match:
        return str(match.group("group")), int(match.group("index"))
    tail = text.rsplit(":", 1)[-1]
    try:
        return "unknown", int(tail)
    except ValueError:
        return "unknown", 0


def majority_label(values: np.ndarray) -> Tuple[int, int]:
    labels = np.asarray(values, dtype=int).reshape(-1)
    counts = np.bincount(labels, minlength=3)
    tied = np.flatnonzero(counts == counts.max())
    if tied.size > 1:
        return int(labels[labels.size // 2]), 1
    return int(tied[0]), 0


def chunk_predictions(
    preds: pd.DataFrame,
    *,
    seconds: float,
    window_shift_seconds: float,
    windows_per_chunk: Optional[int],
    drop_incomplete_chunks: bool,
) -> pd.DataFrame:
    if windows_per_chunk is None:
        windows_per_chunk = max(1, int(round(float(seconds) / max(1e-6, float(window_shift_seconds)))))
    if int(windows_per_chunk) <= 0:
        raise ValueError("--windows-per-chunk must be positive.")
    rows: List[Dict[str, object]] = []
    df = validate_predictions(preds)
    df = df.copy()
    parsed = df["sample_id"].map(parse_sample_id)
    if "group" in df.columns:
        df["chunk_group"] = df["group"].fillna("").astype(str)
        missing_group = df["chunk_group"].isin({"", "nan", "None"})
        df.loc[missing_group, "chunk_group"] = [g for g, _idx in parsed[missing_group]]
    else:
        df["chunk_group"] = [g for g, _idx in parsed]
    if "raw_index" in df.columns:
        df["chunk_raw_index"] = pd.to_numeric(df["raw_index"], errors="coerce")
        missing_index = df["chunk_raw_index"].isna()
        df.loc[missing_index, "chunk_raw_index"] = [idx for _g, idx in parsed[missing_index]]
        df["chunk_raw_index"] = df["chunk_raw_index"].astype(int)
    else:
        df["chunk_raw_index"] = [idx for _g, idx in parsed]
    sort_cols = ["model_name", "subject", "seed", "chunk_group", "chunk_raw_index", "sample_id"]
    for (model, subject, seed, group), g in df.sort_values(sort_cols).groupby(
        ["model_name", "subject", "seed", "chunk_group"],
        dropna=False,
        sort=False,
    ):
        g = g.reset_index(drop=True)
        for chunk_id, st in enumerate(range(0, len(g), int(windows_per_chunk))):
            en = min(st + int(windows_per_chunk), len(g))
            if bool(drop_incomplete_chunks) and en - st < int(windows_per_chunk):
                continue
            sub = g.iloc[st:en]
            probs = sub[["prob_rest", "prob_low", "prob_high"]].to_numpy(dtype=float)
            p = probs.mean(axis=0)
            p = p / max(float(p.sum()), 1e-12)
            true_class, tied = majority_label(sub["true_class"].to_numpy(dtype=int))
            sample_id = f"{subject}:{group}:chunk60:{int(seed):03d}:{chunk_id:06d}"
            rows.append(
                {
                    "scope": "one_min",
                    "subject": str(subject),
                    "seed": int(seed),
                    "sample_id": sample_id,
                    "true_class": int(true_class),
                    "predicted_class": int(np.argmax(p)),
                    "prob_rest": float(p[0]),
                    "prob_low": float(p[1]),
                    "prob_high": float(p[2]),
                    "model_name": str(model),
                    "checkpoint_or_run_path": " | ".join(sorted(set(sub["checkpoint_or_run_path"].astype(str)))),
                    "raw_condition": "majority",
                    "chunk_group": str(group),
                    "chunk_id": int(chunk_id),
                    "start_sample_id": str(sub["sample_id"].iloc[0]),
                    "end_sample_id": str(sub["sample_id"].iloc[-1]),
                    "start_raw_index": int(sub["chunk_raw_index"].iloc[0]),
                    "end_raw_index": int(sub["chunk_raw_index"].iloc[-1]),
                    "n_source_windows": int(len(sub)),
                    "expected_windows_per_chunk": int(windows_per_chunk),
                    "incomplete_chunk": int(len(sub) < int(windows_per_chunk)),
                    "tie_true_label": int(tied),
                    "source_result_file": " | ".join(sorted(set(sub.get("source_result_file", pd.Series([""])).astype(str)))),
                }
            )
    if not rows:
        raise RuntimeError("No one-minute chunks were produced.")
    return pd.DataFrame(rows)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate MWL prediction rows into non-overlapping one-minute chunks.")
    parser.add_argument("--prediction-roots", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--subjects", default="")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--window-shift-seconds", type=float, default=20.0)
    parser.add_argument("--windows-per-chunk", type=int, default=0, help="Override seconds/window_shift; 0 uses the time-derived value.")
    parser.add_argument("--drop-incomplete-chunks", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    roots = [Path(p) for p in args.prediction_roots]
    files = discover_prediction_files(roots)
    preds, load_status = load_all_predictions(files)
    preds = validate_predictions(preds)
    if args.subjects:
        keep = set(parse_list(args.subjects))
        preds = preds[preds["subject"].astype(str).isin(keep)].copy()
    if args.seeds:
        keep_seeds = {int(v) for v in parse_list(args.seeds)}
        preds = preds[preds["seed"].astype(int).isin(keep_seeds)].copy()
    if preds.empty:
        raise RuntimeError("No prediction rows remain after subject/seed filtering.")
    windows_per_chunk = None if int(args.windows_per_chunk) <= 0 else int(args.windows_per_chunk)
    chunks = chunk_predictions(
        preds,
        seconds=float(args.seconds),
        window_shift_seconds=float(args.window_shift_seconds),
        windows_per_chunk=windows_per_chunk,
        drop_incomplete_chunks=bool(args.drop_incomplete_chunks),
    )
    chunks.to_csv(out_dir / "mwl_one_minute_prediction_rows.csv", index=False)
    manifest = {
        "mapping": mapping_manifest(),
        "prediction_roots": [str(p) for p in roots],
        "source_result_files": sorted(preds["source_result_file"].astype(str).unique()),
        "load_status": load_status,
        "subjects": sorted(chunks["subject"].astype(str).unique()),
        "seeds": sorted(int(v) for v in chunks["seed"].unique()),
        "seconds": float(args.seconds),
        "window_shift_seconds": float(args.window_shift_seconds),
        "windows_per_chunk": int(chunks["expected_windows_per_chunk"].iloc[0]),
        "drop_incomplete_chunks": bool(args.drop_incomplete_chunks),
        "true_label_rule": "majority vote within chunk; ties use center source window label and tie_true_label=1",
        "prediction_rule": "mean class probability within chunk, then argmax",
        "git_commit": git_commit(),
        "hostname": platform.node(),
        "python_version": sys.version,
    }
    (out_dir / "mwl_one_minute_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(f"[DONE] wrote {out_dir / 'mwl_one_minute_prediction_rows.csv'}", flush=True)


if __name__ == "__main__":
    main()
