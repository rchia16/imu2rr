#!/usr/bin/env python3
"""Extract HR and short-window HRV features from processed ECG pickles."""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, iirnotch

from config import ECG_FS


def _subject_from_path(path: Path) -> str:
    return path.stem


def _resolve_pickles(data_dir: Path, subjects: Optional[Iterable[str]]) -> List[Path]:
    if subjects:
        return [data_dir / f"{str(sbj)}.pkl" for sbj in subjects]
    return sorted(data_dir.glob("S*.pkl"))


def preprocess_ecg(x: np.ndarray, fs: float) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    finite = np.isfinite(x)
    if x.size == 0 or finite.mean() < 0.5:
        return np.asarray([], dtype=float)
    if not finite.all():
        idx = np.arange(x.size)
        x = np.interp(idx, idx[finite], x[finite])
    x = x - np.nanmedian(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd <= 1e-12:
        return np.asarray([], dtype=float)
    x = x / sd

    nyq = 0.5 * float(fs)
    if fs > 110:
        b_notch, a_notch = iirnotch(w0=50.0, Q=30.0, fs=fs)
        x = filtfilt(b_notch, a_notch, x)
    high = min(20.0, 0.95 * nyq)
    low = min(5.0, high * 0.5)
    if low <= 0 or high <= low:
        return x
    b, a = butter(3, [low / nyq, high / nyq], btype="bandpass")
    return filtfilt(b, a, x)


def ecg_window_features(
    x: np.ndarray,
    *,
    fs: float,
    min_hr_bpm: float,
    max_hr_bpm: float,
    min_valid_fraction: float,
) -> dict:
    if x.size < max(8, int(fs * 5)) or float(np.isfinite(x).mean()) < min_valid_fraction:
        return _nan_features(0)
    y = preprocess_ecg(x, fs)
    if y.size < max(8, int(fs * 5)):
        return _nan_features(0)

    min_distance = max(1, int(round(fs * 60.0 / max_hr_bpm)))
    prominence = max(0.25, 0.35 * float(np.nanstd(y)))
    peaks, _ = find_peaks(y, distance=min_distance, prominence=prominence)
    polarity = 1
    if peaks.size < 3:
        peaks, _ = find_peaks(-y, distance=min_distance, prominence=prominence)
        polarity = -1
    if peaks.size < 3:
        return _nan_features(int(peaks.size), polarity=polarity)

    rri_ms = np.diff(peaks) / float(fs) * 1000.0
    min_rri = 60000.0 / max_hr_bpm
    max_rri = 60000.0 / min_hr_bpm
    rri_ms = rri_ms[(rri_ms >= min_rri) & (rri_ms <= max_rri)]
    if rri_ms.size < 2:
        return _nan_features(int(peaks.size), polarity=polarity)

    diff_ms = np.diff(rri_ms)
    med_rri = float(np.nanmedian(rri_ms))
    return {
        "rri_ms": med_rri,
        "hr_bpm": float(60000.0 / med_rri),
        "sdnn_ms": float(np.nanstd(rri_ms, ddof=1)) if rri_ms.size > 1 else float("nan"),
        "rmssd_ms": float(np.sqrt(np.nanmean(diff_ms * diff_ms))) if diff_ms.size else float("nan"),
        "pnn50": float(np.mean(np.abs(diff_ms) > 50.0)) if diff_ms.size else float("nan"),
        "n_r_peaks": int(peaks.size),
        "n_rri": int(rri_ms.size),
        "r_peak_polarity": int(polarity),
    }


def _nan_features(n_peaks: int, polarity: int = 0) -> dict:
    return {
        "rri_ms": float("nan"),
        "hr_bpm": float("nan"),
        "sdnn_ms": float("nan"),
        "rmssd_ms": float("nan"),
        "pnn50": float("nan"),
        "n_r_peaks": int(n_peaks),
        "n_rri": 0,
        "r_peak_polarity": int(polarity),
    }


def extract_pickle(path: Path, args) -> List[dict]:
    with path.open("rb") as f:
        data = pickle.load(f)
    if "ecg_filt" not in data:
        raise KeyError(f"{path} does not contain 'ecg_filt'")
    ecg = np.asarray(data["ecg_filt"])
    conds = np.asarray(data.get("conds", [""] * ecg.shape[0]))
    subject = str(data.get("subject", _subject_from_path(path)))
    rows = []
    for idx, win in enumerate(ecg):
        feat = ecg_window_features(
            np.asarray(win),
            fs=float(args.fs),
            min_hr_bpm=float(args.min_hr_bpm),
            max_hr_bpm=float(args.max_hr_bpm),
            min_valid_fraction=float(args.min_valid_fraction),
        )
        rows.append(
            {
                "subject": subject,
                "window_idx": int(idx),
                "condition": str(conds[idx]) if idx < len(conds) else "",
                **feat,
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract ECG-derived HR and short-window HRV from processed pickles.")
    p.add_argument("--data-dir", default="/projects/BLVMob/imu-rr-seated/Data/levels")
    p.add_argument("--subjects", nargs="*", default=None)
    p.add_argument("--out-csv", default="ecg_hrv_features.csv")
    p.add_argument("--fs", type=float, default=float(ECG_FS))
    p.add_argument("--min-hr-bpm", type=float, default=35.0)
    p.add_argument("--max-hr-bpm", type=float, default=220.0)
    p.add_argument("--min-valid-fraction", type=float, default=0.8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    paths = _resolve_pickles(Path(args.data_dir), args.subjects)
    rows = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        rows.extend(extract_pickle(path, args))
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "subject",
        "window_idx",
        "condition",
        "rri_ms",
        "hr_bpm",
        "sdnn_ms",
        "rmssd_ms",
        "pnn50",
        "n_r_peaks",
        "n_rri",
        "r_peak_polarity",
    ]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    valid_hr = int(np.isfinite([row["hr_bpm"] for row in rows]).sum()) if rows else 0
    print(f"[OK] wrote {out} rows={len(rows)} valid_hr={valid_hr}")


if __name__ == "__main__":
    main()
