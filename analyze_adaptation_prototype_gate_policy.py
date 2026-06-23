#!/usr/bin/env python3
"""Strict no-label prototype/OOD safety gate for fixed-alpha RR adaptation.

Keeps alpha hard-coded (default alpha_075) and learns whether to apply it from
source-subject prototype/OOD diagnostics.  Labels are used only for pseudo-target
source gains and held-out evaluation, never as input features.
"""
from __future__ import annotations

import argparse, json, math
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

ALLOWED_PREFIXES = (
    "feature_adapt_", "feature_align_", "profile_stats_", "profile_vector_",
    "profile_gate_", "profile_delta_norm", "raw_profile_", "norm_profile_",
    "hybrid_", "unsup_", "readout_affine_", "rr_probe_n_", "eval_stft_confidence",
)
LEAKAGE_VERSION = "strict_v4_prototype_gate_no_labels_no_residuals_no_overshoot"
LEAKY = (
    "mae", "rmse", "corr", "residual", "direction_dot", "direction_agree",
    "reduces_abs_error", "delta_mae", "overshoot", "oracle", "uses_target",
    "label", "rr_true", "ground_truth", "y_true", "target_rr", "true_rr",
)
LEAKY_NS = ("eval_direction", "cal_direction", "profile_oracle")
DEFAULT_CANDIDATES = "none adapt_mean_alpha_050 adapt_mean_alpha_075 adapt_mean_alpha_100 profile_film_init_only profile_film_unsup_sparc direct_stft_rr hybrid_probe_stft_conf"


def toks(s: str) -> List[str]:
    return [x.strip() for x in str(s).replace(",", " ").split() if x.strip()]


def is_leaky(name: str) -> bool:
    n = str(name).lower()
    return any(k in n for k in LEAKY) or any(n.startswith(k) or k in n for k in LEAKY_NS)


def feature_cols(df: pd.DataFrame, min_frac: float = 0.05) -> Tuple[List[str], List[str], List[str]]:
    selected, excluded_leaky, excluded_sparse = [], [], []
    for c in sorted([c for c in df.columns if any(str(c).startswith(p) for p in ALLOWED_PREFIXES)]):
        if is_leaky(c):
            excluded_leaky.append(c); continue
        v = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if v.notna().mean() < min_frac or v.nunique(dropna=True) <= 1:
            excluded_sparse.append(c); continue
        selected.append(c)
    bad = [c for c in selected if is_leaky(c)]
    if bad:
        raise RuntimeError("Leaky selected features: " + ", ".join(bad))
    return selected, excluded_leaky, excluded_sparse


def prep(df: pd.DataFrame, candidates: Sequence[str]) -> pd.DataFrame:
    df = df[df["mode"].astype(str).isin(candidates)].copy()
    for c in ["rr_probe_pre_mae", "rr_probe_post_mae"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["rr_probe_pre_mae", "rr_probe_post_mae"])
    df["subject"] = df["subject"].astype(str); df["mode"] = df["mode"].astype(str)
    df["delta_mae"] = df["rr_probe_post_mae"] - df["rr_probe_pre_mae"]
    return df


def mode_summary(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("mode", as_index=False).agg(
        n_subjects=("subject", "nunique"),
        post_mae_mean=("rr_probe_post_mae", "mean"),
        delta_mae_mean=("delta_mae", "mean"),
        subjects_improved=("delta_mae", lambda x: int((x < 0).sum())),
        subjects_worse=("delta_mae", lambda x: int((x > 0).sum())),
    ).sort_values("post_mae_mean")


def oracle(df: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for s, sub in df.groupby("subject"):
        best = sub.sort_values("rr_probe_post_mae").iloc[0]
        none = sub[sub.mode == "none"].iloc[0]
        rows.append({"subject": s, "oracle_mode": best.mode, "oracle_post_mae": best.rr_probe_post_mae,
                     "none_post_mae": none.rr_probe_post_mae,
                     "oracle_delta_vs_none": best.rr_probe_post_mae - none.rr_probe_post_mae})
    return pd.DataFrame(rows)


def standardize(train: pd.DataFrame, test: pd.DataFrame, cols: Sequence[str]):
    xtr = pd.DataFrame({c: pd.to_numeric(train[c], errors="coerce") for c in cols}).replace([np.inf,-np.inf],np.nan)
    med = xtr.median().fillna(0.0)
    xtr = xtr.fillna(med).to_numpy(float)
    xte = pd.DataFrame({c: pd.to_numeric(test[c], errors="coerce") for c in cols}).replace([np.inf,-np.inf],np.nan).fillna(med).to_numpy(float)
    mu, sd = xtr.mean(0), xtr.std(0)+1e-6
    return (xtr-mu)/sd, (xte-mu)/sd


def distances(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(((x[:,None,:]-y[None,:,:])**2).mean(2), 0.0))


def ridge_fit(x, y, alpha=10.0):
    x = np.nan_to_num(np.asarray(x, float)); y = np.nan_to_num(np.asarray(y, float))
    mu, sd = x.mean(0), x.std(0)+1e-6
    xs = (x-mu)/sd
    X = np.c_[np.ones(len(xs)), xs]
    reg = np.eye(X.shape[1])*alpha; reg[0,0]=0
    coef = np.linalg.solve(X.T@X+reg, X.T@y)
    return coef, mu, sd


def ridge_pred(model, x):
    coef, mu, sd = model
    x = np.nan_to_num(np.asarray(x, float)); xs = (x-mu)/sd
    return np.c_[np.ones(len(xs)), xs] @ coef


def proto_features(xtr, xte, gains, safe, train_subjects, k=3, q=0.95):
    d = distances(xte, xtr).ravel(); order = np.argsort(d); k=min(k,len(order)); nn=order[:k]
    leave = distances(xtr, xtr); np.fill_diagonal(leave, np.inf)
    th = np.quantile(np.min(leave,1), q) if len(order)>1 else np.inf
    temp = max(np.median(d[np.isfinite(d)]), 1e-6)
    w = np.exp(-d/temp); w = w/max(w.sum(),1e-12)
    safe_x = xtr[safe]; harm_x = xtr[~safe]
    safe_dist = float(distances(xte, safe_x.mean(0,keepdims=True))[0,0]) if len(safe_x) else float("nan")
    harm_dist = float(distances(xte, harm_x.mean(0,keepdims=True))[0,0]) if len(harm_x) else float("nan")
    ent = float(-(w*np.log(w+1e-12)).sum()/math.log(len(w))) if len(w)>1 else 0.0
    return {
        "nearest_subject": train_subjects[int(order[0])], "nearest_dist": float(d[order[0]]),
        "knn_safe_frac": float(safe[nn].mean()), "soft_safe_prob": float((w*safe.astype(float)).sum()),
        "soft_gain": float((w*gains).sum()), "entropy": ent, "safe_dist": safe_dist,
        "harm_dist": harm_dist, "ood_threshold": float(th), "ood_reject": int(d[order[0]]>th),
        "vec": np.array([[d[order[0]], d[order[1]] if len(order)>1 else d[order[0]], float(safe[nn].mean()), float((w*safe.astype(float)).sum()), float((w*gains).sum()), ent, safe_dist if np.isfinite(safe_dist) else d[order[0]], harm_dist if np.isfinite(harm_dist) else d[order[0]], float(d[order[0]]>th)]])
    }


def evaluate_gate(df: pd.DataFrame, cols: Sequence[str], alpha_mode: str, min_gain: float, safe_threshold: float, reject_ood: bool, ridge_alpha: float, k: int, q: float):
    rows=[]; subjects=sorted(df.subject.unique())
    for held in subjects:
        train_sub=[s for s in subjects if s!=held]
        train_rows=[]; gains=[]; safe=[]
        for s in train_sub:
            r=df[(df.subject==s)&(df.mode==alpha_mode)]
            n=df[(df.subject==s)&(df.mode=="none")]
            if r.empty or n.empty: continue
            train_rows.append(r.iloc[0]); g=float(n.rr_probe_post_mae.iloc[0]-r.rr_probe_post_mae.iloc[0]); gains.append(g); safe.append(g>=min_gain)
        test=df[(df.subject==held)&(df.mode==alpha_mode)]
        none=df[(df.subject==held)&(df.mode=="none")]
        if test.empty or none.empty or len(train_rows)<2: continue
        train_df=pd.DataFrame(train_rows); test_df=pd.DataFrame([test.iloc[0]])
        xtr,xte=standardize(train_df,test_df,cols)
        gains=np.asarray(gains,float); safe=np.asarray(safe,bool)
        pf=proto_features(xtr,xte,gains,safe,train_sub,k,q)
        # Ridge on raw features + leave-one-source prototype descriptors
        proto_train=[]
        for i in range(len(train_sub)):
            mask=np.ones(len(train_sub),bool); mask[i]=False
            pfi=proto_features(xtr[mask], xtr[i:i+1], gains[mask], safe[mask], [s for j,s in enumerate(train_sub) if mask[j]], k, q)
            proto_train.append(pfi["vec"].ravel())
        xtr_aug=np.c_[xtr, np.vstack(proto_train)]; xte_aug=np.c_[xte, pf["vec"]]
        pred_gain=float(ridge_pred(ridge_fit(xtr_aug,gains,ridge_alpha),xte_aug)[0])
        proto_pass = (pf["soft_safe_prob"] >= safe_threshold) and (pf["soft_gain"] >= min_gain)
        ridge_pass = pred_gain >= min_gain
        if reject_ood:
            proto_pass = proto_pass and not bool(pf["ood_reject"])
            ridge_pass = ridge_pass and not bool(pf["ood_reject"])
        none_mae=float(none.rr_probe_post_mae.iloc[0]); alpha_mae=float(test.rr_probe_post_mae.iloc[0]); actual_gain=none_mae-alpha_mae
        for name, passed, score in [("prototype_gate_"+alpha_mode, proto_pass, pf["soft_gain"]), ("prototype_ridge_gate_"+alpha_mode, ridge_pass, pred_gain)]:
            rows.append({"subject":held,"policy":name,"alpha_mode":alpha_mode,"selected_mode":alpha_mode if passed else "none",
                         "selected_post_mae":alpha_mae if passed else none_mae,"none_post_mae":none_mae,"alpha_post_mae":alpha_mae,
                         "selected_delta_vs_none":(alpha_mae if passed else none_mae)-none_mae,
                         "actual_alpha_gain_vs_none":actual_gain,"actual_alpha_safe":int(actual_gain>=min_gain),"gate_pass":int(passed),
                         "gate_score":score,"pred_gain_ridge":pred_gain,"proto_soft_gain":pf["soft_gain"],"proto_soft_safe_prob":pf["soft_safe_prob"],
                         "proto_nearest_subject":pf["nearest_subject"],"proto_nearest_dist":pf["nearest_dist"],"proto_entropy":pf["entropy"],"proto_ood_reject":pf["ood_reject"]})
    return rows


def summarize_policy(rows: pd.DataFrame, modes: pd.DataFrame) -> pd.DataFrame:
    out=[]
    for p, sub in rows.groupby("policy"):
        out.append({"policy":p,"n_subjects":sub.subject.nunique(),"post_mae_mean":sub.selected_post_mae.mean(),"delta_vs_none_mean":sub.selected_delta_vs_none.mean(),"subjects_improved_vs_none":int((sub.selected_delta_vs_none<0).sum()),"subjects_worse_vs_none":int((sub.selected_delta_vs_none>0).sum()),"gate_pass_rate":sub.gate_pass.mean()})
    for _, r in modes.iterrows():
        out.append({"policy":r["mode"],"n_subjects":r["n_subjects"],"post_mae_mean":r["post_mae_mean"],"delta_vs_none_mean":r["delta_mae_mean"],"subjects_improved_vs_none":r["subjects_improved"],"subjects_worse_vs_none":r["subjects_worse"],"gate_pass_rate":np.nan})
    return pd.DataFrame(out).sort_values("post_mae_mean")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--subject-rows",required=True,type=Path); ap.add_argument("--out-dir",required=True,type=Path)
    ap.add_argument("--candidates",default=DEFAULT_CANDIDATES); ap.add_argument("--alpha-modes",default="adapt_mean_alpha_075")
    ap.add_argument("--min-gain",type=float,default=0.02); ap.add_argument("--safe-threshold",type=float,default=0.55)
    ap.add_argument("--knn-k",type=int,default=3); ap.add_argument("--ood-quantile",type=float,default=0.95); ap.add_argument("--ridge-alpha",type=float,default=10.0)
    ap.add_argument("--reject-ood",action="store_true",default=True); ap.add_argument("--no-reject-ood",dest="reject_ood",action="store_false")
    args=ap.parse_args()
    cand=sorted(set(toks(args.candidates)+toks(args.alpha_modes)+["none"]))
    df=prep(pd.read_csv(args.subject_rows), cand)
    cols, excl_leaky, excl_sparse=feature_cols(df)
    if not cols: raise SystemExit("No strict no-label features survived filtering")
    modes=mode_summary(df[df.mode.isin(toks(args.candidates))]); orc=oracle(df[df.mode.isin(toks(args.candidates))])
    rows=[]
    for a in toks(args.alpha_modes):
        rows.extend(evaluate_gate(df,cols,a,args.min_gain,args.safe_threshold,args.reject_ood,args.ridge_alpha,args.knn_k,args.ood_quantile))
    by=pd.DataFrame(rows); summ=summarize_policy(by,modes)
    args.out_dir.mkdir(parents=True,exist_ok=True)
    modes.to_csv(args.out_dir/"adaptation_mode_summary.csv",index=False); orc.to_csv(args.out_dir/"adaptation_oracle_by_subject.csv",index=False)
    by.to_csv(args.out_dir/"adaptation_prototype_gate_by_subject.csv",index=False); summ.to_csv(args.out_dir/"adaptation_prototype_gate_summary.csv",index=False)
    json.dump({"features":cols,"excluded_leaky_features":excl_leaky,"excluded_non_numeric_or_sparse_features":excl_sparse,"leakage_filter_version":LEAKAGE_VERSION,"alpha_modes":toks(args.alpha_modes),"strict_no_label":True}, open(args.out_dir/"adaptation_prototype_gate_features.json","w"), indent=2)
    with pd.option_context("display.max_columns",80,"display.width",220):
        print("\n=== Prototype gate summary ==="); print(summ)
        print(f"[INFO] used {len(cols)} features; excluded {len(excl_leaky)} leaky diagnostics")

if __name__=="__main__": main()
