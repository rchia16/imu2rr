#!/usr/bin/env python
"""Minimal LDA-on-embeddings test for one held-out subject and one checkpoint."""

import argparse
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report, f1_score

from config import SBJ_PROCESSED_DIR
from imu_mwl_classify import (
    TASK_CHOICES,
    _build_loso_loaders_l_to_l,
    _collect_embeddings,
    _default_subjects,
    _infer_task_from_group,
)
from utils import RunInfo, extract_run_timestamp, infer_backbone, infer_method, load_yaml


def main():
    p = argparse.ArgumentParser(description="Train LDA on frozen embeddings for one LOSO subject.")
    p.add_argument("--subject", required=True, help="Held-out subject, e.g. S12")
    p.add_argument("--run-yaml", required=True, help="Path to args/run YAML for model config")
    p.add_argument("--ckpt", required=True, help="Exact checkpoint path to load")
    p.add_argument("--backbone", default="", help="Override backbone if not inferable")
    p.add_argument("--data-str", default="", help="Override data_str; defaults to YAML or imu_filt")
    p.add_argument("--train-data-group", default="levels", choices=["mr", "level", "levels"])
    p.add_argument("--test-data-group", default="levels", choices=["mr", "level", "levels"])
    p.add_argument("--downstream-task", default="", choices=["", *TASK_CHOICES])
    p.add_argument("--include-levels-in-train", action="store_true")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    run_yaml = Path(args.run_yaml)
    ckpt = Path(args.ckpt)
    cfg = load_yaml(run_yaml) or {}
    data_str = args.data_str or str(cfg.get("data_str", "imu_filt"))
    backbone = args.backbone or infer_backbone(str(cfg.get("backbone", "")), run_yaml.parent)
    method = infer_method(str(cfg.get("method", "")), run_yaml.parent)
    downstream_task = args.downstream_task or _infer_task_from_group(args.train_data_group)

    run = RunInfo(
        run_dir=run_yaml.parent,
        run_yaml=run_yaml,
        run_ts=extract_run_timestamp(run_yaml),
        cfg=cfg,
        method=method,
        backbone=backbone,
        subject=args.subject,
    )

    train_loader, test_loader = _build_loso_loaders_l_to_l(
        args.subject,
        _default_subjects(),
        data_str=data_str,
        batch_size=args.batch_size,
        data_dir=SBJ_PROCESSED_DIR,
        train_data_group=args.train_data_group,
        test_data_group=args.test_data_group,
        task=downstream_task,
        include_levels_in_train=args.include_levels_in_train,
    )

    X_train, y_train = _collect_embeddings(run, train_loader, args.device, ckpt_path_override=ckpt)
    X_test, y_test = _collect_embeddings(run, test_loader, args.device, ckpt_path_override=ckpt)

    clf = LinearDiscriminantAnalysis()
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    print(f"subject={args.subject}")
    print(f"backbone={backbone}")
    print(f"ckpt={ckpt}")
    print(f"train_shape={X_train.shape} test_shape={X_test.shape}")
    print(f"train_labels={np.unique(y_train, return_counts=True)}")
    print(f"test_labels={np.unique(y_test, return_counts=True)}")
    print(f"acc={accuracy_score(y_test, pred):.4f}")
    print(f"f1_macro={f1_score(y_test, pred, average='macro'):.4f}")
    print(classification_report(y_test, pred, digits=4))


if __name__ == "__main__":
    main()
