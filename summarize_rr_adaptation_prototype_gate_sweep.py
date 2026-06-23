#!/usr/bin/env python3
"""Unique-name wrapper for summarising adaptation/prototype-gate sweeps."""
from __future__ import annotations
try:
    from summarize_rr_adaptation_alpha_hat_sweep import main
except ImportError:
    from summarize_rr_unsup_ladder_config_sweep_adaptation import main
if __name__ == "__main__":
    main()
