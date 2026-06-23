#!/usr/bin/env python3
"""Unique-name wrapper for the exact adaptation/prototype-gate sweep."""
from __future__ import annotations
try:
    from vit_pressure_crossmodal_stft_rr_adaptation_alpha_hat_sweep import main
except ImportError:
    from vit_pressure_crossmodal_stft_rr_unsup_ladder_config_sweep_adaptation import main
if __name__ == "__main__":
    main()
