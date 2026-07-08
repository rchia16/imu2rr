#!/usr/bin/env python3
"""Compatibility wrapper for the unsupervised RR adaptation ladder.

The older profile-TTT entrypoints import this module name directly. In this
checkout, the implementation lives in
``vit_pressure_crossmodal_stft_rr_adaptation_alpha_hat_sweep``. Re-exporting it
keeps those entrypoints working without copying an older ladder implementation
over the local one.
"""

from importlib import import_module

_impl = import_module("vit_pressure_crossmodal_stft_rr_adaptation_alpha_hat_sweep")

for _name in dir(_impl):
    if _name.startswith("__") and _name.endswith("__"):
        continue
    globals()[_name] = getattr(_impl, _name)

