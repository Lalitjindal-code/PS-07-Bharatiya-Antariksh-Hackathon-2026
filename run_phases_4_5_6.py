"""
run_phases_4_5_6.py
====================
Convenience script that runs Phases 4-6 end-to-end on Kepler-10 (KIC 11904151)
using cached data, prints all results, and saves all plots.

This is a diagnostic script — not part of the pipeline proper.
The proper orchestrator is pipeline.py (Phase 11).

Run with:
    python run_phases_4_5_6.py
"""

from __future__ import annotations
import logging
import numpy as np

from data_loader import download_lightcurve, preprocess
from detrend import run_detrending
from identify import (
    build_period_grid, build_duration_grid,
    run_bls, bin_lc_for_bls, phase_fold,
    plot_periodogram, plot_phase_fold,
    compare_to_known, KNOWN_PARAMS,
)
from characterize import run_characterization

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

TARGET = "KIC 11904151"
MISSION = "Kepler"
BIN_CADENCE_MIN = 30.0

print("\n" + "=" * 65)
print("  END-TO-END: PHASES 4-6 on Kepler-10b (KIC 11904151)")
print("=" * 65 + "\n")

# ---- Phase 2-3: Data ----
print("[Phase 2-3] Loading and preprocessing ...")
lc = download_lightcurve(TARGET, mission=MISSION)
time, flux, flux_err = preprocess(lc)
cadence_min = float(np.median(np.diff(time)) * 24 * 60)
baseline = float(time[-1] - time[0])
print(f"  {len(time):,} clean cadences | {baseline:.1f}-day baseline | {cadence_min:.2f}-min cadence\n")

# ---- Phase 4: Detrending ----
print("[Phase 4] Detrending (Savitzky-Golay, window=3×max_duration) ...")
_, detrended, window_pts = run_detrending(
    time=time, flux=flux,
    period_max_days=baseline / 3.0,
    cadence_minutes=cadence_min,
    method="savgol",
    target_id=TARGET,
    save_plot=True,
)
print(f"  Window: {window_pts} pts | Detrended std: {detrended.std():.4e}\n")

# ---- Phase 5: Period Search ----
print("[Phase 5] BLS period search (binned to 30-min cadence) ...")
time_bls, det_bls, err_bls = bin_lc_for_bls(
    time, detrended, flux_err, target_cadence_min=BIN_CADENCE_MIN
)
print(f"  BLS input: {len(time_bls):,} binned points")

period_grid = build_period_grid(baseline)
duration_grid = build_duration_grid()
bls_result, best_signal = run_bls(time_bls, det_bls, err_bls, period_grid, duration_grid)

# Phase-fold
phase, flux_folded = phase_fold(time, detrended, best_signal["period"], best_signal["t0"])
best_signal["phase"] = phase
best_signal["flux_folded"] = flux_folded

# Plots
from pathlib import Path
PLOTS = Path("plots")
plot_periodogram(bls_result, best_signal["period"], TARGET,
                 save_path=PLOTS / "KIC_11904151_periodogram.png")
plot_phase_fold(phase, flux_folded, best_signal, TARGET,
                save_path=PLOTS / "KIC_11904151_phasefold.png")

comparison_bls = compare_to_known(best_signal, TARGET)

print("\n[Phase 5] BLS Results:")
print(f"  Period  : {best_signal['period']:.5f} d")
print(f"  t0      : {best_signal['t0']:.4f} d")
print(f"  Depth   : {best_signal['depth']*1e6:.1f} ppm")
print(f"  Duration: {best_signal['duration']*24:.3f} h")
print(f"  Power   : {best_signal['power']:.2f}")

# ---- Phase 6: Characterization ----
print("\n[Phase 6] batman + lmfit transit model fit ...")
fit_params, comparison_fit = run_characterization(
    time=time,
    flux=detrended,
    flux_err=flux_err,
    best_signal=best_signal,
    target_id=TARGET,
    save_plot=True,
)

print("\n[Phase 6] Fit Parameters (1-sigma from covariance):")
important_keys = [
    "depth_ppm_val", "depth_ppm_err",
    "duration_h_val",
    "rp_val", "rp_err",
    "a_rs_val", "a_rs_err",
    "inc_val", "inc_err",
    "u1_val", "u1_err",
    "u2_val", "u2_err",
    "baseline_val", "baseline_err",
    "redchi", "fit_ok",
]
for k in important_keys:
    if k in fit_params:
        v = fit_params[k]
        if isinstance(v, float):
            print(f"  {k:<22}: {v:.6g}")
        else:
            print(f"  {k:<22}: {v}")

print("\n" + "=" * 65)
print("  PHASES 4-6 COMPLETE. Plots in plots/")
print("=" * 65 + "\n")
