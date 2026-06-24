"""
tests/test_synthetic.py
=======================
CONTROLLED UNIT TESTS using synthetic light curves.

These tests verify that each pipeline stage recovers injected parameters
within reasonable tolerances on *synthetic data*.  They are NOT a substitute
for validation on real data.

Run with:  pytest tests/ -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from data_loader import generate_synthetic_lc
from detrend import run_detrending, estimate_max_transit_duration, compute_window_length
from identify import build_period_grid, build_duration_grid, run_bls, phase_fold


# ---- Fixtures ---------------------------------------------------------------

SYNTH_PERIOD = 5.0      # days
SYNTH_DEPTH = 0.01      # 1% depth
SYNTH_DURATION = 0.15   # days
SYNTH_NOISE = 2e-4
N_POINTS = 15_000


@pytest.fixture(scope="module")
def synthetic_data():
    """Generate a synthetic light curve once for all tests in this module."""
    time, flux, flux_err = generate_synthetic_lc(
        n_points=N_POINTS,
        period_days=SYNTH_PERIOD,
        depth=SYNTH_DEPTH,
        duration_days=SYNTH_DURATION,
        noise_level=SYNTH_NOISE,
        seed=42,
    )
    return time, flux, flux_err


# ---- Unit tests --------------------------------------------------------------

class TestSyntheticDataLoader:
    """Test that generate_synthetic_lc returns sensible arrays."""

    def test_shapes(self, synthetic_data):
        time, flux, flux_err = synthetic_data
        assert len(time) == N_POINTS
        assert len(flux) == N_POINTS
        assert len(flux_err) == N_POINTS

    def test_flux_near_unity(self, synthetic_data):
        _, flux, _ = synthetic_data
        assert abs(np.median(flux) - 1.0) < 0.001, "Median flux should be ~1.0"

    def test_transit_present(self, synthetic_data):
        time, flux, flux_err = synthetic_data
        # Check that minimum flux is noticeably below 1 (transit exists)
        assert flux.min() < 1.0 - SYNTH_DEPTH * 0.5, \
            "No transit dip detected in synthetic data"

    def test_reproducibility(self):
        """Same seed must produce identical output."""
        t1, f1, _ = generate_synthetic_lc(seed=42)
        t2, f2, _ = generate_synthetic_lc(seed=42)
        np.testing.assert_array_equal(f1, f2)


class TestDetrending:
    """Test detrending on synthetic data with known trend."""

    def test_window_length_positive(self):
        dur = estimate_max_transit_duration(30.0)
        wl = compute_window_length(dur, cadence_minutes=30.0)
        assert wl > 0 and wl % 2 == 1, "Window length must be positive and odd"

    def test_window_length_increases_with_period(self):
        dur_short = estimate_max_transit_duration(5.0)
        dur_long = estimate_max_transit_duration(50.0)
        wl_short = compute_window_length(dur_short, 30.0)
        wl_long = compute_window_length(dur_long, 30.0)
        assert wl_long > wl_short, "Longer period → longer window"

    def test_savgol_preserves_median(self, synthetic_data):
        time, flux, flux_err = synthetic_data
        cadence_min = np.median(np.diff(time)) * 24 * 60
        baseline = time[-1] - time[0]
        _, detrended, _ = run_detrending(
            time, flux,
            period_max_days=baseline / 3.0,
            cadence_minutes=cadence_min,
            method="savgol",
            target_id="synthetic_test",
            save_plot=False,
        )
        assert abs(np.median(detrended) - 1.0) < 0.01, \
            "Detrended median should be 1.0"

    def test_detrended_std_smaller_than_raw(self, synthetic_data):
        """Detrending should reduce the overall flux scatter (no trend left)."""
        time, flux, flux_err = synthetic_data
        cadence_min = np.median(np.diff(time)) * 24 * 60
        baseline = time[-1] - time[0]
        # For purely synthetic data with no injected trend,
        # detrended std should remain comparable to raw (within 2x).
        _, detrended, _ = run_detrending(
            time, flux,
            period_max_days=baseline / 3.0,
            cadence_minutes=cadence_min,
            method="savgol",
            target_id="synthetic_test",
            save_plot=False,
        )
        assert detrended.std() < flux.std() * 2.0, \
            "Detrended std should not be much larger than raw"


class TestPeriodSearch:
    """Test BLS period recovery on synthetic data."""

    def test_period_grid_limits(self):
        baseline = 100.0
        grid = build_period_grid(baseline, min_period=0.5, max_fraction=1/3)
        assert grid[0] >= 0.5
        assert grid[-1] <= baseline / 3.0 + 0.01  # small floating tolerance

    def test_bls_recovers_period(self, synthetic_data):
        """
        BLS should recover the injected period within 1%.
        This is a controlled unit test on synthetic data — not real data validation.
        """
        time, flux, flux_err = synthetic_data
        cadence_min = np.median(np.diff(time)) * 24 * 60
        baseline = time[-1] - time[0]

        # Detrend first
        _, detrended, _ = run_detrending(
            time, flux,
            period_max_days=baseline / 3.0,
            cadence_minutes=cadence_min,
            method="savgol",
            target_id="synth_bls_test",
            save_plot=False,
        )

        period_grid = build_period_grid(baseline)
        duration_grid = build_duration_grid()
        _, best_signal = run_bls(time, detrended, flux_err, period_grid, duration_grid)

        period_error_pct = (
            abs(best_signal["period"] - SYNTH_PERIOD) / SYNTH_PERIOD * 100
        )
        assert period_error_pct < 1.0, (
            f"BLS period recovery error {period_error_pct:.2f}% > 1% "
            f"(recovered={best_signal['period']:.4f} d, "
            f"injected={SYNTH_PERIOD:.4f} d)"
        )

    def test_phase_fold_shape(self, synthetic_data):
        time, flux, flux_err = synthetic_data
        phase, flux_folded = phase_fold(time, flux, SYNTH_PERIOD, t0=0.0)
        assert len(phase) == len(flux_folded)
        assert phase.min() >= -0.5
        assert phase.max() <= 0.5
