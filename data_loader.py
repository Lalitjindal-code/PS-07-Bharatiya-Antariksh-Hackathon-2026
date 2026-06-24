"""
data_loader.py
==============
Phase 2 & 3 of the exoplanet transit detection pipeline.

Responsibilities
----------------
1. Download real TESS / Kepler light curves via ``lightkurve``.
2. Stitch multiple sectors / quarters into a single baseline.
3. Apply quality masking, NaN removal, and iterative sigma-clipping.
4. Provide a *clearly labelled* synthetic light-curve generator for
   controlled unit tests ONLY — never used as the primary validation target.

Usage (standalone smoke-test)
------------------------------
    python data_loader.py --target "KIC 11904151" --mission Kepler

References
----------
- Lightkurve Collaboration (2018): https://docs.lightkurve.org
- Kepler Data Processing Handbook, §2.3 (quality flags)
"""

from __future__ import annotations

import argparse
import logging
import os
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import lightkurve as lk
from astropy.timeseries import TimeSeries
import astropy.units as u

# ---------------------------------------------------------------------------
# Module-level configuration
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

DEFAULT_CACHE_DIR = Path(__file__).parent / "data"
DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Random seed for all stochastic operations in this module
RNG_SEED: int = 42
rng = np.random.default_rng(RNG_SEED)


# ---------------------------------------------------------------------------
# Real-data acquisition
# ---------------------------------------------------------------------------

def download_lightcurve(
    target_id: str,
    mission: str = "Kepler",
    exptime: Optional[int] = None,
    author: Optional[str] = None,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    max_download: int = 20,
) -> lk.LightCurve:
    """
    Search and download a light curve collection for *target_id*, then stitch
    all available quarters / sectors into a single ``LightCurve``.

    Parameters
    ----------
    target_id : str
        Any identifier accepted by ``lightkurve.search_lightcurve()``,
        e.g. "KIC 11904151", "TIC 261136679", "Kepler-10".
    mission : str
        "Kepler", "K2", or "TESS".
    exptime : int, optional
        Exposure time in seconds.  Pass ``None`` to accept any cadence.
    author : str, optional
        Pipeline author, e.g. "Kepler", "SPOC".  ``None`` = any.
    cache_dir : Path
        Directory where downloaded FITS files are cached.
    max_download : int
        Maximum number of sectors / quarters to download (avoids runaway
        downloads on targets with many quarters).

    Returns
    -------
    lk.LightCurve
        A single stitched light curve.  Time is in BKJD or BTJD days.

    Raises
    ------
    ValueError
        If no light curves are found for *target_id*.
    RuntimeError
        If the download or stitching step fails unexpectedly.
    """
    logger.info("Searching for '%s' in mission=%s ...", target_id, mission)

    search_kwargs: dict = {"mission": mission}
    if exptime is not None:
        search_kwargs["exptime"] = exptime
    if author is not None:
        search_kwargs["author"] = author

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = lk.search_lightcurve(target_id, **search_kwargs)

    if len(result) == 0:
        raise ValueError(
            f"No light curves found for target='{target_id}' "
            f"mission='{mission}'.  Check the target ID and mission."
        )

    logger.info("Found %d light curve file(s).  Downloading up to %d ...",
                len(result), max_download)

    # Limit downloads to avoid very long waits on multi-year targets
    result = result[:max_download]

    try:
        lc_collection = result.download_all(
            cache=True,
            download_dir=str(cache_dir),
        )
    except Exception as exc:
        raise RuntimeError(
            f"Download failed for '{target_id}': {exc}"
        ) from exc

    # Stitch: normalize each quarter to unit median before joining
    # (this suppresses inter-quarter flux offsets from different apertures)
    logger.info("Stitching %d segments ...", len(lc_collection))
    try:
        lc_stitched: lk.LightCurve = lc_collection.stitch()
    except Exception as exc:
        raise RuntimeError(
            f"Stitching failed: {exc}"
        ) from exc

    logger.info(
        "Stitched light curve: %d points, baseline=%.2f days",
        len(lc_stitched),
        float(lc_stitched.time.value[-1] - lc_stitched.time.value[0]),
    )
    return lc_stitched


def extract_arrays(
    lc: lk.LightCurve,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract plain NumPy arrays from a ``LightCurve`` object.

    Parameters
    ----------
    lc : lk.LightCurve
        Stitched light curve from ``download_lightcurve()``.

    Returns
    -------
    time : np.ndarray, shape (N,)
        Time stamps in days (BKJD or BTJD).
    flux : np.ndarray, shape (N,)
        Normalised flux (SAP or PDCSAP depending on pipeline).
    flux_err : np.ndarray, shape (N,)
        Per-point flux uncertainty.
    quality : np.ndarray, shape (N,), dtype int
        Quality bitmask flags (0 = clean).
    """
    time = lc.time.value.astype(np.float64)
    flux = lc.flux.value.astype(np.float64)

    # flux_err may be absent in some pipelines
    if hasattr(lc, "flux_err") and lc.flux_err is not None:
        flux_err = lc.flux_err.value.astype(np.float64)
    else:
        logger.warning("flux_err not available; estimating from flux scatter.")
        flux_err = np.full_like(flux, np.nanstd(flux))

    # quality flag may be absent (e.g. after stitching)
    if hasattr(lc, "quality") and lc.quality is not None:
        quality = np.array(lc.quality, dtype=int)
    else:
        quality = np.zeros(len(flux), dtype=int)

    return time, flux, flux_err, quality


def apply_quality_mask(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    quality: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Remove cadences where quality != 0.

    Quality bitmask conventions differ between Kepler and TESS; any non-zero
    flag is treated conservatively as a bad point.

    Parameters
    ----------
    time, flux, flux_err, quality : np.ndarray
        Raw arrays from ``extract_arrays()``.

    Returns
    -------
    time, flux, flux_err : np.ndarray
        Arrays with bad-quality cadences removed.
    """
    good = quality == 0
    n_removed = int(np.sum(~good))
    logger.info("Quality mask: removing %d / %d bad-quality cadences.",
                n_removed, len(flux))
    return time[good], flux[good], flux_err[good]


def drop_nans(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Drop any cadence where ``time``, ``flux``, or ``flux_err`` is NaN or Inf.

    Parameters
    ----------
    time, flux, flux_err : np.ndarray

    Returns
    -------
    time, flux, flux_err : np.ndarray
        Arrays with NaN / Inf rows removed.
    """
    finite = (
        np.isfinite(time)
        & np.isfinite(flux)
        & np.isfinite(flux_err)
    )
    n_removed = int(np.sum(~finite))
    if n_removed:
        logger.info("NaN/Inf removal: dropping %d cadences.", n_removed)
    return time[finite], flux[finite], flux_err[finite]


def sigma_clip_lc(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    sigma: float = 5.0,
    n_passes: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Iterative sigma-clipping outlier removal.

    In each pass, points further than *sigma* × σ from the median flux are
    masked out.  Three passes are used to handle clustered outliers that
    would inflate the standard deviation in a single pass.

    Parameters
    ----------
    time, flux, flux_err : np.ndarray
        Pre-cleaned arrays (after NaN removal and quality masking).
    sigma : float
        Clipping threshold in units of the current standard deviation.
    n_passes : int
        Number of iterative clipping passes.

    Returns
    -------
    time, flux, flux_err : np.ndarray
        Arrays with outliers removed.
    """
    mask = np.ones(len(flux), dtype=bool)
    for i in range(n_passes):
        median = np.median(flux[mask])
        std = np.std(flux[mask])
        new_mask = np.abs(flux - median) < sigma * std
        n_clipped = int(np.sum(mask & ~new_mask))
        if n_clipped:
            logger.info("Sigma-clip pass %d/%d: removed %d outliers (>%.0f-sigma).",
                        i + 1, n_passes, n_clipped, sigma)
        mask = mask & new_mask

    return time[mask], flux[mask], flux_err[mask]


def preprocess(
    lc: lk.LightCurve,
    sigma: float = 5.0,
    n_clip_passes: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Full Phase 3 preprocessing pipeline: extract → quality mask → NaN drop
    → sigma clip.

    Parameters
    ----------
    lc : lk.LightCurve
        Raw stitched light curve from ``download_lightcurve()``.
    sigma : float
        Sigma-clipping threshold.
    n_clip_passes : int
        Number of sigma-clip passes.

    Returns
    -------
    time, flux, flux_err : np.ndarray
        Clean arrays ready for detrending.
    """
    time, flux, flux_err, quality = extract_arrays(lc)
    time, flux, flux_err = apply_quality_mask(time, flux, flux_err, quality)
    time, flux, flux_err = drop_nans(time, flux, flux_err)
    time, flux, flux_err = sigma_clip_lc(
        time, flux, flux_err, sigma=sigma, n_passes=n_clip_passes
    )
    logger.info("Preprocessing complete: %d clean cadences remain.", len(time))
    return time, flux, flux_err


# ---------------------------------------------------------------------------
# Synthetic light-curve generator — UNIT TEST UTILITY ONLY
# ---------------------------------------------------------------------------

def generate_synthetic_lc(
    n_points: int = 10_000,
    period_days: float = 3.5,
    depth: float = 0.01,
    duration_days: float = 0.1,
    noise_level: float = 3e-4,
    seed: int = RNG_SEED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    **CONTROLLED UNIT TEST UTILITY — NOT a substitute for real data.**

    Generate a synthetic light curve with a simple box-shaped transit injected
    into Gaussian noise.  Used exclusively in ``tests/test_synthetic.py`` to
    verify that each pipeline stage recovers the injected parameters.

    Transit model: flat box (depth × duration).
    Noise model: i.i.d. Gaussian with amplitude *noise_level*.

    Parameters
    ----------
    n_points : int
        Number of cadences.
    period_days : float
        Orbital period [days].
    depth : float
        Transit depth as a fractional flux decrement (e.g. 0.01 = 1%).
    duration_days : float
        Transit duration [days].
    noise_level : float
        1-σ Gaussian noise level (fraction of unit flux).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    time : np.ndarray, shape (n_points,)
        Time array [days], starting at 0.
    flux : np.ndarray, shape (n_points,)
        Normalised flux including transit signal + Gaussian noise.
    flux_err : np.ndarray, shape (n_points,)
        Per-point flux uncertainties (constant = noise_level).
    """
    # IMPORTANT: This is synthetic data for unit testing only.
    # Results on synthetic data do NOT represent real pipeline performance.
    local_rng = np.random.default_rng(seed)

    cadence = 30.0 / 60.0 / 24.0  # 30-minute cadence in days
    time = np.arange(n_points) * cadence

    # Create box transit signal
    phase = (time % period_days) / period_days  # [0, 1)
    half_dur = duration_days / (2.0 * period_days)
    in_transit = (phase < half_dur) | (phase > 1.0 - half_dur)

    flux = np.ones(n_points)
    flux[in_transit] -= depth

    # Add Gaussian noise
    flux += local_rng.normal(0.0, noise_level, n_points)
    flux_err = np.full(n_points, noise_level)

    logger.info(
        "[SYNTHETIC] Generated %d-point light curve: "
        "period=%.3f d, depth=%.4f, duration=%.3f d, noise=%.1e  "
        "— THIS IS A UNIT-TEST FIXTURE, NOT REAL DATA.",
        n_points, period_days, depth, duration_days, noise_level,
    )
    return time, flux, flux_err


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Download and preprocess a real TESS/Kepler light curve."
    )
    p.add_argument("--target", default="KIC 11904151",
                   help="Target identifier (default: Kepler-10, KIC 11904151)")
    p.add_argument("--mission", default="Kepler",
                   choices=["Kepler", "K2", "TESS"],
                   help="Space mission")
    p.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR),
                   help="Directory to cache FITS files")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()

    print("\n" + "=" * 60)
    print("  DATA LOADER SMOKE TEST — REAL DATA")
    print("=" * 60)
    print(f"  Target  : {args.target}")
    print(f"  Mission : {args.mission}")
    print("=" * 60 + "\n")

    lc = download_lightcurve(
        target_id=args.target,
        mission=args.mission,
        cache_dir=Path(args.cache_dir),
    )
    time, flux, flux_err = preprocess(lc)

    baseline = time[-1] - time[0]
    print("\n--- RESULTS ---")
    print(f"  Clean cadences     : {len(time):,}")
    print(f"  Baseline           : {baseline:.2f} days")
    print(f"  Cadence (median)   : {np.median(np.diff(time)) * 24 * 60:.2f} min")
    print(f"  Flux range         : [{flux.min():.6f}, {flux.max():.6f}]")
    print(f"  Median flux_err    : {np.median(flux_err):.2e}")
    print("\nSMOKE TEST PASSED — data_loader.py is working correctly.\n")
