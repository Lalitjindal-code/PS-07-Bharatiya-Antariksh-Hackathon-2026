"""
classify.py  —  Phase 9-10: Feature engineering + inference.
Loads the pre-trained RandomForestClassifier and returns a prediction
for any candidate signal described by pipeline outputs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent / "models"
CLF_PATH   = MODELS_DIR / "rf_classifier.joblib"
IMP_PATH   = MODELS_DIR / "imputer.joblib"

# Feature order MUST match what the model was trained on (train_classifier.py)
FEATURE_NAMES = [
    "tce_depth",        # transit depth [ppm]
    "tce_duration",     # transit duration [hours]
    "tce_period",       # orbital period [days]
    "tce_model_snr",    # model SNR
    "tce_bin_oedp_stat",# odd-even depth difference statistic
    "tce_impact",       # orbital impact parameter b = (a/Rs)*cos(inc)
    "tce_prad",         # planet radius proxy [R_earth]
]

# Label mapping used during training
LABEL_MAP = {"PC": "planet_candidate", "AFP": "eclipsing_binary_or_false_positive", "NTP": "noise"}


def load_classifier() -> tuple:
    """Load (classifier, imputer) from disk. Raises FileNotFoundError if missing."""
    if not CLF_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {CLF_PATH}. Run train_classifier.py first."
        )
    clf = joblib.load(CLF_PATH)
    imp = joblib.load(IMP_PATH) if IMP_PATH.exists() else None
    logger.info("Loaded classifier from %s", CLF_PATH)
    return clf, imp


def build_feature_vector(
    depth_ppm: float,
    duration_h: float,
    period_d: float,
    snr: float,
    odd_even_stat: float = np.nan,
    impact_b: float = np.nan,
    prad_earth: float = np.nan,
) -> np.ndarray:
    """
    Assemble feature vector in the order expected by the model.

    Parameters
    ----------
    depth_ppm     : transit depth [ppm]
    duration_h    : transit duration [hours]
    period_d      : orbital period [days]
    snr           : signal-to-noise ratio
    odd_even_stat : |odd_depth - even_depth| / pooled_std  (0 if not computed)
    impact_b      : orbital impact parameter b = (a/Rs)*cos(inc)
    prad_earth    : planet radius in Earth radii  (nan -> imputed by model)

    Returns
    -------
    np.ndarray shape (1, 7)
    """
    vec = np.array([[
        depth_ppm,
        duration_h,
        period_d,
        snr,
        odd_even_stat,
        impact_b,
        prad_earth,
    ]], dtype=float)
    return vec


def classify_candidate(
    feature_vec: np.ndarray,
    clf=None,
    imp=None,
) -> Dict:
    """
    Run inference on one feature vector.

    Parameters
    ----------
    feature_vec : np.ndarray  shape (1, 7)
    clf         : loaded sklearn classifier (loaded if None)
    imp         : loaded SimpleImputer     (loaded if None)

    Returns
    -------
    dict with: classification, classification_confidence, class_probabilities
    """
    if clf is None or imp is None:
        clf, imp = load_classifier()

    X = feature_vec.copy()
    if imp is not None:
        X = imp.transform(X)

    pred_label = clf.predict(X)[0]
    proba      = clf.predict_proba(X)[0]
    confidence = float(proba.max())

    class_probs = {cls: float(p) for cls, p in zip(clf.classes_, proba)}

    return {
        "classification":            LABEL_MAP.get(pred_label, pred_label),
        "classification_raw_label":  pred_label,
        "classification_confidence": confidence,
        "class_probabilities":       class_probs,
    }


def classify_from_pipeline_outputs(
    best_signal: dict,
    fit_params: dict,
    snr_result: dict,
    vet_results: dict,
    clf=None,
    imp=None,
    star_radius_rsun: float = 1.0,
) -> Dict:
    """
    Convenience wrapper: extract features from pipeline dicts and classify.

    Parameters
    ----------
    best_signal       : from identify.run_bls()
    fit_params        : from characterize.run_characterization()
    snr_result        : from significance.run_significance()
    vet_results       : from vet.run_vetting()  (test_results dict)
    star_radius_rsun  : stellar radius [R_sun]; used for planet radius calculation.
                        Default 1.0 (generic solar). Pass TIC/KIC value for accuracy.
    """
    rp_rs      = float(fit_params.get("rp_val", np.nan))
    a_rs       = float(fit_params.get("a_rs_val", np.nan))
    inc_deg    = float(fit_params.get("inc_val", 90.0))

    # Use refined fit values for depth and duration (more accurate than raw BLS)
    fit_depth_ppm = float(fit_params.get("depth_ppm_val", np.nan))
    fit_dur_h     = float(fit_params.get("duration_h_val", np.nan))
    depth_ppm  = fit_depth_ppm if np.isfinite(fit_depth_ppm) else float(best_signal.get("depth", 0.0)) * 1e6
    duration_h = fit_dur_h    if np.isfinite(fit_dur_h)     else float(best_signal.get("duration", 0.0)) * 24.0

    period_d    = float(best_signal.get("period", 0.0))
    snr         = float(snr_result.get("snr", np.nan))

    # Real orbital impact parameter b = (a/Rs) * cos(inc)
    impact_b = a_rs * np.cos(np.radians(inc_deg)) if np.isfinite(a_rs) and np.isfinite(inc_deg) else np.nan

    oe = vet_results.get("odd_even", {})
    odd_even_stat = float(oe.get("depth_diff_sigma", np.nan))

    # Planet radius: Rp/Rs * R_star. Default 1.0 R_sun; caller should pass target's R_star.
    prad_earth = rp_rs * 109.076 * star_radius_rsun if np.isfinite(rp_rs) else np.nan

    fvec = build_feature_vector(
        depth_ppm, duration_h, period_d, snr, odd_even_stat, impact_b, prad_earth
    )
    result = classify_candidate(fvec, clf=clf, imp=imp)
    result["features_used"] = {
        name: float(val) for name, val in zip(FEATURE_NAMES, fvec[0])
    }
    return result
