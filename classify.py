"""
classify.py  —  Phase 9-10: Feature engineering + inference.
Loads the pre-trained RandomForestClassifier and returns a prediction
for any candidate signal described by pipeline outputs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Any

import joblib
import numpy as np

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent / "models"
CLF_PATH   = MODELS_DIR / "rf_classifier.joblib"
IMP_PATH   = MODELS_DIR / "imputer.joblib"
CNN_PATH   = MODELS_DIR / "cnn_classifier.pt"

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


def load_cnn_classifier() -> Optional[Any]:
    """Load CNN classifier if available."""
    if not CNN_PATH.exists():
        return None
    try:
        import torch
        from train_cnn import DualViewCNN
        model = DualViewCNN()
        model.load_state_dict(torch.load(CNN_PATH, map_location=torch.device('cpu')))
        model.eval()
        logger.info("Loaded CNN classifier from %s", CNN_PATH)
        return model
    except Exception as e:
        logger.warning("Failed to load CNN classifier: %s", e)
        return None


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

    # Use the best available depth in this priority order:
    # 1. empirical phase-fold depth (most accurate for shallow transits)
    # 2. batman model fit depth
    # 3. raw BLS box depth
    empirical_ppm  = float(best_signal.get("empirical_depth_ppm", 0.0))
    fit_depth_ppm  = float(fit_params.get("depth_ppm_val", np.nan))
    bls_depth_ppm  = float(best_signal.get("depth", 0.0)) * 1e6
    fit_dur_h      = float(fit_params.get("duration_h_val", np.nan))

    # Pick best depth: empirical > batman if empirical is substantially larger
    if empirical_ppm > 10.0 and (not np.isfinite(fit_depth_ppm) or fit_depth_ppm < 5.0 or empirical_ppm > fit_depth_ppm * 1.5):
        depth_ppm = empirical_ppm
        logger.info("Using empirical depth %.1f ppm for classification (batman=%.1f ppm).",
                    empirical_ppm, fit_depth_ppm if np.isfinite(fit_depth_ppm) else 0.0)
    elif np.isfinite(fit_depth_ppm) and fit_depth_ppm >= 1.0:
        depth_ppm = fit_depth_ppm
    else:
        depth_ppm = max(bls_depth_ppm, empirical_ppm)

    duration_h = fit_dur_h if np.isfinite(fit_dur_h) and fit_dur_h > 0 else float(best_signal.get("duration", 0.0)) * 24.0
    period_d   = float(best_signal.get("period", 0.0))
    snr        = float(snr_result.get("snr", np.nan))

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

    # -----------------------------------------------------------------------
    # Vetting-score probability adjustment
    # If the classical vetting tests strongly favour a planet (all 3 pass),
    # apply a small boost to the PC probability to correct for RF bias.
    # Conversely, penalise PC if vetting score is very negative.
    # This keeps the ML result primary while incorporating physical evidence.
    # -----------------------------------------------------------------------
    vet_score = 0
    for test_key in ["odd_even", "secondary", "centroid"]:
        vet_score += vet_results.get(test_key, {}).get("score", 0)

    probs = result["class_probabilities"]
    if vet_score >= 2 and "PC" in probs and "AFP" in probs:
        # Strong vetting pass: shift 10% from AFP toward PC
        boost = min(0.10, probs.get("AFP", 0.0))
        probs["PC"]  = min(1.0, probs.get("PC", 0.0)  + boost)
        probs["AFP"] = max(0.0, probs.get("AFP", 0.0) - boost)
        logger.info("Vetting boost (+%.0f%% PC): all vetting tests passed.", boost * 100)
    elif vet_score <= -2 and "AFP" in probs and "PC" in probs:
        # Strong vetting fail: shift 10% from PC toward AFP
        penalty = min(0.10, probs.get("PC", 0.0))
        probs["AFP"] = min(1.0, probs.get("AFP", 0.0) + penalty)
        probs["PC"]  = max(0.0, probs.get("PC", 0.0)  - penalty)
        logger.info("Vetting penalty (-%.0f%% PC): vetting tests failed.", penalty * 100)

    # Re-determine top class after adjustment
    if probs:
        top_raw = max(probs, key=lambda k: probs[k])
        result["classification"] = LABEL_MAP.get(top_raw, top_raw)
        result["classification_confidence"] = float(probs[top_raw])
        result["class_probabilities"] = probs

    result["features_used"] = {
        name: float(val) for name, val in zip(FEATURE_NAMES, fvec[0])
    }
    result["features_used"]["empirical_depth_ppm"] = empirical_ppm


    # --- CNN Classification Integration ---
    cnn_model = load_cnn_classifier()
    if cnn_model is not None:
        phase_f = best_signal.get("phase", None)
        flux_f = best_signal.get("flux_folded", None)
        if phase_f is not None and flux_f is not None:
            try:
                import torch
                # Interpolate to uniform global bins (2001 bins from -0.5 to +0.5)
                global_grid = np.linspace(-0.5, 0.5, 2001)
                global_view = np.interp(global_grid, phase_f, flux_f)
                
                # Interpolate to uniform local bins (201 bins from -0.15 to +0.15)
                local_grid = np.linspace(-0.15, 0.15, 201)
                local_view = np.interp(local_grid, phase_f, flux_f)
                
                # Form Torch tensors
                g_tensor = torch.tensor(global_view, dtype=torch.float32).unsqueeze(0) # (1, 2001)
                l_tensor = torch.tensor(local_view, dtype=torch.float32).unsqueeze(0) # (1, 201)
                
                # Scalars: depth, duration, period, snr, impact
                _imp = impact_b if (np.isfinite(impact_b) and not np.isnan(impact_b)) else 0.0
                _snr = snr if (np.isfinite(snr) and not np.isnan(snr)) else 5.0
                s_tensor = torch.tensor([[depth_ppm, duration_h, period_d, _snr, _imp]], dtype=torch.float32)
                
                with torch.no_grad():
                    logits = cnn_model(g_tensor, l_tensor, s_tensor)
                    probs = torch.softmax(logits, dim=1).numpy()[0]
                    
                label_map = ["PC", "AFP", "NTP"]
                cnn_classes = {"PC": "planet_candidate", "AFP": "eclipsing_binary_or_false_positive", "NTP": "noise"}
                
                best_idx = np.argmax(probs)
                best_class = label_map[best_idx]
                
                result["cnn_classification"] = cnn_classes.get(best_class, best_class)
                result["cnn_confidence"] = float(probs[best_idx])
                result["cnn_class_probabilities"] = {
                    cnn_classes.get(label_map[i]): float(p) for i, p in enumerate(probs)
                }
            except Exception as e:
                logger.warning("CNN inference failed: %s", e)

    return result
