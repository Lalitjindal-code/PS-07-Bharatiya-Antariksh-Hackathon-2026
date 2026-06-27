"""
prepare_cnn_data.py
===================
Prepares phase-folded global (2001 bins) and local (201 bins) views
from Kepler TCE dataset parameters for training the CNN classifier.
"""

import logging
import os
from pathlib import Path
import numpy as np
import pandas as pd
from data_loader import generate_synthetic_lc, preprocess
from identify import phase_fold

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = Path(__file__).parent / "tce_data.csv"
OUTPUT_PATH = Path(__file__).parent / "models" / "cnn_dataset.npz"
OUTPUT_PATH.parent.mkdir(exist_ok=True)

GLOBAL_BINS = 2001
LOCAL_BINS = 201

def generate_folded_views(period, duration_h, depth_ppm, impact_b, label, seed=42):
    """
    Synthesize global and local folded light curve views based on TCE parameters.
    PC: U-shaped / trapezoid flat-bottom transit.
    AFP: V-shaped transit (simulating binary stars).
    NTP: Pure noise or stellar activity.
    """
    # 30-minute cadence over 10 days to get clean folding baseline
    n_points = 2000
    cadence = 0.02  # days
    time = np.arange(n_points) * cadence
    
    # Calculate noise level from Kepler typical parameters
    noise_level = 1e-4
    flux = np.ones(n_points)
    
    depth = depth_ppm / 1e6
    duration = duration_h / 24.0
    
    # Generate phase fold coordinates
    phase_raw = ((time - 0.1) % period) / period
    phase = np.where(phase_raw > 0.5, phase_raw - 1.0, phase_raw)
    
    # Generate signal shape based on label
    if label == "PC":
        # Trapezoid shape (flat bottom)
        half_dur = duration / 2.0
        ingress = duration * 0.1
        in_transit = np.abs(phase) < half_dur
        flux[in_transit] -= depth
        # smooth ingress/egress
        ingress_mask = (np.abs(phase) >= half_dur - ingress) & (np.abs(phase) <= half_dur)
        if np.any(ingress_mask):
            dist = (half_dur - np.abs(phase[ingress_mask])) / ingress
            flux[ingress_mask] = 1.0 - depth * dist
            
    elif label == "AFP":
        # V-shape profile (typical for grazing EBs)
        half_dur = duration / 2.0
        in_transit = np.abs(phase) < half_dur
        if np.any(in_transit):
            flux[in_transit] -= depth * (1.0 - np.abs(phase[in_transit]) / half_dur)
            
    else:  # NTP (Noise)
        # No signal injected, just noise
        pass

    # Add Gaussian noise
    rng = np.random.default_rng(seed)
    flux += rng.normal(0, noise_level, n_points)
    
    # Fold
    phase_f, flux_f = phase_fold(time, flux, period, 0.1)
    
    # Interpolate to uniform global bins (2001 bins from -0.5 to +0.5)
    global_grid = np.linspace(-0.5, 0.5, GLOBAL_BINS)
    global_view = np.interp(global_grid, phase_f, flux_f)
    
    # Interpolate to uniform local bins (201 bins from -0.15 to +0.15)
    local_grid = np.linspace(-0.15, 0.15, LOCAL_BINS)
    local_view = np.interp(local_grid, phase_f, flux_f)
    
    return global_view, local_view

def prepare_dataset(max_samples=2000):
    """
    Load catalog data, construct dataset, and save as compressed .npz.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"TCE data not found at {DATA_PATH}")
        
    df = pd.read_csv(DATA_PATH)
    # Filter valid labels
    df = df[df["av_training_set"].isin(["PC", "AFP", "NTP"])].copy()
    
    # Subsample if necessary to balance dataset and run quickly
    if len(df) > max_samples:
        df = df.groupby("av_training_set", group_keys=False).apply(
            lambda x: x.sample(min(len(x), max_samples // 3), random_state=42)
        )
        
    logger.info("Preparing CNN dataset with %d targets...", len(df))
    
    global_data = []
    local_data = []
    labels = []
    scalars = []
    
    for idx, row in df.iterrows():
        try:
            period = float(row["tce_period"])
            duration = float(row["tce_duration"])
            depth = float(row["tce_depth"])
            impact = float(row["tce_impact"]) if pd.notna(row["tce_impact"]) else 0.0
            snr = float(row["tce_model_snr"]) if pd.notna(row["tce_model_snr"]) else 5.0
            label = row["av_training_set"]
            
            g_view, l_view = generate_folded_views(period, duration, depth, impact, label, seed=idx)
            
            global_data.append(g_view)
            local_data.append(l_view)
            labels.append(label)
            # Store scalar features: depth, duration, period, snr, impact
            scalars.append([depth, duration, period, snr, impact])
        except Exception as e:
            logger.warning("Failed processing row %d: %s", idx, e)
            
    global_arr = np.array(global_data, dtype=np.float32)
    local_arr = np.array(local_data, dtype=np.float32)
    scalars_arr = np.array(scalars, dtype=np.float32)
    
    # Label encoding
    label_mapping = {"PC": 0, "AFP": 1, "NTP": 2}
    labels_encoded = np.array([label_mapping[l] for l in labels], dtype=np.int64)
    
    # Save datasets
    np.savez_compressed(
        OUTPUT_PATH,
        global_view=global_arr,
        local_view=local_arr,
        scalars=scalars_arr,
        labels=labels_encoded
    )
    logger.info("Dataset prepared and saved to %s", OUTPUT_PATH)

if __name__ == "__main__":
    prepare_dataset()
