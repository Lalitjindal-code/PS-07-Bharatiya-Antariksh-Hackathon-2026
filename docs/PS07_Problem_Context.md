# PS-07 — Bharatiya Antariksh Hackathon 2026

**Problem Title:** AI-enabled Detection of Exoplanets from Noisy Astronomical Light Curves
**Organizers:** ISRO + Physical Research Laboratory, Ahmedabad
**Team:** Rishikesh Sharma, Churchil Dwivedi, Neelam JSSV Prasad, Kapil Kumar

---

## Core Concept: Transit Photometry

When a planet passes in front of its star, the star's brightness dips periodically.

**Key Parameters to detect/measure:**
- **Event Depth** ≈ (R_planet / R_star)²
- **Event Duration**
- **Orbital Period**

**Missions:** TESS · Kepler · K2

---

## Problem Statement — 5 Tasks

### 01. Detrending
Remove noise and systematic variations from raw stellar light curve data to reveal underlying signals.
- Input: raw noisy flux (red/blue plots shown)
- Output: clean detrended flux

### 02. Identifying the Events
Spot periodic dips in light curves that could indicate astrophysical phenomena.
- Find transit dips via BLS period search
- Phase-fold to confirm periodicity
- **Orbital Period** = interval between two consecutive transit events

### 03. Characterization
Measure and estimate key parameters for detected events:
- Transit depth (ppm)
- Orbital period (days)
- Transit duration (hours)

### 04. Classification
Develop a framework to categorize dips into:
- Planetary transits
- Eclipsing binaries
- Blends
- Other astrophysical signals

### 05. Statistical Significance
Provide signal-to-noise ratios and formal significance levels for all identified astronomical findings.

---

## Dataset Provided

- TESS raw light curves (Target / unknown data)
- A curated dataset for different classifiers to train the AI model (Kepler TCE table → `tce_data.csv`)

---

## Our Pipeline Mapping

| PS Task | Our Module | Phase |
|---|---|---|
| Detrending | `detrend.py` | Phase 4 |
| Identification | `identify.py` | Phase 5 |
| Characterization | `characterize.py` | Phase 6 |
| Classification | `classify.py` + `train_classifier.py` | Phase 9-10 |
| Statistical Significance | `significance.py` | Phase 8 |
| Vetting (EB rejection) | `vet.py` | Phase 7 |
| End-to-end | `pipeline.py` | Phase 11 |
| Demo | `streamlit_app.py` | Phase 12 |

## Reference Target: Kepler-10b (KIC 11904151)
- Period: 0.8375243 d
- Depth: ~152 ppm  (= (Rp/Rs)² where Rp≈1.47 R_earth, Rs≈1.065 R_sun)
- Duration: 1.811 h
- Mission: Kepler long-cadence (29.4 min)

---

## Further Steps and Expectations (from organizers)

- **04.** Develop the AI-based classifier based on transit shape parameters and train it with the given known datasets.
- **05.** Apply it to the provided unknown datasets, identify and classify the events.
- **06.** Provide the basic parameters with the significance level.

---

## Official Problem Description

Develop an AI-based data analysis pipeline capable of automatically detecting exoplanet transit signals from noisy astronomical light curve data.

**Details:** Exoplanet detection through transit photometry requires identification of extremely small brightness variations in stars. For light curves in crowded fields, there can be significant contaminations from:
- Stellar blending by foreground/background sources in the aperture
- Intrinsic noise from detector response
- Transiting planet across host star's disk
- Eclipsing stellar companion in binary star systems
- Starspots

Different phenomena give rise to distinct features in light curves which become difficult to disentangle in noisy crowded-field datasets.

---

## Transit Shape Fitting — Expected Output (Slide 03)

Characterize the event and estimate shape parameters:

| Parameter | Example Value |
|---|---|
| Baseline Flux (f₀) | 1.000259 |
| Transit Depth | 1.4619% (14618.6 ppm) |
| Total Duration (T_tot) | 2.633 hours |
| Ingress/Egress (T_in) | 0.444 hours |
| Flat Bottom Duration | 1.745 hours |

---

## Expected AI Pipeline Flow (from organizers' diagram)

```
Noisy Light Curve
  → Denoising (Autoencoder / SG filter)
  → Feature Extraction
  → Transit Detection Classifier
  → Output: Detection Probability (e.g. 0.97)
```

**Detection Result format expected:**
- Transit Detected: Yes/No
- Confidence score
- Period (days)
- Depth (%)
- Duration (hours)
- Phase-folded light curve plot with model fit overlay
