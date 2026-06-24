"""
streamlit_app.py  —  Phase 12: Interactive demo for the PS-07 pipeline.

Run with:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PS-07 | Exoplanet Transit Detector",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background: #0b0f1a; color: #e0e6f0; }

    .hero {
        background: linear-gradient(135deg, #0d1b2a 0%, #1a2a4a 50%, #0d2235 100%);
        border: 1px solid #1e3a5f;
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
    }
    .hero h1 {
        font-size: 2rem; font-weight: 700;
        background: linear-gradient(90deg, #4fc3f7, #81d4fa, #b3e5fc);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin: 0 0 0.3rem 0;
    }
    .hero p { color: #90a4ae; margin: 0; font-size: 0.95rem; }

    .metric-card {
        background: #111827;
        border: 1px solid #1e3a5f;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .metric-card .label { font-size: 0.75rem; color: #607d8b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-card .value { font-size: 1.7rem; font-weight: 700; color: #4fc3f7; margin: 0.2rem 0; }
    .metric-card .sub   { font-size: 0.78rem; color: #546e7a; }

    .verdict-planet  { background: linear-gradient(135deg,#0d2d1a,#0a3d2e); border:1px solid #1b5e20; border-radius:10px; padding:1rem; }
    .verdict-binary  { background: linear-gradient(135deg,#2d1a0d,#3d2a0a); border:1px solid #5e3a1b; border-radius:10px; padding:1rem; }
    .verdict-noise   { background: linear-gradient(135deg,#1a1a2d,#2a2a3d); border:1px solid #3a3a5e; border-radius:10px; padding:1rem; }

    .vet-pass { color: #66bb6a; font-weight: 600; }
    .vet-fail { color: #ef5350; font-weight: 600; }
    .vet-na   { color: #78909c; }

    .stButton>button {
        background: linear-gradient(135deg, #1565c0, #0288d1);
        color: white; border: none; border-radius: 8px;
        padding: 0.6rem 2rem; font-weight: 600; font-size: 1rem;
        width: 100%; transition: all 0.2s;
    }
    .stButton>button:hover { opacity: 0.85; transform: translateY(-1px); }

    div[data-testid="stExpander"] {
        background: #111827; border: 1px solid #1e3a5f; border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero">
  <h1>🪐 Exoplanet Transit Detector</h1>
  <p>Bharatiya Antariksh Hackathon 2026 &nbsp;·&nbsp; Problem Statement 07 &nbsp;·&nbsp; AI-based light curve analysis pipeline</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar — inputs
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    target_id = st.text_input("Target ID", value="KIC 11904151",
                               help="KIC, TIC, or Kepler ID — e.g. KIC 11904151 (Kepler-10)")
    mission   = st.selectbox("Mission", ["Kepler", "K2", "TESS"])

    st.divider()
    st.markdown("### Stellar Parameters")
    star_r = st.number_input("Stellar Radius [R☉]", value=1.065, step=0.01)
    star_m = st.number_input("Stellar Mass [M☉]",   value=0.895, step=0.01)

    st.divider()
    st.markdown("### Analysis Options")
    skip_fap   = st.checkbox("Skip bootstrap FAP (faster)", value=True)
    n_fap      = st.slider("FAP bootstrap trials", 100, 1000, 200, 100, disabled=skip_fap)
    save_plots = st.checkbox("Save plots to disk", value=True)

    st.divider()
    run_btn = st.button("🚀 Run Pipeline", type="primary")

# ---------------------------------------------------------------------------
# Quick-reference known planets
# ---------------------------------------------------------------------------
KNOWN = {
    "KIC 11904151": {"period": 0.8375243, "depth_ppm": 1470.0, "duration_h": 1.811,
                      "note": "Kepler-10b — confirmed hot rocky super-Earth"},
}

with st.expander("📚 Quick-reference: Known planet targets"):
    for tid, info in KNOWN.items():
        st.markdown(
            f"**{tid}** — {info['note']}  \n"
            f"Period: {info['period']} d | Depth: {info['depth_ppm']} ppm | Duration: {info['duration_h']} h"
        )

# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------
if run_btn:
    with st.spinner("Running pipeline — this may take a few minutes …"):
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from pipeline import run_pipeline
            result = run_pipeline(
                target_id        = target_id.strip(),
                mission          = mission,
                n_fap_trials     = n_fap,
                skip_fap         = skip_fap,
                star_radius_rsun = star_r,
                star_mass_msun   = star_m,
                save_plots       = save_plots,
                rng_seed         = 42,
            )
            st.session_state["result"]    = result
            st.session_state["target_id"] = target_id.strip()
        except Exception as exc:
            st.error(f"Pipeline error: {exc}")
            st.exception(exc)
            st.stop()

# ---------------------------------------------------------------------------
# Display results (if available)
# ---------------------------------------------------------------------------
result: dict = st.session_state.get("result", {})

if result:
    st.markdown("---")
    st.markdown("## 📊 Results")

    # --- Classification verdict banner ---
    clf    = result.get("classification", "unknown")
    conf   = result.get("classification_confidence", 0.0) * 100
    snr    = result.get("snr", float("nan"))
    fap    = result.get("false_alarm_probability", float("nan"))

    verdict_class = (
        "verdict-planet" if "planet" in clf else
        "verdict-binary" if "binary" in clf or "false" in clf else
        "verdict-noise"
    )
    verdict_emoji = "✅" if "planet" in clf else "⚠️" if "binary" in clf or "false" in clf else "❌"

    st.markdown(f"""
    <div class="{verdict_class}" style="margin-bottom:1.5rem">
      <h3 style="margin:0">{verdict_emoji} {clf.replace("_", " ").title()}</h3>
      <p style="margin:0.3rem 0 0 0; opacity:0.75">Confidence: {conf:.1f}% &nbsp;|&nbsp; {result.get('vetting_verdict','')}</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Key metrics grid ---
    col1, col2, col3, col4 = st.columns(4)
    def _mc(col, label, value, sub=""):
        col.markdown(f"""
        <div class="metric-card">
          <div class="label">{label}</div>
          <div class="value">{value}</div>
          <div class="sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    _mc(col1, "Period", f"{result['period_days']:.5f} d", f"±{result.get('period_uncertainty',0):.1e} d")
    _mc(col2, "Depth",  f"{result['depth_pct']:.4f} %",  f"±{result.get('depth_uncertainty_pct',0):.1e} %")
    _mc(col3, "Duration", f"{result['duration_hours']:.3f} h", "transit duration")
    _mc(col4, "SNR", f"{snr:.1f}" if np.isfinite(snr) else "—", f"FAP={fap:.4f}" if np.isfinite(fap) else "FAP not computed")

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Two-column layout: vetting + probabilities ---
    c_left, c_right = st.columns(2)

    with c_left:
        st.markdown("### 🔍 Vetting Flags")
        vet = result.get("vetting", {})
        def _flag(name, passed, pass_msg, fail_msg):
            icon  = '<span class="vet-pass">✔</span>' if passed else '<span class="vet-fail">✘</span>'
            label = pass_msg if passed else fail_msg
            st.markdown(f"{icon} **{name}** — {label}", unsafe_allow_html=True)

        _flag("Odd-Even Test",
              vet.get("odd_even_consistent", True),
              "Depths consistent (planet-like)",
              "Depths differ (possible EB!)")
        _flag("Secondary Eclipse",
              not vet.get("secondary_eclipse_detected", False),
              "No secondary eclipse",
              "Secondary eclipse detected (possible EB!)")
        _flag("Centroid Shift",
              not vet.get("centroid_shift_detected", False),
              "No centroid shift",
              "Centroid shifted (possible blend!)")

        st.markdown(f"**Vetting score:** {result.get('vetting_score', '?')} / 5")

    with c_right:
        st.markdown("### 🤖 Class Probabilities")
        probs = result.get("class_probabilities", {})
        label_map = {"PC": "Planet Candidate", "AFP": "Eclipsing Binary / FP", "NTP": "Noise"}
        colors = {"PC": "#4fc3f7", "AFP": "#ef9a9a", "NTP": "#b0bec5"}
        for k, v in sorted(probs.items(), key=lambda x: -x[1]):
            bar_pct = int(v * 100)
            label   = label_map.get(k, k)
            color   = colors.get(k, "#78909c")
            st.markdown(f"**{label}** — {v*100:.1f}%")
            st.markdown(
                f'<div style="background:#1e2a3a;border-radius:6px;height:12px;margin-bottom:8px">'
                f'<div style="background:{color};width:{bar_pct}%;height:100%;border-radius:6px"></div>'
                f'</div>', unsafe_allow_html=True
            )

    # --- Known-value comparison ---
    if "known_value_comparison" in result:
        st.markdown("### 📐 Recovery vs. Published Values")
        kvc = result["known_value_comparison"]
        cols = st.columns(3)
        for col, (pub_k, rec_k, err_k, unit) in zip(cols, [
            ("published_period_d",   "recovered_period_d",   "period_error_pct",   "d"),
            ("published_depth_ppm",  "recovered_depth_ppm",  "depth_error_pct",    "ppm"),
            ("published_duration_h", "recovered_duration_h", "duration_error_pct", "h"),
        ]):
            label = pub_k.split("_")[1].title()
            err   = kvc[err_k]
            status = "🟢" if err < 5 else "🟡" if err < 15 else "🔴"
            col.metric(
                label=f"{status} {label}",
                value=f"{kvc[rec_k]} {unit}",
                delta=f"{err:.1f}% vs published",
            )

    # --- Plots ---
    PLOTS_DIR = Path(__file__).parent / "plots"
    tag = result.get("target_id", "target").replace(" ", "_")
    plot_files = {
        "Detrended Light Curve": PLOTS_DIR / f"{tag}_detrending.png",
        "BLS Periodogram":       PLOTS_DIR / f"{tag}_periodogram.png",
        "Phase-Folded Transit":  PLOTS_DIR / f"{tag}_phasefold.png",
        "Transit Model Fit":     PLOTS_DIR / f"{tag}_transit_fit.png",
        "Vetting Summary":       PLOTS_DIR / f"{tag}_vetting.png",
        "FAP Distribution":      PLOTS_DIR / f"{tag}_fap_distribution.png",
    }

    available = {k: v for k, v in plot_files.items() if v.exists()}
    if available:
        st.markdown("### 📈 Plots")
        names = list(available.keys())
        tab_objects = st.tabs(names)
        for tab, name in zip(tab_objects, names):
            with tab:
                st.image(str(available[name]), use_column_width=True)

    # --- Raw JSON ---
    with st.expander("📄 Raw JSON result"):
        st.json(result)

    # --- Download button ---
    json_str = json.dumps(result, indent=2, default=str)
    st.download_button(
        "⬇️  Download result JSON",
        data=json_str,
        file_name=f"{tag}_result.json",
        mime="application/json",
    )

else:
    # Placeholder state
    st.markdown("""
    <div style="text-align:center; padding:4rem; opacity:0.4;">
      <div style="font-size:4rem">🔭</div>
      <p style="font-size:1.1rem">Enter a target ID in the sidebar and click <strong>Run Pipeline</strong></p>
    </div>
    """, unsafe_allow_html=True)

    # Show pre-existing results if any
    RESULTS_DIR = Path(__file__).parent / "results"
    prev = sorted(RESULTS_DIR.glob("*.json"))
    if prev:
        st.markdown("### 🗂️ Previous results")
        for p in prev:
            with st.expander(p.stem):
                st.json(json.loads(p.read_text()))
