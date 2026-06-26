# PS-07 Code Review — File-by-File Bug Report (Verified)

> Methodology: every numeric claim below was independently re-derived (Kepler's
> 3rd law, (Rp/Rs)^2 depth formula) before being called a "bug" — nothing here
> is asserted without a check. Confidence level is stated for each finding.

---

## EXECUTIVE SUMMARY — Ranked by Severity

| # | Severity | File(s) | One-line summary |
|---|---|---|---|
| 1 | 🔴 CRITICAL (confirmed) | `batman_wrapper.py` + `characterize.py` | Fallback model NEVER receives the real time array — this is the direct cause of `Depth=0.0±nan` |
| 2 | 🔴 CRITICAL (confirmed) | `pipeline.py`, `streamlit_app.py` | "Known" reference depth = 1470 ppm is wrong; physics says ~152 ppm (matches `identify.py`'s own number) — 3 files disagree with each other |
| 3 | 🟠 SIGNIFICANT (confirmed) | `detrend.py` | Max-transit-duration physics constant is wrong by ~7.3x |
| 4 | 🟡 MODERATE (confirmed) | `classify.py` | Classifier fed raw/biased BLS depth+duration instead of the refined fit values it already has access to |
| 5 | 🟡 MODERATE (likely) | `classify.py` | `tce_impact` feature may hold the wrong physical quantity vs. how the (unseen) training script defines it |
| 6 | 🟡 MODERATE (conceptual) | `vet.py` | "Centroid shift" test does not test centroid/spatial position at all — it's a time-symmetry check |
| 7 | 🟢 MINOR (confirmed) | `vet.py` | Odd/even transit labels are swapped (cosmetic, doesn't break the statistic) |
| 8 | 🟢 MINOR (confirmed) | `streamlit_app.py` | Looks for `_transit_fit.png`, actual file is `_transit_model.png` — plot tab silently never shows |
| 9 | 🟢 MINOR | `pipeline.py` | `odd_even_consistent` flag treats "inconclusive" the same as "pass" |
| 10 | ⚪ MISSING FILE | `train_classifier.py` | Not provided — cannot verify data leakage / real metrics, the single biggest ML-honesty risk |

---

## File 1: `data_loader.py` — Phases 1-3 (Setup, Acquisition, Preprocessing)

**Verdict: structurally sound, no critical bugs.**

- Iterative sigma-clipping (median + std, 3 passes) is logically correct.
- `quality == 0` filtering is a defensible, conservative choice — it under-uses
  the full bitmask semantics described in the reference doc (Part B2), but it
  errs on the safe side (only removes data that's flagged clean), so it's not
  wrong, just stricter than necessary.
- **Note:** confirm `download_lightcurve()` isn't silently mixing long- and
  short-cadence data for the same target across quarters — `lightkurve`'s
  `.stitch()` can produce duplicate/overlapping timestamps if both cadence
  types exist and `exptime` isn't pinned. Add an `assert np.all(np.diff(time) > 0)`
  right after stitching to catch this early.

---

## File 2: `detrend.py` — Phase 4 (Detrending)

**Verdict: the gap-segmentation fix (from the earlier review) is correctly
implemented. ✅ But a separate, new bug was found in the physics.**

### 🟠 Bug: `estimate_max_transit_duration()` constant is wrong by ~7.3x

The function computes:
```python
T_max_days = 0.0104 * period_days ** (1/3)
```
**Verification (Kepler's 3rd law derivation):**
```
T_dur = (P/pi) * (R_star/a),  a^3 = G*M_star*P^2/(4*pi^2)
=> T_dur = (R_star/pi) * (4*pi^2/(G*M_star))^(1/3) * P^(1/3)
```
Plugging in Sun-like values (R_star=R_sun, M_star=M_sun) and converting units
gives a coefficient of **≈0.0758** (not 0.0104). Sanity check against a known
real number: for P=365.25 days (Earth-Sun), this formula correctly predicts a
**~13-hour** central transit duration — the textbook value. The code's
constant predicts only **~1.8 hours** for the same case — about 7.3x too small.

**Why this matters:** this "max duration" feeds directly into the
`window = 3 x max_duration` safety margin (Part B3). If the estimate is 7x too
small, the safety margin can end up *shorter* than the real transit for
longer-period systems — meaning the filter can eat the very signal it was
supposed to protect, for exactly the systems the margin was designed to guard.

**Cross-file consistency check:** `vet.py`'s `test_duration_period_consistency()`
independently implements the *correct* formula (`T_max = P*R_star/(pi*a)`,
confirmed correct). **The codebase currently has two different formulas for
the same physical quantity, in two files, that disagree by ~7x.**

**Fix:**
```python
T_max_days = 0.0758 * period_days ** (1/3)   # or, better: import and reuse
                                              # vet.py's verified formula here
                                              # instead of a second hardcoded one
```

---

## File 3: `identify.py` — Phase 5 (BLS Identification)

**Verdict: solid implementation.** Correctly inverts flux for BLS, checks for
2x/0.5x period aliases, builds a sensible period/duration grid.

- `KNOWN_PARAMS["KIC 11904151"]`: `period=0.83749070 d`, `depth_ppm=152.0`,
  `duration_h=1.811`. **This depth value is the correct one** — see File 8
  below for the independent verification.
- The large depth/duration discrepancy seen in the actual screenshot
  (39 ppm / 0.96h recovered vs. 152 ppm / 1.811h published) is **expected
  algorithmic behavior, not a bug**: with Kepler long-cadence (29.4 min) and a
  ~1.8-hour transit, only ~3-4 raw points fall inside any single transit. A
  box-model fit (BLS) on marginally-resolved transits like this systematically
  **dilutes depth and underestimates duration** — this is precisely why Phase
  6 (the physical transit-model fit) exists, and precisely why its current
  failure (File 4) is the most urgent thing to fix — once it works, depth/duration
  should land much closer to the published values.

---

## File 4: `batman_wrapper.py` + `characterize.py` — Phase 6 (Characterization)

### 🔴 CRITICAL BUG (CONFIRMED): the fallback model never receives the real time array

This is the direct, traceable root cause of `Depth = 0.0 ± nan ppm` from your
screenshot.

**The chain of the bug:**

1. In `characterize.py`'s `_transit_residual()` (the function lmfit calls on
   every optimizer iteration):
   ```python
   model_obj = make_batman_model(bp, t_abs)
   model_flux = eval_model(model_obj, bp) * v["baseline"]
   ```
   `t_abs` (the real, correct time array) is passed into `make_batman_model`,
   but **`eval_model()` is never given `t_abs`** — its signature is
   `eval_model(model, params)`, with no time argument at all.

2. In `batman_wrapper.py`, when the real `batman` package is unavailable, the
   fallback path inside `eval_model()` does:
   ```python
   model.params = p_dict
   return model.light_curve(model.params.get("time", np.array([0.0])))
   ```
   `p_dict` never contains a `"time"` key (confirmed by reading
   `make_batman_model()`'s fallback branch — it only sets `t0, per, rp, a, inc,
   u, limb_dark`). So `.get("time", np.array([0.0]))` **always** falls through
   to the default — a single-element array `[0.0]`.

3. The model is therefore evaluated at exactly one point (t=0) instead of
   across the real ~hundreds/thousands of data points. NumPy then
   **silently broadcasts** this single value against the full `flux` array in
   the residual calculation (`flux - model_flux`) — this does NOT raise a
   shape error, it just subtracts the same constant from every point.

4. With a residual that is identical to "data minus a single constant," the
   optimizer's best move to minimize chi-square is to push that constant
   toward the data's mean (~1.0) — which physically means **shrinking the
   transit depth toward zero**. This exactly matches `Rp/Rs = 0.0000` and
   `Depth = 0.0`. The uncertainty comes out as `nan` because the fit has
   driven a bounded parameter (`rp`) to its lower limit, where the
   covariance matrix becomes singular.

**This bug fires only when the real `batman` package is not actually
importable in the running environment** — which lines up with the
"Cannot find module `batman`" message seen in the Problems panel. **Confirm
with `pip list` inside the actual `.venv`** — if `batman-package` truly isn't
installed there, this fallback path is exactly what's running.

**The fix (two changes):**
```python
# batman_wrapper.py — give eval_model the real time array
def eval_model(model, params: TransitParams, time: np.ndarray) -> np.ndarray:
    if BATMAN_AVAILABLE:
        ...
        return model.light_curve(bp)
    else:
        p_dict = {...}
        model.params = p_dict
        return model.light_curve(time)        # <- use the real array, not a dict lookup

# characterize.py — pass t_abs through at both call sites
model_flux = eval_model(model_obj, bp, t_abs) * v["baseline"]   # in _transit_residual()
model_flux = eval_model(model_obj, bp, t_abs) * v["baseline"]   # in compute_model_curve()
```

### 🟡 Secondary issue: unverified "<0.1% error" claim on the limb-darkening approximation

The fallback model's docstring claims its first-order limb-darkening
correction is accurate to "<0.1%". Real quadratic limb-darkening changes the
*shape* of ingress/egress, not just a uniform scaling of depth — a constant
correction factor cannot capture that shape change. This specific number
looks asserted rather than tested. **Recommendation:** either remove the
specific number, or actually validate it (compare this approximation's output
against real `batman` output on a grid of test cases) before keeping the claim.

### 🟢 Minor: fragile `and/or` idiom
```python
dur_in_phase = a_val and (1.0 / (np.pi * a_val)) or 0.05
```
If `a_val` is ever `nan` (truthy in Python!), this silently returns `nan`
instead of the intended `0.05` fallback — the classic Python `x and y or z`
pitfall. Use an explicit `if/else`.

### 🟢 Disclosed, lower-priority gap
`duration_err` / `ingress_err` are permanently set to `nan` (explicitly
commented as skipped, not hidden) — fine as an honestly-disclosed limitation,
but flag it in your README per the original prompt's instructions.

---

## File 5: `vet.py` — Phase 7 (Vetting)

**Verdict: mostly strong; one conceptual issue worth flagging clearly, one
cosmetic bug.**

### 🟡 Conceptual issue: `test_centroid_shift()` does not test centroid/spatial position

The function computes a **flux-weighted mean PHASE** (a time-domain quantity)
in- and out-of-transit, and calls the difference a "centroid shift." A real
centroid check (Part B6 of the reference doc) requires the flux-weighted
**spatial (pixel x/y) position** from the Target Pixel File — a fundamentally
different measurement. This proxy is mathematically closer to a transit
"time-symmetry" check than to blending detection, and a real eclipsing-binary
blend would not necessarily trigger it at all.

This is disclosed as a "simplified proxy" in the docstring, which is good
practice — but given that the official PS-07 problem statement explicitly
names "stellar blending" as a contamination source to address, this disclosure
should be made more prominent (e.g., explicitly reporting
`"centroid_shift_detected": "not_tested (proxy only, no pixel data used)"`
rather than a clean True/False) so it cannot be mistaken for a real blending
check by a reader skimming the output.

**Recommended fix (if time allows):** implement the real version using
`lightkurve`'s Target Pixel File access (Part B6's exact formula) for targets
where pixel data is available; fall back to "not_tested" otherwise — not to
this proxy's True/False.

### 🟢 Minor, confirmed: odd/even labels are swapped
```python
odd_flux  = flux[(in_transit) & (transit_num % 2 == 0)]   # this is actually EVEN
even_flux = flux[(in_transit) & (transit_num % 2 == 1)]   # this is actually ODD
```
The statistical test itself is symmetric and still valid (it doesn't matter
which group is "labeled" odd vs. even for the |Δ|/σ comparison), but the
printed/plotted labels are backwards relative to their names. Swap the two
conditions so the labels match reality.

### ✅ Good: `test_duration_period_consistency()` — correct physics, correct,
realistic Kepler-10 stellar parameters (R≈1.065 R_sun, M≈0.895 M_sun, both in
the right range for the published literature values).

### ✅ Good: `test_transit_shape()` — BIC-based trapezoid-vs-V-shape comparison
is a legitimate, standard approach; correctly implemented.

---

## File 6: `significance.py` — Phase 8 (Statistical Significance)

**Verdict: the best-implemented file in the codebase.** Explicitly follows the
reference doc's Part B7 (cited directly in comments), uses circular-shift
bootstrap (correct choice over naive shuffling for preserving red-noise
structure), robust MAD-based sigma, and correctly warns when `n_trials < 1000`.

- 🟢 Minor: the function's own top docstring says "shuffle the flux values"
  while the actual implementation does a circular roll (the better,
  intentional choice per the inline comment a few lines below) — just a stale
  docstring, update the wording.
- 🟢 Minor: both the `significance.py` and `pipeline.py` CLIs default
  `n_fap_trials` to 200, which the code's own logged warning says is
  "unreliable" (<1000). Change the CLI default to at least 1000, or make the
  quick/unreliable mode something the user must opt into explicitly.

---

## File 7: `classify.py` — Phase 9-10 (Classification)

### 🟡 Bug: classifier is fed the less-accurate depth/duration

```python
depth_ppm  = float(best_signal.get("depth", 0.0)) * 1e6     # raw BLS box estimate
duration_h = float(best_signal.get("duration", 0.0)) * 24.0 # raw BLS box estimate
...
rp_rs = float(fit_params.get("rp_val", np.nan))              # refined batman/lmfit fit
```
`fit_params` (the Phase 6 output) is passed into this function and used for
`rp_rs`, but **not** for `depth_ppm`/`duration_h` — even though Phase 6 exists
specifically to produce more accurate depth/duration than BLS's raw box-fit
(see File 3's note on box-model dilution bias). Once File 4's bug is fixed,
this function should source depth and duration from `fit_params` too, for
consistency and accuracy.

### 🟡 Likely bug: `tce_impact` feature may not hold what the model was trained on

```python
"tce_impact",   # impact parameter proxy (Rp/Rs)
```
`FEATURE_NAMES` deliberately mirrors the real NASA Kepler TCE catalog's column
names (a good, authentic choice) — but in the **real** TCE catalog,
`tce_impact` is the orbital **impact parameter b** (`b = (a/Rs)*cos(i)`,
typically 0-1+), a different physical quantity from `Rp/Rs` (typically
0.01-0.2). If `train_classifier.py` trained on the real catalog's `tce_impact`
column, then feeding `Rp/Rs` into that same feature slot at inference time
silently feeds the model out-of-distribution values for that feature — every
prediction would be subtly degraded without any visible error.
**This cannot be fully confirmed without seeing `train_classifier.py`** (not
provided) — check what that script actually used for this column, and either
rename this slot to whatever it truly represents, or compute a real impact
parameter (`b = a_rs_val * cos(inc_val)` — both available from
`fit_params` once File 4 is fixed) instead of reusing Rp/Rs.

### 🟢 Minor: `prad_earth` assumes a Sun-like star
```python
prad_earth = rp_rs * 109.0 if np.isfinite(rp_rs) else np.nan
```
This implicitly assumes `R_star = 1 R_sun`. `vet.py` already carries the real
value for this exact target (`1.065 R_sun`) — thread that value through here
instead of the hardcoded assumption for a more accurate planet-radius feature.

### ⚪ Missing file: `train_classifier.py`
Not included in this batch. This is the file most likely to contain the
data-leakage and "too-good-to-be-true accuracy" risks described in Part B8 of
the reference doc. **Please share it for review before trusting any reported
classifier accuracy/precision/recall numbers.**

---

## File 8: `pipeline.py` — Phase 11 (Orchestration & Output)

### 🔴 CRITICAL BUG (CONFIRMED): wrong "known" depth value, contradicts another file

```python
KNOWN_PARAMS = {
    "KIC 11904151": {"period": 0.8375243, "depth_ppm": 1470.0, "duration_h": 1.811},
}
```
Compare to `identify.py`'s own `KNOWN_PARAMS` for the **same target**:
`depth_ppm = 152.0`.

**Independent verification (which one is correct):** Kepler-10b's published
radius is ≈1.47 Earth radii; the host star's radius (already used elsewhere in
this codebase, in `vet.py`) is ≈1.065 R_sun.
```
depth = (Rp/Rs)^2
Rp = 1.47 R_earth = 1.47 x 6371 km = 9365 km
Rs = 1.065 R_sun  = 1.065 x 696000 km = 741,240 km
depth = (9365 / 741240)^2 = (0.01263)^2 = 1.595e-4 = ~160 ppm
```
This confirms **`identify.py`'s value (152 ppm) is correct** (within normal
literature rounding), and **`pipeline.py`'s value (1470 ppm) is wrong by
roughly 9-10x**. Notably, `1470` looks suspiciously like Kepler-10b's radius
figure "1.47" (R_earth) with a misplaced decimal/extra zero — consistent with
a copy-paste/unit mix-up rather than a deliberate value.

**Impact:** any "known-value comparison" / `depth_error_pct` printed by this
pipeline is currently being checked against a number that is itself wrong by
~10x — which could make a genuinely good fit look like it has a huge error,
or (worse) make a genuinely bad fit look closer to "correct" than it is,
purely by coincidence.

**Fix:** change `depth_ppm` to `152.0` (or re-derive from the latest NASA
Exoplanet Archive entry) — and see the consolidation recommendation below so
this can't drift out of sync again.

### 🟢 Minor: `odd_even_consistent` flag conflates "inconclusive" with "pass"
```python
"odd_even_consistent": vet_tests.get("odd_even", {}).get("score", 0) >= 0,
```
`vet.py`'s scoring convention is `+1=pass, 0=inconclusive, -1=fail`. Using
`>= 0` means an inconclusive result (e.g., too few transits to split) reports
as `True` (consistent) — overstating what was actually established. Use
`== 1` if you want this flag to mean "positively confirmed," and report
inconclusive results as a separate third state rather than folding them into
"pass."

---

## File 9: `streamlit_app.py` — Phase 12 (Demo)

### 🟢 Confirmed bug: plot filename mismatch
```python
"Transit Model Fit": PLOTS_DIR / f"{tag}_transit_fit.png",
```
`characterize.py` actually saves this plot as `{tag}_transit_model.png`
(confirmed directly from your screenshot's tab title). Because the app only
displays a tab if the file `.exists()`, the "Transit Model Fit" tab will
**silently never appear** in the demo — no error, it just won't be there.
Fix the filename string to match.

### 🔴 Same critical data bug, now duplicated a third time
```python
KNOWN = {
    "KIC 11904151": {"period": 0.8375243, "depth_ppm": 1470.0, "duration_h": 1.811, ...},
}
```
Same wrong `1470.0` value as `pipeline.py` (File 8) — now duplicated in a
**third** location. See the consolidation fix below.

### ✅ Good practice: errors are shown, not swallowed
```python
except Exception as exc:
    st.error(f"Pipeline error: {exc}")
    st.exception(exc)
    st.stop()
```
This is exactly the right pattern — contrast with the silent `except` inside
`batman_wrapper.py` (File 4). Good instinct here; apply the same visibility to
every other `try/except` in the codebase.

---

## CROSS-FILE CONSISTENCY ISSUES (root cause of bugs #2 and #3)

The codebase currently has the same "ground truth" data duplicated in three
places, and the same physics formula duplicated in two places — and in both
cases, the copies disagree with each other:

| Quantity | Where it lives | Values found |
|---|---|---|
| Kepler-10b known depth_ppm | `identify.py`, `pipeline.py`, `streamlit_app.py` | 152.0 vs **1470.0** vs **1470.0** (two wrong copies) |
| Max transit duration formula | `detrend.py`, `vet.py` | Two different constants, disagree by ~7.3x |

**Structural fix (do this, not just the individual number/constant fixes
above):** create a single shared module, e.g. `reference_targets.py` and
`physics_utils.py`, that every other file imports from. Right now, fixing the
number in one file does not fix it in the other two — exactly what happened
here. This is the same root cause behind two of your three most serious bugs.

---

## RECOMMENDED FIX ORDER

1. **`batman_wrapper.py` / `characterize.py`** — thread `time` through
   `eval_model()` (fixes the `Depth=0.0±nan` bug). Re-run on KIC 11904151 and
   confirm depth/duration land much closer to 152 ppm / 1.811 h.
2. **Fix `pipeline.py` + `streamlit_app.py`'s known depth value** to 152.0 ppm,
   and consolidate all three `KNOWN_PARAMS` copies into one shared file.
3. **Fix `detrend.py`'s duration-estimate constant** (0.0104 -> 0.0758, or
   better, import `vet.py`'s already-correct formula instead of keeping two).
4. **Fix `streamlit_app.py`'s plot filename** (`_transit_fit.png` ->
   `_transit_model.png`).
5. Re-run the full pipeline end-to-end and re-check Part D's 10-point test
   checklist from the technical reference doc before trusting any output.
6. Share `train_classifier.py` for review before trusting classifier metrics.
7. Address the `vet.py` odd/even label swap and centroid-test honesty note —
   lower priority, doesn't block a working demo, but matters for the "nothing
   fake" standard before final submission.
