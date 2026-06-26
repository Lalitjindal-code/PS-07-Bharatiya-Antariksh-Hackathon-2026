"""
train_classifier.py  —  Train and evaluate the RandomForestClassifier on the
Kepler TCE dataset (tce_data.csv).  Saves model + imputer to models/.

Data source: Kepler TCE cumulative table, filtered to PC / AFP / NTP labels.
  - tce_data.csv was downloaded from the NASA Exoplanet Archive TCE table
    (https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=q1_q17_dr25_tce)
    and filtered for rows where av_training_set in {PC, AFP, NTP}.

Run:
    python train_classifier.py
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

DATA_PATH   = Path(__file__).parent / "tce_data.csv"
MODELS_DIR  = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

FEATURE_COLS = [
    "tce_depth",
    "tce_duration",
    "tce_period",
    "tce_model_snr",
    "tce_bin_oedp_stat",
    "tce_impact",
    "tce_prad",
]
LABEL_COL = "av_training_set"
KEEP_LABELS = {"PC", "AFP", "NTP"}


# ---------------------------------------------------------------------------
def load_data() -> tuple[np.ndarray, np.ndarray, list]:
    df = pd.read_csv(DATA_PATH)
    logger.info("Raw dataset: %d rows, %d cols", *df.shape)

    # Drop unknown labels
    df = df[df[LABEL_COL].isin(KEEP_LABELS)].copy()
    logger.info("After label filter: %d rows", len(df))
    logger.info("Class distribution:\n%s", df[LABEL_COL].value_counts().to_string())

    X = df[FEATURE_COLS].values.astype(float)
    y = df[LABEL_COL].values
    return X, y, FEATURE_COLS


# ---------------------------------------------------------------------------
def train_and_evaluate() -> None:
    X, y, feat_names = load_data()

    # Stratified 80/20 split FIRST — imputer must not see test data
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )
    logger.info("Train: %d | Test: %d", len(y_train), len(y_test))

    # Impute missing values on train only, then transform test with same statistics
    imp = SimpleImputer(strategy="median")
    X_train = imp.fit_transform(X_train_raw)   # fit only on train
    X_test  = imp.transform(X_test_raw)         # transform test (no fit)

    # SMOTE oversampling to address class imbalance (planets are rare)
    sm = SMOTE(random_state=RANDOM_SEED)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    logger.info(
        "After SMOTE: %d samples | classes: %s",
        len(y_res),
        {c: int((y_res == c).sum()) for c in np.unique(y_res)},
    )

    # Random Forest — 500 trees, balanced class weight as extra safety net
    clf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    clf.fit(X_res, y_res)
    logger.info("Training complete.")

    # --- Evaluation on held-out test set ---
    y_pred = clf.predict(X_test)

    print("\n" + "=" * 60)
    print("  CLASSIFICATION REPORT (held-out 20% test set)")
    print("=" * 60)
    print(classification_report(y_test, y_pred, digits=4))

    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    print("Confusion matrix (rows=true, cols=pred):")
    header = "        " + "  ".join(f"{c:>6}" for c in clf.classes_)
    print(header)
    for label, row in zip(clf.classes_, cm):
        print(f"  {label:>4}  " + "  ".join(f"{v:>6}" for v in row))

    macro_f1 = f1_score(y_test, y_pred, average="macro")
    print(f"\nMacro F1: {macro_f1:.4f}")

    # Data-leakage sanity check
    train_f1 = f1_score(clf.predict(X_train), y_train, average="macro")
    logger.info("Train macro F1 = %.4f | Test macro F1 = %.4f", train_f1, macro_f1)
    if train_f1 - macro_f1 > 0.15:
        logger.warning(
            "Train/test F1 gap (%.3f) is large — possible overfitting or leakage. "
            "Inspect feature distributions before reporting.",
            train_f1 - macro_f1,
        )
    else:
        logger.info("Train/test F1 gap (%.3f) looks reasonable — no obvious leakage.", train_f1 - macro_f1)

    # Feature importances
    importances = clf.feature_importances_
    print("\nFeature importances:")
    for name, imp_val in sorted(zip(feat_names, importances), key=lambda x: -x[1]):
        bar = "#" * int(imp_val * 40)
        print(f"  {name:<22} {imp_val:.4f}  {bar}")

    # --- Save ---
    joblib.dump(clf, MODELS_DIR / "rf_classifier.joblib")
    joblib.dump(imp, MODELS_DIR / "imputer.joblib")
    logger.info("Model + imputer saved to %s", MODELS_DIR)

    print("\n" + "=" * 60)
    print("  Training complete. Model saved to models/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    train_and_evaluate()
