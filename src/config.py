"""
Central configuration for the Storm Severity Mining pipeline.
All paths, hyperparameters, and constants live here for reproducibility.
"""

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW     = PROJECT_ROOT / "data" / "raw"
DATA_PROC    = PROJECT_ROOT / "data" / "processed"
OUT_FIG      = PROJECT_ROOT / "output" / "figures"
OUT_MODEL    = PROJECT_ROOT / "output" / "models"
OUT_RESULT   = PROJECT_ROOT / "output" / "results"

for d in [DATA_RAW, DATA_PROC, OUT_FIG, OUT_MODEL, OUT_RESULT]:
    d.mkdir(parents=True, exist_ok=True)

# ── Data parameters ────────────────────────────────────────────────────────
NOAA_BASE_URL = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"
YEAR_RANGE    = range(1996, 2025)  # 1996-2024 inclusive (stable data)
RANDOM_SEED   = 42

# ── Preprocessing ──────────────────────────────────────────────────────────
DAMAGE_TIERS = {
    "quantile_probs": [0.25, 0.50, 0.75],
    "labels": ["Low", "Medium", "High", "Catastrophic"],
}

SEASON_MAP = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Fall", 10: "Fall", 11: "Fall",
}

# ── Classification ─────────────────────────────────────────────────────────
CV_FOLDS      = 10
SMOTE_STRATEGY = "auto"

CLASSIFIER_PARAMS = {
    "decision_tree": {
        "max_depth": [3, 4, 5, 6, 7, 8, 10],
        "min_samples_split": [5, 10, 20, 50],
        "min_samples_leaf": [5, 10, 20, 50],
        "criterion": ["gini", "entropy"],
        "class_weight": ["balanced", None],
    },
    "random_forest": {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [10, 15, 20, None],
        "min_samples_split": [2, 5, 10],
        "max_features": ["sqrt", "log2"],
        "class_weight": ["balanced", "balanced_subsample", None],
    },
    "knn": {
        "n_neighbors": [3, 5, 7, 11, 15, 21],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"],
    },
    "svm": {
        "C": [0.1, 1, 10],
        "gamma": ["scale", "auto"],
        "kernel": ["rbf"],
    },
    "neural_network": {
        "hidden_layers": [64, 32],
        "dropout": [0.3, 0.2],
        "epochs": 50,
        "batch_size": 512,
        "learning_rate": 0.001,
    },
}

# ── Clustering ─────────────────────────────────────────────────────────────
DBSCAN_EPS_RANGE    = [0.003, 0.005, 0.007, 0.01, 0.012, 0.015]  # radians (~19–95 km)
DBSCAN_MINPTS_RANGE = [20, 30, 50, 75, 100]
KMEANS_K_RANGE      = range(2, 16)

# ── Association Mining ─────────────────────────────────────────────────────
APRIORI_MIN_SUPPORT    = 0.01
APRIORI_MIN_CONFIDENCE = 0.3
APRIORI_MIN_LIFT       = 1.0

# ── Outlier Detection ──────────────────────────────────────────────────────
ZSCORE_THRESHOLD       = 3.0
ISO_FOREST_CONTAMINATION = 0.05
LOF_N_NEIGHBORS        = 20
LOF_CONTAMINATION      = 0.05

# ── Autoencoder ────────────────────────────────────────────────────────────
AE_ENCODING_DIM = 6
AE_EPOCHS       = 50
AE_BATCH_SIZE   = 256
AE_ANOMALY_PCTL = 95  # percentile threshold for reconstruction error
