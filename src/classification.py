"""
Classification pipeline (Chapter IX-X).

Predicts damage severity tier (Low / Medium / High / Catastrophic) using:
  1. Decision Tree (with pruning via cost-complexity)
  2. Random Forest (with feature importance)
  3. k-Nearest Neighbors
  4. Support Vector Machine (RBF kernel)
  5. XGBoost gradient boosting
  6. LightGBM gradient boosting
  7. Feedforward Neural Network (PyTorch)

Evaluation methodology (research-grade):
  - Proper 80/20 stratified holdout split before ANY model fitting
  - 10-fold stratified cross-validation on training set only
  - Final metrics reported on held-out test set (never seen during training)
  - SMOTE applied within each CV fold (no leakage)
  - Reports: Accuracy, Macro-F1, Weighted-F1, ROC-AUC, Precision, Recall
  - Visualizations: confusion matrices, multi-class ROC, PR curves,
                    calibration plots, learning curves, feature importance
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import warnings
import joblib
import hashlib

from sklearn.model_selection import (
    StratifiedKFold, RandomizedSearchCV, train_test_split, learning_curve
)
from sklearn.preprocessing import (
    LabelEncoder, StandardScaler, OneHotEncoder, label_binarize
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    f1_score, accuracy_score, precision_score, recall_score,
    roc_curve, precision_recall_curve, average_precision_score,
    brier_score_loss
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from src.config import (
    OUT_FIG, OUT_MODEL, OUT_RESULT, RANDOM_SEED, CV_FOLDS, CLASSIFIER_PARAMS
)
from src.utils import logger, save_figure, save_results

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
np.random.seed(RANDOM_SEED)

# Bump this whenever the feature set or preprocessing logic changes.
# Old caches with a different version are automatically skipped and retrained.
CACHE_VERSION = "v3"


# ── Module-level PyTorch class (must be at module level for pickling) ──────

class _StormNet:
    """Lazy-import wrapper so classification.py doesn't require torch at import."""
    _cls = None

    @classmethod
    def get_class(cls):
        if cls._cls is None:
            import torch.nn as nn

            class StormNet(nn.Module):
                def __init__(self, n_in, n_out):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(n_in, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.3),
                        nn.Linear(256, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.3),
                        nn.Linear(128, 64),  nn.ReLU(), nn.BatchNorm1d(64),  nn.Dropout(0.2),
                        nn.Linear(64, 32),   nn.ReLU(), nn.Dropout(0.1),
                        nn.Linear(32, n_out),
                    )
                def forward(self, x):
                    return self.net(x)
            cls._cls = StormNet
        return cls._cls


# ═══════════════════════════════════════════════════════════════════════════
# Feature Engineering
# ═══════════════════════════════════════════════════════════════════════════

def prepare_classification_data(df):
    """
    Prepare features and target for classification.

    Features (22 total, up from 16):
      Categorical : EVENT_TYPE, SEASON
      Core numeric: DURATION_MIN, LOG_DURATION, HOUR, MONTH, YEAR, DECADE,
                    DAY_OF_WEEK, INJURIES_DIRECT, INJURIES_INDIRECT,
                    DEATHS_DIRECT, DEATHS_INDIRECT, HUMAN_IMPACT
      Spatial     : BEGIN_LAT, BEGIN_LON, IS_COASTAL, PATH_LENGTH_DEG
      Storm type  : MAGNITUDE, HAS_MAGNITUDE, EF_NUMERIC
      Episode     : EPISODE_SIZE
    """
    clf_df = df[df["DAMAGE_CLASS"] != "None"].copy()
    logger.info(f"Classification dataset: {len(clf_df):,} records (excluding no-damage events)")

    # ── Engineer new features ──────────────────────────────────────────────

    # EPISODE_SIZE: how many events share this episode (proxy for storm-system scale)
    if "EPISODE_ID" in clf_df.columns:
        ep_sizes = df["EPISODE_ID"].map(df.groupby("EPISODE_ID").size())
        clf_df["EPISODE_SIZE"] = ep_sizes.reindex(clf_df.index).fillna(1).clip(upper=50)
    else:
        clf_df["EPISODE_SIZE"] = 1

    # DECADE: captures climate-change trend (1990s, 2000s, 2010s, 2020s)
    if "YEAR" in clf_df.columns:
        clf_df["DECADE"] = (clf_df["YEAR"] // 10 * 10).astype(float)

    # IS_COASTAL: simple geographic coastal indicator
    if "BEGIN_LAT" in clf_df.columns and "BEGIN_LON" in clf_df.columns:
        clf_df["IS_COASTAL"] = (
            (clf_df["BEGIN_LON"] > -82) |                                    # East coast
            ((clf_df["BEGIN_LAT"] < 32) & (clf_df["BEGIN_LON"] > -98)) |    # Gulf coast
            (clf_df["BEGIN_LON"] < -117)                                      # West coast
        ).astype(float)

    # LOG_DURATION: log-normalise the heavily right-skewed duration
    if "DURATION_MIN" in clf_df.columns:
        clf_df["LOG_DURATION"] = np.log1p(clf_df["DURATION_MIN"])

    # HAS_MAGNITUDE: binary flag — distinguishes "not applicable" from actual zero
    if "MAGNITUDE" in clf_df.columns:
        clf_df["HAS_MAGNITUDE"] = (clf_df["MAGNITUDE"].notna() & (clf_df["MAGNITUDE"] > 0)).astype(float)

    # EF_NUMERIC: already created in preprocess.py; fallback if absent
    if "EF_NUMERIC" not in clf_df.columns:
        clf_df["EF_NUMERIC"] = np.nan

    # ── Select feature columns ─────────────────────────────────────────────
    feature_cols = [
        # Categorical
        "EVENT_TYPE", "SEASON",
        # Core numeric
        "DURATION_MIN", "LOG_DURATION", "HOUR", "MONTH", "YEAR", "DECADE",
        "DAY_OF_WEEK",
        "INJURIES_DIRECT", "INJURIES_INDIRECT", "DEATHS_DIRECT", "DEATHS_INDIRECT",
        "HUMAN_IMPACT",
        # Spatial
        "BEGIN_LAT", "BEGIN_LON", "IS_COASTAL", "PATH_LENGTH_DEG",
        # Storm-type specific
        "MAGNITUDE", "HAS_MAGNITUDE", "EF_NUMERIC",
        # Episode scale
        "EPISODE_SIZE",
    ]
    available = [c for c in feature_cols if c in clf_df.columns]
    target = "DAMAGE_CLASS"

    subset = clf_df[available + [target]].copy()

    # ── Impute missing numerics ────────────────────────────────────────────
    if "MAGNITUDE" in subset.columns:
        subset["MAGNITUDE"] = subset["MAGNITUDE"].fillna(0)
    if "EF_NUMERIC" in subset.columns:
        subset["EF_NUMERIC"] = subset["EF_NUMERIC"].fillna(-1)   # -1 = "not a tornado"
    if "PATH_LENGTH_DEG" in subset.columns:
        subset["PATH_LENGTH_DEG"] = subset["PATH_LENGTH_DEG"].fillna(0)
    for geo_col in ["BEGIN_LAT", "BEGIN_LON"]:
        if geo_col in subset.columns:
            if "STATE" in clf_df.columns:
                state_medians = clf_df.loc[subset.index].groupby("STATE")[geo_col].transform("median")
                subset[geo_col] = subset[geo_col].fillna(state_medians)
            subset[geo_col] = subset[geo_col].fillna(subset[geo_col].median())
    for col in ["IS_COASTAL", "LOG_DURATION", "HAS_MAGNITUDE", "EPISODE_SIZE", "DECADE"]:
        if col in subset.columns:
            subset[col] = subset[col].fillna(0)

    before = len(subset)
    subset = subset.dropna()
    if before != len(subset):
        logger.info(f"Dropped {before - len(subset):,} remaining NaN rows")
    logger.info(f"Classification records after imputation: {len(subset):,}")

    X = subset[available]
    y = subset[target]

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # Encode target with enforced ordinal order
    le = LabelEncoder()
    le.classes_ = np.array(["Low", "Medium", "High", "Catastrophic"])
    y_encoded = le.transform(y)

    logger.info(f"Features: {len(available)} ({len(cat_cols)} categorical, {len(num_cols)} numeric)")
    logger.info(f"Class distribution:\n{pd.Series(y).value_counts().to_string()}")

    return X, y_encoded, le, cat_cols, num_cols


def build_preprocessor(cat_cols, num_cols):
    """Build sklearn column transformer for mixed data."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                  min_frequency=500, sparse_output=False), cat_cols),
        ],
        remainder="drop",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Model Training Functions
# ═══════════════════════════════════════════════════════════════════════════

def train_decision_tree(X, y, preprocessor, cv):
    logger.info("Training Decision Tree...")
    pipe = ImbPipeline([
        ("preprocess", preprocessor),
        ("smote", SMOTE(random_state=RANDOM_SEED)),
        ("clf", DecisionTreeClassifier(random_state=RANDOM_SEED)),
    ])
    param_dist = {
        "clf__max_depth": CLASSIFIER_PARAMS["decision_tree"]["max_depth"],
        "clf__min_samples_split": CLASSIFIER_PARAMS["decision_tree"]["min_samples_split"],
        "clf__min_samples_leaf": CLASSIFIER_PARAMS["decision_tree"]["min_samples_leaf"],
        "clf__criterion": CLASSIFIER_PARAMS["decision_tree"]["criterion"],
        "clf__class_weight": CLASSIFIER_PARAMS["decision_tree"]["class_weight"],
    }
    search = RandomizedSearchCV(
        pipe, param_dist, n_iter=30, cv=cv, scoring="f1_macro",
        random_state=RANDOM_SEED, n_jobs=4, verbose=0,
    )
    search.fit(X, y)
    logger.info(f"  Best params: {search.best_params_}")
    logger.info(f"  CV Macro-F1: {search.best_score_:.4f}")
    return search.best_estimator_, search


def train_random_forest(X, y, preprocessor, cv):
    logger.info("Training Random Forest...")
    pipe = ImbPipeline([
        ("preprocess", preprocessor),
        ("smote", SMOTE(random_state=RANDOM_SEED)),
        ("clf", RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1)),
    ])
    param_dist = {
        "clf__n_estimators": CLASSIFIER_PARAMS["random_forest"]["n_estimators"],
        "clf__max_depth": CLASSIFIER_PARAMS["random_forest"]["max_depth"],
        "clf__min_samples_split": CLASSIFIER_PARAMS["random_forest"]["min_samples_split"],
        "clf__max_features": CLASSIFIER_PARAMS["random_forest"]["max_features"],
        "clf__class_weight": CLASSIFIER_PARAMS["random_forest"]["class_weight"],
    }
    search = RandomizedSearchCV(
        pipe, param_dist, n_iter=20, cv=cv, scoring="f1_macro",
        random_state=RANDOM_SEED, n_jobs=4, verbose=0,
    )
    search.fit(X, y)
    logger.info(f"  Best params: {search.best_params_}")
    logger.info(f"  CV Macro-F1: {search.best_score_:.4f}")
    return search.best_estimator_, search


def train_knn(X, y, preprocessor, cv):
    logger.info("Training k-NN...")
    # k-NN is O(n²) at predict time — cap at 40k for memory safety on 16 GB
    n_max = 40_000
    if len(X) > n_max:
        logger.info(f"  Subsampling to {n_max} for k-NN training")
        idx = np.random.choice(len(X), n_max, replace=False)
        X_sub, y_sub = X.iloc[idx], y[idx]
    else:
        X_sub, y_sub = X, y

    knn_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    pipe = ImbPipeline([
        ("preprocess", preprocessor),
        ("smote", SMOTE(random_state=RANDOM_SEED, k_neighbors=3)),
        ("clf", KNeighborsClassifier()),
    ])
    param_dist = {
        "clf__n_neighbors": CLASSIFIER_PARAMS["knn"]["n_neighbors"],
        "clf__weights": CLASSIFIER_PARAMS["knn"]["weights"],
        "clf__metric": CLASSIFIER_PARAMS["knn"]["metric"],
    }
    search = RandomizedSearchCV(
        pipe, param_dist, n_iter=8, cv=knn_cv, scoring="f1_macro",
        random_state=RANDOM_SEED, n_jobs=4, verbose=0,
    )
    search.fit(X_sub, y_sub)
    logger.info(f"  Best params: {search.best_params_}")
    logger.info(f"  CV Macro-F1: {search.best_score_:.4f}")
    return search.best_estimator_, search


def train_svm(X, y, preprocessor, cv):
    logger.info("Training SVM (RBF)...")
    # SVM is O(n²) — aggressive cap; 15k is plenty for a strong signal
    n_max = 15_000
    if len(X) > n_max:
        logger.info(f"  Subsampling to {n_max} for SVM training")
        idx = np.random.choice(len(X), n_max, replace=False)
        X_sub, y_sub = X.iloc[idx], y[idx]
    else:
        X_sub, y_sub = X, y

    svm_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
    pipe = ImbPipeline([
        ("preprocess", preprocessor),
        ("smote", SMOTE(random_state=RANDOM_SEED, k_neighbors=3)),
        ("clf", SVC(probability=True, random_state=RANDOM_SEED, cache_size=1000)),
    ])
    param_dist = {
        "clf__C": CLASSIFIER_PARAMS["svm"]["C"],
        "clf__gamma": CLASSIFIER_PARAMS["svm"]["gamma"],
    }
    search = RandomizedSearchCV(
        pipe, param_dist, n_iter=6, cv=svm_cv, scoring="f1_macro",
        random_state=RANDOM_SEED, n_jobs=-1, verbose=0,
    )
    search.fit(X_sub, y_sub)
    logger.info(f"  Best params: {search.best_params_}")
    logger.info(f"  CV Macro-F1: {search.best_score_:.4f}")
    return search.best_estimator_, search


def train_xgboost(X, y, preprocessor, cv):
    try:
        from xgboost import XGBClassifier
    except Exception:
        logger.warning("XGBoost not available, skipping")
        return None, None

    logger.info("Training XGBoost...")
    pipe = ImbPipeline([
        ("preprocess", preprocessor),
        ("smote", SMOTE(random_state=RANDOM_SEED)),
        ("clf", XGBClassifier(
            tree_method="hist", random_state=RANDOM_SEED,
            eval_metric="mlogloss", verbosity=0, n_jobs=-1,
        )),
    ])
    param_dist = {
        "clf__n_estimators": [200, 300, 500],
        "clf__max_depth": [4, 6, 8, 10],
        "clf__learning_rate": [0.01, 0.05, 0.1],
        "clf__subsample": [0.7, 0.8, 0.9],
        "clf__colsample_bytree": [0.7, 0.8, 1.0],
        "clf__min_child_weight": [1, 3, 5],
    }
    search = RandomizedSearchCV(
        pipe, param_dist, n_iter=20, cv=cv, scoring="f1_macro",
        random_state=RANDOM_SEED, n_jobs=4, verbose=0,
    )
    search.fit(X, y)
    logger.info(f"  Best params: {search.best_params_}")
    logger.info(f"  CV Macro-F1: {search.best_score_:.4f}")
    return search.best_estimator_, search


def train_lightgbm(X, y, preprocessor, cv):
    try:
        from lightgbm import LGBMClassifier
    except Exception:
        logger.warning("LightGBM not available, skipping")
        return None, None

    logger.info("Training LightGBM...")
    pipe = ImbPipeline([
        ("preprocess", preprocessor),
        ("smote", SMOTE(random_state=RANDOM_SEED)),
        ("clf", LGBMClassifier(random_state=RANDOM_SEED, verbose=-1, n_jobs=-1)),
    ])
    param_dist = {
        "clf__n_estimators": [200, 300, 500],
        "clf__max_depth": [4, 6, 8, -1],
        "clf__learning_rate": [0.01, 0.05, 0.1],
        "clf__num_leaves": [31, 50, 80, 127],
        "clf__subsample": [0.7, 0.8, 0.9],
        "clf__colsample_bytree": [0.7, 0.8, 1.0],
    }
    search = RandomizedSearchCV(
        pipe, param_dist, n_iter=20, cv=cv, scoring="f1_macro",
        random_state=RANDOM_SEED, n_jobs=4, verbose=0,
    )
    search.fit(X, y)
    logger.info(f"  Best params: {search.best_params_}")
    logger.info(f"  CV Macro-F1: {search.best_score_:.4f}")
    return search.best_estimator_, search


def train_neural_network(X, y, preprocessor, n_classes):
    """Train NN in isolated subprocess to avoid loky/OpenMP deadlocks."""
    logger.info("Training Neural Network...")
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        logger.warning("PyTorch not available, skipping neural network")
        return None, None

    logger.info("  Preprocessing for NN...")
    X_processed = preprocessor.fit_transform(X)
    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()
    X_processed = np.asarray(X_processed, dtype=np.float32)
    logger.info(f"  Preprocessed: {X_processed.shape}, dtype={X_processed.dtype}")

    smote = SMOTE(random_state=RANDOM_SEED)
    X_res, y_res = smote.fit_resample(X_processed, y)
    logger.info(f"  After SMOTE: {X_res.shape[0]:,} samples, {X_res.shape[1]} features")

    from sklearn.model_selection import train_test_split as _tts
    X_tr, X_val, y_tr, y_val = _tts(
        X_res, y_res, test_size=0.2, random_state=RANDOM_SEED, stratify=y_res
    )
    logger.info(f"  Train: {X_tr.shape}, Val: {X_val.shape}")

    data_path   = str(OUT_MODEL / "_nn_train_data.npz")
    result_path = str(OUT_MODEL / "_nn_train_result.pt")
    np.savez(data_path, X_train=X_tr, y_train=y_tr, X_val=X_val, y_val=y_val)

    import subprocess, sys, os
    from pathlib import Path
    script_path = str(Path(__file__).parent / "_nn_train.py")
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
                "OPENBLAS_NUM_THREADS": "4", "VECLIB_MAXIMUM_THREADS": "4"})

    logger.info("  Starting NN training in isolated subprocess...")
    try:
        proc = subprocess.run(
            [sys.executable, script_path, data_path, result_path, str(n_classes)],
            env=env, timeout=600, capture_output=True, text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        for line in proc.stdout.strip().split("\n"):
            if line.strip():
                logger.info(f"  [NN] {line}")
        if proc.returncode != 0:
            logger.warning(f"  NN subprocess failed (exit {proc.returncode})")
            for line in proc.stderr.strip().split("\n")[-10:]:
                logger.warning(f"  [NN stderr] {line}")
            return None, None
    except subprocess.TimeoutExpired:
        logger.warning("  NN training timed out (10 min), skipping")
        return None, None

    result = torch.load(result_path, weights_only=False)
    StormNet = _StormNet.get_class()
    n_features = result["n_features"]
    model = StormNet(n_features, n_classes)
    model.load_state_dict(result["state_dict"])

    train_losses = result["train_losses"]
    val_losses   = result["val_losses"]
    train_accs   = result["train_accs"]
    val_accs     = result["val_accs"]
    val_f1       = result["val_f1"]
    logger.info(f"  NN Validation Macro-F1: {val_f1:.4f}")

    # Training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(train_losses, label="Train")
    axes[0].plot(val_losses,   label="Validation")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss"); axes[0].legend()
    axes[1].plot(train_accs, label="Train")
    axes[1].plot(val_accs,   label="Validation")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training Accuracy"); axes[1].legend()
    fig.suptitle("Neural Network Training Curves", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "15_nn_training_curves", OUT_FIG)
    plt.close(fig)

    nn_result = {
        "model": model,
        "preprocessor": preprocessor,
        "n_features": n_features,
        "n_classes": n_classes,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_f1": val_f1,
    }
    try:
        os.remove(data_path)
    except OSError:
        pass

    return nn_result, {"train_losses": train_losses, "val_losses": val_losses}


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation — proper held-out test set
# ═══════════════════════════════════════════════════════════════════════════

def _predict_nn(nn_dict, X, y=None):
    """Run NN inference via subprocess, return (y_pred, y_proba)."""
    import torch, subprocess, sys, os
    from pathlib import Path

    nn_model_obj  = nn_dict["model"]
    nn_preprocessor = nn_dict["preprocessor"]

    X_proc = nn_preprocessor.transform(X)
    if hasattr(X_proc, "toarray"):
        X_proc = X_proc.toarray()
    X_proc = np.asarray(X_proc, dtype=np.float32)

    eval_data   = str(OUT_MODEL / "_nn_eval_data.npz")
    eval_model  = str(OUT_MODEL / "_nn_eval_model.pt")
    eval_result = str(OUT_MODEL / "_nn_eval_result.npz")

    np.savez(eval_data, X=X_proc, y=y if y is not None else np.zeros(len(X_proc), dtype=int))
    torch.save({
        "state_dict": nn_model_obj.state_dict(),
        "n_features": nn_dict["n_features"],
        "n_classes":  nn_dict["n_classes"],
    }, eval_model)

    eval_script = f'''
import numpy as np, torch, torch.nn as nn
torch.set_num_threads(4)
data = np.load("{eval_data}")
X, y_true = data["X"], data["y"]
info = torch.load("{eval_model}", weights_only=False)

class StormNet(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.3),
            nn.Linear(128, 64),  nn.ReLU(), nn.BatchNorm1d(64),  nn.Dropout(0.2),
            nn.Linear(64, 32),   nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, n_out),
        )
    def forward(self, x):
        return self.net(x)

model = StormNet(info["n_features"], info["n_classes"])
model.load_state_dict(info["state_dict"])
model.eval()
with torch.no_grad():
    out   = model(torch.FloatTensor(X))
    proba = torch.softmax(out, dim=1).numpy()
    preds = out.argmax(1).numpy()
np.savez("{eval_result}", y_pred=preds, y_pred_proba=proba)
print("done", flush=True)
'''
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "4"
    proc = subprocess.run(
        [sys.executable, "-c", eval_script],
        env=env, timeout=120, capture_output=True, text=True,
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    if proc.returncode != 0:
        logger.warning(f"  NN eval subprocess failed: {proc.stderr[-300:]}")
        return None, None

    result_data = np.load(eval_result)
    y_pred      = result_data["y_pred"]
    y_proba     = result_data["y_pred_proba"]

    for f in [eval_data, eval_model, eval_result]:
        try:
            os.remove(f)
        except OSError:
            pass

    return y_pred, y_proba


def evaluate_all_models(models, X_train, X_test, y_train, y_test, le, preprocessor, cv):
    """
    Evaluate all models on the held-out test set.

    Metrics reported:
      - Test Accuracy, Test Macro-F1, Test Weighted-F1, Test ROC-AUC
      - CV Macro-F1 (from RandomizedSearchCV — training data only)
      - Precision & Recall (macro)

    Plots generated:
      - Confusion matrix per model
      - Multi-class ROC curves (all models)
      - Precision-Recall curves (all models)
      - Calibration plot (all models)
      - Classification report heatmap (best model)
    """
    class_names = le.classes_
    n_classes   = len(class_names)
    results     = []

    roc_data = {}   # For combined ROC plot
    pr_data  = {}   # For combined PR plot
    cal_data = {}   # For calibration plot

    for name, (model, search) in models.items():
        if model is None:
            continue

        logger.info(f"Evaluating {name} on held-out test set...")

        # ── Get predictions ────────────────────────────────────────────────
        if name == "Neural Network":
            try:
                y_pred, y_proba = _predict_nn(model, X_test, y_test)
                if y_pred is None:
                    continue
            except Exception as e:
                logger.warning(f"  NN evaluation failed: {e}")
                continue
            cv_f1 = model.get("val_f1", np.nan) if isinstance(model, dict) else np.nan
        else:
            y_pred  = model.predict(X_test)
            try:
                y_proba = model.predict_proba(X_test)
            except Exception:
                y_proba = None
            cv_f1 = search.best_score_ if search is not None else np.nan

        # ── Compute metrics on test set ────────────────────────────────────
        acc        = accuracy_score(y_test, y_pred)
        f1_macro   = f1_score(y_test, y_pred, average="macro")
        f1_weighted= f1_score(y_test, y_pred, average="weighted")
        prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec_macro  = recall_score(y_test, y_pred, average="macro",  zero_division=0)

        try:
            roc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
        except Exception:
            roc = np.nan

        results.append({
            "Model":           name,
            "Test_Accuracy":   acc,
            "Test_Macro_F1":   f1_macro,
            "CV_Macro_F1":     cv_f1,
            "Test_Weighted_F1":f1_weighted,
            "Test_Precision":  prec_macro,
            "Test_Recall":     rec_macro,
            "Test_ROC_AUC":    roc,
        })

        logger.info(f"  Test Accuracy={acc:.4f}, Macro-F1={f1_macro:.4f}, "
                    f"ROC-AUC={roc:.4f}, CV-F1={cv_f1:.4f}")

        # ── Confusion matrix ───────────────────────────────────────────────
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=class_names, yticklabels=class_names)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix: {name}\n(Held-Out Test Set)", fontweight="bold")
        fig.tight_layout()
        save_figure(fig, f"16_cm_{name.lower().replace(' ', '_')}", OUT_FIG)
        plt.close(fig)

        # ── Collect ROC / PR / calibration data ───────────────────────────
        if y_proba is not None:
            y_bin = label_binarize(y_test, classes=np.arange(n_classes))
            roc_data[name] = (y_bin, y_proba)
            pr_data[name]  = (y_bin, y_proba)
            cal_data[name] = (y_test, y_proba)

    # ── Summary table ──────────────────────────────────────────────────────
    results_df = pd.DataFrame(results).sort_values("Test_Macro_F1", ascending=False)
    save_results(results_df, "classification_model_comparison", OUT_RESULT)

    logger.info("\n" + "=" * 70)
    logger.info("MODEL COMPARISON (Held-Out Test Set — 20% stratified holdout)")
    logger.info("=" * 70)
    logger.info(results_df.to_string(index=False))

    # ── Comparison bar chart ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    x     = np.arange(len(results_df))
    width = 0.18
    metrics_plot = ["Test_Accuracy", "Test_Macro_F1", "CV_Macro_F1", "Test_Weighted_F1", "Test_ROC_AUC"]
    colors_plot  = ["#2196F3", "#F44336", "#FF9800", "#4CAF50", "#9C27B0"]
    labels_plot  = ["Test Accuracy", "Test Macro-F1", "CV Macro-F1", "Test Weighted-F1", "Test ROC-AUC"]
    for i, (metric, color, label) in enumerate(zip(metrics_plot, colors_plot, labels_plot)):
        if metric in results_df.columns:
            vals = results_df[metric].values
            ax.bar(x + i * width, vals, width, label=label, color=color, alpha=0.85)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Classification Model Comparison\n(20% Held-Out Test Set)",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(results_df["Model"], rotation=20, ha="right")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, "17_model_comparison", OUT_FIG)
    plt.close(fig)

    # ── Multi-class ROC curves (one-vs-rest) ──────────────────────────────
    if roc_data:
        _plot_roc_curves(roc_data, class_names)

    # ── Precision-Recall curves ────────────────────────────────────────────
    if pr_data:
        _plot_pr_curves(pr_data, class_names)

    # ── Reliability / Calibration diagram ─────────────────────────────────
    if cal_data:
        _plot_calibration(cal_data, n_classes)

    # ── Classification report heatmap (best model) ─────────────────────────
    if len(results_df) > 0:
        best_name  = results_df.iloc[0]["Model"]
        best_model = models[best_name][0]
        _plot_classification_report(best_model, best_name, X_test, y_test,
                                    class_names, preprocessor)

    return results_df


def _plot_roc_curves(roc_data, class_names):
    """Multi-class ROC curves — one subplot per class, all models overlaid."""
    n_classes = len(class_names)
    fig, axes = plt.subplots(1, n_classes, figsize=(5 * n_classes, 5))
    if n_classes == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, len(roc_data)))

    for cls_idx, cls_name in enumerate(class_names):
        ax = axes[cls_idx]
        for (name, (y_bin, y_proba)), color in zip(roc_data.items(), colors):
            try:
                fpr, tpr, _ = roc_curve(y_bin[:, cls_idx], y_proba[:, cls_idx])
                auc_val = roc_auc_score(y_bin[:, cls_idx], y_proba[:, cls_idx])
                ax.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})",
                        color=color, linewidth=1.8)
            except Exception:
                pass
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC — {cls_name}", fontweight="bold")
        ax.legend(fontsize=7, loc="lower right")
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])

    fig.suptitle("Multi-Class ROC Curves (One-vs-Rest, Held-Out Test Set)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "17b_roc_curves", OUT_FIG)
    plt.close(fig)


def _plot_pr_curves(pr_data, class_names):
    """Precision-Recall curves — one subplot per class."""
    n_classes = len(class_names)
    fig, axes = plt.subplots(1, n_classes, figsize=(5 * n_classes, 5))
    if n_classes == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, len(pr_data)))

    for cls_idx, cls_name in enumerate(class_names):
        ax = axes[cls_idx]
        for (name, (y_bin, y_proba)), color in zip(pr_data.items(), colors):
            try:
                prec, rec, _ = precision_recall_curve(y_bin[:, cls_idx], y_proba[:, cls_idx])
                ap = average_precision_score(y_bin[:, cls_idx], y_proba[:, cls_idx])
                ax.plot(rec, prec, label=f"{name} (AP={ap:.3f})",
                        color=color, linewidth=1.8)
            except Exception:
                pass
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"PR — {cls_name}", fontweight="bold")
        ax.legend(fontsize=7, loc="upper right")
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])

    fig.suptitle("Precision-Recall Curves (Held-Out Test Set)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "17c_pr_curves", OUT_FIG)
    plt.close(fig)


def _plot_calibration(cal_data, n_classes):
    """Reliability diagram — how well predicted probabilities match empirical frequencies."""
    fig, axes = plt.subplots(1, n_classes, figsize=(5 * n_classes, 5))
    if n_classes == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, len(cal_data)))

    class_labels = ["Low", "Medium", "High", "Catastrophic"]
    for cls_idx in range(n_classes):
        ax = axes[cls_idx]
        for (name, (y_test, y_proba)), color in zip(cal_data.items(), colors):
            try:
                y_bin_cls = (y_test == cls_idx).astype(int)
                prob_true, prob_pred = calibration_curve(
                    y_bin_cls, y_proba[:, cls_idx], n_bins=10
                )
                ax.plot(prob_pred, prob_true, "s-", label=name, color=color,
                        linewidth=1.5, markersize=4)
            except Exception:
                pass
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Perfect calibration")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title(f"Calibration — {class_labels[cls_idx]}", fontweight="bold")
        ax.legend(fontsize=7)
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1])

    fig.suptitle("Probability Calibration (Reliability Diagrams)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "17d_calibration", OUT_FIG)
    plt.close(fig)


def _plot_classification_report(model, name, X_test, y_test, class_names, preprocessor):
    """Full classification report as a heatmap for the best model."""
    if name == "Neural Network":
        return

    try:
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        report_df = pd.DataFrame(report).T.drop(
            columns=["support"], errors="ignore"
        ).drop(
            index=["accuracy", "macro avg", "weighted avg"], errors="ignore"
        )

        fig, ax = plt.subplots(figsize=(7, 4))
        sns.heatmap(report_df.astype(float), annot=True, fmt=".3f", cmap="YlGnBu",
                    ax=ax, vmin=0, vmax=1, linewidths=0.5)
        ax.set_title(f"Classification Report: {name}\n(Held-Out Test Set)",
                     fontweight="bold")
        ax.set_xlabel("Metric")
        ax.set_ylabel("Class")
        fig.tight_layout()
        save_figure(fig, "17e_classification_report", OUT_FIG)
        plt.close(fig)
    except Exception as e:
        logger.warning(f"Classification report plot failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# Feature Importance
# ═══════════════════════════════════════════════════════════════════════════

def plot_feature_importance(rf_model, X_test, y_test, cat_cols, num_cols, preprocessor):
    """Feature importance from Random Forest + permutation importance."""
    logger.info("Computing feature importance...")

    try:
        ohe = preprocessor.named_transformers_["cat"]
        cat_feature_names = ohe.get_feature_names_out(cat_cols).tolist()
    except Exception:
        cat_feature_names = []
    feature_names = num_cols + cat_feature_names

    rf_clf = rf_model.named_steps["clf"]
    importances = rf_clf.feature_importances_

    if len(feature_names) == len(importances):
        fi_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances,
        }).sort_values("Importance", ascending=False).head(25)

        fig, ax = plt.subplots(figsize=(10, 9))
        colors_fi = sns.color_palette("viridis", len(fi_df))
        ax.barh(fi_df["Feature"][::-1], fi_df["Importance"][::-1], color=colors_fi[::-1])
        ax.set_xlabel("Feature Importance (Gini)", fontsize=11)
        ax.set_title("Random Forest Feature Importance (Top 25)", fontweight="bold")
        fig.tight_layout()
        save_figure(fig, "18_feature_importance_rf", OUT_FIG)
        plt.close(fig)
        save_results(fi_df, "feature_importance_rf", OUT_RESULT)

    # Learning curve for the best tree-based model
    _plot_learning_curve(rf_model, X_test, y_test)

    # Permutation importance on test set (subsampled for speed on 16 GB Mac)
    try:
        perm_n = min(20_000, len(X_test))
        perm_idx = np.random.choice(len(X_test), perm_n, replace=False)
        X_perm = X_test.iloc[perm_idx]
        y_perm = y_test[perm_idx]
        perm_imp = permutation_importance(
            rf_model, X_perm, y_perm, n_repeats=5, random_state=RANDOM_SEED,
            scoring="f1_macro", n_jobs=4,
        )
        orig_features = X_perm.columns.tolist()
        perm_df = pd.DataFrame({
            "Feature": orig_features,
            "Importance_Mean": perm_imp.importances_mean,
            "Importance_Std":  perm_imp.importances_std,
        }).sort_values("Importance_Mean", ascending=False)

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(perm_df["Feature"][::-1], perm_df["Importance_Mean"][::-1],
                xerr=perm_df["Importance_Std"][::-1],
                color=sns.color_palette("magma", len(perm_df)), capsize=3)
        ax.set_xlabel("Permutation Importance (Macro-F1 decrease on test set)", fontsize=11)
        ax.set_title("Permutation Feature Importance (Test Set)", fontweight="bold")
        fig.tight_layout()
        save_figure(fig, "19_feature_importance_perm", OUT_FIG)
        plt.close(fig)
        save_results(perm_df, "feature_importance_permutation", OUT_RESULT)
    except Exception as e:
        logger.warning(f"Permutation importance failed: {e}")


def _plot_learning_curve(model, X, y):
    """Learning curve: training size vs. CV score.
    Subsampled to 60k max for tractability on 16 GB Mac Mini.
    """
    try:
        logger.info("Computing learning curve (subsampled for speed)...")
        n_lc = min(60_000, len(X))
        if len(X) > n_lc:
            lc_idx = np.random.choice(len(X), n_lc, replace=False)
            X_lc, y_lc = X.iloc[lc_idx], y[lc_idx]
        else:
            X_lc, y_lc = X, y

        train_sizes, train_scores, val_scores = learning_curve(
            model, X_lc, y_lc,
            train_sizes=np.linspace(0.15, 1.0, 5),
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED),
            scoring="f1_macro",
            n_jobs=4,
        )
        train_mean = train_scores.mean(axis=1)
        train_std  = train_scores.std(axis=1)
        val_mean   = val_scores.mean(axis=1)
        val_std    = val_scores.std(axis=1)

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color="#2196F3")
        ax.fill_between(train_sizes, val_mean   - val_std,   val_mean   + val_std,  alpha=0.15, color="#F44336")
        ax.plot(train_sizes, train_mean, "o-", color="#2196F3", label="Training score", linewidth=2)
        ax.plot(train_sizes, val_mean,   "s-", color="#F44336", label="CV score",       linewidth=2)
        ax.set_xlabel("Training Set Size", fontsize=12)
        ax.set_ylabel("Macro-F1 Score",    fontsize=12)
        ax.set_title("Learning Curve — Random Forest\n(shaded = ±1 std dev)",
                     fontsize=13, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        save_figure(fig, "19b_learning_curve", OUT_FIG)
        plt.close(fig)
    except Exception as e:
        logger.warning(f"Learning curve failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# Caching helpers
# ═══════════════════════════════════════════════════════════════════════════

def _cache_path(name):
    safe = name.lower().replace(" ", "_").replace("-", "_")
    return OUT_MODEL / f"cached_{safe}_{CACHE_VERSION}.joblib"


def _save_model_cache(name, model, search):
    path = _cache_path(name)
    joblib.dump({"model": model, "search": search}, path)
    logger.info(f"  Cached {name} → {path.name}")


def _load_model_cache(name):
    path = _cache_path(name)
    if path.exists():
        data = joblib.load(path)
        logger.info(f"  Loaded cached {name} from {path.name}")
        return data["model"], data["search"]
    return None, None


# ═══════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_classification(df: pd.DataFrame):
    """
    Full classification pipeline with research-grade evaluation.

    Key design decisions:
      1. 80/20 stratified holdout split created FIRST, before any model sees data
      2. All hyperparameter tuning (RandomizedSearchCV) done on X_train only
      3. Test set used ONCE at the end for final evaluation
      4. SMOTE applied inside CV folds (no leakage into validation folds)
      5. All models cached with version tag — retrained when features change
    """
    logger.info("=" * 70)
    logger.info("CLASSIFICATION PIPELINE  (research-grade evaluation)")
    logger.info("=" * 70)

    X, y, le, cat_cols, num_cols = prepare_classification_data(df)
    preprocessor = build_preprocessor(cat_cols, num_cols)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    # ── CRITICAL: Create holdout BEFORE any model fitting ─────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_SEED
    )
    logger.info(f"Train set: {len(X_train):,} samples | "
                f"Holdout test set: {len(X_test):,} samples (20%, never seen during training)")

    # Save train/test split indices for reproducibility
    split_info = pd.DataFrame({
        "train_size": [len(X_train)],
        "test_size":  [len(X_test)],
        "cache_version": [CACHE_VERSION],
        "n_features": [X.shape[1]],
        "n_classes":  [len(le.classes_)],
        "class_names": [str(list(le.classes_))],
    })
    save_results(split_info, "classification_split_info", OUT_RESULT)

    # ── Train or load cached models ────────────────────────────────────────
    models = {}

    def train_or_load(name, train_fn, *args):
        cached_model, cached_search = _load_model_cache(name)
        if cached_model is not None:
            return cached_model, cached_search
        model, search = train_fn(*args)
        if model is not None:
            _save_model_cache(name, model, search)
        return model, search

    # All sklearn models trained on X_train / y_train
    dt_model,  dt_search  = train_or_load("Decision Tree",  train_decision_tree,  X_train, y_train, preprocessor, cv)
    models["Decision Tree"] = (dt_model, dt_search)

    rf_model,  rf_search  = train_or_load("Random Forest",  train_random_forest,  X_train, y_train, preprocessor, cv)
    models["Random Forest"] = (rf_model, rf_search)

    knn_model, knn_search = train_or_load("k-NN",           train_knn,            X_train, y_train, preprocessor, cv)
    models["k-NN"] = (knn_model, knn_search)

    svm_model, svm_search = train_or_load("SVM",            train_svm,            X_train, y_train, preprocessor, cv)
    models["SVM"] = (svm_model, svm_search)

    xgb_model, xgb_search = train_or_load("XGBoost",        train_xgboost,        X_train, y_train, preprocessor, cv)
    if xgb_model is not None:
        models["XGBoost"] = (xgb_model, xgb_search)

    lgb_model, lgb_search = train_or_load("LightGBM",       train_lightgbm,       X_train, y_train, preprocessor, cv)
    if lgb_model is not None:
        models["LightGBM"] = (lgb_model, lgb_search)

    # Shut down loky workers before PyTorch NN to avoid deadlocks
    try:
        from joblib.externals.loky import get_reusable_executor
        get_reusable_executor().shutdown(wait=True)
    except Exception:
        pass
    import gc
    gc.collect()

    # Neural Network — separate caching (torch.save, not joblib)
    nn_cache_pt = OUT_MODEL / f"cached_neural_network_{CACHE_VERSION}.pt"
    if nn_cache_pt.exists():
        logger.info("  Loading cached Neural Network...")
        import torch
        nn_data = torch.load(str(nn_cache_pt), weights_only=False)
        StormNet = _StormNet.get_class()
        nn_pt_model = StormNet(nn_data["n_features"], nn_data["n_classes"])
        nn_pt_model.load_state_dict(nn_data["state_dict"])
        preprocessor.fit(X_train)   # ensure preprocessor is fitted
        nn_model = {
            "model": nn_pt_model,
            "preprocessor": preprocessor,
            "n_features": nn_data["n_features"],
            "n_classes":  nn_data["n_classes"],
            "val_f1": nn_data.get("val_f1"),
        }
        nn_history = nn_data.get("history", {})
        logger.info(f"  Cached NN loaded (val F1={nn_data.get('val_f1', 'N/A')})")
    else:
        nn_model, nn_history = train_neural_network(
            X_train, y_train, preprocessor, n_classes=len(le.classes_)
        )
        if nn_model is not None:
            import torch
            torch.save({
                "state_dict": nn_model["model"].state_dict(),
                "n_features": nn_model["n_features"],
                "n_classes":  nn_model["n_classes"],
                "val_f1":     nn_model.get("val_f1"),
                "history":    nn_history,
            }, str(nn_cache_pt))
            logger.info(f"  Cached Neural Network → {nn_cache_pt.name}")

    models["Neural Network"] = (nn_model, nn_history)

    # ── Evaluate all models on held-out test set ───────────────────────────
    results = evaluate_all_models(
        models, X_train, X_test, y_train, y_test, le, preprocessor, cv
    )

    # ── Feature importance (from RF, evaluated on test set) ────────────────
    if rf_model is not None:
        plot_feature_importance(
            rf_model, X_test, y_test, cat_cols, num_cols,
            rf_model.named_steps["preprocess"],
        )

    # ── Save best model ────────────────────────────────────────────────────
    best_name  = results.iloc[0]["Model"]
    best_model = models[best_name][0]
    if best_name != "Neural Network":
        model_path = OUT_MODEL / "best_classifier.joblib"
        joblib.dump(best_model, model_path)
        logger.info(f"Best model ({best_name}) saved → {model_path}")
    else:
        logger.info(f"Best model ({best_name}) already saved as {nn_cache_pt.name}")

    joblib.dump(le, OUT_MODEL / "label_encoder.joblib")

    return models, results


if __name__ == "__main__":
    from src.config import DATA_PROC
    df = pd.read_parquet(DATA_PROC / "storms_processed.parquet")
    run_classification(df)
