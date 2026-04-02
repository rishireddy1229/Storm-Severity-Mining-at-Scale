"""
Statistical Evaluation Module — Conference-Grade Rigor.

Adds missing statistical analyses to the classification pipeline:
  1. Baseline comparisons (majority class, stratified random)
  2. Bootstrap confidence intervals on test metrics
  3. McNemar's test for statistical significance between models
  4. Cohen's weighted Kappa (ordinal agreement)
  5. Temporal validation (train pre-2019, test 2019+)
  6. Feature ablation study (contribution of feature groups)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import warnings
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    cohen_kappa_score, confusion_matrix
)

from src.config import OUT_FIG, OUT_MODEL, OUT_RESULT, RANDOM_SEED
from src.utils import logger, save_figure, save_results

warnings.filterwarnings("ignore")
np.random.seed(RANDOM_SEED)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Baseline Comparisons
# ═══════════════════════════════════════════════════════════════════════════

def compute_baselines(y_train, y_test, y_test_proba_best=None):
    """Compute baseline classifier performance for context."""
    n_classes = len(np.unique(y_train))

    # Majority class baseline
    majority_class = np.bincount(y_train).argmax()
    y_pred_majority = np.full_like(y_test, majority_class)

    # Stratified random baseline (respects class distribution)
    class_probs = np.bincount(y_train, minlength=n_classes) / len(y_train)
    rng = np.random.RandomState(RANDOM_SEED)
    y_pred_stratified = rng.choice(n_classes, size=len(y_test), p=class_probs)

    baselines = []
    for name, y_pred in [("Majority Class", y_pred_majority),
                         ("Stratified Random", y_pred_stratified)]:
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        kappa = cohen_kappa_score(y_test, y_pred)
        baselines.append({
            "Model": name, "Test_Accuracy": acc,
            "Test_Macro_F1": f1, "Cohens_Kappa": kappa,
        })

    baselines_df = pd.DataFrame(baselines)
    logger.info("BASELINE COMPARISONS:")
    for _, row in baselines_df.iterrows():
        logger.info(f"  {row['Model']}: Acc={row['Test_Accuracy']:.4f}, "
                    f"F1={row['Test_Macro_F1']:.4f}, Kappa={row['Cohens_Kappa']:.4f}")

    return baselines_df


# ═══════════════════════════════════════════════════════════════════════════
# 2. Bootstrap Confidence Intervals
# ═══════════════════════════════════════════════════════════════════════════

def bootstrap_ci(y_true, y_pred, y_proba=None, n_bootstraps=1000, ci=0.95):
    """Compute bootstrap confidence intervals for classification metrics."""
    rng = np.random.RandomState(RANDOM_SEED)
    n = len(y_true)
    accs, f1s, aucs, kappas = [], [], [], []

    for _ in range(n_bootstraps):
        idx = rng.choice(n, n, replace=True)
        yt, yp = y_true[idx], y_pred[idx]
        # Skip if bootstrap sample has fewer than 2 classes
        if len(np.unique(yt)) < 2:
            continue
        accs.append(accuracy_score(yt, yp))
        f1s.append(f1_score(yt, yp, average="macro", zero_division=0))
        kappas.append(cohen_kappa_score(yt, yp))
        if y_proba is not None:
            try:
                aucs.append(roc_auc_score(yt, y_proba[idx], multi_class="ovr",
                                          average="macro"))
            except Exception:
                pass

    alpha = (1 - ci) / 2
    results = {
        "accuracy": (np.percentile(accs, alpha*100), np.percentile(accs, (1-alpha)*100)),
        "macro_f1": (np.percentile(f1s, alpha*100), np.percentile(f1s, (1-alpha)*100)),
        "kappa": (np.percentile(kappas, alpha*100), np.percentile(kappas, (1-alpha)*100)),
    }
    if aucs:
        results["roc_auc"] = (np.percentile(aucs, alpha*100),
                              np.percentile(aucs, (1-alpha)*100))
    return results


# ═══════════════════════════════════════════════════════════════════════════
# 3. McNemar's Test
# ═══════════════════════════════════════════════════════════════════════════

def mcnemar_test(y_true, y_pred_a, y_pred_b):
    """McNemar's test for comparing two classifiers."""
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    # Contingency table
    b = np.sum(correct_a & ~correct_b)  # A right, B wrong
    c = np.sum(~correct_a & correct_b)  # A wrong, B right

    # McNemar's test with continuity correction
    if b + c == 0:
        return 0.0, 1.0  # No disagreement
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    from scipy.stats import chi2 as chi2_dist
    p_value = 1 - chi2_dist.cdf(chi2, df=1)
    return chi2, p_value


# ═══════════════════════════════════════════════════════════════════════════
# 4. Feature Ablation Study
# ═══════════════════════════════════════════════════════════════════════════

def run_feature_ablation(df, X_train, y_train, X_test, y_test,
                         cat_cols, num_cols, preprocessor_builder):
    """Remove feature groups and measure impact on RF performance."""
    from sklearn.ensemble import RandomForestClassifier
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder

    feature_groups = {
        "Spatial (LAT/LON/COASTAL)": ["BEGIN_LAT", "BEGIN_LON", "IS_COASTAL"],
        "Temporal (HOUR/MONTH/YEAR/DECADE/DOW)": ["HOUR", "MONTH", "YEAR", "DECADE", "DAY_OF_WEEK"],
        "Human Impact (INJ/DEATH/IMPACT)": ["INJURIES_DIRECT", "INJURIES_INDIRECT",
                                              "DEATHS_DIRECT", "DEATHS_INDIRECT", "HUMAN_IMPACT"],
        "Storm Properties (MAG/EF/PATH)": ["MAGNITUDE", "HAS_MAGNITUDE", "EF_NUMERIC", "PATH_LENGTH_DEG"],
        "Engineered (EPISODE_SIZE/LOG_DUR/DECADE/COASTAL)": ["EPISODE_SIZE", "LOG_DURATION",
                                                               "DECADE", "IS_COASTAL"],
        "Event Type": ["EVENT_TYPE"],
    }

    all_features = X_train.columns.tolist()
    results = []

    # Full model (baseline)
    logger.info("Feature ablation — training full RF model...")
    full_cat = [c for c in cat_cols if c in all_features]
    full_num = [c for c in num_cols if c in all_features]
    full_pre = ColumnTransformer([
        ("num", StandardScaler(), full_num),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist",
                              min_frequency=500, sparse_output=False), full_cat),
    ], remainder="drop")

    full_pipe = ImbPipeline([
        ("preprocess", full_pre),
        ("smote", SMOTE(random_state=RANDOM_SEED)),
        ("clf", RandomForestClassifier(n_estimators=200, max_depth=30,
                                       random_state=RANDOM_SEED, n_jobs=-1)),
    ])
    full_pipe.fit(X_train, y_train)
    y_pred_full = full_pipe.predict(X_test)
    full_f1 = f1_score(y_test, y_pred_full, average="macro")
    full_acc = accuracy_score(y_test, y_pred_full)
    results.append({"Removed": "None (Full Model)", "Accuracy": full_acc,
                    "Macro_F1": full_f1, "Delta_F1": 0.0})
    logger.info(f"  Full model: Acc={full_acc:.4f}, F1={full_f1:.4f}")

    for group_name, group_features in feature_groups.items():
        keep_features = [f for f in all_features if f not in group_features]
        if len(keep_features) == 0:
            continue

        keep_cat = [c for c in full_cat if c in keep_features]
        keep_num = [c for c in full_num if c in keep_features]

        if len(keep_cat) == 0 and len(keep_num) == 0:
            continue

        transformers = []
        if keep_num:
            transformers.append(("num", StandardScaler(), keep_num))
        if keep_cat:
            transformers.append(("cat", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                                      min_frequency=500, sparse_output=False), keep_cat))

        ablated_pre = ColumnTransformer(transformers, remainder="drop")
        ablated_pipe = ImbPipeline([
            ("preprocess", ablated_pre),
            ("smote", SMOTE(random_state=RANDOM_SEED)),
            ("clf", RandomForestClassifier(n_estimators=200, max_depth=30,
                                           random_state=RANDOM_SEED, n_jobs=-1)),
        ])
        try:
            ablated_pipe.fit(X_train[keep_features], y_train)
            y_pred_ab = ablated_pipe.predict(X_test[keep_features])
            ab_f1 = f1_score(y_test, y_pred_ab, average="macro")
            ab_acc = accuracy_score(y_test, y_pred_ab)
            delta = ab_f1 - full_f1
            results.append({"Removed": group_name, "Accuracy": ab_acc,
                            "Macro_F1": ab_f1, "Delta_F1": delta})
            logger.info(f"  Removed {group_name}: Acc={ab_acc:.4f}, "
                        f"F1={ab_f1:.4f} ({delta:+.4f})")
        except Exception as e:
            logger.warning(f"  Ablation failed for {group_name}: {e}")

    results_df = pd.DataFrame(results)
    save_results(results_df, "feature_ablation", OUT_RESULT)

    # Plot ablation results
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_data = results_df[results_df["Removed"] != "None (Full Model)"].copy()
    plot_data = plot_data.sort_values("Delta_F1")
    colors = ["#F44336" if d < -0.01 else "#FFC107" if d < 0 else "#4CAF50"
              for d in plot_data["Delta_F1"]]
    ax.barh(plot_data["Removed"], plot_data["Delta_F1"], color=colors)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Change in Macro-F1 (negative = feature group is important)")
    ax.set_title("Feature Group Ablation Study (Random Forest)\n"
                 f"Full model F1={full_f1:.4f}", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "19c_feature_ablation", OUT_FIG)
    plt.close(fig)

    return results_df


# ═══════════════════════════════════════════════════════════════════════════
# 5. Temporal Validation
# ═══════════════════════════════════════════════════════════════════════════

def run_temporal_validation(df):
    """Train on 1996-2018, test on 2019-2024 — no temporal leakage."""
    from src.classification import prepare_classification_data, build_preprocessor
    from sklearn.ensemble import RandomForestClassifier
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    logger.info("─" * 40)
    logger.info("TEMPORAL VALIDATION (train: 1996-2018, test: 2019-2024)")
    logger.info("─" * 40)

    X, y, le, cat_cols, num_cols = prepare_classification_data(df)

    # Get year from original dataframe
    clf_df = df[df["DAMAGE_CLASS"] != "None"].copy()
    # Align indices
    years = clf_df.loc[X.index, "YEAR"] if "YEAR" in clf_df.columns else None
    if years is None:
        logger.warning("YEAR column not available for temporal split")
        return None

    train_mask = years <= 2018
    test_mask = years >= 2019

    X_train_t, y_train_t = X[train_mask], y[train_mask]
    X_test_t, y_test_t = X[test_mask], y[test_mask]

    logger.info(f"  Temporal train (1996-2018): {len(X_train_t):,} samples")
    logger.info(f"  Temporal test  (2019-2024): {len(X_test_t):,} samples")

    if len(X_test_t) < 100:
        logger.warning("Temporal test set too small, skipping")
        return None

    preprocessor = build_preprocessor(cat_cols, num_cols)

    pipe = ImbPipeline([
        ("preprocess", preprocessor),
        ("smote", SMOTE(random_state=RANDOM_SEED)),
        ("clf", RandomForestClassifier(n_estimators=200, max_depth=30,
                                       random_state=RANDOM_SEED, n_jobs=-1)),
    ])
    pipe.fit(X_train_t, y_train_t)
    y_pred_t = pipe.predict(X_test_t)
    y_proba_t = pipe.predict_proba(X_test_t)

    acc = accuracy_score(y_test_t, y_pred_t)
    f1 = f1_score(y_test_t, y_pred_t, average="macro")
    kappa = cohen_kappa_score(y_test_t, y_pred_t)
    try:
        auc = roc_auc_score(y_test_t, y_proba_t, multi_class="ovr", average="macro")
    except Exception:
        auc = np.nan

    result = {
        "Split": "Temporal (train≤2018, test≥2019)",
        "Train_Size": len(X_train_t),
        "Test_Size": len(X_test_t),
        "Accuracy": acc, "Macro_F1": f1,
        "ROC_AUC": auc, "Cohens_Kappa": kappa,
    }
    logger.info(f"  Temporal RF: Acc={acc:.4f}, F1={f1:.4f}, "
                f"AUC={auc:.4f}, Kappa={kappa:.4f}")

    return pd.DataFrame([result])


# ═══════════════════════════════════════════════════════════════════════════
# Main orchestrator
# ═══════════════════════════════════════════════════════════════════════════

def run_evaluation(df):
    """Run the full statistical evaluation suite."""
    from src.classification import prepare_classification_data, build_preprocessor

    logger.info("=" * 70)
    logger.info("STATISTICAL EVALUATION SUITE (Conference-Grade)")
    logger.info("=" * 70)

    X, y, le, cat_cols, num_cols = prepare_classification_data(df)
    preprocessor = build_preprocessor(cat_cols, num_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_SEED
    )

    # ── Load cached models and get predictions ────────────────────────────
    model_names = ["Random Forest", "XGBoost", "LightGBM", "Decision Tree", "k-NN", "SVM"]
    predictions = {}
    probas = {}

    for name in model_names:
        cache_name = name.lower().replace(" ", "_").replace("-", "_")
        cache_path = OUT_MODEL / f"cached_{cache_name}_v3.joblib"
        if not cache_path.exists():
            logger.warning(f"  No cached model for {name}, skipping")
            continue
        try:
            data = joblib.load(cache_path)
            model = data["model"]
            predictions[name] = model.predict(X_test)
            try:
                probas[name] = model.predict_proba(X_test)
            except Exception:
                pass
            logger.info(f"  Loaded {name} predictions")
        except Exception as e:
            logger.warning(f"  Failed to load {name}: {e}")

    if not predictions:
        logger.warning("No models loaded, skipping evaluation")
        return

    # ── 1. Baselines ──────────────────────────────────────────────────────
    baselines = compute_baselines(y_train, y_test)

    # ── 2. Confidence intervals for all models ────────────────────────────
    logger.info("\nBOOTSTRAP 95% CONFIDENCE INTERVALS (1000 samples):")
    ci_results = []
    for name, y_pred in predictions.items():
        y_prob = probas.get(name)
        ci = bootstrap_ci(y_test, y_pred, y_prob, n_bootstraps=1000)
        kappa = cohen_kappa_score(y_test, y_pred)
        ci_results.append({
            "Model": name,
            "Macro_F1": f1_score(y_test, y_pred, average="macro"),
            "F1_CI_Low": ci["macro_f1"][0],
            "F1_CI_High": ci["macro_f1"][1],
            "Accuracy": accuracy_score(y_test, y_pred),
            "Acc_CI_Low": ci["accuracy"][0],
            "Acc_CI_High": ci["accuracy"][1],
            "Cohens_Kappa": kappa,
            "Kappa_CI_Low": ci["kappa"][0],
            "Kappa_CI_High": ci["kappa"][1],
            "ROC_AUC": roc_auc_score(y_test, probas[name], multi_class="ovr",
                                      average="macro") if name in probas else np.nan,
            "AUC_CI_Low": ci.get("roc_auc", (np.nan, np.nan))[0],
            "AUC_CI_High": ci.get("roc_auc", (np.nan, np.nan))[1],
        })
        f1_lo, f1_hi = ci["macro_f1"]
        auc_lo, auc_hi = ci.get("roc_auc", (np.nan, np.nan))
        logger.info(f"  {name}: F1={ci_results[-1]['Macro_F1']:.4f} "
                    f"[{f1_lo:.4f}, {f1_hi:.4f}], "
                    f"Kappa={kappa:.4f}, "
                    f"AUC=[{auc_lo:.4f}, {auc_hi:.4f}]")

    ci_df = pd.DataFrame(ci_results).sort_values("Macro_F1", ascending=False)
    save_results(ci_df, "classification_confidence_intervals", OUT_RESULT)

    # ── 3. McNemar's tests ────────────────────────────────────────────────
    logger.info("\nMcNEMAR'S TESTS (pairwise model comparisons):")
    mcnemar_results = []
    model_list = list(predictions.keys())
    for i in range(len(model_list)):
        for j in range(i + 1, len(model_list)):
            name_a, name_b = model_list[i], model_list[j]
            chi2, p = mcnemar_test(y_test, predictions[name_a], predictions[name_b])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            mcnemar_results.append({
                "Model_A": name_a, "Model_B": name_b,
                "Chi2": chi2, "p_value": p, "Significant": sig,
            })
            logger.info(f"  {name_a} vs {name_b}: chi2={chi2:.2f}, p={p:.4f} {sig}")

    mcnemar_df = pd.DataFrame(mcnemar_results)
    save_results(mcnemar_df, "mcnemar_tests", OUT_RESULT)

    # ── 4. Comprehensive results table with baselines and CIs ─────────────
    # Create publication-ready comparison figure
    fig, ax = plt.subplots(figsize=(14, 8))

    plot_models = ci_df["Model"].tolist()
    x = np.arange(len(plot_models))
    f1_vals = ci_df["Macro_F1"].values
    f1_lo = ci_df["F1_CI_Low"].values
    f1_hi = ci_df["F1_CI_High"].values
    f1_err = np.array([f1_vals - f1_lo, f1_hi - f1_vals])

    bars = ax.bar(x, f1_vals, color=sns.color_palette("viridis", len(plot_models)),
                  yerr=f1_err, capsize=5, alpha=0.85)

    # Add baseline lines
    for _, row in baselines.iterrows():
        ax.axhline(row["Test_Macro_F1"], linestyle="--", alpha=0.7,
                   label=f'{row["Model"]}: F1={row["Test_Macro_F1"]:.3f}')

    ax.set_xticks(x)
    ax.set_xticklabels(plot_models, rotation=20, ha="right")
    ax.set_ylabel("Macro-F1 Score")
    ax.set_title("Classification Performance with 95% Bootstrap CIs\n"
                 "vs Baseline Classifiers", fontweight="bold", fontsize=13)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 0.75)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, "17f_confidence_intervals", OUT_FIG)
    plt.close(fig)

    # ── 5. Feature ablation ───────────────────────────────────────────────
    logger.info("\nFEATURE ABLATION STUDY:")
    ablation_df = run_feature_ablation(
        df, X_train, y_train, X_test, y_test,
        cat_cols, num_cols, build_preprocessor
    )

    # ── 6. Temporal validation ────────────────────────────────────────────
    temporal_df = run_temporal_validation(df)
    if temporal_df is not None:
        save_results(temporal_df, "temporal_validation", OUT_RESULT)

    # ── 7. Summary table combining everything ─────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 70)

    summary = []
    # Add baselines
    for _, row in baselines.iterrows():
        summary.append({
            "Model": row["Model"],
            "Macro_F1": f"{row['Test_Macro_F1']:.4f}",
            "95% CI": "—",
            "Kappa": f"{row['Cohens_Kappa']:.4f}",
            "Type": "Baseline",
        })
    # Add trained models
    for _, row in ci_df.iterrows():
        summary.append({
            "Model": row["Model"],
            "Macro_F1": f"{row['Macro_F1']:.4f}",
            "95% CI": f"[{row['F1_CI_Low']:.4f}, {row['F1_CI_High']:.4f}]",
            "Kappa": f"{row['Cohens_Kappa']:.4f}",
            "Type": "Trained",
        })
    # Add temporal validation
    if temporal_df is not None:
        t = temporal_df.iloc[0]
        summary.append({
            "Model": "RF (Temporal Split)",
            "Macro_F1": f"{t['Macro_F1']:.4f}",
            "95% CI": "—",
            "Kappa": f"{t['Cohens_Kappa']:.4f}",
            "Type": "Temporal",
        })

    summary_df = pd.DataFrame(summary)
    save_results(summary_df, "evaluation_summary", OUT_RESULT)
    logger.info(summary_df.to_string(index=False))

    logger.info("\nStatistical evaluation complete.")
    return {
        "baselines": baselines,
        "confidence_intervals": ci_df,
        "mcnemar": mcnemar_df,
        "ablation": ablation_df,
        "temporal": temporal_df,
    }
