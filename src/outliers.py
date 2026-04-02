"""
Outlier Detection (Chapter VII).

Implements:
  1. Z-score based outlier detection (per event type)
  2. Isolation Forest (global anomaly detection)
  3. Local Outlier Factor (LOF)
  4. Outlier characterization and comparison across methods
  5. Consensus outliers (flagged by ≥2 methods)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from src.config import (
    OUT_FIG, OUT_RESULT, RANDOM_SEED,
    ZSCORE_THRESHOLD, ISO_FOREST_CONTAMINATION, LOF_N_NEIGHBORS, LOF_CONTAMINATION,
)
from src.utils import logger, save_figure, save_results


def prepare_outlier_features(df):
    """Prepare feature matrix for outlier detection."""
    feature_cols = ["TOTAL_DAMAGE", "DURATION_MIN", "INJURIES_DIRECT",
                    "DEATHS_DIRECT", "BEGIN_LAT", "BEGIN_LON"]
    available = [c for c in feature_cols if c in df.columns]

    subset = df[df["TOTAL_DAMAGE"] > 0][available].dropna(subset=available).copy()
    logger.info(f"Outlier detection dataset: {len(subset):,} records, {len(available)} features")
    return subset, available


# ═══════════════════════════════════════════════════════════════════════════
# 1. Z-score outliers (per event type)
# ═══════════════════════════════════════════════════════════════════════════

def detect_zscore_outliers(df):
    """Z-score based outlier detection per event type."""
    logger.info("─" * 40)
    logger.info("Z-Score Outlier Detection")
    logger.info("─" * 40)

    damage_events = df[df["TOTAL_DAMAGE"] > 0].copy()
    damage_events["Z_DAMAGE"] = damage_events.groupby("EVENT_TYPE")["TOTAL_DAMAGE"].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    )

    outliers = damage_events[damage_events["Z_DAMAGE"].abs() > ZSCORE_THRESHOLD].copy()
    logger.info(f"Z-score outliers (|z| > {ZSCORE_THRESHOLD}): {len(outliers):,} "
                f"({len(outliers)/len(damage_events)*100:.2f}%)")

    # Distribution of z-scores
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(damage_events["Z_DAMAGE"].clip(-10, 10), bins=100,
                 color="steelblue", alpha=0.7, edgecolor="white")
    axes[0].axvline(x=ZSCORE_THRESHOLD, color="red", linestyle="--",
                    label=f"z = ±{ZSCORE_THRESHOLD}")
    axes[0].axvline(x=-ZSCORE_THRESHOLD, color="red", linestyle="--")
    axes[0].set_xlabel("Z-Score (damage within event type)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Z-Score Distribution", fontweight="bold")
    axes[0].legend()

    # Outliers by event type
    outlier_types = outliers["EVENT_TYPE"].value_counts().head(10)
    axes[1].barh(outlier_types.index[::-1], outlier_types.values[::-1],
                 color=sns.color_palette("Reds_r", 10))
    axes[1].set_xlabel("Number of Z-Score Outliers")
    axes[1].set_title("Outliers by Event Type", fontweight="bold")

    fig.tight_layout()
    save_figure(fig, "29_zscore_outliers", OUT_FIG)
    plt.close(fig)

    return outliers.index


# ═══════════════════════════════════════════════════════════════════════════
# 2. Isolation Forest
# ═══════════════════════════════════════════════════════════════════════════

def detect_isolation_forest(df):
    """Isolation Forest anomaly detection."""
    logger.info("─" * 40)
    logger.info("Isolation Forest")
    logger.info("─" * 40)

    subset, feature_cols = prepare_outlier_features(df)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(subset)

    iso = IsolationForest(
        contamination=ISO_FOREST_CONTAMINATION,
        random_state=RANDOM_SEED,
        n_estimators=200,
        n_jobs=-1,
    )
    labels = iso.fit_predict(X_scaled)
    scores = iso.decision_function(X_scaled)

    outliers = subset.index[labels == -1]
    logger.info(f"Isolation Forest outliers: {len(outliers):,} "
                f"({len(outliers)/len(subset)*100:.2f}%)")

    # Anomaly score distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(scores, bins=100, color="teal", alpha=0.7, edgecolor="white")
    threshold = np.percentile(scores, ISO_FOREST_CONTAMINATION * 100)
    ax.axvline(x=threshold, color="red", linestyle="--",
               label=f"Threshold ({ISO_FOREST_CONTAMINATION*100:.0f}th percentile)")
    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Count")
    ax.set_title("Isolation Forest Anomaly Score Distribution", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, "30_isolation_forest_scores", OUT_FIG)
    plt.close(fig)

    return outliers, scores


# ═══════════════════════════════════════════════════════════════════════════
# 3. Local Outlier Factor (LOF)
# ═══════════════════════════════════════════════════════════════════════════

def detect_lof(df):
    """Local Outlier Factor detection."""
    logger.info("─" * 40)
    logger.info("Local Outlier Factor (LOF)")
    logger.info("─" * 40)

    subset, feature_cols = prepare_outlier_features(df)

    # LOF is memory-intensive — subsample if needed
    max_n = 20000
    if len(subset) > max_n:
        logger.info(f"Subsampling to {max_n} for LOF")
        subset = subset.sample(max_n, random_state=RANDOM_SEED)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(subset)

    lof = LocalOutlierFactor(
        n_neighbors=LOF_N_NEIGHBORS,
        contamination=LOF_CONTAMINATION,
        n_jobs=-1,
    )
    labels = lof.fit_predict(X_scaled)
    scores = -lof.negative_outlier_factor_  # Higher = more anomalous

    outliers = subset.index[labels == -1]
    logger.info(f"LOF outliers: {len(outliers):,} ({len(outliers)/len(subset)*100:.2f}%)")

    # LOF score distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(scores.clip(0, 5), bins=100, color="darkorange", alpha=0.7, edgecolor="white")
    ax.axvline(x=np.percentile(scores, (1 - LOF_CONTAMINATION) * 100),
               color="red", linestyle="--", label="Threshold")
    ax.set_xlabel("LOF Score")
    ax.set_ylabel("Count")
    ax.set_title("Local Outlier Factor Score Distribution", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, "31_lof_scores", OUT_FIG)
    plt.close(fig)

    return outliers, scores


# ═══════════════════════════════════════════════════════════════════════════
# 4. Consensus & characterization
# ═══════════════════════════════════════════════════════════════════════════

def consensus_outliers(df, zscore_idx, isoforest_idx, lof_idx):
    """
    Find consensus outliers (flagged by ≥2 methods) and characterize them.
    """
    logger.info("─" * 40)
    logger.info("Consensus Outlier Analysis")
    logger.info("─" * 40)

    # Count how many methods flag each point
    all_idx = set(zscore_idx) | set(isoforest_idx) | set(lof_idx)
    counts = {}
    for idx in all_idx:
        c = 0
        if idx in zscore_idx: c += 1
        if idx in isoforest_idx: c += 1
        if idx in lof_idx: c += 1
        counts[idx] = c

    # Consensus = flagged by ≥ 2 methods
    consensus_idx = [idx for idx, c in counts.items() if c >= 2]
    all_three = [idx for idx, c in counts.items() if c == 3]

    logger.info(f"Outliers by method:")
    logger.info(f"  Z-score: {len(zscore_idx):,}")
    logger.info(f"  Isolation Forest: {len(isoforest_idx):,}")
    logger.info(f"  LOF: {len(lof_idx):,}")
    logger.info(f"  Consensus (≥2 methods): {len(consensus_idx):,}")
    logger.info(f"  All three methods: {len(all_three):,}")

    # Venn diagram (approximate with bar chart)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Method overlap
    method_counts = {
        "Z-score only": len(set(zscore_idx) - set(isoforest_idx) - set(lof_idx)),
        "IsoForest only": len(set(isoforest_idx) - set(zscore_idx) - set(lof_idx)),
        "LOF only": len(set(lof_idx) - set(zscore_idx) - set(isoforest_idx)),
        "≥2 methods": len(consensus_idx),
        "All 3 methods": len(all_three),
    }
    axes[0].barh(list(method_counts.keys()), list(method_counts.values()),
                 color=["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0"])
    axes[0].set_xlabel("Number of Outliers")
    axes[0].set_title("Outlier Detection Method Overlap", fontweight="bold")

    # Characterize consensus outliers
    if consensus_idx:
        consensus_events = df.loc[df.index.isin(consensus_idx)]
        type_counts = consensus_events["EVENT_TYPE"].value_counts().head(10)
        axes[1].barh(type_counts.index[::-1], type_counts.values[::-1],
                     color=sns.color_palette("Spectral", 10))
        axes[1].set_xlabel("Number of Consensus Outliers")
        axes[1].set_title("Consensus Outliers by Event Type", fontweight="bold")

    fig.tight_layout()
    save_figure(fig, "32_consensus_outliers", OUT_FIG)
    plt.close(fig)

    # Save top outliers with details
    if consensus_idx:
        top_outliers = df.loc[df.index.isin(consensus_idx)].nlargest(50, "TOTAL_DAMAGE")
        cols_to_save = ["EVENT_TYPE", "STATE", "TOTAL_DAMAGE", "TOTAL_DEATHS",
                        "TOTAL_INJURIES", "DURATION_MIN", "BEGIN_LAT", "BEGIN_LON",
                        "YEAR", "MONTH", "DAMAGE_CLASS"]
        available_cols = [c for c in cols_to_save if c in top_outliers.columns]
        save_results(top_outliers[available_cols], "top_consensus_outliers", OUT_RESULT)

        logger.info("\nTop 10 most extreme consensus outliers:")
        for _, row in top_outliers.head(10).iterrows():
            logger.info(f"  {row.get('EVENT_TYPE', 'N/A')} in {row.get('STATE', 'N/A')} "
                       f"({row.get('YEAR', 'N/A')}): "
                       f"${row.get('TOTAL_DAMAGE', 0):,.0f} damage, "
                       f"{row.get('TOTAL_DEATHS', 0)} deaths")

    return consensus_idx


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def run_outlier_detection(df: pd.DataFrame):
    """Run the full outlier detection pipeline."""
    logger.info("=" * 60)
    logger.info("OUTLIER DETECTION")
    logger.info("=" * 60)

    zscore_idx = detect_zscore_outliers(df)
    isoforest_idx, iso_scores = detect_isolation_forest(df)
    lof_idx, lof_scores = detect_lof(df)
    consensus = consensus_outliers(df, zscore_idx, isoforest_idx, lof_idx)

    logger.info("Outlier detection complete.")
    return {
        "zscore_outliers": zscore_idx,
        "isoforest_outliers": isoforest_idx,
        "lof_outliers": lof_idx,
        "consensus_outliers": consensus,
    }


if __name__ == "__main__":
    from src.config import DATA_PROC
    df = pd.read_parquet(DATA_PROC / "storms_processed.parquet")
    run_outlier_detection(df)
