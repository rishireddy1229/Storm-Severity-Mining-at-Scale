"""
Clustering and Spatial Mining (Chapters VIII, XII, XIV).

Implements:
  1. DBSCAN on geocoordinates — discovers natural "hazard zones"
  2. K-Means on event severity profiles — with elbow + silhouette analysis
  3. Hierarchical clustering on event-type profiles — with dendrogram
  4. Cluster profiling — damage stats per cluster
  5. Spatial autocorrelation — Moran's I on damage per region
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import joblib
from collections import Counter

from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist

from src.config import (
    OUT_FIG, OUT_MODEL, OUT_RESULT, RANDOM_SEED,
    DBSCAN_EPS_RANGE, DBSCAN_MINPTS_RANGE, KMEANS_K_RANGE,
)
from src.utils import logger, save_figure, save_results


# ═══════════════════════════════════════════════════════════════════════════
# 1. DBSCAN — Spatial Hazard Zone Discovery
# ═══════════════════════════════════════════════════════════════════════════

def run_dbscan(df):
    """
    DBSCAN on high-damage event coordinates to find natural hazard zones.
    Includes parameter sensitivity analysis.
    """
    logger.info("─" * 40)
    logger.info("DBSCAN: Spatial Hazard Zone Discovery")
    logger.info("─" * 40)

    # Select high-damage events (top 25% of damage-causing events)
    damage_events = df[df["TOTAL_DAMAGE"] > 0]
    q75 = damage_events["TOTAL_DAMAGE"].quantile(0.75)
    high_damage = df[
        (df["TOTAL_DAMAGE"] >= q75) &
        df["BEGIN_LAT"].notna() &
        df["BEGIN_LON"].notna()
    ].copy()
    logger.info(f"High-damage events (≥ ${q75:,.0f}): {len(high_damage):,}")

    # Filter to Continental US — excludes Alaska, Hawaii, Puerto Rico,
    # and territories that create trivial macro-clusters
    conus_mask = (
        high_damage["BEGIN_LAT"].between(24.5, 49.5) &
        high_damage["BEGIN_LON"].between(-125.0, -66.5)
    )
    high_damage = high_damage[conus_mask].copy()
    logger.info(f"Continental US high-damage events: {len(high_damage):,}")

    coords = high_damage[["BEGIN_LAT", "BEGIN_LON"]].values

    # ── Parameter sensitivity analysis ─────────────────────────────────────
    logger.info("Running DBSCAN parameter sensitivity analysis...")
    # Subsample for sensitivity if dataset is large (DBSCAN is O(n log n))
    max_sens = 20000
    if len(coords) > max_sens:
        sens_idx = np.random.choice(len(coords), max_sens, replace=False)
        coords_sens = coords[sens_idx]
        logger.info(f"  Subsampled to {max_sens:,} for sensitivity analysis")
    else:
        coords_sens = coords

    coords_sens_rad = np.radians(coords_sens)
    total_combos = len(DBSCAN_EPS_RANGE) * len(DBSCAN_MINPTS_RANGE)
    sensitivity = []
    combo_i = 0
    for eps in DBSCAN_EPS_RANGE:
        for minpts in DBSCAN_MINPTS_RANGE:
            combo_i += 1
            db = DBSCAN(eps=eps, min_samples=minpts, metric="haversine", n_jobs=-1)
            labels = db.fit_predict(coords_sens_rad)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            noise_pct = (labels == -1).sum() / len(labels) * 100
            sil = silhouette_score(
                coords_sens_rad, labels, metric="haversine",
                sample_size=min(5000, len(coords_sens))
            ) if n_clusters >= 2 else -1
            sensitivity.append({
                "eps": eps, "minPts": minpts,
                "n_clusters": n_clusters, "noise_pct": noise_pct,
                "silhouette": sil,
            })
            logger.info(f"  [{combo_i}/{total_combos}] eps={eps}, minPts={minpts} → "
                        f"{n_clusters} clusters, sil={sil:.4f}")

    sens_df = pd.DataFrame(sensitivity)
    save_results(sens_df, "dbscan_parameter_sensitivity", OUT_RESULT)

    # Plot sensitivity heatmap
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric, title in zip(
        axes,
        ["n_clusters", "silhouette"],
        ["Number of Clusters", "Silhouette Score"],
    ):
        pivot = sens_df.pivot(index="minPts", columns="eps", values=metric)
        sns.heatmap(pivot, annot=True, fmt=".2f" if "sil" in metric else ".0f",
                    cmap="YlGnBu", ax=ax)
        ax.set_title(f"DBSCAN: {title}", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "20_dbscan_sensitivity", OUT_FIG)
    plt.close(fig)

    # ── Select best configuration (domain-aware) ─────────────────────────
    # Require 5-20 clusters with reasonable noise, then pick best silhouette
    viable = sens_df[
        (sens_df["n_clusters"] >= 5) &
        (sens_df["n_clusters"] <= 20) &
        (sens_df["noise_pct"] < 40) &
        (sens_df["silhouette"] > -0.5)
    ]
    if len(viable) > 0:
        best = viable.loc[viable["silhouette"].idxmax()]
    else:
        # Fallback: most clusters with positive silhouette
        fallback = sens_df[sens_df["silhouette"] > 0]
        if len(fallback) > 0:
            best = fallback.loc[fallback["n_clusters"].idxmax()]
        else:
            best = sens_df.loc[sens_df["n_clusters"].idxmax()]
        logger.warning(f"No config with 5-20 clusters; fallback to "
                       f"{int(best['n_clusters'])} clusters")
    best_eps = best["eps"]
    best_minpts = int(best["minPts"])
    logger.info(f"Selected DBSCAN: eps={best_eps}, minPts={best_minpts}, "
                f"clusters={int(best['n_clusters'])}, silhouette={best['silhouette']:.4f}")

    # Scale minPts proportionally to full dataset size to maintain cluster structure
    scale_factor = len(high_damage) / len(coords_sens) if len(coords_sens) < len(high_damage) else 1.0
    scaled_minpts = int(best_minpts * scale_factor)
    logger.info(f"Scaling minPts {best_minpts} → {scaled_minpts} for full dataset "
                f"(scale={scale_factor:.1f}x)")
    coords_rad = np.radians(coords)
    db_final = DBSCAN(eps=best_eps, min_samples=scaled_minpts, metric="haversine", n_jobs=-1)
    high_damage["CLUSTER"] = db_final.fit_predict(coords_rad)

    n_clusters = high_damage["CLUSTER"].nunique() - (1 if -1 in high_damage["CLUSTER"].values else 0)
    noise_n = (high_damage["CLUSTER"] == -1).sum()
    logger.info(f"Final: {n_clusters} clusters, {noise_n} noise points "
                f"({noise_n/len(high_damage)*100:.1f}%)")

    # ── Visualize clusters ─────────────────────────────────────────────────
    clustered = high_damage[high_damage["CLUSTER"] >= 0]
    fig, ax = plt.subplots(figsize=(16, 10))
    scatter = ax.scatter(
        clustered["BEGIN_LON"], clustered["BEGIN_LAT"],
        c=clustered["CLUSTER"], cmap="tab20", s=5, alpha=0.5,
    )
    # Noise in gray
    noise = high_damage[high_damage["CLUSTER"] == -1]
    ax.scatter(noise["BEGIN_LON"], noise["BEGIN_LAT"],
               c="lightgray", s=1, alpha=0.2, label="Noise")
    ax.set_xlim(-130, -65)
    ax.set_ylim(24, 50)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"DBSCAN Spatial Hazard Zones ({n_clusters} clusters)",
                 fontsize=14, fontweight="bold")
    ax.legend(markerscale=5)
    fig.tight_layout()
    save_figure(fig, "21_dbscan_clusters", OUT_FIG)
    plt.close(fig)

    # ── Cluster profiles ───────────────────────────────────────────────────
    profiles = clustered.groupby("CLUSTER").agg(
        n_events=("EVENT_ID", "count"),
        mean_damage=("TOTAL_DAMAGE", "mean"),
        median_damage=("TOTAL_DAMAGE", "median"),
        total_damage=("TOTAL_DAMAGE", "sum"),
        mean_lat=("BEGIN_LAT", "mean"),
        mean_lon=("BEGIN_LON", "mean"),
        total_deaths=("TOTAL_DEATHS", "sum"),
        total_injuries=("TOTAL_INJURIES", "sum"),
        dominant_event=("EVENT_TYPE", lambda x: x.mode().iloc[0] if len(x) > 0 and len(x.mode()) > 0 else ""),
        n_states=("STATE", "nunique"),
    ).sort_values("total_damage", ascending=False)
    save_results(profiles.reset_index(), "dbscan_cluster_profiles", OUT_RESULT)

    logger.info("Top 5 hazard zone profiles:")
    for _, row in profiles.head(5).iterrows():
        logger.info(f"  Cluster {row.name}: {row['n_events']} events, "
                    f"${row['total_damage']/1e6:.0f}M damage, "
                    f"dominant: {row['dominant_event']}")

    return high_damage, profiles


# ═══════════════════════════════════════════════════════════════════════════
# 2. K-Means on event severity profiles
# ═══════════════════════════════════════════════════════════════════════════

def run_kmeans(df):
    """
    K-Means clustering on event severity profiles.
    Includes elbow method + silhouette analysis.
    """
    logger.info("─" * 40)
    logger.info("K-Means: Event Severity Profiles")
    logger.info("─" * 40)

    features = ["TOTAL_DAMAGE", "DURATION_MIN", "INJURIES_DIRECT",
                "DEATHS_DIRECT", "HUMAN_IMPACT"]
    available = [c for c in features if c in df.columns]

    km_data = df[df["TOTAL_DAMAGE"] > 0][available].dropna().copy()
    # Log-transform skewed features
    for col in ["TOTAL_DAMAGE", "HUMAN_IMPACT"]:
        if col in km_data.columns:
            km_data[col] = np.log1p(km_data[col])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(km_data)
    logger.info(f"K-Means data: {X_scaled.shape[0]:,} samples, {X_scaled.shape[1]} features")

    # Subsample for validation loop if dataset is large
    max_km = 100000
    if len(X_scaled) > max_km:
        km_idx = np.random.choice(len(X_scaled), max_km, replace=False)
        X_km_val = X_scaled[km_idx]
        logger.info(f"  Subsampled to {max_km:,} for K-Means validation")
    else:
        X_km_val = X_scaled

    # ── Elbow + Silhouette analysis ────────────────────────────────────────
    inertias = []
    silhouettes = []
    ch_scores = []

    for k in KMEANS_K_RANGE:
        km = KMeans(n_clusters=k, n_init=25, random_state=RANDOM_SEED)
        labels = km.fit_predict(X_km_val)
        inertias.append(km.inertia_)
        if k >= 2 and len(np.unique(labels)) >= 2:
            try:
                sil = silhouette_score(X_km_val, labels,
                                       sample_size=min(10000, len(X_km_val)),
                                       random_state=RANDOM_SEED)
                ch = calinski_harabasz_score(X_km_val, labels)
                silhouettes.append(sil)
                ch_scores.append(ch)
            except ValueError:
                silhouettes.append(0)
                ch_scores.append(0)
        else:
            silhouettes.append(0)
            ch_scores.append(0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    k_vals = list(KMEANS_K_RANGE)

    axes[0].plot(k_vals, inertias, "bo-", linewidth=2)
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Inertia (WSS)")
    axes[0].set_title("Elbow Method", fontweight="bold")

    axes[1].plot(k_vals, silhouettes, "rs-", linewidth=2)
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title("Silhouette Analysis", fontweight="bold")

    axes[2].plot(k_vals, ch_scores, "g^-", linewidth=2)
    axes[2].set_xlabel("k")
    axes[2].set_ylabel("Calinski-Harabasz Score")
    axes[2].set_title("Calinski-Harabasz Index", fontweight="bold")

    fig.suptitle("K-Means Cluster Validation Metrics", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "22_kmeans_validation", OUT_FIG)
    plt.close(fig)

    # ── Optimal k (highest silhouette) ─────────────────────────────────────
    optimal_k = k_vals[np.argmax(silhouettes)]
    logger.info(f"Optimal k by silhouette: {optimal_k}")

    km_final = KMeans(n_clusters=optimal_k, n_init=25, random_state=RANDOM_SEED)
    km_labels = km_final.fit_predict(X_scaled)

    # Silhouette plot (subsample for large datasets — silhouette_samples is O(n^2))
    fig, ax = plt.subplots(figsize=(10, 7))
    max_sil = 30000
    if len(X_scaled) > max_sil:
        sil_idx = np.random.choice(len(X_scaled), max_sil, replace=False)
        X_sil = X_scaled[sil_idx]
        labels_sil = km_labels[sil_idx]
    else:
        X_sil = X_scaled
        labels_sil = km_labels
    sil_vals = silhouette_samples(X_sil, labels_sil)
    y_lower = 10
    for i in range(optimal_k):
        cluster_sil = sil_vals[labels_sil == i]
        cluster_sil.sort()
        y_upper = y_lower + len(cluster_sil)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_sil, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * len(cluster_sil), str(i))
        y_lower = y_upper + 10
    mean_sil = silhouette_score(X_scaled, km_labels, sample_size=min(10000, len(X_scaled)))
    ax.axvline(x=mean_sil, color="red", linestyle="--",
               label=f"Mean: {mean_sil:.3f}")
    ax.set_xlabel("Silhouette Coefficient")
    ax.set_ylabel("Cluster")
    ax.set_title(f"Silhouette Plot (k={optimal_k})", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, "23_kmeans_silhouette", OUT_FIG)
    plt.close(fig)

    # Save model
    joblib.dump(km_final, OUT_MODEL / "kmeans_model.joblib")
    joblib.dump(scaler, OUT_MODEL / "kmeans_scaler.joblib")

    return km_final, km_labels, X_scaled, km_data


# ═══════════════════════════════════════════════════════════════════════════
# 3. Hierarchical Clustering on event-type profiles
# ═══════════════════════════════════════════════════════════════════════════

def run_hierarchical(df):
    """
    Hierarchical clustering on aggregated event-type profiles.
    Reveals which event types have similar severity characteristics.
    """
    logger.info("─" * 40)
    logger.info("Hierarchical Clustering: Event Type Profiles")
    logger.info("─" * 40)

    type_profiles = df.groupby("EVENT_TYPE").agg(
        avg_damage=("TOTAL_DAMAGE", "mean"),
        median_damage=("TOTAL_DAMAGE", "median"),
        avg_duration=("DURATION_MIN", "mean"),
        avg_injuries=("INJURIES_DIRECT", "mean"),
        avg_deaths=("DEATHS_DIRECT", "mean"),
        event_count=("EVENT_ID", "count"),
        pct_catastrophic=("DAMAGE_CLASS", lambda x: (x == "Catastrophic").mean() * 100),
    ).dropna()

    # Filter to event types with enough data
    type_profiles = type_profiles[type_profiles["event_count"] >= 100]
    logger.info(f"Event types with ≥100 events: {len(type_profiles)}")

    # Scale features
    features_to_cluster = ["avg_damage", "median_damage", "avg_duration",
                           "avg_injuries", "avg_deaths", "pct_catastrophic"]
    X_hc = StandardScaler().fit_transform(type_profiles[features_to_cluster])

    # Linkage
    Z = linkage(X_hc, method="ward")

    # Dendrogram
    fig, ax = plt.subplots(figsize=(16, 10))
    dendrogram(Z, labels=type_profiles.index.tolist(), ax=ax,
               leaf_rotation=90, leaf_font_size=9, color_threshold=0.7 * max(Z[:, 2]))
    ax.set_ylabel("Ward Distance")
    ax.set_title("Event Type Dendrogram (Ward Linkage)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "24_hierarchical_dendrogram", OUT_FIG)
    plt.close(fig)

    # Cut at k=5 clusters
    hc_labels = fcluster(Z, t=5, criterion="maxclust")
    type_profiles["HC_CLUSTER"] = hc_labels

    # Profile each cluster
    for c in sorted(type_profiles["HC_CLUSTER"].unique()):
        members = type_profiles[type_profiles["HC_CLUSTER"] == c].index.tolist()
        logger.info(f"  HC Cluster {c}: {members}")

    save_results(type_profiles.reset_index(), "hierarchical_cluster_profiles", OUT_RESULT)

    return type_profiles, Z


# ═══════════════════════════════════════════════════════════════════════════
# 4. Spatial autocorrelation (Moran's I)
# ═══════════════════════════════════════════════════════════════════════════

def compute_morans_i(df):
    """
    Compute Moran's I spatial autocorrelation on state-level damage.
    Tests whether nearby states have similar damage levels.
    """
    logger.info("─" * 40)
    logger.info("Spatial Autocorrelation: Moran's I")
    logger.info("─" * 40)

    try:
        from libpysal.weights import Queen, KNN
        from esda.moran import Moran
        import geopandas as gpd
    except ImportError:
        logger.warning("libpysal/esda/geopandas not available. "
                       "Computing approximate spatial autocorrelation with scipy.")

        # Fallback: compute state centroids from data and use distance-based weights
        state_stats = df.groupby("STATE").agg(
            mean_damage=("TOTAL_DAMAGE", "mean"),
            mean_lat=("BEGIN_LAT", "mean"),
            mean_lon=("BEGIN_LON", "mean"),
        ).dropna()

        if len(state_stats) < 5:
            logger.warning("Not enough states with coordinate data for Moran's I")
            return None

        from scipy.spatial.distance import cdist
        coords = state_stats[["mean_lat", "mean_lon"]].values
        dist_matrix = cdist(coords, coords)

        # Binary weight: neighbors within ~500km (~4.5 degrees)
        W = (dist_matrix < 4.5).astype(float)
        np.fill_diagonal(W, 0)
        # Row-standardize
        row_sums = W.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        W = W / row_sums

        y = state_stats["mean_damage"].values
        y_centered = y - y.mean()
        n = len(y)

        numerator = n * (y_centered @ W @ y_centered)
        denominator = (y_centered @ y_centered) * W.sum()
        morans_i = numerator / denominator if denominator != 0 else 0

        # Permutation test
        np.random.seed(RANDOM_SEED)
        n_perm = 999
        perm_i = np.zeros(n_perm)
        for p in range(n_perm):
            y_perm = np.random.permutation(y_centered)
            num_p = n * (y_perm @ W @ y_perm)
            den_p = (y_perm @ y_perm) * W.sum()
            perm_i[p] = num_p / den_p if den_p != 0 else 0

        p_value = (np.sum(np.abs(perm_i) >= np.abs(morans_i)) + 1) / (n_perm + 1)

        logger.info(f"Moran's I (state-level damage): {morans_i:.4f}")
        logger.info(f"Pseudo p-value ({n_perm} permutations): {p_value:.4f}")
        logger.info(f"Interpretation: {'Significant spatial clustering' if p_value < 0.05 else 'No significant spatial pattern'}")

        result = {"Morans_I": morans_i, "p_value": p_value, "n_states": n}
        save_results(pd.DataFrame([result]), "morans_i_result", OUT_RESULT)
        return result

    return None


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def run_clustering(df: pd.DataFrame):
    """Run the full clustering pipeline."""
    logger.info("=" * 60)
    logger.info("CLUSTERING & SPATIAL MINING")
    logger.info("=" * 60)

    dbscan_data, dbscan_profiles = run_dbscan(df)
    km_model, km_labels, X_km, km_data = run_kmeans(df)
    hc_profiles, hc_linkage = run_hierarchical(df)
    morans = compute_morans_i(df)

    logger.info("Clustering complete.")
    return {
        "dbscan_data": dbscan_data,
        "dbscan_profiles": dbscan_profiles,
        "kmeans_model": km_model,
        "kmeans_labels": km_labels,
        "hc_profiles": hc_profiles,
        "morans_i": morans,
    }


if __name__ == "__main__":
    from src.config import DATA_PROC
    df = pd.read_parquet(DATA_PROC / "storms_processed.parquet")
    run_clustering(df)
