"""
Storm Hazard Profile Integration — the project's NOVEL contribution.

Combines all three mining components into region-level hazard profiles:
  1. Spatial clusters (from DBSCAN) define the regions
  2. Association rules inform escalation risk per region
  3. Classification models provide predicted severity
  4. Outlier scores flag anomalous events

Output: A composite "Storm Hazard Score" per region that captures:
  - Historical damage intensity
  - Event diversity / cascading risk
  - Temporal pattern strength
  - Human impact severity
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.config import OUT_FIG, OUT_RESULT
from src.utils import logger, save_figure, save_results


def build_region_profiles(df, dbscan_data):
    """
    Build comprehensive profiles for each DBSCAN spatial cluster.

    Each cluster becomes a "hazard zone" with multi-dimensional risk scores.
    """
    logger.info("─" * 40)
    logger.info("Building Region Hazard Profiles")
    logger.info("─" * 40)

    clustered = dbscan_data[dbscan_data["CLUSTER"] >= 0].copy()
    if len(clustered) == 0:
        logger.warning("No DBSCAN clusters available")
        return None

    profiles = []
    for cluster_id in sorted(clustered["CLUSTER"].unique()):
        cluster_events = clustered[clustered["CLUSTER"] == cluster_id]

        profile = {
            "cluster_id": cluster_id,
            "n_events": len(cluster_events),
            "centroid_lat": cluster_events["BEGIN_LAT"].mean(),
            "centroid_lon": cluster_events["BEGIN_LON"].mean(),
            "spatial_spread_deg": np.sqrt(
                cluster_events["BEGIN_LAT"].var() + cluster_events["BEGIN_LON"].var()
            ),

            # Damage intensity metrics
            "total_damage": cluster_events["TOTAL_DAMAGE"].sum(),
            "mean_damage": cluster_events["TOTAL_DAMAGE"].mean(),
            "median_damage": cluster_events["TOTAL_DAMAGE"].median(),
            "max_damage": cluster_events["TOTAL_DAMAGE"].max(),
            "damage_iqr": (cluster_events["TOTAL_DAMAGE"].quantile(0.75) -
                          cluster_events["TOTAL_DAMAGE"].quantile(0.25)),

            # Human impact
            "total_deaths": cluster_events["TOTAL_DEATHS"].sum(),
            "total_injuries": cluster_events["TOTAL_INJURIES"].sum(),
            "human_impact_score": (cluster_events["TOTAL_DEATHS"].sum() * 10 +
                                  cluster_events["TOTAL_INJURIES"].sum()),

            # Event diversity (Shannon entropy of event types)
            "n_event_types": cluster_events["EVENT_TYPE"].nunique(),
            "dominant_event": (cluster_events["EVENT_TYPE"].mode().iloc[0]
                              if len(cluster_events["EVENT_TYPE"].mode()) > 0 else "N/A"),
            "event_diversity": _shannon_entropy(cluster_events["EVENT_TYPE"]),

            # Temporal span
            "n_years": cluster_events["YEAR"].nunique() if "YEAR" in cluster_events else 0,
            "events_per_year": (len(cluster_events) / max(1, cluster_events["YEAR"].nunique())
                               if "YEAR" in cluster_events else 0),

            # Severity tier distribution
            "pct_catastrophic": (cluster_events["DAMAGE_CLASS"] == "Catastrophic").mean() * 100,
            "pct_high": (cluster_events["DAMAGE_CLASS"] == "High").mean() * 100,

            # States involved
            "n_states": cluster_events["STATE"].nunique() if "STATE" in cluster_events else 0,
            "primary_state": (cluster_events["STATE"].mode().iloc[0]
                            if "STATE" in cluster_events and len(cluster_events["STATE"].mode()) > 0
                            else "N/A"),
        }
        profiles.append(profile)

    profiles_df = pd.DataFrame(profiles)
    logger.info(f"Built profiles for {len(profiles_df)} hazard zones")

    return profiles_df


def _shannon_entropy(series):
    """Compute Shannon entropy of a categorical series."""
    probs = series.value_counts(normalize=True)
    return -np.sum(probs * np.log2(probs + 1e-10))


def compute_composite_hazard_score(profiles_df):
    """
    Compute a composite hazard score from multiple risk dimensions.

    Dimensions:
      1. Damage Intensity — mean/median damage
      2. Human Impact — deaths + injuries weighted
      3. Cascading Risk — event type diversity (entropy)
      4. Frequency — events per year
      5. Catastrophic Proportion — % of high/catastrophic events

    Each dimension is min-max normalized to [0, 1], then weighted-averaged.
    """
    logger.info("Computing composite hazard scores...")

    scaler = MinMaxScaler()
    dimensions = {
        "damage_intensity": profiles_df["mean_damage"],
        "human_impact": profiles_df["human_impact_score"],
        "cascading_risk": profiles_df["event_diversity"],
        "frequency": profiles_df["events_per_year"],
        "catastrophic_pct": profiles_df["pct_catastrophic"],
    }

    # Weights (sum to 1) — reflects domain importance
    weights = {
        "damage_intensity": 0.25,
        "human_impact": 0.25,
        "cascading_risk": 0.20,
        "frequency": 0.15,
        "catastrophic_pct": 0.15,
    }

    normalized = {}
    for dim_name, values in dimensions.items():
        vals = values.values.reshape(-1, 1)
        if vals.max() > vals.min():
            normalized[dim_name] = scaler.fit_transform(vals).flatten()
        else:
            normalized[dim_name] = np.zeros(len(vals))
        profiles_df[f"norm_{dim_name}"] = normalized[dim_name]

    # Composite score
    profiles_df["hazard_score"] = sum(
        normalized[dim] * weight for dim, weight in weights.items()
    )

    # Rank
    profiles_df["hazard_rank"] = profiles_df["hazard_score"].rank(
        ascending=False, method="min"
    ).astype(int)

    # Categorize
    profiles_df["hazard_tier"] = pd.cut(
        profiles_df["hazard_score"],
        bins=[0, 0.25, 0.5, 0.75, 1.01],
        labels=["Low Risk", "Moderate Risk", "High Risk", "Extreme Risk"],
    )

    save_results(profiles_df, "hazard_zone_profiles", OUT_RESULT)

    logger.info("\nHazard Zone Rankings:")
    for _, row in profiles_df.sort_values("hazard_rank").head(10).iterrows():
        logger.info(f"  Rank {row['hazard_rank']}: Cluster {row['cluster_id']} "
                    f"({row['primary_state']}) — Score={row['hazard_score']:.3f} "
                    f"[{row['hazard_tier']}] — {row['n_events']} events, "
                    f"${row['total_damage']/1e6:.0f}M damage")

    return profiles_df


def visualize_hazard_profiles(profiles_df):
    """Generate comprehensive hazard profile visualizations."""
    if profiles_df is None or len(profiles_df) == 0:
        return

    # ── 1. Radar chart for top hazard zones ────────────────────────────────
    top5 = profiles_df.nsmallest(5, "hazard_rank")
    dims = ["norm_damage_intensity", "norm_human_impact", "norm_cascading_risk",
            "norm_frequency", "norm_catastrophic_pct"]
    dim_labels = ["Damage\nIntensity", "Human\nImpact", "Cascading\nRisk",
                  "Frequency", "Catastrophic\n%"]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    colors = sns.color_palette("husl", len(top5))
    for idx, (_, row) in enumerate(top5.iterrows()):
        values = [row[d] for d in dims] + [row[dims[0]]]
        ax.plot(angles, values, "o-", linewidth=2, color=colors[idx],
                label=f"Cluster {int(row['cluster_id'])} ({row['primary_state']})")
        ax.fill(angles, values, alpha=0.1, color=colors[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dim_labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title("Top 5 Hazard Zone Profiles (Radar Chart)", fontsize=14,
                 fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    fig.tight_layout()
    save_figure(fig, "37_hazard_radar", OUT_FIG)
    plt.close(fig)

    # ── 2. Hazard score geographic scatter ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 10))
    scatter = ax.scatter(
        profiles_df["centroid_lon"], profiles_df["centroid_lat"],
        c=profiles_df["hazard_score"], cmap="YlOrRd",
        s=profiles_df["n_events"] / profiles_df["n_events"].max() * 500 + 20,
        alpha=0.7, edgecolors="black", linewidth=0.5,
    )
    ax.set_xlim(-130, -65)
    ax.set_ylim(24, 50)
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.set_title("Storm Hazard Zones: Composite Risk Score\n"
                 "(size ∝ event count, color = hazard score)",
                 fontsize=14, fontweight="bold")
    plt.colorbar(scatter, ax=ax, label="Hazard Score", shrink=0.7)

    # Annotate top 5
    for _, row in top5.iterrows():
        ax.annotate(f"#{int(row['hazard_rank'])}",
                   (row["centroid_lon"], row["centroid_lat"]),
                   fontsize=10, fontweight="bold", color="red",
                   textcoords="offset points", xytext=(10, 10))
    fig.tight_layout()
    save_figure(fig, "38_hazard_zones_map", OUT_FIG)
    plt.close(fig)

    # ── 3. Dimension heatmap ───────────────────────────────────────────────
    top15 = profiles_df.nsmallest(15, "hazard_rank")
    heatmap_data = top15.set_index(
        top15.apply(lambda r: f"C{int(r['cluster_id'])} ({r['primary_state']})", axis=1)
    )[dims]
    heatmap_data.columns = dim_labels

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax,
                linewidths=0.5, vmin=0, vmax=1)
    ax.set_title("Hazard Zone Risk Dimensions (Top 15)", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "39_hazard_heatmap", OUT_FIG)
    plt.close(fig)

    # ── 4. Score distribution ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(profiles_df["hazard_score"], bins=20, color="coral",
                 alpha=0.7, edgecolor="white")
    axes[0].set_xlabel("Hazard Score")
    axes[0].set_ylabel("Number of Zones")
    axes[0].set_title("Distribution of Hazard Scores", fontweight="bold")

    tier_counts = profiles_df["hazard_tier"].value_counts()
    colors_tier = {"Low Risk": "#4CAF50", "Moderate Risk": "#FFC107",
                   "High Risk": "#FF9800", "Extreme Risk": "#F44336"}
    axes[1].bar(tier_counts.index, tier_counts.values,
                color=[colors_tier.get(t, "gray") for t in tier_counts.index])
    axes[1].set_xlabel("Risk Tier")
    axes[1].set_ylabel("Number of Zones")
    axes[1].set_title("Hazard Zone Risk Tier Distribution", fontweight="bold")

    fig.tight_layout()
    save_figure(fig, "40_hazard_distribution", OUT_FIG)
    plt.close(fig)


def enrich_with_association_patterns(profiles_df, association_results, df, dbscan_data):
    """
    Enrich hazard profiles with association mining insights.
    For each cluster, find the most common co-occurrence patterns.
    """
    if association_results is None or profiles_df is None:
        return profiles_df

    logger.info("Enriching profiles with association patterns...")

    bigrams = association_results.get("bigrams")
    if bigrams is None or len(bigrams) == 0:
        return profiles_df

    clustered = dbscan_data[dbscan_data["CLUSTER"] >= 0]

    cluster_patterns = []
    for cluster_id in profiles_df["cluster_id"]:
        cluster_events = clustered[clustered["CLUSTER"] == cluster_id]
        # Get episodes in this cluster
        if "EPISODE_ID" in cluster_events.columns:
            episode_ids = cluster_events["EPISODE_ID"].dropna().unique()
            # Count event type transitions in these episodes
            cluster_df = df[df["EPISODE_ID"].isin(episode_ids)].sort_values(
                ["EPISODE_ID", "BEGIN_DATE_TIME"]
            )
            transitions = []
            for _, group in cluster_df.groupby("EPISODE_ID"):
                types = group["EVENT_TYPE"].tolist()
                for i in range(len(types) - 1):
                    if types[i] != types[i + 1]:
                        transitions.append(f"{types[i]} → {types[i+1]}")

            if transitions:
                from collections import Counter
                top_transition = Counter(transitions).most_common(1)[0]
                cluster_patterns.append({
                    "cluster_id": cluster_id,
                    "top_escalation": top_transition[0],
                    "escalation_count": top_transition[1],
                    "n_unique_transitions": len(set(transitions)),
                })
            else:
                cluster_patterns.append({
                    "cluster_id": cluster_id,
                    "top_escalation": "N/A",
                    "escalation_count": 0,
                    "n_unique_transitions": 0,
                })

    if cluster_patterns:
        patterns_df = pd.DataFrame(cluster_patterns)
        profiles_df = profiles_df.merge(patterns_df, on="cluster_id", how="left")
        save_results(profiles_df, "hazard_zone_profiles_enriched", OUT_RESULT)

    return profiles_df


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def run_hazard_profiles(df, dbscan_data, association_results=None):
    """Build and visualize storm hazard profiles."""
    logger.info("=" * 60)
    logger.info("STORM HAZARD PROFILE INTEGRATION")
    logger.info("=" * 60)

    profiles = build_region_profiles(df, dbscan_data)
    if profiles is None:
        return None

    profiles = compute_composite_hazard_score(profiles)
    profiles = enrich_with_association_patterns(profiles, association_results, df, dbscan_data)
    visualize_hazard_profiles(profiles)

    logger.info("Hazard profile integration complete.")
    return profiles
