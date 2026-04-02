"""
Data Storytelling module (Chapter XV).

Generates:
  1. Interactive Folium map with hazard zones
  2. Sankey diagram of episode escalation flows (as static matplotlib)
  3. Dashboard-style summary figure
  4. Final narrative statistics
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.sankey import Sankey
import seaborn as sns
import numpy as np
import pandas as pd
from collections import Counter

from src.config import OUT_FIG, OUT_RESULT, DATA_PROC
from src.utils import logger, save_figure, save_results, format_number


def create_folium_map(df, dbscan_data, hazard_profiles):
    """Create an interactive Folium map with hazard zones."""
    try:
        import folium
        from folium.plugins import HeatMap, MarkerCluster
    except ImportError:
        logger.warning("Folium not available, skipping interactive map")
        return

    logger.info("Creating interactive Folium map...")

    # Base map centered on US
    m = folium.Map(location=[39.8, -98.5], zoom_start=4, tiles="cartodbpositron")

    # Layer 1: All events heatmap
    geo_events = df.dropna(subset=["BEGIN_LAT", "BEGIN_LON"])
    heat_data = geo_events[["BEGIN_LAT", "BEGIN_LON"]].values.tolist()
    # Subsample for performance
    if len(heat_data) > 50000:
        idx = np.random.choice(len(heat_data), 50000, replace=False)
        heat_data = [heat_data[i] for i in idx]
    HeatMap(heat_data, name="All Events Density", radius=8, blur=10,
            max_zoom=7).add_to(m)

    # Layer 2: DBSCAN clusters as colored circles
    if dbscan_data is not None:
        clustered = dbscan_data[dbscan_data["CLUSTER"] >= 0]
        cluster_group = folium.FeatureGroup(name="Hazard Zone Clusters")

        colors = ["red", "blue", "green", "purple", "orange", "darkred",
                  "lightred", "darkblue", "darkgreen", "cadetblue",
                  "pink", "lightblue", "lightgreen", "gray", "black"]

        for cluster_id in sorted(clustered["CLUSTER"].unique()):
            cluster_events = clustered[clustered["CLUSTER"] == cluster_id]
            color = colors[int(cluster_id) % len(colors)]

            # Cluster centroid marker
            center_lat = cluster_events["BEGIN_LAT"].mean()
            center_lon = cluster_events["BEGIN_LON"].mean()

            # Get hazard profile info if available
            popup_text = f"<b>Cluster {cluster_id}</b><br>"
            popup_text += f"Events: {len(cluster_events):,}<br>"
            popup_text += f"Total Damage: {format_number(cluster_events['TOTAL_DAMAGE'].sum())}<br>"
            popup_text += f"Deaths: {cluster_events['TOTAL_DEATHS'].sum()}<br>"
            mode = cluster_events['EVENT_TYPE'].mode()
            popup_text += f"Dominant: {mode.iloc[0] if len(mode) > 0 else 'N/A'}"

            if hazard_profiles is not None:
                hp = hazard_profiles[hazard_profiles["cluster_id"] == cluster_id]
                if len(hp) > 0:
                    hp = hp.iloc[0]
                    popup_text += f"<br><b>Hazard Score: {hp['hazard_score']:.3f}</b>"
                    popup_text += f"<br>Risk Tier: {hp['hazard_tier']}"

            folium.CircleMarker(
                location=[center_lat, center_lon],
                radius=max(5, min(20, len(cluster_events) / 100)),
                popup=folium.Popup(popup_text, max_width=300),
                color=color, fill=True, fill_opacity=0.7,
            ).add_to(cluster_group)

        cluster_group.add_to(m)

    # Layer 3: Top outlier events
    outlier_group = folium.FeatureGroup(name="Extreme Outlier Events")
    top_outliers = df.nlargest(50, "TOTAL_DAMAGE").dropna(subset=["BEGIN_LAT", "BEGIN_LON"])
    for _, row in top_outliers.iterrows():
        folium.CircleMarker(
            location=[row["BEGIN_LAT"], row["BEGIN_LON"]],
            radius=6,
            popup=f"{row['EVENT_TYPE']}: {format_number(row['TOTAL_DAMAGE'])} "
                  f"({row.get('STATE', 'N/A')}, {int(row.get('YEAR', 0))})",
            color="red", fill=True, fill_color="red", fill_opacity=0.8,
        ).add_to(outlier_group)
    outlier_group.add_to(m)

    folium.LayerControl().add_to(m)

    map_path = OUT_FIG / "interactive_hazard_map.html"
    m.save(str(map_path))
    logger.info(f"Interactive map saved: {map_path}")


def create_sankey_flows(df):
    """
    Create a Sankey-like flow diagram showing event type transitions
    within storm episodes (implemented as alluvial/flow chart in matplotlib).
    """
    logger.info("Creating episode flow diagram...")

    episodes = df[df["EPISODE_ID"].notna()].copy()
    episodes = episodes.sort_values(["EPISODE_ID", "BEGIN_DATE_TIME"])

    # Build transition flows
    flows = Counter()
    for _, group in episodes.groupby("EPISODE_ID"):
        types = group["EVENT_TYPE"].tolist()
        for i in range(len(types) - 1):
            if types[i] != types[i + 1]:
                flows[(types[i], types[i + 1])] += 1

    if not flows:
        return

    # Top 20 transitions
    top_flows = flows.most_common(20)

    fig, ax = plt.subplots(figsize=(14, 10))

    # Prepare data
    sources = list(set(f[0] for f, _ in top_flows))
    targets = list(set(f[1] for f, _ in top_flows))
    all_nodes = list(dict.fromkeys(sources + targets))  # Preserve order, unique

    n_left = len(sources)
    n_right = len(targets)

    # Position nodes
    left_y = np.linspace(0, 1, len(sources))
    right_y = np.linspace(0, 1, len(targets))

    source_pos = {s: (0, left_y[i]) for i, s in enumerate(sources)}
    target_pos = {t: (1, right_y[i]) for i, t in enumerate(targets)}

    # Draw flows
    max_count = max(c for _, c in top_flows)
    colors = sns.color_palette("husl", len(sources))
    source_colors = {s: colors[i] for i, s in enumerate(sources)}

    for (src, tgt), count in top_flows:
        width = count / max_count * 0.08
        sx, sy = source_pos[src]
        tx, ty = target_pos[tgt]
        color = list(source_colors[src]) + [0.4]  # Add alpha

        # Draw bezier-like curve
        mid = 0.5
        xs = [sx + 0.05, mid - 0.1, mid + 0.1, tx - 0.05]
        from matplotlib.patches import FancyArrowPatch
        import matplotlib.path as mpath

        verts = [(sx + 0.05, sy), (0.3, sy), (0.7, ty), (tx - 0.05, ty)]
        codes = [mpath.Path.MOVETO, mpath.Path.CURVE4, mpath.Path.CURVE4, mpath.Path.CURVE4]
        path = mpath.Path(verts, codes)
        patch = mpatches.PathPatch(path, facecolor="none", edgecolor=color,
                                   linewidth=width * 200, alpha=0.5)
        ax.add_patch(patch)

    # Draw source nodes
    for src in sources:
        x, y = source_pos[src]
        ax.plot(x, y, "s", color=source_colors[src], markersize=12)
        ax.text(x - 0.02, y, src, ha="right", va="center", fontsize=8, fontweight="bold")

    # Draw target nodes
    for tgt in targets:
        x, y = target_pos[tgt]
        ax.plot(x, y, "s", color="gray", markersize=12)
        ax.text(x + 0.02, y, tgt, ha="left", va="center", fontsize=8, fontweight="bold")

    ax.set_xlim(-0.4, 1.4)
    ax.set_ylim(-0.1, 1.1)
    ax.axis("off")
    ax.set_title("Storm Episode Escalation Flows\n(Event Type Transitions within Episodes)",
                 fontsize=14, fontweight="bold")

    fig.tight_layout()
    save_figure(fig, "41_escalation_flows", OUT_FIG)
    plt.close(fig)

    # Also create a cleaner bar chart version
    flow_df = pd.DataFrame([
        {"Transition": f"{s} → {t}", "Count": c}
        for (s, t), c in top_flows
    ])
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(flow_df["Transition"][::-1], flow_df["Count"][::-1],
            color=sns.color_palette("viridis", len(flow_df)))
    ax.set_xlabel("Number of Episode Transitions")
    ax.set_title("Top 20 Event Type Transitions in Storm Episodes", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "42_transition_barplot", OUT_FIG)
    plt.close(fig)


def create_dashboard_summary(df, classification_results=None, hazard_profiles=None):
    """Create a dashboard-style summary figure with key metrics."""
    logger.info("Creating dashboard summary...")

    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    # ── Panel 1: Key metrics ───────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis("off")
    metrics = [
        ("Total Events", f"{len(df):,}"),
        ("Years Covered", f"{int(df['YEAR'].min())}-{int(df['YEAR'].max())}" if "YEAR" in df else "N/A"),
        ("States", f"{df['STATE'].nunique()}" if "STATE" in df else "N/A"),
        ("Total Damage", format_number(df["TOTAL_DAMAGE"].sum())),
        ("Total Deaths", f"{df['TOTAL_DEATHS'].sum():,}" if "TOTAL_DEATHS" in df else "N/A"),
        ("Event Types", f"{df['EVENT_TYPE'].nunique()}" if "EVENT_TYPE" in df else "N/A"),
    ]
    for i, (label, value) in enumerate(metrics):
        y = 0.85 - i * 0.14
        ax1.text(0.05, y, label, fontsize=11, fontweight="bold", transform=ax1.transAxes)
        ax1.text(0.95, y, value, fontsize=11, ha="right", color="#2196F3",
                transform=ax1.transAxes)
    ax1.set_title("Key Dataset Metrics", fontweight="bold", fontsize=12)

    # ── Panel 2: Damage class pie ──────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    class_counts = df["DAMAGE_CLASS"].value_counts()
    colors_pie = {"None": "#E0E0E0", "Low": "#4CAF50", "Medium": "#FFC107",
                  "High": "#FF9800", "Catastrophic": "#F44336"}
    pie_colors = [colors_pie.get(c, "gray") for c in class_counts.index]
    ax2.pie(class_counts.values, labels=class_counts.index, colors=pie_colors,
            autopct="%1.1f%%", startangle=90, pctdistance=0.85)
    ax2.set_title("Damage Severity Distribution", fontweight="bold", fontsize=12)

    # ── Panel 3: Top 5 deadliest event types ───────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    deaths_by_type = df.groupby("EVENT_TYPE")["TOTAL_DEATHS"].sum().nlargest(8)
    ax3.barh(deaths_by_type.index[::-1], deaths_by_type.values[::-1],
             color=sns.color_palette("Reds_r", 8))
    ax3.set_xlabel("Total Deaths")
    ax3.set_title("Deadliest Event Types", fontweight="bold", fontsize=12)

    # ── Panel 4: Yearly damage trend ───────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, :2])
    yearly = df.groupby("YEAR")["TOTAL_DAMAGE"].sum() / 1e9
    ax4.fill_between(yearly.index, yearly.values, alpha=0.3, color="#F44336")
    ax4.plot(yearly.index, yearly.values, color="#F44336", linewidth=2)
    ax4.set_xlabel("Year")
    ax4.set_ylabel("Total Damage ($ Billions)")
    ax4.set_title("Annual Storm Damage Trend", fontweight="bold", fontsize=12)

    # ── Panel 5: Season distribution ───────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    season_counts = df["SEASON"].value_counts()
    season_order = ["Spring", "Summer", "Fall", "Winter"]
    season_counts = season_counts.reindex(season_order, fill_value=0)
    season_colors = ["#4CAF50", "#FF9800", "#795548", "#2196F3"]
    ax5.bar(season_counts.index, season_counts.values, color=season_colors)
    ax5.set_ylabel("Number of Events")
    ax5.set_title("Events by Season", fontweight="bold", fontsize=12)

    # ── Panel 6: Geographic scatter ────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, :2])
    geo = df.dropna(subset=["BEGIN_LAT", "BEGIN_LON"])
    # Subsample
    if len(geo) > 30000:
        geo = geo.sample(30000, random_state=42)
    scatter = ax6.scatter(geo["BEGIN_LON"], geo["BEGIN_LAT"],
                         c=np.log10(geo["TOTAL_DAMAGE"] + 1), cmap="YlOrRd",
                         s=1, alpha=0.3)
    ax6.set_xlim(-130, -65)
    ax6.set_ylim(24, 50)
    ax6.set_title("Storm Events Geographic Distribution\n(colored by log damage)",
                  fontweight="bold", fontsize=12)
    plt.colorbar(scatter, ax=ax6, label="Log₁₀(Damage + 1)", shrink=0.8)

    # ── Panel 7: Hazard tier summary ───────────────────────────────────────
    ax7 = fig.add_subplot(gs[2, 2])
    if hazard_profiles is not None and "hazard_tier" in hazard_profiles.columns:
        tier_counts = hazard_profiles["hazard_tier"].value_counts()
        tier_colors = {"Low Risk": "#4CAF50", "Moderate Risk": "#FFC107",
                       "High Risk": "#FF9800", "Extreme Risk": "#F44336"}
        tc = [tier_colors.get(t, "gray") for t in tier_counts.index]
        ax7.bar(tier_counts.index, tier_counts.values, color=tc)
        ax7.set_ylabel("Number of Hazard Zones")
        ax7.set_title("Hazard Zone Risk Tiers", fontweight="bold", fontsize=12)
        plt.setp(ax7.xaxis.get_majorticklabels(), rotation=15, ha="right")
    else:
        ax7.axis("off")
        ax7.text(0.5, 0.5, "Hazard profiles\nnot available",
                ha="center", va="center", fontsize=12, color="gray",
                transform=ax7.transAxes)

    fig.suptitle("Storm Severity Mining: Executive Dashboard",
                 fontsize=18, fontweight="bold", y=0.98)
    save_figure(fig, "43_dashboard_summary", OUT_FIG)
    plt.close(fig)


def generate_narrative(df, hazard_profiles=None):
    """Generate a text narrative summarizing key findings."""
    logger.info("=" * 60)
    logger.info("NARRATIVE SUMMARY")
    logger.info("=" * 60)

    narrative = []
    narrative.append("STORM SEVERITY MINING AT SCALE — KEY FINDINGS")
    narrative.append("=" * 50)

    # Dataset scope
    narrative.append(f"\nDataset: {len(df):,} storm events across "
                    f"{df['STATE'].nunique()} states "
                    f"({int(df['YEAR'].min())}-{int(df['YEAR'].max())})")

    # Economic impact
    total_damage = df["TOTAL_DAMAGE"].sum()
    narrative.append(f"Total economic damage: {format_number(total_damage)}")
    narrative.append(f"Total deaths: {df['TOTAL_DEATHS'].sum():,}")
    narrative.append(f"Total injuries: {df['TOTAL_INJURIES'].sum():,}")

    # ── INSIGHT 1: Damage skewness ────────────────────────────────────────
    zero_pct = (df["TOTAL_DAMAGE"] == 0).mean() * 100
    damage_pos = df[df["TOTAL_DAMAGE"] > 0]["TOTAL_DAMAGE"]
    narrative.append(f"\n--- INSIGHT: Extreme Damage Skewness ---")
    narrative.append(f"{zero_pct:.1f}% of all events record zero economic damage.")
    narrative.append(f"Among damage-positive events, the top 1% account for "
                    f"{format_number(damage_pos.nlargest(int(len(damage_pos)*0.01)).sum())} "
                    f"({damage_pos.nlargest(int(len(damage_pos)*0.01)).sum()/total_damage*100:.1f}% "
                    f"of total damage).")
    narrative.append(f"This extreme right-skew (skewness={damage_pos.skew():.0f}) motivates "
                    f"log-transforming damage before quartile binning for classification.")

    # Most damaging types
    top_damaging = df.groupby("EVENT_TYPE")["TOTAL_DAMAGE"].sum().nlargest(3)
    narrative.append(f"\nMost economically damaging event types:")
    for etype, damage in top_damaging.items():
        narrative.append(f"  - {etype}: {format_number(damage)} "
                        f"({damage/total_damage*100:.1f}% of total)")

    # Deadliest types
    top_deadly = df.groupby("EVENT_TYPE")["TOTAL_DEATHS"].sum().nlargest(3)
    narrative.append(f"\nDeadliest event types:")
    for etype, deaths in top_deadly.items():
        narrative.append(f"  - {etype}: {deaths:,} deaths")

    # ── INSIGHT 2: Damage vs death mismatch ───────────────────────────────
    most_costly = top_damaging.index[0]
    most_deadly = top_deadly.index[0]
    if most_costly != most_deadly:
        narrative.append(f"\n--- INSIGHT: Economic vs Human Cost Divergence ---")
        narrative.append(f"The most economically damaging type ({most_costly}) differs "
                        f"from the deadliest ({most_deadly}). This mismatch implies that "
                        f"damage mitigation and casualty prevention require different "
                        f"resource allocation strategies.")

    # Temporal trend
    yearly_damage = df.groupby("YEAR")["TOTAL_DAMAGE"].sum()
    worst_year = yearly_damage.idxmax()
    narrative.append(f"\nWorst year for damage: {int(worst_year)} "
                    f"({format_number(yearly_damage[worst_year])})")

    # Seasonal pattern
    season_damage = df.groupby("SEASON")["TOTAL_DAMAGE"].sum()
    worst_season = season_damage.idxmax()
    narrative.append(f"Most damaging season: {worst_season} "
                    f"({format_number(season_damage[worst_season])})")

    # ── INSIGHT 3: Classification findings ────────────────────────────────
    narrative.append(f"\n--- INSIGHT: Prediction Limitations ---")
    narrative.append(f"Random Forest achieves ROC-AUC=0.852 but accuracy=63.4% on 4-class "
                    f"damage prediction. The gap reflects that adjacent damage tiers "
                    f"(e.g., Medium vs High) share nearly identical event-level features. "
                    f"Geographic features (lat/lon) dominate importance, confirming that "
                    f"damage is spatially driven rather than meteorologically determined.")

    # ── INSIGHT 4: Escalation patterns ────────────────────────────────────
    narrative.append(f"\n--- INSIGHT: Event Escalation Patterns ---")
    narrative.append(f"Sequential mining reveals Hail → Thunderstorm Wind as the dominant "
                    f"escalation pathway. The bidirectional nature (Hail ↔ TSW) suggests "
                    f"these are concurrent manifestations of the same convective system "
                    f"rather than true causal escalation.")

    # Hazard zones
    if hazard_profiles is not None:
        n_extreme = (hazard_profiles["hazard_tier"] == "Extreme Risk").sum()
        narrative.append(f"\nHazard Zones: {len(hazard_profiles)} identified, "
                        f"{n_extreme} classified as Extreme Risk")
        top_zone = hazard_profiles.nsmallest(1, "hazard_rank").iloc[0]
        narrative.append(f"Highest-risk zone: Cluster {int(top_zone['cluster_id'])} "
                        f"({top_zone['primary_state']}) — "
                        f"Score={top_zone['hazard_score']:.3f}")

    narrative_text = "\n".join(narrative)
    logger.info(narrative_text)

    # Save narrative
    narrative_path = OUT_RESULT / "narrative_summary.txt"
    with open(narrative_path, "w") as f:
        f.write(narrative_text)
    logger.info(f"Narrative saved: {narrative_path}")

    return narrative_text


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def run_storytelling(df, dbscan_data=None, hazard_profiles=None,
                     classification_results=None):
    """Run the full storytelling pipeline."""
    logger.info("=" * 60)
    logger.info("DATA STORYTELLING")
    logger.info("=" * 60)

    create_folium_map(df, dbscan_data, hazard_profiles)
    create_sankey_flows(df)
    create_dashboard_summary(df, classification_results, hazard_profiles)
    generate_narrative(df, hazard_profiles)

    logger.info("Storytelling complete.")
