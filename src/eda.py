"""
Exploratory Data Analysis (EDA) for the Storm Events dataset.

Generates publication-quality figures:
  1. Yearly event count trend
  2. Top 15 event types by frequency
  3. Damage distribution by event type (log-scale box plots)
  4. Seasonal distribution heatmap (event_type x season)
  5. Geographic heatmap of event density
  6. Correlation matrix of numeric features
  7. Missing value analysis
  8. Temporal patterns (hourly, monthly, day-of-week)
  9. Damage vs. human impact scatter
  10. State-level damage ranking
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
import pandas as pd

from src.config import OUT_FIG, OUT_RESULT
from src.utils import logger, save_figure, save_results, format_number

sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE = "viridis"


def plot_yearly_trend(df):
    """1. Yearly event count trend with damage overlay."""
    fig, ax1 = plt.subplots(figsize=(14, 5))
    yearly = df.groupby("YEAR").agg(
        count=("EVENT_ID", "count"),
        total_damage=("TOTAL_DAMAGE", "sum"),
    ).reset_index()

    color1 = "#2196F3"
    ax1.bar(yearly["YEAR"], yearly["count"], color=color1, alpha=0.7, label="Event Count")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Number of Events", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = "#F44336"
    ax2.plot(yearly["YEAR"], yearly["total_damage"] / 1e9, color=color2, linewidth=2.5,
             marker="o", markersize=4, label="Total Damage ($B)")
    ax2.set_ylabel("Total Damage ($ Billions)", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    fig.suptitle("U.S. Storm Events: Annual Count and Economic Damage (1996-2024)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "01_yearly_trend", OUT_FIG)
    plt.close(fig)


def plot_top_event_types(df):
    """2. Top 15 event types by frequency."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    top15 = df["EVENT_TYPE"].value_counts().head(15)
    colors = sns.color_palette("Blues_r", 15)
    axes[0].barh(top15.index[::-1], top15.values[::-1], color=colors[::-1])
    axes[0].set_xlabel("Number of Events")
    axes[0].set_title("Top 15 Event Types by Frequency", fontweight="bold")
    for i, v in enumerate(top15.values[::-1]):
        axes[0].text(v + top15.max() * 0.01, i, f"{v:,}", va="center", fontsize=9)

    # By total damage
    top15_dmg = (df.groupby("EVENT_TYPE")["TOTAL_DAMAGE"].sum()
                 .sort_values(ascending=False).head(15))
    colors2 = sns.color_palette("Reds_r", 15)
    axes[1].barh(top15_dmg.index[::-1], top15_dmg.values[::-1] / 1e9, color=colors2[::-1])
    axes[1].set_xlabel("Total Damage ($ Billions)")
    axes[1].set_title("Top 15 Event Types by Total Damage", fontweight="bold")

    fig.tight_layout()
    save_figure(fig, "02_top_event_types", OUT_FIG)
    plt.close(fig)


def plot_damage_distribution(df):
    """3. Damage distribution by event type (log-scale box/violin)."""
    top10 = df["EVENT_TYPE"].value_counts().head(10).index
    subset = df[df["EVENT_TYPE"].isin(top10) & (df["TOTAL_DAMAGE"] > 0)].copy()
    subset["LOG_DAMAGE"] = np.log10(subset["TOTAL_DAMAGE"])

    fig, ax = plt.subplots(figsize=(14, 7))
    order = (subset.groupby("EVENT_TYPE")["LOG_DAMAGE"].median()
             .sort_values(ascending=False).index)
    sns.violinplot(data=subset, y="EVENT_TYPE", x="LOG_DAMAGE", order=order,
                   palette="YlOrRd", inner="quartile", ax=ax, cut=0)
    ax.set_xlabel("Log₁₀(Total Damage $)")
    ax.set_ylabel("")
    ax.set_title("Damage Distribution by Event Type (Events with Damage > 0)",
                 fontweight="bold")
    ax.axvline(x=6, color="red", linestyle="--", alpha=0.5, label="$1M threshold")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, "03_damage_distribution", OUT_FIG)
    plt.close(fig)


def plot_seasonal_heatmap(df):
    """4. Seasonal distribution heatmap (event_type x season)."""
    top12 = df["EVENT_TYPE"].value_counts().head(12).index
    subset = df[df["EVENT_TYPE"].isin(top12)]
    season_order = ["Winter", "Spring", "Summer", "Fall"]
    ct = pd.crosstab(subset["EVENT_TYPE"], subset["SEASON"])
    ct = ct.reindex(columns=season_order, fill_value=0)
    # Normalize per event type (row-wise percentage)
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(ct_pct, annot=True, fmt=".0f", cmap="YlOrBr", ax=ax,
                linewidths=0.5, cbar_kws={"label": "% of Events"})
    ax.set_title("Seasonal Distribution of Top Event Types (% within type)",
                 fontweight="bold")
    ax.set_ylabel("")
    fig.tight_layout()
    save_figure(fig, "04_seasonal_heatmap", OUT_FIG)
    plt.close(fig)


def plot_geographic_density(df):
    """5. Geographic scatter/hexbin of event density."""
    geo = df.dropna(subset=["BEGIN_LAT", "BEGIN_LON"])

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Hexbin of all events
    hb = axes[0].hexbin(geo["BEGIN_LON"], geo["BEGIN_LAT"], gridsize=80,
                        cmap="YlOrRd", mincnt=1, bins="log")
    axes[0].set_xlim(-130, -65)
    axes[0].set_ylim(24, 50)
    axes[0].set_title("All Storm Events – Log Density", fontweight="bold")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    plt.colorbar(hb, ax=axes[0], label="Log(Count)")

    # High-damage events only
    high = geo[geo["DAMAGE_CLASS"] == "Catastrophic"]
    hb2 = axes[1].hexbin(high["BEGIN_LON"], high["BEGIN_LAT"], gridsize=60,
                         cmap="hot_r", mincnt=1, bins="log")
    axes[1].set_xlim(-130, -65)
    axes[1].set_ylim(24, 50)
    axes[1].set_title("Catastrophic Damage Events – Log Density", fontweight="bold")
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")
    plt.colorbar(hb2, ax=axes[1], label="Log(Count)")

    fig.tight_layout()
    save_figure(fig, "05_geographic_density", OUT_FIG)
    plt.close(fig)


def plot_correlation_matrix(df):
    """6. Correlation matrix of numeric features."""
    numeric_cols = [
        "TOTAL_DAMAGE", "DAMAGE_PROPERTY_NUM", "DAMAGE_CROPS_NUM",
        "DURATION_MIN", "INJURIES_DIRECT", "DEATHS_DIRECT",
        "TOTAL_INJURIES", "TOTAL_DEATHS", "HUMAN_IMPACT",
        "BEGIN_LAT", "BEGIN_LON", "MAGNITUDE",
    ]
    available = [c for c in numeric_cols if c in df.columns]
    corr = df[available].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, ax=ax, linewidths=0.5, square=True,
                vmin=-1, vmax=1)
    ax.set_title("Correlation Matrix of Numeric Features", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "06_correlation_matrix", OUT_FIG)
    plt.close(fig)


def plot_missing_values(df):
    """7. Missing value analysis."""
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False).head(25)
    missing_pct = (missing / len(df) * 100)

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(missing_pct.index[::-1], missing_pct.values[::-1],
                   color=sns.color_palette("Oranges_r", len(missing_pct)))
    ax.set_xlabel("% Missing")
    ax.set_title("Top 25 Columns by Missing Value Percentage", fontweight="bold")
    for bar, pct in zip(bars, missing_pct.values[::-1]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f"{pct:.1f}%", va="center", fontsize=9)
    fig.tight_layout()
    save_figure(fig, "07_missing_values", OUT_FIG)
    plt.close(fig)


def plot_temporal_patterns(df):
    """8. Hourly, monthly, and day-of-week patterns."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Hourly
    hourly = df.groupby("HOUR").size()
    axes[0].bar(hourly.index, hourly.values, color="#4CAF50", alpha=0.8)
    axes[0].set_xlabel("Hour of Day")
    axes[0].set_ylabel("Number of Events")
    axes[0].set_title("Events by Hour of Day", fontweight="bold")
    axes[0].set_xticks(range(0, 24, 3))

    # Monthly
    monthly = df.groupby("MONTH").size()
    axes[1].bar(monthly.index, monthly.values, color="#FF9800", alpha=0.8)
    axes[1].set_xlabel("Month")
    axes[1].set_ylabel("Number of Events")
    axes[1].set_title("Events by Month", fontweight="bold")
    axes[1].set_xticks(range(1, 13))
    axes[1].set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"])

    # Day of week
    dow = df.groupby("DAY_OF_WEEK").size()
    axes[2].bar(dow.index, dow.values, color="#9C27B0", alpha=0.8)
    axes[2].set_xlabel("Day of Week")
    axes[2].set_ylabel("Number of Events")
    axes[2].set_title("Events by Day of Week", fontweight="bold")
    axes[2].set_xticks(range(7))
    axes[2].set_xticklabels(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])

    fig.suptitle("Temporal Patterns in U.S. Storm Events", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_figure(fig, "08_temporal_patterns", OUT_FIG)
    plt.close(fig)


def plot_damage_vs_impact(df):
    """9. Damage vs. human impact scatter."""
    subset = df[(df["TOTAL_DAMAGE"] > 0) & (df["HUMAN_IMPACT"] > 0)].copy()
    if len(subset) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        np.log10(subset["TOTAL_DAMAGE"]),
        np.log10(subset["HUMAN_IMPACT"] + 1),
        c=subset["DURATION_MIN"].clip(upper=500),
        cmap="plasma", alpha=0.3, s=8, edgecolors="none",
    )
    ax.set_xlabel("Log₁₀(Total Damage $)")
    ax.set_ylabel("Log₁₀(Human Impact Score + 1)")
    ax.set_title("Economic Damage vs. Human Impact (colored by duration)",
                 fontweight="bold")
    plt.colorbar(scatter, ax=ax, label="Duration (min, clipped at 500)")
    fig.tight_layout()
    save_figure(fig, "09_damage_vs_impact", OUT_FIG)
    plt.close(fig)


def plot_state_ranking(df):
    """10. State-level damage and event count ranking."""
    state_stats = df.groupby("STATE").agg(
        event_count=("EVENT_ID", "count"),
        total_damage=("TOTAL_DAMAGE", "sum"),
        total_deaths=("TOTAL_DEATHS", "sum"),
    ).sort_values("total_damage", ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(state_stats.index[::-1],
                   state_stats["total_damage"].values[::-1] / 1e9,
                   color=sns.color_palette("Reds_r", 20))
    ax.set_xlabel("Total Damage ($ Billions)")
    ax.set_title("Top 20 States by Total Storm Damage (1996-2024)", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "10_state_ranking", OUT_FIG)
    plt.close(fig)


def plot_damage_class_analysis(df):
    """
    Analyze and visualize damage class distribution, skewness, and
    the log-transform binning methodology.
    Addresses the inherent skewness in storm damage data.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel 1: Raw damage distribution (extreme right skew)
    damage_pos = df[df["TOTAL_DAMAGE"] > 0]["TOTAL_DAMAGE"]
    zero_pct = (df["TOTAL_DAMAGE"] == 0).mean() * 100
    axes[0, 0].hist(damage_pos.clip(upper=damage_pos.quantile(0.99)),
                     bins=100, color="#2196F3", alpha=0.7, edgecolor="white")
    axes[0, 0].set_xlabel("Total Damage ($)")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_title(f"Raw Damage Distribution (clipped at 99th pctl)\n"
                          f"Zero-damage events: {zero_pct:.1f}% of all records",
                          fontweight="bold")
    axes[0, 0].xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, p: f"${x/1e6:.1f}M" if x >= 1e6 else f"${x/1e3:.0f}K"))

    # Panel 2: Log-transformed damage (near-normal after transform)
    log_damage = np.log10(damage_pos)
    axes[0, 1].hist(log_damage, bins=80, color="#4CAF50", alpha=0.7,
                     edgecolor="white", density=True)
    # Overlay quartile thresholds
    quantiles = damage_pos.quantile([0.25, 0.50, 0.75]).values
    colors_q = ["#FFC107", "#FF9800", "#F44336"]
    labels_q = ["Q1 (Low/Med)", "Q2 (Med/High)", "Q3 (High/Cat)"]
    for q, c, lbl in zip(quantiles, colors_q, labels_q):
        axes[0, 1].axvline(np.log10(q), color=c, linestyle="--", linewidth=2,
                            label=f"{lbl}: ${q:,.0f}")
    axes[0, 1].set_xlabel("Log₁₀(Total Damage $)")
    axes[0, 1].set_ylabel("Density")
    axes[0, 1].set_title("Log-Transformed Damage with Quartile Thresholds",
                          fontweight="bold")
    axes[0, 1].legend(fontsize=8, loc="upper left")

    # Panel 3: Class size comparison (4 classification classes only)
    class_order = ["Low", "Medium", "High", "Catastrophic"]
    class_counts = df[df["DAMAGE_CLASS"].isin(class_order)]["DAMAGE_CLASS"].value_counts()
    class_counts = class_counts.reindex(class_order)
    colors_cls = ["#4CAF50", "#FFC107", "#FF9800", "#F44336"]
    bars = axes[1, 0].bar(class_counts.index, class_counts.values, color=colors_cls)
    for bar, v in zip(bars, class_counts.values):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, v + 500,
                        f"{v:,}\n({v/class_counts.sum()*100:.1f}%)",
                        ha="center", va="bottom", fontsize=9)
    axes[1, 0].set_ylabel("Number of Events")
    axes[1, 0].set_title("Classification Target Distribution\n"
                          "(zero-damage events excluded from classification)",
                          fontweight="bold")

    # Panel 4: Damage range per class (box plot on log scale)
    subset = df[df["DAMAGE_CLASS"].isin(class_order)].copy()
    subset["LOG_DAMAGE"] = np.log10(subset["TOTAL_DAMAGE"])
    class_order_map = {c: i for i, c in enumerate(class_order)}
    subset["class_order"] = subset["DAMAGE_CLASS"].map(class_order_map)
    subset = subset.sort_values("class_order")
    bp = axes[1, 1].boxplot(
        [subset[subset["DAMAGE_CLASS"] == c]["LOG_DAMAGE"].values for c in class_order],
        labels=class_order, patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], colors_cls):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    axes[1, 1].set_ylabel("Log₁₀(Total Damage $)")
    axes[1, 1].set_title("Damage Range per Classification Tier",
                          fontweight="bold")

    # Add threshold annotations
    for i, (q, lbl) in enumerate(zip(quantiles, ["Q1", "Q2", "Q3"])):
        axes[1, 1].axhline(np.log10(q), color="gray", linestyle=":", alpha=0.5)
        axes[1, 1].text(4.3, np.log10(q), f"${q:,.0f}", fontsize=8, va="center")

    fig.suptitle("Damage Severity Classification: Distribution Analysis",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    save_figure(fig, "03b_damage_class_analysis", OUT_FIG)
    plt.close(fig)

    # Log key statistics
    logger.info("─" * 40)
    logger.info("DAMAGE DISTRIBUTION ANALYSIS")
    logger.info("─" * 40)
    logger.info(f"  Total events: {len(df):,}")
    logger.info(f"  Zero-damage events: {(df['TOTAL_DAMAGE']==0).sum():,} ({zero_pct:.1f}%)")
    logger.info(f"  Damage-positive events: {len(damage_pos):,} ({100-zero_pct:.1f}%)")
    logger.info(f"  Raw damage skewness: {damage_pos.skew():.1f}")
    logger.info(f"  Log-damage skewness: {log_damage.skew():.2f}")
    logger.info(f"  Quartile thresholds: Q1=${quantiles[0]:,.0f}, "
                f"Q2=${quantiles[1]:,.0f}, Q3=${quantiles[2]:,.0f}")
    logger.info(f"  Classification class sizes (excl. None):")
    for c in class_order:
        n = class_counts[c]
        logger.info(f"    {c}: {n:,} ({n/class_counts.sum()*100:.1f}%)")


def generate_summary_statistics(df):
    """Generate and save summary statistics tables."""
    # Event type statistics
    type_stats = df.groupby("EVENT_TYPE").agg(
        count=("EVENT_ID", "count"),
        total_damage=("TOTAL_DAMAGE", "sum"),
        mean_damage=("TOTAL_DAMAGE", "mean"),
        median_damage=("TOTAL_DAMAGE", "median"),
        max_damage=("TOTAL_DAMAGE", "max"),
        total_deaths=("TOTAL_DEATHS", "sum"),
        total_injuries=("TOTAL_INJURIES", "sum"),
        mean_duration=("DURATION_MIN", "mean"),
        pct_with_damage=("TOTAL_DAMAGE", lambda x: (x > 0).mean() * 100),
    ).sort_values("total_damage", ascending=False)
    save_results(type_stats.reset_index(), "eda_event_type_stats", OUT_RESULT)

    # State statistics
    state_stats = df.groupby("STATE").agg(
        count=("EVENT_ID", "count"),
        total_damage=("TOTAL_DAMAGE", "sum"),
        mean_damage=("TOTAL_DAMAGE", "mean"),
        total_deaths=("TOTAL_DEATHS", "sum"),
    ).sort_values("total_damage", ascending=False)
    save_results(state_stats.reset_index(), "eda_state_stats", OUT_RESULT)

    # Damage class distribution
    class_stats = df.groupby("DAMAGE_CLASS").agg(
        count=("EVENT_ID", "count"),
        mean_damage=("TOTAL_DAMAGE", "mean"),
        median_damage=("TOTAL_DAMAGE", "median"),
    )
    save_results(class_stats.reset_index(), "eda_damage_class_stats", OUT_RESULT)

    return type_stats, state_stats


def run_eda(df: pd.DataFrame):
    """Run the complete EDA pipeline."""
    logger.info("=" * 60)
    logger.info("EXPLORATORY DATA ANALYSIS")
    logger.info("=" * 60)

    plot_yearly_trend(df)
    plot_top_event_types(df)
    plot_damage_distribution(df)
    plot_damage_class_analysis(df)
    plot_seasonal_heatmap(df)
    plot_geographic_density(df)
    plot_correlation_matrix(df)
    plot_missing_values(df)
    plot_temporal_patterns(df)
    plot_damage_vs_impact(df)
    plot_state_ranking(df)
    generate_summary_statistics(df)

    logger.info(f"EDA complete. Figures saved to {OUT_FIG}")


if __name__ == "__main__":
    df = pd.read_parquet(DATA_PROC / "storms_processed.parquet")
    run_eda(df)
