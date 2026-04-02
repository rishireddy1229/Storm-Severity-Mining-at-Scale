"""
Preprocessing and feature engineering for the NOAA Storm Events dataset.

Handles:
  - Loading and concatenating all yearly CSVs
  - Damage string parsing (K/M/B suffixes)
  - Datetime parsing and duration computation
  - Season, hour, year extraction
  - Damage severity tier creation (quantile-based)
  - Coordinate cleaning
  - Saving processed parquet file
"""

import glob
import numpy as np
import pandas as pd
from pathlib import Path

from src.config import DATA_RAW, DATA_PROC, DAMAGE_TIERS, SEASON_MAP, RANDOM_SEED
from src.utils import parse_damage_column, month_to_season, parse_ef_scale, logger


def load_raw_data() -> pd.DataFrame:
    """Load and concatenate all raw StormEvents_details CSVs."""
    files = sorted(glob.glob(str(DATA_RAW / "StormEvents_details*.csv")))
    if not files:
        raise FileNotFoundError(f"No detail CSVs found in {DATA_RAW}")
    logger.info(f"Loading {len(files)} CSV files from {DATA_RAW}")

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Skipping {f}: {e}")
    storms = pd.concat(dfs, ignore_index=True)
    logger.info(f"Raw dataset: {len(storms):,} records, {storms.shape[1]} columns")
    return storms


def clean_and_engineer(storms: pd.DataFrame) -> pd.DataFrame:
    """Full preprocessing and feature engineering pipeline."""
    df = storms.copy()

    # ── Standardize column names to uppercase ──────────────────────────────
    df.columns = [c.upper() for c in df.columns]

    # ── Parse damage fields ────────────────────────────────────────────────
    logger.info("Parsing damage fields...")
    df["DAMAGE_PROPERTY_NUM"] = parse_damage_column(df["DAMAGE_PROPERTY"])
    df["DAMAGE_CROPS_NUM"]    = parse_damage_column(df["DAMAGE_CROPS"])
    df["TOTAL_DAMAGE"]        = df["DAMAGE_PROPERTY_NUM"] + df["DAMAGE_CROPS_NUM"]
    df["LOG_DAMAGE"]          = np.where(
        df["TOTAL_DAMAGE"] > 0, np.log10(df["TOTAL_DAMAGE"]), 0
    )

    # ── Parse datetime fields ──────────────────────────────────────────────
    logger.info("Parsing datetime fields...")
    for col in ["BEGIN_DATE_TIME", "END_DATE_TIME"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format="mixed", errors="coerce")

    # ── Duration (minutes) ─────────────────────────────────────────────────
    if "BEGIN_DATE_TIME" in df.columns and "END_DATE_TIME" in df.columns:
        df["DURATION_MIN"] = (
            (df["END_DATE_TIME"] - df["BEGIN_DATE_TIME"]).dt.total_seconds() / 60
        )
        # Clip negative durations and extreme outliers
        df["DURATION_MIN"] = df["DURATION_MIN"].clip(lower=0, upper=14400)  # max 10 days

    # ── Temporal features ──────────────────────────────────────────────────
    if "BEGIN_DATE_TIME" in df.columns:
        df["YEAR"]  = df["BEGIN_DATE_TIME"].dt.year
        df["MONTH"] = df["BEGIN_DATE_TIME"].dt.month
        df["HOUR"]  = df["BEGIN_DATE_TIME"].dt.hour
        df["DAY_OF_WEEK"] = df["BEGIN_DATE_TIME"].dt.dayofweek
        df["SEASON"] = df["MONTH"].map(month_to_season)

    # ── EF-scale numeric ───────────────────────────────────────────────────
    if "TOR_F_SCALE" in df.columns:
        df["EF_NUMERIC"] = df["TOR_F_SCALE"].apply(parse_ef_scale)

    # ── Clean coordinates ──────────────────────────────────────────────────
    for col in ["BEGIN_LAT", "BEGIN_LON", "END_LAT", "END_LON"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Sanity bounds for continental US + territories
    if "BEGIN_LAT" in df.columns:
        mask = (
            (df["BEGIN_LAT"].between(17, 72)) &
            (df["BEGIN_LON"].between(-180, -60))
        )
        invalid_coords = (~mask) & df["BEGIN_LAT"].notna()
        df.loc[invalid_coords, ["BEGIN_LAT", "BEGIN_LON"]] = np.nan
        logger.info(f"Invalidated {invalid_coords.sum()} out-of-bounds coordinates")

    # ── Compute path length (degrees, approximate) ─────────────────────────
    if all(c in df.columns for c in ["BEGIN_LAT", "BEGIN_LON", "END_LAT", "END_LON"]):
        df["PATH_LENGTH_DEG"] = np.sqrt(
            (df["END_LAT"] - df["BEGIN_LAT"])**2 +
            (df["END_LON"] - df["BEGIN_LON"])**2
        )

    # ── Episode size (co-occurrence scale of the storm system) ─────────────
    if "EPISODE_ID" in df.columns:
        ep_sizes = df["EPISODE_ID"].map(df.groupby("EPISODE_ID").size())
        df["EPISODE_SIZE"] = ep_sizes.fillna(1).clip(upper=100).astype(np.float32)
        logger.info(f"  EPISODE_SIZE: mean={df['EPISODE_SIZE'].mean():.1f}, "
                    f"max={df['EPISODE_SIZE'].max():.0f}")

    # ── Decade (climate-trend proxy) ───────────────────────────────────────
    if "YEAR" in df.columns:
        df["DECADE"] = (df["YEAR"] // 10 * 10).astype(np.float32)

    # ── Coastal indicator (geographic heuristic) ───────────────────────────
    if "BEGIN_LAT" in df.columns and "BEGIN_LON" in df.columns:
        df["IS_COASTAL"] = (
            (df["BEGIN_LON"] > -82) |                                   # East coast
            ((df["BEGIN_LAT"] < 32) & (df["BEGIN_LON"] > -98)) |       # Gulf coast
            (df["BEGIN_LON"] < -117)                                     # West coast
        ).astype(np.float32)
        logger.info(f"  IS_COASTAL: {df['IS_COASTAL'].mean()*100:.1f}% of events are coastal")

    # ── Human impact score ─────────────────────────────────────────────────
    for col in ["INJURIES_DIRECT", "INJURIES_INDIRECT", "DEATHS_DIRECT", "DEATHS_INDIRECT"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    df["TOTAL_INJURIES"] = df.get("INJURIES_DIRECT", 0) + df.get("INJURIES_INDIRECT", 0)
    df["TOTAL_DEATHS"]   = df.get("DEATHS_DIRECT", 0) + df.get("DEATHS_INDIRECT", 0)
    df["HUMAN_IMPACT"]   = df["TOTAL_INJURIES"] + 10 * df["TOTAL_DEATHS"]

    # ── Clean EVENT_TYPE ───────────────────────────────────────────────────
    if "EVENT_TYPE" in df.columns:
        df["EVENT_TYPE"] = df["EVENT_TYPE"].str.strip().str.title()

    # ── Clean STATE ────────────────────────────────────────────────────────
    if "STATE" in df.columns:
        df["STATE"] = df["STATE"].str.strip().str.title()

    return df


def create_damage_tiers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create quantile-based damage severity tiers.
    Only events with total_damage > 0 contribute to quantile computation.
    """
    damage_events = df.loc[df["TOTAL_DAMAGE"] > 0, "TOTAL_DAMAGE"]
    quantiles = damage_events.quantile(DAMAGE_TIERS["quantile_probs"]).values
    logger.info(f"Damage quantiles (25/50/75): ${quantiles[0]:,.0f} / ${quantiles[1]:,.0f} / ${quantiles[2]:,.0f}")

    conditions = [
        df["TOTAL_DAMAGE"] == 0,
        df["TOTAL_DAMAGE"] <= quantiles[0],
        df["TOTAL_DAMAGE"] <= quantiles[1],
        df["TOTAL_DAMAGE"] <= quantiles[2],
        df["TOTAL_DAMAGE"] > quantiles[2],
    ]
    choices = ["None", "Low", "Medium", "High", "Catastrophic"]
    df["DAMAGE_CLASS"] = np.select(conditions, choices, default="None")

    # Also store the quantile thresholds for later use
    df.attrs["damage_quantiles"] = quantiles.tolist()

    logger.info("Damage tier distribution:")
    for tier, count in df["DAMAGE_CLASS"].value_counts().items():
        logger.info(f"  {tier}: {count:,} ({count/len(df)*100:.1f}%)")

    return df


def run_preprocessing() -> pd.DataFrame:
    """Full preprocessing pipeline: load → clean → engineer → save."""
    # Load
    storms = load_raw_data()

    # Clean and engineer features
    storms = clean_and_engineer(storms)

    # Create damage tiers
    storms = create_damage_tiers(storms)

    # Save processed data
    out_path = DATA_PROC / "storms_processed.parquet"
    storms.to_parquet(out_path, index=False)
    logger.info(f"Saved processed data: {out_path} ({len(storms):,} records)")

    # Also save a summary
    summary = {
        "total_records": len(storms),
        "year_range": f"{storms['YEAR'].min()}-{storms['YEAR'].max()}" if "YEAR" in storms else "N/A",
        "n_event_types": storms["EVENT_TYPE"].nunique() if "EVENT_TYPE" in storms else 0,
        "n_states": storms["STATE"].nunique() if "STATE" in storms else 0,
        "pct_with_coords": (storms["BEGIN_LAT"].notna().mean() * 100) if "BEGIN_LAT" in storms else 0,
        "pct_with_damage": ((storms["TOTAL_DAMAGE"] > 0).mean() * 100),
        "total_economic_damage": storms["TOTAL_DAMAGE"].sum(),
        "total_deaths": storms["TOTAL_DEATHS"].sum() if "TOTAL_DEATHS" in storms else 0,
    }
    logger.info("Dataset summary:")
    for k, v in summary.items():
        logger.info(f"  {k}: {v}")

    return storms


if __name__ == "__main__":
    run_preprocessing()
