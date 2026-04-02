"""
Shared utility functions for the Storm Severity Mining pipeline.
"""

import re
import numpy as np
import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("storm_mining")


# ── Damage string parser ──────────────────────────────────────────────────
_DAMAGE_RE = re.compile(r"^([\d.]+)\s*([KMBkmb])?$")

def parse_damage(s):
    """
    Parse NOAA damage strings like '25K', '1.5M', '0.00B' into floats.
    Returns 0.0 for missing / unparseable values.
    """
    if pd.isna(s) or s == "":
        return 0.0
    s = str(s).strip().upper()
    if s in ("0", "0.0", "0.00"):
        return 0.0
    m = _DAMAGE_RE.match(s)
    if m is None:
        try:
            return float(s)
        except ValueError:
            return 0.0
    val = float(m.group(1))
    suffix = m.group(2)
    if suffix == "K":
        return val * 1e3
    elif suffix == "M":
        return val * 1e6
    elif suffix == "B":
        return val * 1e9
    return val


def parse_damage_column(series: pd.Series) -> pd.Series:
    """Vectorised damage parsing for a whole column."""
    return series.apply(parse_damage)


# ── Season mapping ─────────────────────────────────────────────────────────
def month_to_season(month: int) -> str:
    if month in (12, 1, 2):
        return "Winter"
    elif month in (3, 4, 5):
        return "Spring"
    elif month in (6, 7, 8):
        return "Summer"
    else:
        return "Fall"


# ── EF-scale parser ───────────────────────────────────────────────────────
_EF_MAP = {"EF0": 0, "EF1": 1, "EF2": 2, "EF3": 3, "EF4": 4, "EF5": 5,
           "F0": 0, "F1": 1, "F2": 2, "F3": 3, "F4": 4, "F5": 5}

def parse_ef_scale(s):
    if pd.isna(s):
        return np.nan
    return _EF_MAP.get(str(s).strip().upper(), np.nan)


# ── Reporting helpers ──────────────────────────────────────────────────────
def save_figure(fig, name: str, out_dir: Path, dpi: int = 300, tight: bool = True):
    """Save a matplotlib figure to output/figures/."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    if tight:
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    else:
        fig.savefig(path, dpi=dpi, facecolor="white")
    logger.info(f"Saved figure: {path}")
    return path


def save_results(df: pd.DataFrame, name: str, out_dir: Path):
    """Save a DataFrame as CSV to output/results/."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.csv"
    df.to_csv(path, index=False)
    logger.info(f"Saved results: {path}")
    return path


def format_number(n):
    """Format large numbers for display."""
    if abs(n) >= 1e9:
        return f"${n/1e9:.1f}B"
    elif abs(n) >= 1e6:
        return f"${n/1e6:.1f}M"
    elif abs(n) >= 1e3:
        return f"${n/1e3:.0f}K"
    return f"${n:.0f}"
