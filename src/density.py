"""
Density Estimation module (Chapter V).

Performs:
  1. Kernel Density Estimation (KDE) of log-damage by event type
  2. Parametric MLE fitting (lognormal, exponential, Weibull)
  3. Goodness-of-fit comparison (KS-test, AIC/BIC)
  4. Spatial KDE for event locations
  5. Bivariate KDE for damage × duration
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import gaussian_kde

from src.config import OUT_FIG, OUT_RESULT
from src.utils import logger, save_figure, save_results


def kde_by_event_type(df, top_n=10):
    """KDE of log-damage for top event types."""
    top_types = (df[df["TOTAL_DAMAGE"] > 0]["EVENT_TYPE"]
                 .value_counts().head(top_n).index)

    fig, ax = plt.subplots(figsize=(14, 7))
    colors = sns.color_palette("husl", top_n)

    for i, etype in enumerate(top_types):
        data = df[(df["EVENT_TYPE"] == etype) & (df["TOTAL_DAMAGE"] > 0)]["TOTAL_DAMAGE"]
        log_data = np.log10(data)
        kde = gaussian_kde(log_data, bw_method="silverman")
        x_grid = np.linspace(log_data.min() - 0.5, log_data.max() + 0.5, 500)
        ax.plot(x_grid, kde(x_grid), color=colors[i], linewidth=2, label=etype)
        ax.fill_between(x_grid, kde(x_grid), alpha=0.08, color=colors[i])

    ax.set_xlabel("Log₁₀(Total Damage $)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Kernel Density Estimation of Damage by Event Type\n(Silverman bandwidth)",
                 fontsize=14, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)

    # Add reference lines
    for val, label in [(3, "$1K"), (6, "$1M"), (9, "$1B")]:
        ax.axvline(x=val, color="gray", linestyle=":", alpha=0.5)
        ax.text(val, ax.get_ylim()[1] * 0.95, label, fontsize=8, color="gray")

    fig.tight_layout()
    save_figure(fig, "11_kde_damage_by_type", OUT_FIG)
    plt.close(fig)


def parametric_fitting(df):
    """
    Fit parametric distributions to tornado damage and compare with KDE.
    Tests: lognormal, exponential, gamma, Weibull.
    """
    tornado_dmg = df[(df["EVENT_TYPE"] == "Tornado") & (df["TOTAL_DAMAGE"] > 0)]["TOTAL_DAMAGE"].values
    if len(tornado_dmg) < 100:
        logger.warning("Not enough tornado damage data for parametric fitting")
        return

    log_dmg = np.log10(tornado_dmg)

    distributions = {
        "Lognormal": stats.lognorm,
        "Exponential": stats.expon,
        "Gamma": stats.gamma,
        "Weibull": stats.weibull_min,
    }

    fit_results = []
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (name, dist) in enumerate(distributions.items()):
        ax = axes[idx]
        try:
            params = dist.fit(tornado_dmg)
            # KS test
            ks_stat, ks_pval = stats.kstest(tornado_dmg, dist.cdf, args=params)

            # Log-likelihood, AIC, BIC
            ll = np.sum(dist.logpdf(tornado_dmg, *params))
            k = len(params)
            n = len(tornado_dmg)
            aic = 2 * k - 2 * ll
            bic = k * np.log(n) - 2 * ll

            fit_results.append({
                "Distribution": name,
                "KS_Statistic": ks_stat,
                "KS_pvalue": ks_pval,
                "Log_Likelihood": ll,
                "AIC": aic,
                "BIC": bic,
                "n_params": k,
            })

            # Plot
            ax.hist(log_dmg, bins=50, density=True, alpha=0.5, color="steelblue",
                    label="Empirical")
            x_fit = np.linspace(tornado_dmg.min(), tornado_dmg.max(), 1000)
            pdf_vals = dist.pdf(x_fit, *params)
            # Transform to log10 scale
            x_log = np.log10(x_fit)
            # We need to adjust the PDF for the log transform
            ax.plot(x_log, pdf_vals * x_fit * np.log(10), color="red", linewidth=2,
                    label=f"{name} fit")

            # Also plot KDE
            kde = gaussian_kde(log_dmg, bw_method="silverman")
            x_kde = np.linspace(log_dmg.min() - 0.5, log_dmg.max() + 0.5, 500)
            ax.plot(x_kde, kde(x_kde), color="green", linewidth=2, linestyle="--",
                    label="KDE")

            ax.set_title(f"{name}\nKS={ks_stat:.4f}, AIC={aic:.0f}", fontsize=11)
            ax.legend(fontsize=8)
            ax.set_xlabel("Log₁₀(Damage $)")

        except Exception as e:
            logger.warning(f"Failed to fit {name}: {e}")
            ax.text(0.5, 0.5, f"{name}\nFit failed", transform=ax.transAxes,
                    ha="center", va="center")

    fig.suptitle("Parametric Distribution Fitting: Tornado Damage",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "12_parametric_fitting", OUT_FIG)
    plt.close(fig)

    if fit_results:
        results_df = pd.DataFrame(fit_results).sort_values("AIC")
        save_results(results_df, "density_parametric_fits", OUT_RESULT)
        logger.info("Parametric fit comparison (by AIC):")
        for _, row in results_df.iterrows():
            logger.info(f"  {row['Distribution']}: AIC={row['AIC']:.0f}, "
                       f"KS={row['KS_Statistic']:.4f} (p={row['KS_pvalue']:.4f})")
        return results_df


def bandwidth_sensitivity(df):
    """Sensitivity analysis of KDE bandwidth selection."""
    tornado_dmg = df[(df["EVENT_TYPE"] == "Tornado") & (df["TOTAL_DAMAGE"] > 0)]["TOTAL_DAMAGE"]
    if len(tornado_dmg) < 100:
        return

    log_dmg = np.log10(tornado_dmg)

    fig, ax = plt.subplots(figsize=(12, 6))
    bandwidths = [0.05, 0.1, 0.2, "silverman", "scott", 0.5]
    colors = sns.color_palette("Set2", len(bandwidths))
    x_grid = np.linspace(log_dmg.min() - 1, log_dmg.max() + 1, 500)

    ax.hist(log_dmg, bins=80, density=True, alpha=0.3, color="gray", label="Histogram")

    for bw, color in zip(bandwidths, colors):
        kde = gaussian_kde(log_dmg, bw_method=bw)
        ax.plot(x_grid, kde(x_grid), color=color, linewidth=2,
                label=f"bw={bw}")

    ax.set_xlabel("Log₁₀(Total Damage $)")
    ax.set_ylabel("Density")
    ax.set_title("KDE Bandwidth Sensitivity (Tornado Damage)", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, "13_bandwidth_sensitivity", OUT_FIG)
    plt.close(fig)


def bivariate_kde(df):
    """Bivariate KDE: damage × duration."""
    subset = df[(df["TOTAL_DAMAGE"] > 0) &
                (df["DURATION_MIN"] > 0) &
                (df["DURATION_MIN"] < 1440)].copy()  # < 24 hours

    if len(subset) < 100:
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    x = np.log10(subset["TOTAL_DAMAGE"])
    y = np.log10(subset["DURATION_MIN"] + 1)

    ax.hexbin(x, y, gridsize=50, cmap="YlOrRd", mincnt=1, bins="log")
    plt.colorbar(ax.collections[0], ax=ax, label="Log(Count)")

    # Overlay KDE contours
    try:
        from scipy.stats import gaussian_kde as gkde
        xy = np.vstack([x, y])
        kde = gkde(xy)
        xgrid = np.linspace(x.min(), x.max(), 100)
        ygrid = np.linspace(y.min(), y.max(), 100)
        X, Y = np.meshgrid(xgrid, ygrid)
        Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
        ax.contour(X, Y, Z, levels=8, colors="white", linewidths=0.8, alpha=0.7)
    except Exception:
        pass

    ax.set_xlabel("Log₁₀(Total Damage $)")
    ax.set_ylabel("Log₁₀(Duration min + 1)")
    ax.set_title("Bivariate Density: Damage × Duration", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "14_bivariate_kde", OUT_FIG)
    plt.close(fig)


def run_density_estimation(df: pd.DataFrame):
    """Run the full density estimation pipeline."""
    logger.info("=" * 60)
    logger.info("DENSITY ESTIMATION")
    logger.info("=" * 60)

    kde_by_event_type(df)
    parametric_fitting(df)
    bandwidth_sensitivity(df)
    bivariate_kde(df)

    logger.info("Density estimation complete.")


if __name__ == "__main__":
    from src.config import DATA_PROC
    df = pd.read_parquet(DATA_PROC / "storms_processed.parquet")
    run_density_estimation(df)
