"""Reusable EDA plotting helpers.

Origin: Code/02_eda_visualization.ipynb.

Chart conventions (validated palette, see notebook 02):
- categorical series colors come from PALETTE in fixed order, never cycled;
  gray CONTEXT_COLOR is reserved for "other/context" marks, never a series
- one measure per axis (no dual axes); sequential colormap for magnitude
  (cohort heatmap), diverging colormap only for polarity (correlations)
- bars carry direct value labels (relief for the low-contrast palette WARN);
  multi-series plots always get a legend
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 5-slot categorical palette - passes CVD/lightness/chroma validation.
PALETTE = ["#0173B2", "#DE8F05", "#029E73", "#D55E00", "#CC78BC"]
CONTEXT_COLOR = "#949494"  # reserved: context/"other", never a series
SEQUENTIAL_CMAP = "crest"  # magnitude (one hue, light->dark)
DIVERGING_CMAP = "vlag"    # polarity only (correlation), neutral midpoint


def set_style() -> None:
    """Apply the project plot style once per notebook/session."""
    sns.set_theme(style="whitegrid", palette=PALETTE)
    plt.rcParams.update({
        "figure.dpi": 110,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.alpha": 0.35,
        "axes.titlelocation": "left",
        "axes.titleweight": "bold",
    })


def plot_rate_by_category(
    df: pd.DataFrame,
    cat_col: str,
    target_col: str = "Churn",
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Axes:
    """Rate of a binary target per category, with direct % labels and n=.

    Single measure -> single hue (category identity is already on the axis).
    """
    stats = (
        df.groupby(cat_col, observed=True)[target_col]
        .agg(rate="mean", n="size")
        .sort_values("rate", ascending=False)
        .reset_index()
    )
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 0.5 + 0.45 * len(stats)))
    ax.barh(stats[cat_col].astype(str), stats["rate"], color=PALETTE[0], height=0.62)
    for i, row in stats.iterrows():
        ax.text(row["rate"] + 0.004, i, f"{row['rate']:.0%}  (n={row['n']:,})",
                va="center", fontsize=9, color="#333333")
    ax.set_xlim(0, stats["rate"].max() * 1.35)
    ax.invert_yaxis()
    ax.set_xlabel(f"{target_col} rate")
    ax.set_title(title or f"{target_col} rate by {cat_col}")
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    return ax


def plot_dist_by_target(
    df: pd.DataFrame,
    num_col: str,
    target_col: str = "Churn",
    labels: tuple[str, str] = ("Stayed", "Churned"),
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Distribution of a numeric column split by binary target (2 series -> legend)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    for value, label, color in zip((0, 1), labels, PALETTE[:2]):
        sns.kdeplot(df.loc[df[target_col] == value, num_col], fill=True,
                    alpha=0.35, color=color, label=label, ax=ax, warn_singular=False)
    ax.set_title(f"{num_col} by outcome")
    ax.legend(frameon=False)
    return ax


def churn_profile_table(
    df: pd.DataFrame, by: list[str], target_col: str = "Churn"
) -> pd.DataFrame:
    """Segmentation-flavored profiling: rate + size for every combination of `by`."""
    out = (
        df.groupby(by, observed=True)[target_col]
        .agg(rate="mean", n="size")
        .sort_values("rate", ascending=False)
    )
    out["rate"] = out["rate"].round(3)
    return out.reset_index()


def crosstab_heatmap(
    df: pd.DataFrame,
    row_col: str,
    col_col: str,
    target_col: str = "Churn",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Mean target rate for row x column combinations (magnitude -> sequential)."""
    pivot = df.pivot_table(index=row_col, columns=col_col, values=target_col,
                           aggfunc="mean", observed=True)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(pivot, annot=True, fmt=".0%", cmap=SEQUENTIAL_CMAP,
                linewidths=1.5, linecolor="white", cbar=False, ax=ax)
    ax.set_title(f"{target_col} rate: {row_col} x {col_col}")
    return ax


def corr_heatmap(df: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    """Correlation matrix - polarity, so diverging cmap centered at 0."""
    corr = df.select_dtypes("number").corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    if ax is None:
        _, ax = plt.subplots(figsize=(0.7 * len(corr) + 2, 0.6 * len(corr) + 1))
    sns.heatmap(corr, mask=mask, cmap=DIVERGING_CMAP, center=0, vmin=-1, vmax=1,
                annot=True, fmt=".2f", annot_kws={"size": 8},
                linewidths=1.2, linecolor="white", ax=ax)
    ax.set_title("Correlation matrix")
    return ax


def cumulative_churn_by_tenure(
    df: pd.DataFrame,
    group_col: str = "Contract",
    tenure_col: str = "tenure",
    target_col: str = "Churn",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Survival preview: share of a group's churners that left by each tenure month."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))
    for group, color in zip(df[group_col].unique(), PALETTE):
        sub = df[df[group_col] == group]
        churned = sub[sub[target_col] == 1]
        if churned.empty:
            continue
        months = np.arange(0, df[tenure_col].max() + 1)
        cum = [(churned[tenure_col] <= m).mean() for m in months]
        ax.plot(months, cum, label=str(group), color=color, linewidth=2)
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Cumulative share of churners")
    ax.set_title(f"When churners leave, by {group_col}")
    ax.legend(frameon=False, title=group_col)
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0%}")
    return ax


def cohort_retention_heatmap(
    retail: pd.DataFrame, max_periods: int = 12, ax: plt.Axes | None = None
) -> tuple[plt.Axes, pd.DataFrame]:
    """Monthly acquisition-cohort retention (the churn evidence for
    non-contractual data). Returns (ax, retention_table)."""
    df = retail[["CustomerID", "InvoiceDate"]].copy()
    df["order_month"] = df["InvoiceDate"].dt.to_period("M")
    df["cohort"] = df.groupby("CustomerID")["order_month"].transform("min")
    df["period"] = (df["order_month"] - df["cohort"]).apply(lambda d: d.n)
    counts = (
        df.groupby(["cohort", "period"])["CustomerID"].nunique().unstack(fill_value=0)
    )
    retention = counts.div(counts[0], axis=0).iloc[:, : max_periods + 1]
    retention.index = retention.index.astype(str)
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 0.45 * len(retention) + 1.5))
    sns.heatmap(retention, annot=True, fmt=".0%", annot_kws={"size": 7},
                cmap=SEQUENTIAL_CMAP, linewidths=1.0, linecolor="white",
                cbar=False, ax=ax)
    ax.set_title("Cohort retention (share of cohort active n months after first purchase)")
    ax.set_xlabel("Months since first purchase")
    ax.set_ylabel("Acquisition cohort")
    return ax, retention


def treatment_balance_report(
    df: pd.DataFrame,
    treatment_col: str = "treatment_group",
    outcome_col: str = "conversion",
) -> pd.DataFrame:
    """Group sizes + conversion per group + the naive uplift (difference in means)."""
    rep = (
        df.groupby(treatment_col)[outcome_col]
        .agg(n="size", conversions="sum", conversion_rate="mean")
        .reset_index()
    )
    rates = rep.set_index(treatment_col)["conversion_rate"]
    if len(rates) == 2:
        hi, lo = rates.max(), rates.min()
        print(f"Naive average treatment effect: {hi - lo:+.4f} "
              f"({hi:.4f} vs {lo:.4f})")
    return rep
