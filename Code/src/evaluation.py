"""Evaluation: metrics tables, lift/gains, cost thresholds, Qini, cluster
profiles and the Revenue-at-Risk synthesis.

Origin: Code/04_result_analysis.ipynb.

Design rule: every function returns a DataFrame (or (fig, DataFrame)) so
results are inspectable and exportable, never just a printed blob.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)

from .eda import CONTEXT_COLOR, PALETTE


# --------------------------------------------------------------------------
# Lens A - classification
# --------------------------------------------------------------------------

def score_table(models: dict, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """One row per model: ranking metrics (threshold-free) + 0.5-threshold
    diagnostics. Ranking metrics decide the winner; the 0.5 columns only show
    why threshold tuning matters."""
    rows = []
    for name, model in models.items():
        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        rows.append({
            "model": name,
            "roc_auc": roc_auc_score(y_test, proba),
            "pr_auc": average_precision_score(y_test, proba),
            "recall@0.5": recall_score(y_test, pred),
            "precision@0.5": precision_score(y_test, pred),
            "f1@0.5": f1_score(y_test, pred),
        })
    return (pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
            .reset_index(drop=True).round(4))


def plot_roc_pr(models: dict, X_test: pd.DataFrame, y_test: pd.Series):
    """ROC + PR overlays. PR is the honest curve under 27% prevalence."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for (name, model), color in zip(models.items(), PALETTE):
        proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        prec, rec, _ = precision_recall_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        ax1.plot(fpr, tpr, label=f"{name} ({auc:.3f})", color=color, linewidth=2)
        ax2.plot(rec, prec, label=name, color=color, linewidth=2)
    ax1.plot([0, 1], [0, 1], color=CONTEXT_COLOR, linestyle="--", linewidth=1)
    ax2.axhline(y_test.mean(), color=CONTEXT_COLOR, linestyle="--", linewidth=1,
                label=f"prevalence ({y_test.mean():.2f})")
    ax1.set(title="ROC curves", xlabel="False positive rate", ylabel="True positive rate")
    ax2.set(title="Precision-Recall curves", xlabel="Recall", ylabel="Precision")
    for ax in (ax1, ax2):
        ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def lift_table(y_true: pd.Series, proba: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Decile analysis: sort by predicted risk, cut into 10 equal bins.
    'lift' = decile churn rate / base rate; 'cum_capture' = share of ALL
    churners found so far - the capacity-planning column."""
    df = pd.DataFrame({"y": np.asarray(y_true), "proba": proba})
    df["decile"] = pd.qcut(df["proba"].rank(method="first", ascending=False),
                           n_bins, labels=range(1, n_bins + 1))
    base = df["y"].mean()
    out = df.groupby("decile", observed=True).agg(
        n=("y", "size"), churners=("y", "sum"), churn_rate=("y", "mean"),
        min_proba=("proba", "min"),
    )
    out["lift"] = out["churn_rate"] / base
    out["cum_capture"] = out["churners"].cumsum() / out["churners"].sum()
    return out.round(3).reset_index()


def gains_chart(y_true: pd.Series, proba: np.ndarray, ax=None):
    """Cumulative gains vs the random-targeting diagonal."""
    order = np.argsort(-proba)
    y_sorted = np.asarray(y_true)[order]
    frac_customers = np.arange(1, len(y_sorted) + 1) / len(y_sorted)
    frac_churners = np.cumsum(y_sorted) / y_sorted.sum()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(frac_customers, frac_churners, color=PALETTE[0], linewidth=2,
            label="model")
    ax.plot([0, 1], [0, 1], color=CONTEXT_COLOR, linestyle="--", linewidth=1,
            label="random")
    ax.set(title="Cumulative gains", xlabel="Share of customers contacted",
           ylabel="Share of churners captured")
    ax.legend(frameon=False)
    return ax


def cost_optimal_threshold(
    y_true: pd.Series, proba: np.ndarray,
    cost_missed_churner: float = 500.0, cost_offer: float = 50.0,
) -> tuple[float, pd.DataFrame]:
    """Pick the threshold that minimizes total cost:
    FN x cost_missed + (everyone flagged) x cost_offer.
    Costs are parameters, not truths - Stage 4 shows sensitivity to them."""
    y = np.asarray(y_true)
    rows = []
    for t in np.round(np.arange(0.05, 0.96, 0.01), 2):
        pred = (proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
        cost = fn * cost_missed_churner + (tp + fp) * cost_offer
        rows.append({"threshold": t, "flagged": int(tp + fp), "missed": int(fn),
                     "caught": int(tp), "total_cost": cost})
    curve = pd.DataFrame(rows)
    best = float(curve.loc[curve["total_cost"].idxmin(), "threshold"])
    return best, curve


def calibration_table(y_true: pd.Series, proba: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Predicted vs observed churn rate per probability bin - trust check for
    using probabilities as numbers (Revenue-at-Risk multiplies them)."""
    df = pd.DataFrame({"y": np.asarray(y_true), "proba": proba})
    df["bin"] = pd.cut(df["proba"], bins=np.linspace(0, 1, n_bins + 1))
    out = df.groupby("bin", observed=True).agg(
        n=("y", "size"), mean_predicted=("proba", "mean"), observed_rate=("y", "mean"),
    ).dropna()
    return out.round(3).reset_index()


# --------------------------------------------------------------------------
# Lens C - uplift
# --------------------------------------------------------------------------

def qini_report(
    uplift_models: dict, X_val: pd.DataFrame, y_val: pd.Series, t_val: pd.Series
) -> pd.DataFrame:
    """Qini AUC + uplift@10%/@20% for every fitted uplift model."""
    from sklift.metrics import qini_auc_score, uplift_at_k

    rows = []
    for name, model in uplift_models.items():
        uplift = model.predict(X_val)
        rows.append({
            "model": name,
            "qini_auc": qini_auc_score(y_val, uplift, t_val),
            "uplift@10%": uplift_at_k(y_val, uplift, t_val, strategy="overall", k=0.1),
            "uplift@20%": uplift_at_k(y_val, uplift, t_val, strategy="overall", k=0.2),
        })
    return (pd.DataFrame(rows).sort_values("qini_auc", ascending=False)
            .reset_index(drop=True).round(4))


def uplift_by_decile(
    uplift_scores: np.ndarray, y_val: pd.Series, t_val: pd.Series, n_bins: int = 10
) -> pd.DataFrame:
    """Observed uplift (treated rate - control rate) per predicted-uplift decile.
    A good model shows real uplift concentrated in its top deciles."""
    df = pd.DataFrame({"uplift": uplift_scores,
                       "y": np.asarray(y_val), "t": np.asarray(t_val)})
    df["decile"] = pd.qcut(df["uplift"].rank(method="first", ascending=False),
                           n_bins, labels=range(1, n_bins + 1))
    def _observed(g: pd.DataFrame) -> float:
        treated, control = g[g["t"] == 1], g[g["t"] == 0]
        if len(treated) == 0 or len(control) == 0:
            return np.nan
        return treated["y"].mean() - control["y"].mean()
    out = df.groupby("decile", observed=True).apply(
        lambda g: pd.Series({
            "n": len(g),
            "mean_predicted_uplift": g["uplift"].mean(),
            "observed_uplift": _observed(g),
        }), include_groups=False)
    return out.round(4).reset_index()


# --------------------------------------------------------------------------
# Lens D - segmentation
# --------------------------------------------------------------------------

def cluster_profiles(
    df: pd.DataFrame, labels: np.ndarray,
    numeric_cols: list[str], cat_cols: list[str],
    target_col: str | None = None,
) -> pd.DataFrame:
    """One profile row per cluster: size, numeric means, dominant category per
    categorical, and (if given) the target rate - the business-readable card."""
    prof = df.copy()
    prof["cluster"] = labels
    rows = []
    for cluster, g in prof.groupby("cluster"):
        row = {"cluster": cluster, "size": len(g),
               "share": round(len(g) / len(prof), 3)}
        for c in numeric_cols:
            row[f"mean_{c}"] = round(g[c].mean(), 2)
        for c in cat_cols:
            top = g[c].mode().iloc[0]
            row[c] = f"{top} ({(g[c] == top).mean():.0%})"
        if target_col is not None:
            row[f"{target_col}_rate"] = round(g[target_col].mean(), 3)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)


# --------------------------------------------------------------------------
# Synthesis - Revenue at Risk
# --------------------------------------------------------------------------

def revenue_at_risk(
    df: pd.DataFrame, proba_col: str = "churn_probability", clv_col: str = "clv"
) -> tuple[pd.DataFrame, float]:
    """RaR = P(churn) x CLV per customer. Returns (ranked frame, total).
    This is the number the whole suite exists to produce: expected money lost
    if nobody acts, and who to call first."""
    out = df.copy()
    out["revenue_at_risk"] = out[proba_col] * out[clv_col]
    out = out.sort_values("revenue_at_risk", ascending=False)
    return out, float(out["revenue_at_risk"].sum())
