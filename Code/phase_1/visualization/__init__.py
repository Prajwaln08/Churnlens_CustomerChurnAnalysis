"""EDA plotting helpers. Public API re-exported from eda.py."""
from . import eda
from .eda import (
    CONTEXT_COLOR,
    DIVERGING_CMAP,
    PALETTE,
    SEQUENTIAL_CMAP,
    churn_profile_table,
    cohort_retention_heatmap,
    corr_heatmap,
    crosstab_heatmap,
    cumulative_churn_by_tenure,
    plot_dist_by_target,
    plot_rate_by_category,
    set_style,
    treatment_balance_report,
)
