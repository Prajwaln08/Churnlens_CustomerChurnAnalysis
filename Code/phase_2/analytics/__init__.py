"""Analytics stage: the section-14 contract - population AND individual level.

insights   snapshot angle: drivers, lift, segments, revenue-at-risk
journeys   event angle: WHEN users churn in their journey, per time grain,
           per population filter, with individual timing estimates
"""
from . import insights, journeys
from .insights import enrich_and_summarize
from .journeys import (
    build_journeys,
    churn_timing,
    individual_outlook,
    retention_by_segment,
    retention_curve,
    segment_summary,
)
