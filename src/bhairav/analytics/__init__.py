"""Phase 12 + Phase 18: Predictive Analytics & NL Summaries.

Phase 12: crowd forecasting, hotspots, trends.
Phase 18: NL alert summaries, predictive hotspot modeling, resource allocation.
"""
from .engine import AnalyticsEngine
from .forecast import CrowdDensityForecast
from .heatmap import SpatialHeatmap
from .trends import TrendAnalyzer
from .summarizer import NLAlertSummarizer, AlertSummary
from .hotspot import PredictiveHotspot, HotspotZone
from .allocation import ResourceAllocator, ResourceRecommendation

__all__ = [
    "AnalyticsEngine", "CrowdDensityForecast", "SpatialHeatmap",
    "TrendAnalyzer", "NLAlertSummarizer", "AlertSummary",
    "PredictiveHotspot", "HotspotZone", "ResourceAllocator",
    "ResourceRecommendation",
]
