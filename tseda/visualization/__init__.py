"""
tseda.visualization — matplotlib plot suite for time series EDA.

All functions return a :class:`matplotlib.figure.Figure`.  Call
:func:`set_style` once at the start of a session to apply the tseda theme.

Style
-----
.. autofunction:: tseda.visualization.base.set_style

Time plots
----------
.. autofunction:: tseda.visualization.time_plots.plot_series
.. autofunction:: tseda.visualization.time_plots.plot_seasonal_subseries
.. autofunction:: tseda.visualization.time_plots.plot_lag
.. autofunction:: tseda.visualization.time_plots.plot_calendar_heatmap
.. autofunction:: tseda.visualization.time_plots.plot_annual_boxplots
.. autofunction:: tseda.visualization.time_plots.plot_density_ridge

Distribution plots
------------------
.. autofunction:: tseda.visualization.distribution_plots.plot_distribution
.. autofunction:: tseda.visualization.distribution_plots.plot_qq
.. autofunction:: tseda.visualization.distribution_plots.plot_rolling_stats

Correlation plots
-----------------
.. autofunction:: tseda.visualization.correlation_plots.plot_acf_pacf
.. autofunction:: tseda.visualization.correlation_plots.plot_acf_heatmap

Decomposition plots
-------------------
.. autofunction:: tseda.visualization.decomposition_plots.plot_decomposition
.. autofunction:: tseda.visualization.decomposition_plots.plot_strength_radar
.. autofunction:: tseda.visualization.decomposition_plots.plot_residual_diagnostics

Seasonality plots
-----------------
.. autofunction:: tseda.visualization.seasonality_plots.plot_periodogram
.. autofunction:: tseda.visualization.seasonality_plots.plot_polar_seasonal
.. autofunction:: tseda.visualization.seasonality_plots.plot_season_heatmap
.. autofunction:: tseda.visualization.seasonality_plots.plot_monthly_boxplots

Anomaly plots
-------------
.. autofunction:: tseda.visualization.anomaly_plots.plot_anomalies
.. autofunction:: tseda.visualization.anomaly_plots.plot_anomaly_scores
.. autofunction:: tseda.visualization.anomaly_plots.plot_anomaly_heatmap

Changepoint plots
-----------------
.. autofunction:: tseda.visualization.changepoint_plots.plot_changepoints
.. autofunction:: tseda.visualization.changepoint_plots.plot_cusum
.. autofunction:: tseda.visualization.changepoint_plots.plot_segment_means

Quality plots
-------------
.. autofunction:: tseda.visualization.quality_plots.plot_missing_heatmap
.. autofunction:: tseda.visualization.quality_plots.plot_outliers
.. autofunction:: tseda.visualization.quality_plots.plot_outlier_score
"""
from tseda.visualization.base import PALETTE, set_style
from tseda.visualization.anomaly_plots import (
    plot_anomalies,
    plot_anomaly_heatmap,
    plot_anomaly_scores,
)
from tseda.visualization.changepoint_plots import (
    plot_changepoints,
    plot_cusum,
    plot_segment_means,
)
from tseda.visualization.correlation_plots import plot_acf_heatmap, plot_acf_pacf
from tseda.visualization.decomposition_plots import (
    plot_decomposition,
    plot_residual_diagnostics,
    plot_strength_radar,
)
from tseda.visualization.distribution_plots import (
    plot_distribution,
    plot_qq,
    plot_rolling_stats,
)
from tseda.visualization.quality_plots import (
    plot_missing_heatmap,
    plot_outlier_score,
    plot_outliers,
)
from tseda.visualization.seasonality_plots import (
    plot_monthly_boxplots,
    plot_periodogram,
    plot_polar_seasonal,
    plot_season_heatmap,
)
from tseda.visualization.time_plots import (
    plot_annual_boxplots,
    plot_calendar_heatmap,
    plot_density_ridge,
    plot_lag,
    plot_seasonal_subseries,
    plot_series,
)

__all__ = [
    "PALETTE",
    "set_style",
    # time
    "plot_series",
    "plot_seasonal_subseries",
    "plot_lag",
    "plot_calendar_heatmap",
    "plot_annual_boxplots",
    "plot_density_ridge",
    # distribution
    "plot_distribution",
    "plot_qq",
    "plot_rolling_stats",
    # correlation
    "plot_acf_pacf",
    "plot_acf_heatmap",
    # decomposition
    "plot_decomposition",
    "plot_strength_radar",
    "plot_residual_diagnostics",
    # seasonality
    "plot_periodogram",
    "plot_polar_seasonal",
    "plot_season_heatmap",
    "plot_monthly_boxplots",
    # anomaly
    "plot_anomalies",
    "plot_anomaly_scores",
    "plot_anomaly_heatmap",
    # changepoint
    "plot_changepoints",
    "plot_cusum",
    "plot_segment_means",
    # quality
    "plot_missing_heatmap",
    "plot_outliers",
    "plot_outlier_score",
]