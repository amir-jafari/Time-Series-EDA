# examples/

End-to-end usage examples for every tseda module.

## Running examples

From the **repository root**:

```bash
# Individual modules
python examples/00_load_data.py
python examples/01_quality_analysis.py
python examples/02_statistics_analysis.py
python examples/03_decomposition_analysis.py
python examples/04_seasonality_detection.py
python examples/05_anomaly_detection.py

# Full pipeline on any CSV
python examples/run_all.py data/sample/daily_sample.csv --period 7
python examples/run_all.py data/sample/monthly_sample.csv --period 12
python examples/run_all.py data/sample/hourly_sample.csv --period 24

# Your own data
python examples/run_all.py /path/to/your/data.csv \
    --date-col timestamp --value-col price --unit USD
```

## Jupyter Notebooks

Convert any script to a notebook with `jupytext`:

```bash
pip install jupytext
jupytext --to notebook examples/00_load_data.py
```

Or use the `examples/notebooks/` directory — pre-built notebooks will be
added once the visualization module (Phase 10) is complete.

## Script index

| Script | Module | What it shows |
|--------|--------|---------------|
| `00_load_data.py` | `core` | Loading CSV, pandas, numpy; transforms |
| `01_quality_analysis.py` | `quality` | Missing values, outliers, flat lines |
| `02_statistics_analysis.py` | `statistics` | Descriptive stats, ADF/KPSS, ACF/PACF |
| `03_decomposition_analysis.py` | `decomposition` | Classical + STL decomposition |
| `04_seasonality_detection.py` | `seasonality` | FFT + ACF period detection |
| `05_anomaly_detection.py` | `anomaly` | Rolling IQR/Z, STL-residual, GESD |
| `06_changepoint_detection.py` | `changepoint` | CUSUM, binary segmentation *(Phase 7)* |
| `07_features.py` | `features` | Feature extraction *(Phase 8)* |
| `08_forecastability.py` | `forecastability` | Forecast readiness score *(Phase 9)* |
| `09_visualizations.py` | `visualization` | All plot types *(Phase 10)* |
| `run_all.py` | all | Complete EDA pipeline on any CSV |