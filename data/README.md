# data/

Place your time series CSV files here.

## Expected format

```
date,value
2020-01-01,123.4
2020-01-02,125.1
...
```

- **date** column: ISO-8601 date string (YYYY-MM-DD, YYYY-MM-DD HH:MM:SS, etc.)
- **value** column: numeric observations (NaN / blank = missing)
- Additional columns are ignored

## Sample datasets (pre-generated)

| File | Description |
|------|-------------|
| `sample/daily_sample.csv` | 1096 days (3 years), daily — trend + weekly seasonality + anomalies + changepoint |
| `sample/monthly_sample.csv` | 72 months (6 years), monthly — trend + annual seasonality |
| `sample/hourly_sample.csv` | 720 hours (30 days), hourly — daily cycle + noise + missing |

Run all examples against these files:

```bash
cd examples/
python run_all.py ../data/sample/daily_sample.csv --freq D --period 7
```

## Adding your own data

1. Copy your CSV into this directory.
2. Make sure it has a `date` (or `timestamp`) column and a `value` column.
3. Edit `examples/run_all.py` to point to your file, or call each module
   example directly.