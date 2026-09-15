# 08 — Log Autoencoder Anomaly Detection

Train a small neural autoencoder on numeric log features and flag high reconstruction-error events as anomalies.

## Learning goals

- Represent security events as numeric vectors
- Use reconstruction error as an anomaly score
- Export top anomalous events for analyst review

## Layout

```
08-log-autoencoder-anomaly/
├── README.md
├── requirements.txt
├── generate_events.py
├── train_autoencoder.py
├── data/events.csv
└── outputs/
    ├── metrics.json
    ├── top_anomalies.csv
    └── error_hist_stats.json
```

## Setup & run

```bash
cd 08-log-autoencoder-anomaly
pip install -r requirements.txt
python generate_events.py
python train_autoencoder.py
```

Uses scikit-learn `MLPRegressor` as a compact autoencoder (no PyTorch required).

## Sample run (committed)

See `outputs/metrics.json` and `outputs/top_anomalies.csv`.
