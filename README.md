# Log Autoencoder Anomaly Detection

Fit a small MLP autoencoder on “normal” numeric event features and treat high reconstruction error as anomalous.

No PyTorch dependency — uses sklearn's `MLPRegressor` as a compact AE so the demo stays easy to run.

## Run

```bash
pip install -r requirements.txt
python generate_events.py
python train_autoencoder.py
```

## Method in one paragraph

Scale features → train AE only on normal rows → score everyone by MSE(x, x̂) → threshold at the 95th percentile of normal errors → dump top offenders.

## Outputs

- `outputs/metrics.json` (includes ROC-AUC vs planted labels)
- `outputs/top_anomalies.csv`
- `outputs/error_hist_stats.json`

## License

MIT
