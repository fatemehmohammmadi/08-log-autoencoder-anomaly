"""Train an MLP autoencoder and score anomalies by reconstruction error."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

DATA = Path(__file__).parent / "data" / "events.csv"
OUT = Path(__file__).parent / "outputs"
FEATS = ["failed_auth", "bytes_out", "unique_dst", "rare_port_hits", "process_entropy"]


def main() -> None:
    if not DATA.exists():
        raise SystemExit("Run generate_events.py first.")

    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA)
    X = df[FEATS].to_numpy(dtype=float)
    y = (df["label"] == "anomaly").astype(int).to_numpy()

    # Fit scaler + AE mostly on normal traffic
    normal_mask = df["label"] == "normal"
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train = X_scaled[normal_mask.to_numpy()]

    ae = MLPRegressor(
        hidden_layer_sizes=(8, 3, 8),
        activation="relu",
        max_iter=400,
        random_state=42,
    )
    ae.fit(X_train, X_train)
    recon = ae.predict(X_scaled)
    err = np.mean((X_scaled - recon) ** 2, axis=1)

    df = df.copy()
    df["recon_error"] = err
    threshold = float(np.percentile(err[normal_mask], 95))
    df["flagged"] = df["recon_error"] >= threshold

    auc = float(roc_auc_score(y, err))
    metrics = {
        "roc_auc": round(auc, 4),
        "threshold_p95_normal": round(threshold, 6),
        "flagged_count": int(df["flagged"].sum()),
        "flagged_and_true_anomaly": int(((df["flagged"]) & (df["label"] == "anomaly")).sum()),
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    df.nlargest(25, "recon_error").to_csv(OUT / "top_anomalies.csv", index=False)
    stats = {
        "error_mean": round(float(err.mean()), 6),
        "error_std": round(float(err.std()), 6),
        "error_p50": round(float(np.percentile(err, 50)), 6),
        "error_p95": round(float(np.percentile(err, 95)), 6),
        "error_max": round(float(err.max()), 6),
    }
    (OUT / "error_hist_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
