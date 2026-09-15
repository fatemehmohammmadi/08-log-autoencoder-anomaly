"""Generate synthetic numeric security event features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(33)
OUT = Path(__file__).parent / "data" / "events.csv"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n_norm, n_anom = 900, 100

    normal = pd.DataFrame(
        {
            "failed_auth": RNG.poisson(0.4, n_norm),
            "bytes_out": RNG.lognormal(8.0, 0.4, n_norm),
            "unique_dst": RNG.integers(1, 12, n_norm),
            "rare_port_hits": RNG.poisson(0.2, n_norm),
            "process_entropy": RNG.uniform(1.0, 3.0, n_norm),
            "label": "normal",
        }
    )
    anom = pd.DataFrame(
        {
            "failed_auth": RNG.integers(8, 40, n_anom),
            "bytes_out": RNG.lognormal(12.5, 0.5, n_anom),
            "unique_dst": RNG.integers(40, 200, n_anom),
            "rare_port_hits": RNG.integers(5, 30, n_anom),
            "process_entropy": RNG.uniform(4.0, 6.5, n_anom),
            "label": "anomaly",
        }
    )
    df = pd.concat([normal, anom], ignore_index=True).sample(frac=1.0, random_state=33).reset_index(drop=True)
    df.insert(0, "event_id", [f"E{i:04d}" for i in range(len(df))])
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df)} events -> {OUT}")


if __name__ == "__main__":
    main()
