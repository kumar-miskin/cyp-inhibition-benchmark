"""Summarize frozen predictions. Never trains or selects a model."""

import json
import numpy as np
import pandas as pd
from benchmark import ROOT, ENDPOINTS, SEED


def main():
    truth = pd.read_csv(ROOT / "data" / "cyp-challenge-TEST-UNBLINDED_inhibition.csv")
    train = pd.read_csv(ROOT / "data" / "cyp-challenge-TRAIN_inhibition.csv")
    rng = np.random.default_rng(SEED)
    ix = rng.integers(0, len(truth), size=(1000, len(truth)))
    scores = {}
    summary = {
        "bootstrap": "1,000 paired compound-row resamples; percentile 95% intervals; seeds fixed; exploratory not rank inference",
        "models": {},
        "distribution": {},
    }
    for name in ["mean", "morgan_rf", "atom_gnn"]:
        pred = pd.read_csv(ROOT / "results" / f"{name}_holdout_predictions.csv")
        assert pred.Molecule_Name.equals(truth.Molecule_Name)
        endpoints = []
        for c in ENDPOINTS:
            y = truth[c].to_numpy()
            p = pred[c].to_numpy()
            lo = truth[c + "_conf_low"].to_numpy()
            hi = truth[c + "_conf_high"].to_numpy()
            yb = y[ix]
            lob = lo[ix]
            hib = hi[ix]
            pb = p[ix]
            numerator = np.nansum(
                np.maximum(lob - pb, 0) + np.maximum(pb - hib, 0), axis=1
            )
            avg = np.nanmean(yb, axis=1)[:, None]
            denominator = np.nansum(
                np.maximum(lob - avg, 0) + np.maximum(avg - hib, 0), axis=1
            )
            endpoints.append(numerator / denominator)
        samples = np.mean(endpoints, axis=0)
        scores[name] = samples
        summary["models"][name] = {
            "macro_ST_RAE_95_percentile_interval": np.quantile(
                samples, [0.025, 0.975]
            ).tolist()
        }
    diff = scores["atom_gnn"] - scores["morgan_rf"]
    summary["paired_GNN_minus_RF_ST_RAE"] = {
        "mean": float(diff.mean()),
        "95_percentile_interval": np.quantile(diff, [0.025, 0.975]).tolist(),
        "interpretation": "Positive favors RF. This interval is conditional on one fitted model per method, not training-seed variation.",
    }
    for c in ENDPOINTS:
        summary["distribution"][c] = {
            "train_n": int(train[c].notna().sum()),
            "holdout_n": int(truth[c].notna().sum()),
            "train_pIC50_below4_fraction": float((train[c].dropna() < 4).mean()),
            "holdout_pIC50_below4_fraction": float((truth[c].dropna() < 4).mean()),
        }
    (ROOT / "results" / "bootstrap.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
