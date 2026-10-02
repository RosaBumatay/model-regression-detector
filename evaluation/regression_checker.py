"""Compare candidate metrics against baseline metrics."""
from __future__ import annotations

# Max tolerated absolute drop per metric before it counts as a regression.
DEFAULT_THRESHOLDS = {
    "accuracy": 0.02,
    "precision": 0.03,
    "recall": 0.03,
    "f1": 0.02,
    "roc_auc": 0.02,
}


def check_regression(baseline: dict, candidate: dict, thresholds: dict | None = None) -> dict:
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    details = {}
    for name, base_val in baseline.items():
        if name not in candidate:
            continue
        delta = candidate[name] - base_val
        allowed_drop = thresholds.get(name, 0.02)
        regressed = delta < -allowed_drop
        details[name] = {
            "baseline": round(base_val, 4),
            "candidate": round(candidate[name], 4),
            "delta": round(delta, 4),
            "allowed_drop": allowed_drop,
            "status": "REGRESSION" if regressed else ("IMPROVED" if delta > 0 else "OK"),
        }
    regressed_metrics = [m for m, d in details.items() if d["status"] == "REGRESSION"]
    return {
        "regression_detected": bool(regressed_metrics),
        "regressed_metrics": regressed_metrics,
        "details": details,
    }
