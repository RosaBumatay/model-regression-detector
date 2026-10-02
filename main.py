"""Model regression detector: compare a candidate model to a baseline and check for drift."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from alerts.slack_alert import send_slack_alert
from evaluation.drift_detector import detect_drift
from evaluation.metrics import evaluate_model
from evaluation.regression_checker import check_regression

ROOT = Path(__file__).resolve().parent
DATA_DIR, MODEL_DIR, REPORT_DIR = ROOT / "data", ROOT / "models", ROOT / "reports"
TARGET = "target"


def bootstrap_demo() -> None:
    """Create sample data + models so the pipeline runs out of the box."""
    from sklearn.datasets import make_classification
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression

    X, y = make_classification(n_samples=4000, n_features=10, n_informative=6,
                               random_state=42)
    cols = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols)
    df[TARGET] = y
    base_df, test_df = df.iloc[:2500].copy(), df.iloc[2500:].copy()

    # Simulate production drift on two features.
    test_df["feature_0"] += 1.0
    test_df["feature_3"] *= 1.5

    DATA_DIR.mkdir(exist_ok=True); MODEL_DIR.mkdir(exist_ok=True)
    base_df.to_csv(DATA_DIR / "baseline_data.csv", index=False)
    test_df.to_csv(DATA_DIR / "test_data.csv", index=False)

    Xb, yb = base_df[cols], base_df[TARGET]
    joblib.dump(GradientBoostingClassifier(random_state=0).fit(Xb, yb),
                MODEL_DIR / "baseline_model.pkl")
    # Deliberately weaker candidate to demonstrate a regression.
    joblib.dump(LogisticRegression(max_iter=50, C=0.001).fit(Xb.iloc[:300], yb.iloc[:300]),
                MODEL_DIR / "candidate_model.pkl")
    print("Demo data and models created.")


def run(no_alert: bool = False) -> dict:
    baseline_df = pd.read_csv(DATA_DIR / "baseline_data.csv")
    test_df = pd.read_csv(DATA_DIR / "test_data.csv")
    features = [c for c in test_df.columns if c != TARGET]

    baseline_model = joblib.load(MODEL_DIR / "baseline_model.pkl")
    candidate_model = joblib.load(MODEL_DIR / "candidate_model.pkl")

    X_test, y_test = test_df[features], test_df[TARGET]
    base_metrics = evaluate_model(baseline_model, X_test, y_test)
    cand_metrics = evaluate_model(candidate_model, X_test, y_test)

    regression = check_regression(base_metrics, cand_metrics)
    drift_df = detect_drift(baseline_df, test_df, features)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "baseline_metrics": base_metrics,
        "candidate_metrics": cand_metrics,
        "regression": regression,
        "drift": drift_df.to_dict(orient="records"),
        "drifted_features": drift_df.loc[drift_df["drifted"], "feature"].tolist(),
    }

    REPORT_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for name in (f"report_{stamp}.json", "latest_report.json"):
        (REPORT_DIR / name).write_text(json.dumps(report, indent=2, default=float))

    print(json.dumps(regression["details"], indent=2))
    print("Drifted features:", report["drifted_features"] or "none")
    print("REGRESSION DETECTED" if regression["regression_detected"] else "No regression.")

    if (regression["regression_detected"] or report["drifted_features"]) and not no_alert:
        send_slack_alert(report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model regression detector")
    parser.add_argument("--bootstrap", action="store_true", help="generate demo data/models")
    parser.add_argument("--no-alert", action="store_true", help="skip Slack alert")
    args = parser.parse_args()

    if args.bootstrap or not (MODEL_DIR / "baseline_model.pkl").exists():
        bootstrap_demo()
    report = run(no_alert=args.no_alert)
    raise SystemExit(1 if report["regression"]["regression_detected"] else 0)
