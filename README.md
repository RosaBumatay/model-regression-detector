# model-regression-detector
# Model Regression Detector

Catch bad model releases before they reach production. This tool compares a **candidate** model against a **baseline** model on the same labeled test data, checks the input features for **data drift**, saves a JSON report, and can alert your team on **Slack**.

It supports both **classification** and **regression** models (any scikit-learn-style model saved with `joblib`).

## Features

- **Regression check**: flags a candidate model that performs meaningfully worse than the baseline, with configurable per-metric tolerances.
- **Drift detection**: per-feature PSI (Population Stability Index) and Kolmogorov-Smirnov test between baseline and test data.
- **Reports**: timestamped JSON reports plus a `latest_report.json`.
- **Dashboard**: Streamlit app showing metric comparisons and drift charts.
- **Slack alerts**: webhook notification when a regression or drift is found.
- **CI-friendly**: exits with code `1` when a regression is detected.

## Project Structure

```
model-regression-detector/
├── data/
│   ├── baseline_data.csv        # data the baseline model was trained on (drift reference)
│   └── test_data.csv            # newer labeled data used to evaluate both models
├── models/
│   ├── baseline_model.pkl       # current production model
│   └── candidate_model.pkl      # new model being evaluated
├── evaluation/
│   ├── metrics.py               # classification and regression metrics
│   ├── drift_detector.py        # PSI + KS drift detection
│   └── regression_checker.py    # baseline vs candidate comparison
├── dashboard/
│   └── app.py                   # Streamlit dashboard
├── alerts/
│   └── slack_alert.py           # Slack webhook alerts
├── reports/                     # generated JSON reports
├── requirements.txt
└── main.py                      # entry point
```

## Installation

```bash
git clone https://github.com/<your-username>/model-regression-detector.git
cd model-regression-detector
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Quick Start (demo)

Generate synthetic data and two models (the candidate is deliberately weaker, and two features are deliberately shifted) and run the full pipeline:

```bash
python main.py --bootstrap                     # classification demo
python main.py --task regression --bootstrap   # regression demo
```

Then open the dashboard:

```bash
streamlit run dashboard/app.py
```

## Using Your Own Models and Data

1. Place your data in `data/`:
   - `baseline_data.csv`: the reference dataset (features + label).
   - `test_data.csv`: labeled data to evaluate both models on.
   - The label column must be named `target` (or change `TARGET` in `main.py`). All other columns are treated as features and must be numeric.
2. Save your models with `joblib` to `models/baseline_model.pkl` and `models/candidate_model.pkl`. They must implement `predict` (and ideally `predict_proba` for classification, which enables ROC-AUC).
3. Run:

```bash
python main.py                      # classification (default)
python main.py --task regression    # regression
python main.py --no-alert           # skip the Slack alert
```

### Command-line options

| Flag | Description |
|------|-------------|
| `--task {classification,regression}` | Type of models being compared (default: `classification`) |
| `--bootstrap` | Generate demo data and models (overwrites existing files in `data/` and `models/`) |
| `--no-alert` | Do not send a Slack alert |

## How It Works

1. **Load** the baseline and test data and both models.
2. **Evaluate** both models on `test_data.csv`.
   - Classification: accuracy, precision, recall, F1, ROC-AUC
   - Regression: MAE, RMSE, R²
3. **Check for regression** by comparing each candidate metric to the baseline's.
4. **Detect drift** by comparing each feature's distribution in baseline vs test data.
5. **Write a report** to `reports/`.
6. **Alert** on Slack if there is a regression or drifted features.

### Regression thresholds

Defined in `evaluation/regression_checker.py` (`DEFAULT_THRESHOLDS`):

| Metric type | Rule | Default |
|-------------|------|---------|
| Scores (accuracy, precision, recall, F1, ROC-AUC, R²) | Fail if the score drops by more than an **absolute** amount | 0.02 - 0.03 |
| Errors (MAE, RMSE) | Fail if the error rises by more than a **relative** amount | 5% |

You can override them by passing a `thresholds` dict to `check_regression`.

### Drift thresholds

A feature is flagged as drifted if **PSI > 0.25** or the **KS test p-value < 0.01** (configurable in `detect_drift`).

| PSI | Interpretation |
|-----|----------------|
| < 0.1 | Stable |
| 0.1 - 0.25 | Moderate shift |
| > 0.25 | Major shift |

## Slack Alerts

Create an [incoming webhook](https://api.slack.com/messaging/webhooks) and export its URL:

```bash
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/XXX/YYY/ZZZ"
python main.py
```

If the variable isn't set, the alert message is printed to the console instead.

## Using in CI

`main.py` exits with code `1` when a regression is detected, so it can gate a deployment:

```yaml
# .github/workflows/model-check.yml
name: Model regression check
on: [pull_request]
jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt
      - run: python main.py --no-alert
```

## Limitations

- Drift detection supports **numeric features only**.
- Features are tested **independently**; changes in relationships between features are not detected.
- Drift is measured on **inputs**, not on labels or label-feature relationships.
- Results are only as good as your test data: it should be labeled and representative of production.

## Security Note

`.pkl` files are loaded with `joblib`/pickle, which can execute arbitrary code. Only load model files from sources you trust.

## License

This project is licensed under the MIT License.
