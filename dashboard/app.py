"""Streamlit dashboard. Run: streamlit run dashboard/app.py"""
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

REPORT_PATH = Path(__file__).resolve().parent.parent / "reports" / "latest_report.json"

st.set_page_config(page_title="Model Regression Detector", layout="wide")
st.title("Model Regression Detector")

if not REPORT_PATH.exists():
    st.warning("No report found. Run `python main.py` first.")
    st.stop()

report = json.loads(REPORT_PATH.read_text())
reg = report["regression"]

st.caption(f"Report generated: {report['timestamp']}")
if reg["regression_detected"]:
    st.error(f"Regression detected in: {', '.join(reg['regressed_metrics'])}")
else:
    st.success("No regression detected.")

st.subheader("Metrics: baseline vs candidate")
metrics_df = pd.DataFrame(reg["details"]).T.reset_index().rename(columns={"index": "metric"})
st.dataframe(metrics_df, use_container_width=True)
long_df = metrics_df.melt(id_vars="metric", value_vars=["baseline", "candidate"],
                          var_name="model", value_name="score")
st.plotly_chart(px.bar(long_df, x="metric", y="score", color="model", barmode="group"),
                use_container_width=True)

st.subheader("Feature drift")
drift_df = pd.DataFrame(report["drift"])
c1, c2 = st.columns([1, 2])
c1.metric("Drifted features", f"{len(report['drifted_features'])} / {len(drift_df)}")
c1.dataframe(drift_df[["feature", "psi", "ks_p_value", "drifted"]], use_container_width=True)
fig = px.bar(drift_df, x="feature", y="psi", color="drifted")
fig.add_hline(y=0.25, line_dash="dash", annotation_text="PSI threshold")
c2.plotly_chart(fig, use_container_width=True)
