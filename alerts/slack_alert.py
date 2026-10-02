"""Slack alerting via incoming webhook (set SLACK_WEBHOOK_URL)."""
from __future__ import annotations

import os

import requests


def build_message(report: dict) -> str:
    reg = report["regression"]
    lines = [":rotating_light: *Model regression detected*" if reg["regression_detected"]
             else ":white_check_mark: *Candidate model passed regression checks*"]
    for m in reg["regressed_metrics"]:
        d = reg["details"][m]
        lines.append(f"• `{m}`: {d['baseline']} → {d['candidate']} (Δ {d['delta']})")
    drifted = report.get("drifted_features", [])
    if drifted:
        lines.append(f":warning: Drift in {len(drifted)} feature(s): {', '.join(drifted[:10])}")
    return "\n".join(lines)


def send_slack_alert(report: dict, webhook_url: str | None = None) -> bool:
    webhook_url = webhook_url or os.getenv("SLACK_WEBHOOK_URL")
    message = build_message(report)
    if not webhook_url:
        print("[alert] SLACK_WEBHOOK_URL not set; printing instead:\n" + message)
        return False
    try:
        resp = requests.post(webhook_url, json={"text": message}, timeout=10)
        resp.raise_for_status()
        return True
    except requests.RequestException as exc:
        print(f"[alert] Failed to send Slack alert: {exc}")
        return False
