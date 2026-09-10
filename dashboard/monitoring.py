"""
Performance Monitoring Dashboard (separate Streamlit page)
==========================================================
Tracks rolling AUROC, sensitivity, alert rate, drift score, and
model version timeline for the performance monitoring use case.
"""

from __future__ import annotations

import sys
from pathlib import Path
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


def render_monitoring_page(pipeline) -> None:
    st.title("📊 EWS Performance Monitor")
    st.caption("Real-time monitoring of model performance, drift, and alert rates.")

    status = pipeline.status()

    # ── KPI Row ──────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.metric("Active Model", status["active_model"].title())
    with c2:
        auroc = status.get("online_auroc", 0)
        st.metric("Online AUROC", f"{auroc:.3f}", delta=f"{auroc - status.get('baseline_auroc', 0):.3f}")
    with c3:
        st.metric("Baseline AUROC", f"{status.get('baseline_auroc', 0):.3f}")
    with c4:
        st.metric("Rollbacks", status["rollback_count"])
    with c5:
        st.metric("Drift Events", status["total_drift_events"])
    with c6:
        alert_count = pipeline.audit.alert_count(last_n_hours=24)
        st.metric("Alerts (24h)", alert_count)

    st.divider()

    # ── Rolling AUROC over time ───────────────────────────────────────────
    recent = pipeline.recent_results(200)
    if recent:
        df = pd.DataFrame(recent)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Rolling AUROC")
            if "online_rolling_auroc" in df.columns:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df["timestamp"], y=df["online_rolling_auroc"],
                    mode="lines", name="Online AUROC",
                    line=dict(color="#00B4D8"),
                ))
                base_auroc = status.get("baseline_auroc", 0)
                if base_auroc > 0:
                    fig.add_hline(y=base_auroc, line_dash="dash", line_color="#90BE6D",
                                  annotation_text=f"Baseline ({base_auroc:.3f})")
                    fig.add_hline(y=base_auroc - 0.05, line_dash="dot", line_color="#FF4B4B",
                                  annotation_text="Rollback threshold")
                fig.update_layout(
                    height=250, plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                    font=dict(color="#fafafa"), margin=dict(l=20, r=20, t=20, b=20),
                    yaxis=dict(range=[0, 1]),
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Risk Score Distribution")
            if "risk_score" in df.columns:
                fig2 = px.histogram(
                    df, x="risk_score", nbins=30,
                    color_discrete_sequence=["#F77F00"],
                )
                fig2.add_vline(x=0.65, line_dash="dash", line_color="#FF4B4B",
                               annotation_text="Alert threshold")
                fig2.update_layout(
                    height=250, plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                    font=dict(color="#fafafa"), margin=dict(l=20, r=20, t=20, b=20),
                    showlegend=False,
                )
                st.plotly_chart(fig2, use_container_width=True)

    # ── Model Version Timeline ────────────────────────────────────────────
    st.subheader("Model Snapshot Registry")
    versions = pipeline.registry.list_versions()
    if versions:
        vdf = pd.DataFrame(versions)
        vdf = vdf[["version", "timestamp", "reason", "n_predictions", "n_updates", "rolling_auroc"]]
        st.dataframe(vdf, use_container_width=True, hide_index=True)
    else:
        st.info("No snapshots yet.")

    # ── Rollback Events ───────────────────────────────────────────────────
    rollback_events = pipeline.selector.rollback_events
    if rollback_events:
        st.subheader("Rollback Events")
        st.dataframe(pd.DataFrame(rollback_events), use_container_width=True, hide_index=True)

    # ── Drift Events ──────────────────────────────────────────────────────
    drift_events = pipeline.drift_detector.drift_events
    if drift_events:
        st.subheader("Drift Event Log")
        st.dataframe(pd.DataFrame(drift_events[-30:]), use_container_width=True, hide_index=True)

    # ── Pipeline Stats ────────────────────────────────────────────────────
    st.subheader("Pipeline Statistics")
    stats_data = {
        "Metric": [
            "Events Processed", "Predictions Made", "Alerts Fired",
            "Patients Tracked", "Model Snapshots", "Online Model Updates",
        ],
        "Value": [
            f"{status['n_events_processed']:,}",
            f"{status['n_predictions']:,}",
            f"{status['n_alerts']:,}",
            f"{status['n_patients_tracked']:,}",
            f"{status['model_snapshots']:,}",
            f"{pipeline.online_model.n_updates:,}",
        ],
    }
    st.table(pd.DataFrame(stats_data).set_index("Metric"))
