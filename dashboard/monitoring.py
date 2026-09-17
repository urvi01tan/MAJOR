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

    # ── KPI row ────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    with c1:
        st.metric("Active Model", status["active_model"].title())
    with c2:
        auroc = status.get("online_auroc", 0)
        base_auroc = status.get("baseline_auroc", 0)
        delta = f"{auroc - base_auroc:+.3f}" if base_auroc > 0 and base_auroc == base_auroc else None
        st.metric("Online AUROC", f"{auroc:.3f}", delta=delta)
    with c3:
        st.metric("Baseline AUROC",
                  f"{base_auroc:.3f}" if base_auroc == base_auroc and base_auroc > 0 else "—")
    with c4:
        st.metric("Rollbacks", status["rollback_count"])
    with c5:
        st.metric("Drift Events", status["total_drift_events"])
    with c6:
        alert_count = pipeline.audit.alert_count(last_n_hours=24)
        st.metric("Alerts (24h)", alert_count)
    with c7:
        rate = pipeline.alert_rate_per_hour(window_hours=1.0)
        st.metric("Alerts/hr", f"{rate:.1f}")

    st.divider()

    # ── Charts row 1 ──────────────────────────────────────────────────────
    recent = pipeline.recent_results(300)
    if recent:
        df = pd.DataFrame(recent)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Rolling AUROC vs Baseline")
            if "online_rolling_auroc" in df.columns:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df["timestamp"], y=df["online_rolling_auroc"],
                    mode="lines", name="Online AUROC",
                    line=dict(color="#00B4D8", width=2),
                    fill="tozeroy", fillcolor="rgba(0,180,216,0.06)",
                ))
                if base_auroc > 0 and base_auroc == base_auroc:
                    fig.add_hline(
                        y=base_auroc, line_dash="dash", line_color="#90BE6D",
                        annotation_text=f"Baseline ({base_auroc:.3f})",
                        annotation_font_color="#90BE6D",
                    )
                    rollback_thr = base_auroc - 0.05
                    fig.add_hline(
                        y=rollback_thr, line_dash="dot", line_color="#FF4B4B",
                        annotation_text=f"Rollback threshold ({rollback_thr:.3f})",
                        annotation_font_color="#FF4B4B",
                    )
                fig.update_layout(
                    height=280, plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                    font=dict(color="#fafafa"), margin=dict(l=20, r=20, t=20, b=20),
                    yaxis=dict(range=[0, 1], gridcolor="#333"),
                    xaxis=dict(showgrid=False),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Risk Score Distribution")
            if "risk_score" in df.columns:
                fig2 = px.histogram(
                    df, x="risk_score", nbins=30,
                    color_discrete_sequence=["#F77F00"],
                )
                thr = status.get("alert_threshold", 0.65)
                fig2.add_vline(x=thr, line_dash="dash", line_color="#FF4B4B",
                               annotation_text=f"Alert {thr:.0%}",
                               annotation_font_color="#FF4B4B")
                fig2.update_layout(
                    height=280, plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                    font=dict(color="#fafafa"), margin=dict(l=20, r=20, t=20, b=20),
                    showlegend=False,
                    xaxis=dict(title="Risk Score", tickformat=".0%"),
                    yaxis=dict(title="Count", gridcolor="#333"),
                )
                st.plotly_chart(fig2, use_container_width=True)

        # ── Charts row 2 ──────────────────────────────────────────────────
        col3, col4 = st.columns(2)

        with col3:
            st.subheader("Online vs Baseline Score (Scatter)")
            if "online_score" in df.columns and "baseline_score" in df.columns:
                sample = df.sample(min(300, len(df)), random_state=42)
                alert_col = sample.get("alert", pd.Series([False] * len(sample)))
                colors = ["#FF4B4B" if a else "#00B4D8" for a in alert_col]
                fig3 = go.Figure(go.Scatter(
                    x=sample["baseline_score"],
                    y=sample["online_score"],
                    mode="markers",
                    marker=dict(color=colors, size=5, opacity=0.7),
                    text=[
                        f"Patient: {r.get('patient_id','')}<br>"
                        f"Online: {r.get('online_score',0):.2%}<br>"
                        f"Baseline: {r.get('baseline_score',0):.2%}"
                        for _, r in sample.iterrows()
                    ],
                    hovertemplate="%{text}<extra></extra>",
                ))
                # Diagonal reference line
                fig3.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                               line=dict(color="#555", dash="dot"))
                fig3.add_vline(x=thr, line_dash="dot", line_color="#FF4B4B", opacity=0.5)
                fig3.add_hline(y=thr, line_dash="dot", line_color="#FF4B4B", opacity=0.5)
                fig3.update_layout(
                    height=280, plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                    font=dict(color="#fafafa"), margin=dict(l=20, r=20, t=20, b=20),
                    xaxis=dict(title="Baseline Score", range=[0, 1], tickformat=".0%", gridcolor="#333"),
                    yaxis=dict(title="Online Score", range=[0, 1], tickformat=".0%", gridcolor="#333"),
                )
                st.plotly_chart(fig3, use_container_width=True)
                st.caption("🔴 = alert fired  🔵 = no alert  Dashed diagonal = perfect agreement")

        with col4:
            st.subheader("Alert Precision-Recall (from feedback)")
            summary = pipeline.shift_summary()
            tp = summary.get("true_positives", 0)
            fp = summary.get("false_positives", 0)
            total_alerts = status.get("n_alerts", 0)
            fn = max(0, total_alerts - tp - fp)  # unconfirmed = assume FN proxy

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall_proxy = tp / (tp + fn) if (tp + fn) > 0 else 0

            if tp + fp > 0:
                pr_fig = go.Figure()
                pr_fig.add_trace(go.Bar(
                    name="True Positives", x=["Feedback breakdown"], y=[tp],
                    marker_color="#00CC44",
                ))
                pr_fig.add_trace(go.Bar(
                    name="False Positives", x=["Feedback breakdown"], y=[fp],
                    marker_color="#FF4B4B",
                ))
                pr_fig.add_trace(go.Bar(
                    name="Unconfirmed", x=["Feedback breakdown"], y=[fn],
                    marker_color="#555",
                ))
                pr_fig.update_layout(
                    barmode="stack", height=220,
                    plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                    font=dict(color="#fafafa"),
                    margin=dict(l=20, r=20, t=20, b=20),
                    legend=dict(orientation="h", y=-0.3),
                )
                st.plotly_chart(pr_fig, use_container_width=True)
                st.caption(
                    f"PPV (precision): **{precision:.1%}**  ·  "
                    f"Recall proxy: **{recall_proxy:.1%}**  ·  "
                    f"Total alerts: **{total_alerts:,}**"
                )
            else:
                st.info("No clinician feedback yet — mark alerts as True/False on the Alerts page.")

    # ── Model Snapshot Registry ────────────────────────────────────────────
    st.divider()
    st.subheader("📸 Model Snapshot Registry")
    versions = pipeline.registry.list_versions()
    if versions:
        vdf = pd.DataFrame(versions)
        cols_to_show = [c for c in ["version", "timestamp", "reason", "n_predictions", "n_updates", "rolling_auroc"] if c in vdf.columns]
        st.dataframe(vdf[cols_to_show], use_container_width=True, hide_index=True)
    else:
        st.info("No snapshots yet.")

    # ── Rollback & drift events ────────────────────────────────────────────
    col_r, col_d = st.columns(2)
    with col_r:
        rollback_events = pipeline.selector.rollback_events
        if rollback_events:
            st.subheader("⚠️ Rollback Events")
            st.dataframe(pd.DataFrame(rollback_events), use_container_width=True, hide_index=True)

    with col_d:
        drift_events = pipeline.drift_detector.drift_events
        if drift_events:
            st.subheader("📡 Drift Event Log")
            st.dataframe(pd.DataFrame(drift_events[-30:]), use_container_width=True, hide_index=True)

    # ── Pipeline statistics ────────────────────────────────────────────────
    st.divider()
    st.subheader("Pipeline Statistics")
    stats_data = {
        "Metric": [
            "Events Processed", "Predictions Made", "Alerts Fired",
            "Patients Tracked", "Model Snapshots", "Online Model Updates",
            "Alert Rate (last hr)",
        ],
        "Value": [
            f"{status['n_events_processed']:,}",
            f"{status['n_predictions']:,}",
            f"{status['n_alerts']:,}",
            f"{status['n_patients_tracked']:,}",
            f"{status['model_snapshots']:,}",
            f"{pipeline.online_model.n_updates:,}",
            f"{pipeline.alert_rate_per_hour(1.0):.1f} / hr",
        ],
    }
    st.table(pd.DataFrame(stats_data).set_index("Metric"))
