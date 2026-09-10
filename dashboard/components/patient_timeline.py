"""
Dashboard Components: Patient Timeline
"""

from __future__ import annotations
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime


def render_patient_timeline(recent_results: list[dict], selected_patient: str | None = None) -> None:
    """Renders risk score timeline for selected patient."""
    if not recent_results:
        st.info("No predictions yet. Start the stream simulation to see data.")
        return

    # Filter for selected patient
    patient_ids = list({r["patient_id"] for r in recent_results})
    if not patient_ids:
        return

    if selected_patient is None or selected_patient not in patient_ids:
        selected_patient = patient_ids[0]

    chosen = st.selectbox("Select patient", patient_ids, index=patient_ids.index(selected_patient))
    patient_results = [r for r in recent_results if r["patient_id"] == chosen]

    if not patient_results:
        st.warning(f"No predictions yet for patient {chosen}")
        return

    timestamps = [r.get("timestamp", "") for r in patient_results]
    risk_scores = [r.get("risk_score", 0) for r in patient_results]
    online_scores = [r.get("online_score", 0) for r in patient_results]
    baseline_scores = [r.get("baseline_score", 0) for r in patient_results]
    alerts = [r.get("alert", False) for r in patient_results]
    alert_times = [t for t, a in zip(timestamps, alerts) if a]
    alert_scores = [s for s, a in zip(risk_scores, alerts) if a]

    threshold = patient_results[0].get("alert_threshold", 0.65)

    fig = go.Figure()

    # Shaded risk zone
    fig.add_hrect(
        y0=threshold, y1=1.0,
        fillcolor="rgba(255,75,75,0.08)", line_width=0,
        annotation_text="Alert Zone", annotation_position="top right",
    )

    # Threshold line
    fig.add_hline(
        y=threshold, line_dash="dash", line_color="#FF4B4B",
        annotation_text=f"Alert threshold ({threshold:.0%})",
        annotation_font_color="#FF4B4B",
    )

    # Online and baseline scores
    fig.add_trace(go.Scatter(
        x=timestamps, y=online_scores,
        mode="lines", name="Online Model",
        line=dict(color="#00B4D8", width=1.5, dash="dot"),
        opacity=0.6,
    ))
    fig.add_trace(go.Scatter(
        x=timestamps, y=baseline_scores,
        mode="lines", name="Baseline",
        line=dict(color="#90BE6D", width=1.5, dash="dot"),
        opacity=0.6,
    ))

    # Active (selected) risk score
    fig.add_trace(go.Scatter(
        x=timestamps, y=risk_scores,
        mode="lines", name="Active Risk Score",
        line=dict(color="#F77F00", width=2.5),
        fill="tozeroy", fillcolor="rgba(247,127,0,0.05)",
    ))

    # Alert markers
    if alert_times:
        fig.add_trace(go.Scatter(
            x=alert_times, y=alert_scores,
            mode="markers", name="Alert",
            marker=dict(color="#FF4B4B", size=12, symbol="triangle-up"),
        ))

    fig.update_layout(
        title=f"Risk Timeline — Patient {chosen[:8]}",
        xaxis_title="Time",
        yaxis_title="Deterioration Risk Score",
        yaxis=dict(range=[0, 1], tickformat=".0%"),
        height=350,
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#333")
    st.plotly_chart(fig, use_container_width=True)
