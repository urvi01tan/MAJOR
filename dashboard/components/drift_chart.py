"""
Dashboard Components: Drift Chart
"""

from __future__ import annotations
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def render_drift_chart(drift_detector, selector_status: dict) -> None:
    """Renders drift detector time-series and rollback timeline."""
    st.subheader("📡 Drift Detection & Model Status")

    series = drift_detector.drift_score_series()
    drift_events = drift_detector.drift_events

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        status_color = "🟢" if selector_status["active_model"] == "online" else "🟡"
        st.metric("Active Model", f"{status_color} {selector_status['active_model'].title()}")
    with col2:
        st.metric("Online AUROC", f"{selector_status.get('online_auroc', 0):.3f}")
    with col3:
        st.metric("Baseline AUROC", f"{selector_status.get('baseline_auroc', 0):.3f}")
    with col4:
        st.metric("Drift Events", drift_detector.total_drift_count)

    # ADWIN scores chart
    adwin_scores = series.get("adwin_scores", [])
    ph_sums = series.get("ph_sums", [])
    error_rates = series.get("error_rates", [])

    if adwin_scores:
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            subplot_titles=("Prediction Score (ADWIN)", "Page-Hinkley Sum", "Prediction Error Rate (DDM)"),
            vertical_spacing=0.08,
        )

        fig.add_trace(
            go.Scatter(y=adwin_scores, mode="lines", name="Risk Score",
                       line=dict(color="#00B4D8", width=1.5)), row=1, col=1
        )
        if ph_sums:
            fig.add_trace(
                go.Scatter(y=ph_sums, mode="lines", name="PH Sum",
                           line=dict(color="#F77F00", width=1.5)), row=2, col=1
            )
            # Threshold line
            fig.add_hline(
                y=drift_detector._ph.threshold, line_dash="dash",
                line_color="#FF4B4B", annotation_text="PH Threshold", row=2, col=1
            )
        if error_rates:
            fig.add_trace(
                go.Scatter(y=error_rates, mode="lines", name="Error",
                           line=dict(color="#E63946", width=1)), row=3, col=1
            )

        # Mark drift events
        for de in drift_events[-20:]:
            fig.add_vline(
                x=len(adwin_scores) - 1,
                line_dash="dot", line_color="#FF4B4B",
                annotation_text=f"{de['detector']}", row=1, col=1
            )

        fig.update_layout(
            height=450,
            showlegend=False,
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font=dict(color="#fafafa"),
            margin=dict(l=40, r=20, t=60, b=20),
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(gridcolor="#333")
        st.plotly_chart(fig, use_container_width=True)

    # Drift event log table
    if drift_events:
        st.markdown("**Recent Drift Events**")
        df = st.dataframe(
            drift_events[-10:],
            use_container_width=True,
            hide_index=True,
        )
