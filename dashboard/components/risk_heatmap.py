"""Risk Heatmap — patient × time matrix of deterioration risk scores."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def render_risk_heatmap(pipeline) -> None:
    st.title("🌡️ Patient Risk Heatmap")
    st.caption(
        "Each cell shows the latest ML risk score per patient across the last N "
        "prediction steps. Red = high risk. Use this to spot clusters of deteriorating patients."
    )

    col_ctrl1, col_ctrl2 = st.columns([2, 1])
    with col_ctrl1:
        max_patients = st.slider("Max patients shown", 10, 100, 30, step=5)
    with col_ctrl2:
        sort_by = st.selectbox("Sort rows by", ["Current risk (desc)", "Patient ID"])

    history = pipeline.all_patient_risk_history(last_n=40)
    if not history:
        st.info("No data yet. Start the simulation from the sidebar.")
        return

    # Build matrix: rows = patients, cols = time steps
    all_pids = list(history.keys())

    # Sort
    if sort_by == "Current risk (desc)":
        latest_risk = {
            pid: (hist[-1]["risk_score"] if hist else 0)
            for pid, hist in history.items()
        }
        all_pids = sorted(all_pids, key=lambda p: latest_risk.get(p, 0), reverse=True)

    pids = all_pids[:max_patients]

    max_len = max((len(history[p]) for p in pids), default=1)
    matrix = []
    for pid in pids:
        hist = history[pid]
        row = [None] * (max_len - len(hist)) + [h["risk_score"] for h in hist]
        matrix.append(row)

    # Color scale
    colorscale = [
        [0.0,  "#0d1b2a"],
        [0.30, "#004e8c"],
        [0.55, "#f77f00"],
        [0.75, "#e63946"],
        [1.0,  "#ff0a0a"],
    ]

    # Hover text
    hover = []
    for i, pid in enumerate(pids):
        hist = history[pid]
        pad = max_len - len(hist)
        row_hover = [""] * pad
        for h in hist:
            ts = str(h.get("timestamp", ""))[:19]
            rs = h.get("risk_score", 0)
            n2 = h.get("news2", "—")
            sev = h.get("severity", "—")
            row_hover.append(f"Patient: {pid}<br>Time: {ts}<br>Risk: {rs:.1%}<br>NEWS2: {n2}<br>Severity: {sev}")
        hover.append(row_hover)

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            y=[str(p)[:12] for p in pids],
            colorscale=colorscale,
            zmin=0,
            zmax=1,
            text=hover,
            hovertemplate="%{text}<extra></extra>",
            showscale=True,
            colorbar=dict(
                title="Risk Score",
                tickformat=".0%",
                tickvals=[0, 0.25, 0.5, 0.65, 0.8, 1.0],
                ticktext=["0%", "25%", "50%", "65% (Alert)", "80%", "100%"],
                len=0.9,
            ),
        )
    )
    fig.update_layout(
        height=max(350, 20 * len(pids)),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="#fafafa", size=11),
        xaxis=dict(title="Prediction step (oldest → newest)", showgrid=False, tickvals=[]),
        yaxis=dict(title="Patient ID", autorange="reversed", tickfont=dict(size=10)),
        margin=dict(l=120, r=40, t=20, b=40),
    )
    # Alert threshold line
    fig.add_vline(
        x=max_len - 1,
        line_color="rgba(255,255,255,0.0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Summary table ─────────────────────────────────────────────────────
    st.subheader("Current Risk Summary")
    census = pipeline.census()[:max_patients]
    if census:
        summary_rows = []
        for p in census:
            pid = p.get("patient_id", "")
            score = p.get("risk_score", 0)
            sev = p.get("severity", "stable")
            # Color code severity
            sev_colors = {
                "critical": "🔴",
                "high": "🟠",
                "watch": "🟡",
                "stable": "🟢",
            }
            summary_rows.append({
                "Status": sev_colors.get(sev, "⚪"),
                "Patient": str(pid)[:14],
                "Risk Score": f"{score:.1%}",
                "Severity": sev.upper(),
                "NEWS2": p.get("news2", "—"),
                "qSOFA": p.get("qsofa", "—"),
                "SOFA": p.get("sofa", "—"),
                "Model": p.get("model_used", "—"),
            })
        df = pd.DataFrame(summary_rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
