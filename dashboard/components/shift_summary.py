"""End-of-shift handoff report with PPV, recall, and severity donut chart."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def render_shift_summary(pipeline) -> None:
    st.title("📋 Shift Handoff")
    st.caption(
        "Snapshot for charge-nurse handover: who is at risk, "
        "how many alerts fired, and model health for the shift."
    )

    summary = pipeline.shift_summary()
    bands = summary.get("risk_bands", {})
    status = pipeline.status()

    # ── Key metrics ────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Patients tracked", summary.get("n_patients", 0))
    c2.metric("🔴 Critical", bands.get("critical", 0))
    c3.metric("🟠 High", bands.get("high", 0))
    c4.metric("Alerts (total)", summary.get("n_alerts", 0))
    c5.metric("Watchlist", summary.get("n_watchlist", 0))

    c6, c7, c8, c9, c10 = st.columns(5)
    c6.metric("True alerts marked", summary.get("true_positives", 0))
    c7.metric("False alerts marked", summary.get("false_positives", 0))

    ppv = summary.get("ppv")
    c8.metric("PPV (from feedback)", f"{ppv:.0%}" if ppv is not None else "—")

    online_auroc = status.get("online_auroc", 0)
    baseline_auroc = status.get("baseline_auroc", float("nan"))
    auroc_delta = None
    if baseline_auroc == baseline_auroc and baseline_auroc > 0:
        auroc_delta = f"{online_auroc - baseline_auroc:+.3f} vs baseline"
    c9.metric("Online AUROC", f"{online_auroc:.3f}", delta=auroc_delta)
    c10.metric("Active Model", str(status.get("active_model", "—")).title())

    # ── Severity donut chart ──────────────────────────────────────────────
    st.divider()
    col_donut, col_table = st.columns([1, 2])

    with col_donut:
        st.subheader("Risk Band Distribution")
        labels = ["Critical", "High", "Watch", "Stable"]
        values = [
            bands.get("critical", 0),
            bands.get("high", 0),
            bands.get("watch", 0),
            bands.get("stable", 0),
        ]
        donut_colors = ["#FF0A0A", "#FF4B4B", "#FFA500", "#00CC44"]

        fig_donut = go.Figure(go.Pie(
            labels=labels,
            values=values,
            hole=0.55,
            marker=dict(colors=donut_colors, line=dict(color="#0e1117", width=2)),
            textinfo="label+percent",
            textfont=dict(color="#fff", size=12),
        ))
        fig_donut.update_layout(
            height=280,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#fafafa"),
            showlegend=False,
            margin=dict(l=10, r=10, t=20, b=10),
            annotations=[dict(
                text=f"{sum(values)}<br>patients",
                x=0.5, y=0.5, font_size=14,
                font_color="#fff", showarrow=False,
            )],
        )
        st.plotly_chart(fig_donut, use_container_width=True)

        # Alert rate
        rate = pipeline.alert_rate_per_hour(window_hours=1.0)
        st.metric("Alert rate (last hour)", f"{rate:.1f} / hr")
        st.metric("Drift events", status.get("total_drift_events", 0))
        st.metric("Model rollbacks", status.get("rollback_count", 0))

    with col_table:
        st.subheader("Highest-risk patients this shift")
        top = pipeline.census()[:20]
        if not top:
            st.info("No census data yet.")
        else:
            rows = []
            for p in top:
                sev = p.get("severity", "stable")
                sev_icon = {"critical": "🔴", "high": "🟠", "watch": "🟡", "stable": "🟢"}.get(sev, "⚪")
                rows.append({
                    "": sev_icon,
                    "Patient": p.get("patient_id"),
                    "Risk": f"{p.get('risk_score', 0):.1%}",
                    "NEWS2": p.get("news2"),
                    "qSOFA": p.get("qsofa"),
                    "SOFA": p.get("sofa"),
                    "Severity": sev.upper(),
                    "Why": str((p.get("explanation") or {}).get("summary", ""))[:50],
                })
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
            )

    # ── Alert quality ─────────────────────────────────────────────────────
    st.divider()
    st.subheader("Alert Quality (clinician feedback)")
    tp = summary.get("true_positives", 0)
    fp = summary.get("false_positives", 0)
    total_feedback = tp + fp

    if total_feedback > 0:
        quality_fig = go.Figure(go.Bar(
            x=["True Alerts", "False Alerts"],
            y=[tp, fp],
            marker_color=["#00CC44", "#FF4B4B"],
            text=[tp, fp],
            textposition="outside",
        ))
        quality_fig.update_layout(
            height=220,
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font=dict(color="#fafafa"),
            showlegend=False,
            margin=dict(l=20, r=20, t=20, b=20),
            yaxis=dict(title="Count"),
        )
        st.plotly_chart(quality_fig, use_container_width=True)
    else:
        st.info("No clinician feedback yet. Mark alerts as True/False from the Patient Alerts page.")

    # ── Recent feedback ────────────────────────────────────────────────────
    feedback = pipeline.audit.recent_feedback(20)
    if feedback:
        st.subheader("Recent clinician feedback")
        st.dataframe(pd.DataFrame(feedback), use_container_width=True, hide_index=True)
