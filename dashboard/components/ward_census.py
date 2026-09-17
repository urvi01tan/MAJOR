"""Ward census: all tracked patients ranked by ML risk plus NEWS2/qSOFA/SIRS/SOFA."""

from __future__ import annotations

import io

import pandas as pd
import plotly.express as px
import streamlit as st


def render_ward_census(pipeline) -> None:
    st.title("🏥 Ward Census")
    st.caption("Every tracked patient, ranked by current ML deterioration risk.")

    patients = pipeline.census()
    if not patients:
        st.info("No patients yet. Start the stream from the sidebar.")
        return

    # ── Summary severity bar chart ─────────────────────────────────────────
    sev_counts = {"critical": 0, "high": 0, "watch": 0, "stable": 0}
    for p in patients:
        sev = p.get("severity", "stable")
        if sev in sev_counts:
            sev_counts[sev] += 1

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("🔴 Critical", sev_counts["critical"])
    s2.metric("🟠 High", sev_counts["high"])
    s3.metric("🟡 Watch", sev_counts["watch"])
    s4.metric("🟢 Stable", sev_counts["stable"])

    # Severity distribution bar chart
    sev_df = pd.DataFrame([
        {"Severity": "Critical", "Count": sev_counts["critical"], "color": "#FF0A0A"},
        {"Severity": "High", "Count": sev_counts["high"], "color": "#FF4B4B"},
        {"Severity": "Watch", "Count": sev_counts["watch"], "color": "#FFA500"},
        {"Severity": "Stable", "Count": sev_counts["stable"], "color": "#00CC44"},
    ])
    fig_bar = px.bar(
        sev_df, x="Severity", y="Count",
        color="Severity",
        color_discrete_map={
            "Critical": "#FF0A0A", "High": "#FF4B4B",
            "Watch": "#FFA500", "Stable": "#00CC44",
        },
        text="Count",
    )
    fig_bar.update_layout(
        height=200,
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=30),
        xaxis=dict(title=""),
        yaxis=dict(title="Patients"),
    )
    fig_bar.update_traces(textposition="outside")
    st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()

    # ── Filter controls ────────────────────────────────────────────────────
    fc1, fc2, fc3 = st.columns([2, 2, 1])
    with fc1:
        query = st.text_input("🔍 Search patient ID")
    with fc2:
        band = st.multiselect(
            "Filter severity",
            ["critical", "high", "watch", "stable"],
            default=["critical", "high", "watch", "stable"],
        )
    with fc3:
        show_all_vitals = st.checkbox("Show labs", value=False)

    # ── Build table ────────────────────────────────────────────────────────
    rows = []
    for p in patients:
        pid = p.get("patient_id", "")
        if query and query.lower() not in str(pid).lower():
            continue
        sev = p.get("severity", "stable")
        if band and sev not in band:
            continue
        vitals = p.get("vitals") or {}

        # Severity emoji
        sev_icon = {"critical": "🔴", "high": "🟠", "watch": "🟡", "stable": "🟢"}.get(sev, "⚪")
        row = {
            "": sev_icon,
            "Patient": str(pid)[:14],
            "ML Risk": p.get("risk_score"),
            "Severity": sev.upper(),
            "NEWS2": p.get("news2"),
            "qSOFA": p.get("qsofa"),
            "SIRS": p.get("sirs"),
            "SOFA": p.get("sofa"),
            "HR": vitals.get("HR"),
            "SBP": vitals.get("SBP"),
            "Resp": vitals.get("Resp"),
            "O2Sat": vitals.get("O2Sat"),
        }
        if show_all_vitals:
            row.update({
                "Temp": vitals.get("Temp"),
                "Lactate": vitals.get("Lactate"),
                "Creatinine": vitals.get("Creatinine"),
                "WBC": vitals.get("WBC"),
                "Platelets": vitals.get("Platelets"),
            })
        row.update({
            "Model": p.get("model_used"),
            "Watchlist": "👁️" if p.get("on_watchlist") else "",
            "Ack": "✓" if p.get("acknowledged") else "",
            "Updated": str(p.get("timestamp", ""))[:19],
        })
        rows.append(row)

    if not rows:
        st.warning("No patients match the current filters.")
        return

    df = pd.DataFrame(rows)
    df["ML Risk"] = df["ML Risk"].map(lambda x: f"{x:.1%}" if x is not None else "—")

    # Numeric formatting
    for col in ["HR", "SBP", "Resp", "O2Sat", "Temp", "Lactate", "Creatinine", "WBC", "Platelets"]:
        if col in df.columns:
            df[col] = df[col].map(lambda x: f"{x:.1f}" if x is not None else "—")

    st.dataframe(df, use_container_width=True, hide_index=True, height=440)

    # ── CSV download ───────────────────────────────────────────────────────
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    st.download_button(
        "📥 Download census CSV",
        data=csv_buf.getvalue().encode(),
        file_name="ward_census.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # ── Watchlist ──────────────────────────────────────────────────────────
    watch = pipeline.watchlist_patients()
    if watch:
        st.divider()
        st.subheader("👁️ Watchlist")
        st.dataframe(
            pd.DataFrame([
                {
                    "Patient": p.get("patient_id"),
                    "Risk": f"{p.get('risk_score', 0):.1%}",
                    "NEWS2": p.get("news2"),
                    "SOFA": p.get("sofa"),
                    "Severity": p.get("severity", "—").upper(),
                }
                for p in watch
            ]),
            use_container_width=True,
            hide_index=True,
        )
