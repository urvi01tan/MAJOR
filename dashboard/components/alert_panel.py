from __future__ import annotations


import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def _sparkline(values: list[float], color: str = "#F77F00", height: int = 60) -> go.Figure:
    fig = go.Figure(go.Scatter(
        y=values,
        mode="lines",
        line=dict(color=color, width=2),
        fill="tozeroy",
        fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.12)",
    ))
    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, range=[0, 1]),
        showlegend=False,
    )
    return fig


def _sofa_badge(score: int | None, risk: str | None) -> str:
    if score is None:
        return "—"
    colors = {"critical": "#FF0A0A", "high": "#FF4B4B", "moderate": "#FFA500", "low": "#FFD166", "none": "#00CC44"}
    c = colors.get(risk or "none", "#888")
    return f"<span style='background:{c};color:#fff;padding:2px 7px;border-radius:4px;font-size:0.8em;font-weight:bold'>{score}</span>"


def render_alert_panel(high_risk_patients: list[dict], audit_logger, pipeline=None) -> None:
    st.subheader("Active High-Risk Alerts")

    # ── Filter controls ───────────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3 = st.columns([2, 1, 1])
    with ctrl1:
        hide_acked = st.checkbox("Hide acknowledged alerts", value=False)
    with ctrl2:
        min_sev = st.selectbox("Min severity", ["watch", "high", "critical"], index=0)
    with ctrl3:
        sort_by = st.selectbox("Sort by", ["Risk ↓", "NEWS2 ↓", "SOFA ↓"])

    sev_order = {"stable": 0, "watch": 1, "high": 2, "critical": 3}
    min_sev_val = sev_order.get(min_sev, 1)

    patients = [
        p for p in high_risk_patients
        if sev_order.get(p.get("severity", "stable"), 0) >= min_sev_val
    ]
    if hide_acked:
        patients = [p for p in patients if not p.get("acknowledged")]

    if sort_by == "NEWS2 ↓":
        patients = sorted(patients, key=lambda p: p.get("news2") or 0, reverse=True)
    elif sort_by == "SOFA ↓":
        patients = sorted(patients, key=lambda p: p.get("sofa") or 0, reverse=True)

    # ── Banner ────────────────────────────────────────────────────────────
    if not patients:
        st.info("ℹ️ No alerts matching severity filter. Lowering filter to 'watch' or starting stream...")
        col_f1, col_f2 = st.columns([1, 2])
        with col_f1:
            if st.button("⚡ Inject High-Risk Patient Data", type="primary", use_container_width=True):
                if pipeline:
                    from src.ingestion.stream_simulator import StreamSimulator
                    sim = StreamSimulator()
                    sim.load_data(n_patients=30)
                    for event in sim.stream(n_patients=30, realtime=False):
                        r = pipeline.process_event(event)
                        if r and r.get("alert"):
                            break
                    st.rerun()
        # Fallback to census patients if any tracked
        if pipeline:
            patients = pipeline.census()[:5]

    if not patients:
        st.warning("No patient data streamed yet. Click '▶ Start' on the sidebar or click '⚡ Inject High-Risk Patient Data' above.")
        return

    critical_count = sum(1 for p in patients if p.get("severity") == "critical")
    glow_color = "rgba(255, 23, 68, 0.45)" if critical_count else "rgba(255, 75, 75, 0.35)"
    border_c = "#FF1744" if critical_count else "#FF4B4B"
    st.markdown(
        f"""<div style='background:linear-gradient(135deg, rgba(255, 23, 68, 0.15) 0%, rgba(200, 16, 46, 0.25) 100%);
        border:1px solid {border_c}; color:#ffffff; padding:12px 22px; border-radius:10px;
        display:inline-flex; align-items:center; gap:10px; font-weight:700; font-size:1.05em;
        box-shadow: 0 0 20px {glow_color}; letter-spacing:0.2px;'>
        <span style='display:inline-block; width:10px; height:10px; border-radius:50%; background:#FF1744; box-shadow:0 0 10px #FF1744;'></span>
        🚨 {len(patients)} HIGH-RISK ALERT PATIENT(S)
        {f'<span style="background:#FF1744; color:#fff; padding:2px 8px; border-radius:12px; font-size:0.8em; margin-left:6px;">{critical_count} CRITICAL</span>' if critical_count else ''}
        </div>""",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

    # ── Patient cards ─────────────────────────────────────────────────────
    for idx, patient in enumerate(patients):
        score = patient.get("risk_score", 0)
        pid = patient.get("patient_id", "Unknown")
        ts = patient.get("timestamp", "")
        model_used = patient.get("model_used", "unknown")
        pred_id = patient.get("prediction_id", "")
        severity = (patient.get("severity") or "high").upper()
        news2 = patient.get("news2")
        qsofa = patient.get("qsofa")
        sirs = patient.get("sirs")
        sofa = patient.get("sofa")
        sofa_risk = patient.get("sofa_risk")
        acked = patient.get("acknowledged", False)
        on_wl = patient.get("on_watchlist", False)

        # Colors
        if score >= 0.80:
            border_color = "#FF1744"
            card_bg = "linear-gradient(135deg, rgba(255, 23, 68, 0.12) 0%, rgba(20, 26, 40, 0.7) 100%)"
        elif score >= 0.65:
            border_color = "#FF5252"
            card_bg = "linear-gradient(135deg, rgba(255, 82, 82, 0.08) 0%, rgba(18, 24, 38, 0.7) 100%)"
        else:
            border_color = "#FFA500"
            card_bg = "linear-gradient(135deg, rgba(255, 165, 0, 0.08) 0%, rgba(16, 22, 36, 0.7) 100%)"

        wl_badge = " 👁️ WATCHLIST" if on_wl else ""
        ack_badge = " ✓ ACK" if acked else ""

        with st.container():
            # Header row
            st.markdown(
                f"""<div style='border-left:4px solid {border_color};
                background:{card_bg}; padding:14px 18px; border-radius:10px;
                border-top: 1px solid rgba(255,255,255,0.06);
                border-right: 1px solid rgba(255,255,255,0.06);
                border-bottom: 1px solid rgba(255,255,255,0.04);
                margin-bottom:8px; box-shadow: 0 4px 16px rgba(0,0,0,0.3);'>
                <span style='font-size:1.15em; font-weight:800; color:#ffffff;'>Patient {pid}</span>
                &nbsp;<span style='color:{border_color}; font-weight:700; background:rgba(255,255,255,0.05); padding:2px 8px; border-radius:6px; font-size:0.85em;'>{severity}</span>
                &nbsp;<span style='color:#7a8ba0; font-size:0.82em; font-family:"JetBrains Mono",monospace;'>{ts}</span>
                <span style='float:right; font-size:1.3em; font-weight:800; color:{border_color}; font-family:"JetBrains Mono",monospace;'>
                {score:.1%}{wl_badge}{ack_badge}
                </span>
                </div>""",
                unsafe_allow_html=True,
            )

            # Score pills row
            sofa_b = _sofa_badge(sofa, sofa_risk)
            st.markdown(
                f"<div style='font-size:0.85em;color:#aaa;margin-bottom:8px;'>"
                f"Model: <code>{model_used}</code> &nbsp;|&nbsp; "
                f"NEWS2: <b>{news2 if news2 is not None else '—'}</b> &nbsp;|&nbsp; "
                f"qSOFA: <b>{qsofa if qsofa is not None else '—'}</b> &nbsp;|&nbsp; "
                f"SIRS: <b>{sirs if sirs is not None else '—'}</b> &nbsp;|&nbsp; "
                f"SOFA: {sofa_b}"
                f"</div>",
                unsafe_allow_html=True,
            )

            # ── Key vitals summary ─────────────────────────────────────────
            vitals = patient.get("vitals", {})
            vital_labels = [
                ("HR", "bpm"), ("MAP", "mmHg"), ("O2Sat", "%"),
                ("Temp", "°C"), ("Resp", "/min"), ("Lactate", "mmol/L")
            ]
            if any(vitals.get(k) is not None for k, _ in vital_labels):
                vcols = st.columns(len(vital_labels))
                for i, (key, unit) in enumerate(vital_labels):
                    val = vitals.get(key)
                    vcols[i].metric(key, f"{val:.1f}" if val is not None else "—", help=unit)

            # ── Sparkline: fetch patient risk history ─────────────────────
            if pipeline is not None:
                history = pipeline.patient_risk_history(pid)
                if len(history) >= 3:
                    spark_col1, spark_col2, spark_col3 = st.columns([3, 2, 2])
                    with spark_col1:
                        st.caption("Risk trend")
                        risk_vals = [h["risk_score"] for h in history]
                        st.plotly_chart(
                            _sparkline(risk_vals, color=border_color),
                            use_container_width=True,
                            config={"displayModeBar": False},
                            key=f"spark_risk_{pid}_{idx}",
                        )
                    with spark_col2:
                        st.caption("NEWS2 trend")
                        news_vals = [(h.get("news2") or 0) / 20.0 for h in history]
                        st.plotly_chart(
                            _sparkline(news_vals, color="#9B5DE5"),
                            use_container_width=True,
                            config={"displayModeBar": False},
                            key=f"spark_news_{pid}_{idx}",
                        )
                    with spark_col3:
                        st.caption("SOFA trend")
                        sofa_vals = [(h.get("sofa") or 0) / 12.0 for h in history]
                        st.plotly_chart(
                            _sparkline(sofa_vals, color="#00B4D8"),
                            use_container_width=True,
                            config={"displayModeBar": False},
                            key=f"spark_sofa_{pid}_{idx}",
                        )

            # ── Explanation ───────────────────────────────────────────────
            exp = patient.get("explanation")
            if exp:
                with st.expander(f"🔍 Why? — {str(exp.get('summary', ''))[:80]}"):
                    st.markdown(f"**{exp.get('summary', '')}**")
                    top = exp.get("top_features", [])
                    if top:
                        rows = []
                        for feat in top[:6]:
                            direction_icon = "⬆️" if feat.get("direction") == "increases_risk" else "⬇️"
                            row_dict = {
                                "Feature": feat.get("plain_english", feat["feature"]),
                                "Ref Range": feat.get("reference_range", "—"),
                                "Value": f"{feat.get('value'):.2f}" if feat.get("value") is not None else "—",
                                "Contribution": f"{feat.get('contribution', 0):.4f}",
                                "Effect": f"{direction_icon} {'Risk ↑' if feat.get('direction') == 'increases_risk' else 'Risk ↓'}",
                            }
                            if feat.get("clinical_justification"):
                                row_dict["Clinical Rationale"] = feat.get("clinical_justification")
                            rows.append(row_dict)
                        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                        case_examples = [f.get("case_example") for f in top[:4] if f.get("case_example")]
                        if case_examples:
                            st.markdown("##### 📖 Clinical Case Examples Justifying Abnormal Flags:")
                            for eg in case_examples:
                                st.markdown(f"- 🩺 *{eg}*")

            # ── AI Clinical Narrative & Insights ──────────────────────────
            with st.expander(f"✨ AI Clinical Insight — Patient {pid}"):
                from src.ai.services import AIAssistService
                ai_service = AIAssistService()
                narrative_res = ai_service.narrative.narrate(patient)
                st.markdown(narrative_res.get("text", "No narrative available."))

            # ── Action buttons ────────────────────────────────────────────
            c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1.8])
            with c1:
                if st.button("✅ True Alert", key=f"tp_{pred_id}_{pid}", type="primary"):
                    if pipeline is not None:
                        pipeline.ingest_feedback(pred_id, pid, feedback="true_positive")
                    st.toast(f"✅ True positive recorded for {pid}", icon="✅")
            with c2:
                if st.button("❌ False Alert", key=f"fp_{pred_id}_{pid}"):
                    if pipeline is not None:
                        pipeline.ingest_feedback(pred_id, pid, feedback="false_positive")
                    st.toast(f"False positive for {pid}", icon="❌")
            with c3:
                if st.button("🔕 Acknowledge", key=f"ack_{pred_id}_{pid}"):
                    if pipeline is not None:
                        pipeline.acknowledge_alert(pid)
                    st.toast(f"Acknowledged {pid}", icon="🔕")
            with c4:
                wl_label = "➖ Watchlist" if on_wl else "👁️ Watchlist"
                if st.button(wl_label, key=f"wl_{pred_id}_{pid}"):
                    if pipeline is not None:
                        on = pipeline.toggle_watchlist(pid)
                        st.toast(f"{'Added to' if on else 'Removed from'} watchlist: {pid}")
            with c5:
                if st.button("💬 Ask AI", key=f"ask_ai_{pred_id}_{pid}"):
                    if "ai_chat_history" not in st.session_state:
                        st.session_state["ai_chat_history"] = []
                    st.session_state["ai_chat_history"].append({
                        "role": "user",
                        "content": f"Explain why patient {pid} is flagged at high risk and what the recommended checks are."
                    })
                    st.toast(f"Focus set to patient {pid}. Switch to '💬 AI Chatbot' tab to chat!", icon="💬")

            st.markdown("---")

    # ── Recent predictions table ──────────────────────────────────────────
    st.subheader("Recent Predictions")
    recent = audit_logger.recent_predictions(50)
    if recent:
        df = pd.DataFrame(recent)
        df["alert_fired"] = df["alert_fired"].map({True: " YES", False: "·"})
        df["risk_score"] = df["risk_score"].map(lambda x: f"{x:.1%}" if x else "—")
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No predictions yet. Start the stream to begin.")
