"""
Dashboard Components: Alert Panel
"""

from __future__ import annotations
import streamlit as st
import pandas as pd
from datetime import datetime


def render_alert_panel(high_risk_patients: list[dict], audit_logger) -> None:
    """
    Renders the active high-risk patient alert panel.
    """
    st.subheader("🚨 Active High-Risk Alerts")

    if not high_risk_patients:
        st.success("✅ No high-risk patients currently detected.")
        return

    # Alert count badge
    st.markdown(
        f"<div style='background:#FF4B4B;color:white;padding:8px 16px;"
        f"border-radius:8px;display:inline-block;font-weight:bold;font-size:1.1em'>"
        f"⚠️ {len(high_risk_patients)} HIGH-RISK PATIENT(S)</div>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    for patient in high_risk_patients:
        score = patient.get("risk_score", 0)
        pid = patient.get("patient_id", "Unknown")
        ts = patient.get("timestamp", "")
        model_used = patient.get("model_used", "unknown")
        pred_id = patient.get("prediction_id", "")

        # Color-coded severity
        if score >= 0.8:
            border_color = "#FF4B4B"
            severity = "CRITICAL"
        elif score >= 0.65:
            border_color = "#FF8C00"
            severity = "HIGH"
        else:
            border_color = "#FFA500"
            severity = "MODERATE"

        with st.container():
            st.markdown(
                f"""<div style='border-left:4px solid {border_color};
                padding:12px 16px;margin-bottom:12px;
                background:rgba(255,75,75,0.05);border-radius:4px'>
                <strong>Patient {pid[:8]}</strong>
                &nbsp;<span style='background:{border_color};color:white;
                padding:2px 8px;border-radius:4px;font-size:0.8em'>{severity}</span>
                &nbsp;&nbsp;Risk Score: <strong>{score:.1%}</strong>
                &nbsp;&nbsp;Model: <code>{model_used}</code>
                &nbsp;&nbsp;<small style='color:#888'>{ts[:19] if ts else ''}</small>
                </div>""",
                unsafe_allow_html=True,
            )

            # Explanation
            exp = patient.get("explanation")
            if exp:
                with st.expander(f"📋 Why? — {exp.get('summary', '')[:80]}…"):
                    st.markdown(f"**{exp.get('summary', '')}**")
                    top = exp.get("top_features", [])
                    if top:
                        rows = []
                        for f in top[:5]:
                            rows.append({
                                "Feature": f.get("plain_english", f["feature"]),
                                "Value": f.get("value"),
                                "Contribution": f.get("contribution"),
                                "Direction": "🔴 Risk↑" if f.get("direction") == "increases_risk" else "🟢 Risk↓",
                            })
                        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            # Clinician feedback buttons
            col1, col2, col3 = st.columns([1, 1, 4])
            with col1:
                tp_key = f"tp_{pred_id}_{pid}"
                if st.button("✅ True Alert", key=tp_key, type="primary"):
                    st.session_state[f"feedback_{pred_id}"] = "true_positive"
                    st.toast(f"Feedback recorded: True Positive for {pid[:8]}")
            with col2:
                fp_key = f"fp_{pred_id}_{pid}"
                if st.button("❌ False Alert", key=fp_key):
                    st.session_state[f"feedback_{pred_id}"] = "false_positive"
                    st.toast(f"Feedback recorded: False Positive for {pid[:8]}")
