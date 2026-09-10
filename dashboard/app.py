"""
Clinician Dashboard — Main Streamlit App
==========================================
Real-time EWS clinician-facing interface.

Pages:
  1. 🏥 Patient Alerts   — active high-risk patient panel with explanations
  2. 📈 Patient Timeline  — risk score history for selected patient
  3. 📊 Performance       — AUROC, drift, model version monitoring
  4. 🔧 Admin             — manual rollback, model controls, audit export

Run with:
  streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="Patient EWS — Real-Time Deterioration Alerts",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.main { background-color: #0e1117; }
.block-container { padding-top: 1.5rem; }

/* Top header bar */
.ews-header {
    background: linear-gradient(135deg, #1a1f2e 0%, #0f1923 100%);
    border-left: 4px solid #00B4D8;
    padding: 16px 24px;
    border-radius: 8px;
    margin-bottom: 20px;
}
.ews-header h1 {
    margin: 0; padding: 0;
    font-size: 1.4em;
    color: #ffffff;
    font-weight: 600;
}
.ews-header small { color: #888; font-size: 0.85em; }

/* Status badge */
.badge-online {
    background: #00CC44; color: white;
    padding: 3px 10px; border-radius: 12px;
    font-size: 0.75em; font-weight: 600;
}
.badge-baseline {
    background: #FFA500; color: white;
    padding: 3px 10px; border-radius: 12px;
    font-size: 0.75em; font-weight: 600;
}
.badge-drift {
    background: #FF4B4B; color: white;
    padding: 3px 10px; border-radius: 12px;
    font-size: 0.75em; font-weight: 600;
    animation: pulse 1s infinite;
}
@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.6; }
    100% { opacity: 1; }
}

/* Metric cards */
.stMetric { background: #1a1f2e; padding: 12px; border-radius: 8px; }

/* Alert card */
.alert-critical { border-left: 4px solid #FF4B4B; background: rgba(255,75,75,0.08); }
.alert-high { border-left: 4px solid #FF8C00; background: rgba(255,140,0,0.08); }
</style>
""", unsafe_allow_html=True)


# ── Pipeline singleton (initialise once per session) ─────────────────────────
@st.cache_resource(show_spinner="Initialising pipeline …")
def get_pipeline():
    from src.pipeline import EWSPipeline
    p = EWSPipeline()
    p.start()
    return p


@st.cache_resource(show_spinner="Loading stream simulator …")
def get_simulator():
    from src.ingestion.stream_simulator import StreamSimulator, SepsisLabelGenerator
    import yaml
    with open("config/settings.yaml") as f:
        cfg = yaml.safe_load(f)
    sim = StreamSimulator()
    return sim, cfg


def start_background_stream(pipeline, simulator):
    """Run stream simulation in background thread, feeding the pipeline."""
    if st.session_state.get("stream_running"):
        return

    def _run():
        n_stays = st.session_state.get("n_stays_override")
        for event in simulator.stream(n_stays=n_stays, realtime=True):
            if not st.session_state.get("stream_running", False):
                break
            pipeline.process_event(event)
            time.sleep(0)  # yield

    st.session_state["stream_running"] = True
    t = threading.Thread(target=_run, daemon=True)
    t.start()


# ── Session state defaults ────────────────────────────────────────────────────
if "stream_running" not in st.session_state:
    st.session_state["stream_running"] = False
if "n_stays_override" not in st.session_state:
    st.session_state["n_stays_override"] = 200


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 Patient EWS")
    st.markdown("Early Warning System")
    st.divider()

    page = st.radio(
        "Navigation",
        ["🚨 Patient Alerts", "📈 Patient Timeline", "📊 Performance", "🔧 Admin"],
        label_visibility="collapsed",
    )
    st.divider()

    # Stream control
    st.markdown("**Stream Control**")
    n_stays = st.number_input("Simulate N ICU stays", min_value=10, max_value=5000,
                               value=200, step=50)
    st.session_state["n_stays_override"] = n_stays

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("▶ Start", use_container_width=True, type="primary"):
            pipeline = get_pipeline()
            simulator, _ = get_simulator()
            simulator.load_data(n_stays=n_stays)
            start_background_stream(pipeline, simulator)
            st.toast("Stream started!")
    with col_b:
        if st.button("⏹ Stop", use_container_width=True):
            st.session_state["stream_running"] = False
            st.toast("Stream stopped.")

    stream_status = "🟢 Running" if st.session_state["stream_running"] else "⚫ Stopped"
    st.caption(f"Stream: {stream_status}")

    st.divider()
    refresh = st.slider("Auto-refresh (s)", 3, 30, 5)
    st.caption("⚠️ Demo only — not for clinical use")


# ── Get live pipeline ─────────────────────────────────────────────────────────
pipeline = get_pipeline()
status = pipeline.status()

# ── Header ────────────────────────────────────────────────────────────────────
active = status["active_model"]
badge_class = "badge-online" if active == "online" else "badge-baseline"
drift_badge = (
    f'<span class="badge-drift">⚡ DRIFT DETECTED ({status["total_drift_events"]})</span>'
    if status["total_drift_events"] > 0
    else ""
)

st.markdown(f"""
<div class="ews-header">
  <h1>🏥 Real-Time Patient Deterioration EWS
  &nbsp;<span class="{badge_class}">{active.upper()} MODEL</span>
  &nbsp;{drift_badge}
  </h1>
  <small>
  Events: {status['n_events_processed']:,} &nbsp;|&nbsp;
  Predictions: {status['n_predictions']:,} &nbsp;|&nbsp;
  Alerts: {status['n_alerts']:,} &nbsp;|&nbsp;
  Patients: {status['n_patients_tracked']:,} &nbsp;|&nbsp;
  AUROC: {status['online_auroc']:.3f} &nbsp;|&nbsp;
  Rollbacks: {status['rollback_count']}
  </small>
</div>
""", unsafe_allow_html=True)


# ── Page routing ──────────────────────────────────────────────────────────────

if page == "🚨 Patient Alerts":
    from dashboard.components.alert_panel import render_alert_panel
    high_risk = pipeline.high_risk_patients()
    render_alert_panel(high_risk, pipeline.audit)

    st.divider()
    st.subheader("📋 Recent Predictions")
    recent = pipeline.audit.recent_predictions(50)
    if recent:
        import pandas as pd
        df = pd.DataFrame(recent)
        df["alert_fired"] = df["alert_fired"].map({True: "🚨 YES", False: "·"})
        df["risk_score"] = df["risk_score"].map(lambda x: f"{x:.1%}" if x else "—")
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No predictions yet. Start the stream to begin.")

elif page == "📈 Patient Timeline":
    from dashboard.components.patient_timeline import render_patient_timeline
    recent = pipeline.recent_results(200)
    render_patient_timeline(recent)

elif page == "📊 Performance":
    from dashboard.monitoring import render_monitoring_page
    from dashboard.components.drift_chart import render_drift_chart
    render_monitoring_page(pipeline)
    st.divider()
    render_drift_chart(pipeline.drift_detector, pipeline.selector.status())

elif page == "🔧 Admin":
    st.title("🔧 System Administration")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model Controls")
        if st.button("⏪ Force Rollback to Baseline", type="secondary"):
            pipeline.selector.force_rollback()
            pipeline.audit.log_drift_event(
                {"detector": "manual", "timestamp": __import__("datetime").datetime.utcnow().isoformat(),
                 "severity": "drift", "metric_value": 0.0},
                model_version=pipeline.online_model.version,
                action_taken="manual_rollback",
            )
            st.warning("Rolled back to baseline model.")

        if st.button("✅ Restore Online Model", type="primary"):
            pipeline.selector.restore_online()
            st.success("Online model restored.")

        if st.button("📸 Take Manual Snapshot"):
            entry = pipeline.registry.snapshot(reason="manual_admin")
            st.success(f"Snapshot saved: v{entry['version']}")

    with col2:
        st.subheader("Audit Export")
        if st.button("📥 Export Predictions CSV"):
            pipeline.audit.export_predictions_csv("logs/predictions_export.csv")
            st.success("Exported to logs/predictions_export.csv")

        st.subheader("Shadow Mode")
        shadow = st.toggle(
            "Shadow Mode (log predictions but don't show alerts)",
            value=pipeline.selector.shadow_mode
        )
        if shadow != pipeline.selector.shadow_mode:
            pipeline.selector.shadow_mode = shadow
            st.info(f"Shadow mode {'enabled' if shadow else 'disabled'}.")

    st.divider()
    st.subheader("System Status")
    import json
    st.json(status)


# ── Auto-refresh ──────────────────────────────────────────────────────────────
time.sleep(refresh)
st.rerun()
