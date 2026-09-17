from __future__ import annotations

import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))


st.set_page_config(
    page_title="SepsisGuard EWS — Real-Time Patient Deterioration Alerts",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background-color: #0a0e1a; }
.block-container { padding-top: 1rem; }

/* Header */
.ews-header {
    background: linear-gradient(135deg, #0f1923 0%, #1a1f35 50%, #0d1b2a 100%);
    border-left: 4px solid #00B4D8;
    border-radius: 10px;
    padding: 18px 26px;
    margin-bottom: 18px;
    box-shadow: 0 4px 20px rgba(0,180,216,0.1);
}
.ews-header h1 {
    margin: 0; padding: 0; font-size: 1.45em;
    color: #ffffff; font-weight: 700;
    letter-spacing: -0.3px;
}
.ews-header .subtitle { color: #7a8ba0; font-size: 0.82em; margin-top: 6px; }
.ews-header .stats { color: #b0c4d8; font-size: 0.84em; margin-top: 8px; }

/* Badges */
.badge-online  { background: linear-gradient(90deg,#00CC44,#00aa33); color:#fff; padding:3px 12px; border-radius:12px; font-size:0.72em; font-weight:700; }
.badge-baseline{ background: linear-gradient(90deg,#FFA500,#e09000); color:#fff; padding:3px 12px; border-radius:12px; font-size:0.72em; font-weight:700; }
.badge-blend   { background: linear-gradient(90deg,#9B5DE5,#7b3db5); color:#fff; padding:3px 12px; border-radius:12px; font-size:0.72em; font-weight:700; }
.badge-drift   {
    background: linear-gradient(90deg,#FF4B4B,#cc2222);
    color:#fff; padding:3px 12px; border-radius:12px;
    font-size:0.72em; font-weight:700;
    animation: pulse 1s infinite;
}
@keyframes pulse { 0%{opacity:1} 50%{opacity:0.65} 100%{opacity:1} }

/* Sidebar */
[data-testid="stSidebar"] { background: linear-gradient(180deg,#0f1923,#0a0e1a); }
[data-testid="stSidebar"] .stRadio > label { color: #c0d0e0; }

/* Metric cards */
[data-testid="metric-container"] {
    background: linear-gradient(135deg,#141929,#0f1923);
    border: 1px solid #1e2a3a;
    border-radius: 10px;
    padding: 14px 16px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}
[data-testid="metric-container"]:hover {
    border-color: #00B4D8;
    box-shadow: 0 4px 16px rgba(0,180,216,0.15);
    transition: all 0.2s ease;
}
</style>
""", unsafe_allow_html=True)


# ── Cache-backed singletons ────────────────────────────────────────────────
@st.cache_resource(show_spinner="Initialising EWS pipeline & loading clinical data …")
def get_pipeline():
    from src.pipeline import EWSPipeline
    from src.ingestion.stream_simulator import StreamSimulator
    p = EWSPipeline()
    p.start()

    # Pre-populate pipeline with data on startup so dashboard opens with live predictions & alerts
    try:
        sim = StreamSimulator()
        sim.load_data(n_patients=50)
        labels = sim.get_patient_labels()
        p.load_labels(labels)
        
        n = 0
        for event in sim.stream(n_patients=50, realtime=False):
            p.process_event(event)
            n += 1
            if n >= 350:
                break
    except Exception as exc:
        print(f"Pre-population warning: {exc}")

    return p


@st.cache_resource(show_spinner="Loading stream simulator …")
def get_simulator():
    from src.ingestion.stream_simulator import StreamSimulator
    return StreamSimulator()


# ── Background stream thread ───────────────────────────────────────────────
def start_background_stream(pipeline, simulator):
    if st.session_state.get("stream_running"):
        return

    def _run():
        n_stays = st.session_state.get("n_stays_override", 80)
        try:
            labels = simulator.get_patient_labels()
            pipeline.load_labels(labels)
        except Exception:
            pass
        for event in simulator.stream(n_patients=n_stays, realtime=False):
            if not st.session_state.get("stream_running", False):
                break
            pipeline.process_event(event)
            time.sleep(0.08)

    st.session_state["stream_running"] = True
    t = threading.Thread(target=_run, daemon=True)
    t.start()


# ── Session state defaults ─────────────────────────────────────────────────
if "stream_running" not in st.session_state:
    st.session_state["stream_running"] = False
if "n_stays_override" not in st.session_state:
    st.session_state["n_stays_override"] = 80

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:12px 0 8px;'>
      <span style='font-size:2.2em'>🏥</span><br>
      <span style='font-size:1.1em;font-weight:700;color:#fff'>SepsisGuard</span><br>
      <span style='font-size:0.75em;color:#7a8ba0'>Early Warning System</span>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🚨 Patient Alerts",
            "🏥 Ward Census",
            "📈 Patient Timeline",
            "🌡️ Risk Heatmap",
            "📋 Shift Handoff",
            "📊 Performance",
            "🃏 Model Card",
            "⚙️ Admin",
        ],
        label_visibility="collapsed",
    )
    st.divider()

    st.markdown("**Stream Control**")
    n_stays = st.number_input(
        "Simulate N ICU stays", min_value=10, max_value=5000, value=80, step=10
    )
    st.session_state["n_stays_override"] = n_stays

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("▶ Start", use_container_width=True, type="primary"):
            pipeline = get_pipeline()
            simulator = get_simulator()
            simulator.load_data(n_patients=n_stays)
            start_background_stream(pipeline, simulator)
            st.toast("✅ Stream started", icon="🚀")
    with col_b:
        if st.button("⏹ Stop", use_container_width=True):
            st.session_state["stream_running"] = False
            st.toast("Stream stopped", icon="⏹")

    stream_status = "🟢 Running" if st.session_state["stream_running"] else "🔴 Stopped"
    st.caption(f"Stream: {stream_status}")

    st.divider()
    refresh = st.slider("Auto-refresh (s)", 3, 30, 5)
    st.caption("⚠️ Research demo only — not for clinical use")

# ── Pipeline status & header ───────────────────────────────────────────────
pipeline = get_pipeline()
status = pipeline.status()
active = status["active_model"]
badge_class = (
    "badge-online" if active == "online"
    else "badge-blend" if active == "blend"
    else "badge-baseline"
)
drift_badge = (
    f'<span class="badge-drift">⚠️ DRIFT ({status["total_drift_events"]})</span>'
    if status["total_drift_events"] > 0
    else ""
)
now_str = datetime.now().strftime("%H:%M:%S")

baseline_auroc = status.get("baseline_auroc", float("nan"))
baseline_auroc_str = f"{baseline_auroc:.3f}" if baseline_auroc == baseline_auroc else "—"

st.markdown(f"""
<div class="ews-header">
  <h1>SepsisGuard — Real-Time Deterioration EWS
  &nbsp;<span class="{badge_class}">{active.upper()} MODEL</span>
  &nbsp;{drift_badge}
  </h1>
  <div class="subtitle">
    PhysioNet/CinC 2019 · Hoeffding Adaptive Tree + Logistic Regression Baseline · NEWS2 / qSOFA / SIRS / SOFA
  </div>
  <div class="stats">
    🕐 {now_str}
    &nbsp;|&nbsp; Events: <strong>{status['n_events_processed']:,}</strong>
    &nbsp;|&nbsp; Predictions: <strong>{status['n_predictions']:,}</strong>
    &nbsp;|&nbsp; Alerts: <strong>{status['n_alerts']:,}</strong>
    &nbsp;|&nbsp; Patients: <strong>{status['n_patients_tracked']:,}</strong>
    &nbsp;|&nbsp; Online AUROC: <strong>{status['online_auroc']:.3f}</strong>
    &nbsp;|&nbsp; Baseline AUROC: <strong>{baseline_auroc_str}</strong>
    &nbsp;|&nbsp; Threshold: <strong>{status.get('alert_threshold', 0.65):.0%}</strong>
    &nbsp;|&nbsp; Updates: <strong>{status.get('n_online_updates', 0):,}</strong>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Page routing ───────────────────────────────────────────────────────────
if page == "🚨 Patient Alerts":
    from dashboard.components.alert_panel import render_alert_panel
    high_risk = pipeline.high_risk_patients()
    render_alert_panel(high_risk, pipeline.audit, pipeline=pipeline)

elif page == "🏥 Ward Census":
    from dashboard.components.ward_census import render_ward_census
    render_ward_census(pipeline)

elif page == "📈 Patient Timeline":
    from dashboard.components.patient_timeline import render_patient_timeline
    recent = pipeline.recent_results(400)
    render_patient_timeline(recent)

elif page == "🌡️ Risk Heatmap":
    from dashboard.components.risk_heatmap import render_risk_heatmap
    render_risk_heatmap(pipeline)

elif page == "📋 Shift Handoff":
    from dashboard.components.shift_summary import render_shift_summary
    render_shift_summary(pipeline)

elif page == "📊 Performance":
    from dashboard.monitoring import render_monitoring_page
    from dashboard.components.drift_chart import render_drift_chart
    render_monitoring_page(pipeline)
    st.divider()
    render_drift_chart(pipeline.drift_detector, pipeline.selector.status())

elif page == "🃏 Model Card":
    from dashboard.components.model_card import render_model_card
    render_model_card(pipeline)

elif page == "⚙️ Admin":
    st.title("⚙️ System Administration")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model Controls")

        # Ensemble strategy selector
        strategy_options = ["auto_rollback", "weighted_blend", "online_only", "baseline_only"]
        current_strategy = pipeline.selector.strategy
        new_strategy = st.selectbox(
            "Ensemble strategy",
            strategy_options,
            index=strategy_options.index(current_strategy) if current_strategy in strategy_options else 0,
            help="auto_rollback: use online, rollback when AUROC drops | weighted_blend: 0.6×online + 0.4×baseline | online_only | baseline_only",
        )
        if new_strategy != pipeline.selector.strategy:
            pipeline.selector.strategy = new_strategy
            st.info(f"Ensemble strategy set to: {new_strategy}")

        if pipeline.selector.strategy == "weighted_blend":
            blend_w = st.slider(
                "Online model blend weight",
                min_value=0.0, max_value=1.0,
                value=float(pipeline.selector.blend_online_weight), step=0.05,
            )
            pipeline.selector.blend_online_weight = blend_w
            pipeline.selector.blend_baseline_weight = round(1.0 - blend_w, 4)
            st.caption(f"Baseline weight: {pipeline.selector.blend_baseline_weight:.2f}")

        st.markdown("---")
        if st.button("🔄 Force Rollback to Baseline", type="secondary"):
            pipeline.selector.force_rollback()
            pipeline.audit.log_drift_event(
                {
                    "detector": "manual",
                    "timestamp": __import__("datetime").datetime.utcnow().isoformat(),
                    "severity": "drift",
                    "metric_value": 0.0,
                },
                model_version=pipeline.online_model.version,
                action_taken="manual_rollback",
            )
            st.warning("⚠️ Rolled back to baseline model.")

        if st.button("✅ Restore Online Model", type="primary"):
            pipeline.selector.restore_online()
            st.success("Online model restored.")

        if st.button("📸 Take Manual Snapshot"):
            entry = pipeline.registry.snapshot(reason="manual_admin")
            st.success(f"Snapshot saved: v{entry['version']}")

        st.subheader("Alert threshold")
        new_thr = st.slider(
            "Risk threshold",
            min_value=0.30, max_value=0.90,
            value=float(pipeline.selector.alert_threshold),
            step=0.05,
        )
        if abs(new_thr - pipeline.selector.alert_threshold) > 1e-9:
            pipeline.set_alert_threshold(new_thr)
            st.info(f"Alert threshold set to {new_thr:.0%}")

    with col2:
        st.subheader("Audit Export")
        if st.button("📥 Export Predictions CSV"):
            pipeline.audit.export_predictions_csv("logs/predictions_export.csv")
            st.success("Exported to logs/predictions_export.csv")

        st.subheader("Shadow Mode")
        shadow = st.toggle(
            "Shadow Mode (log predictions but suppress alerts)",
            value=pipeline.selector.shadow_mode,
        )
        if shadow != pipeline.selector.shadow_mode:
            pipeline.selector.shadow_mode = shadow
            st.info(f"Shadow mode {'🟡 enabled' if shadow else '🟢 disabled'}.")

    st.divider()
    st.subheader("System Status")
    st.json(status)

time.sleep(refresh)
st.rerun()
