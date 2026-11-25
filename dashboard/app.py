from __future__ import annotations

import sys
import threading
from datetime import datetime, timezone
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
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

/* Global typography (FIX: no longer overrides icon fonts via [class*="st-"]) */
html, body, .stApp, .stMarkdown, p, h1, h2, h3, h4, h5, h6, label, button, input, textarea, select,
[data-testid="stMetricLabel"], [data-baseweb="tab"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    -webkit-font-smoothing: antialiased;
}
/* Keep Streamlit's icon font intact (expander arrows, sidebar toggle, etc.) */
[data-testid="stIconMaterial"], .material-icons, .material-symbols-rounded, .material-symbols-outlined {
    font-family: 'Material Symbols Rounded', 'Material Icons' !important;
}

.stApp {
    background: radial-gradient(circle at 12% 8%, rgba(0, 229, 255, 0.05) 0%, transparent 40%),
                radial-gradient(circle at 88% 18%, rgba(124, 77, 255, 0.04) 0%, transparent 45%),
                #070a12 !important;
    color: #e2e8f0;
}

.main { background: transparent !important; }
.block-container { padding-top: 1.1rem; padding-bottom: 2rem; max-width: 96%; }

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #070a12; }
::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #00B4D8; }

/* ── Header Card ── */
.ews-header {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.88) 0%, rgba(10, 16, 30, 0.95) 100%);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(0, 229, 255, 0.18);
    border-left: 5px solid #00E5FF;
    border-radius: 14px;
    padding: 20px 26px;
    margin-bottom: 22px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.08);
    position: relative;
    overflow: hidden;
}
.ews-header::after {
    content: '';
    position: absolute;
    top: 0; right: 0; width: 300px; height: 100%;
    background: radial-gradient(circle at right, rgba(0, 229, 255, 0.06), transparent 70%);
    pointer-events: none;
}
.ews-header h1 {
    margin: 0; padding: 0; font-size: 1.55em;
    color: #ffffff; font-weight: 800;
    letter-spacing: -0.4px;
    display: flex; align-items: center; flex-wrap: wrap; gap: 8px;
}
.ews-header .stats {
    margin-top: 14px;
    display: flex; flex-wrap: wrap; gap: 10px; align-items: center;
}
.stat-chip {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.78em;
    color: #94a3b8;
    display: inline-flex; align-items: center; gap: 5px;
    font-family: 'JetBrains Mono', monospace;
}
.stat-chip strong { color: #f8fafc; font-weight: 600; }
.stat-chip.accent { border-color: rgba(0, 229, 255, 0.35); color: #00E5FF; }
.stat-chip.alert  { border-color: rgba(255, 75, 75, 0.4); color: #FF7B7B; }

/* ── Live Pulse ── */
.pulse-dot {
    display: inline-block; width: 8px; height: 8px; border-radius: 50%;
    background: #00E676; box-shadow: 0 0 10px #00E676;
    animation: pulse-ring 1.8s cubic-bezier(0.455, 0.03, 0.515, 0.955) infinite;
    margin-right: 4px;
}
@keyframes pulse-ring {
    0% { transform: scale(0.9); opacity: 0.8; }
    50% { transform: scale(1.3); opacity: 1; box-shadow: 0 0 14px #00E676; }
    100% { transform: scale(0.9); opacity: 0.8; }
}

/* ── Badges ── */
.badge-online, .badge-baseline, .badge-blend, .badge-drift {
    padding: 3px 12px; border-radius: 20px; font-size: 0.72em; font-weight: 700;
    letter-spacing: 0.4px; white-space: nowrap;
}
.badge-online {
    background: linear-gradient(135deg, rgba(0, 230, 118, 0.2), rgba(0, 204, 68, 0.35));
    border: 1px solid #00E676; color: #00E676;
    box-shadow: 0 0 12px rgba(0, 230, 118, 0.25);
}
.badge-baseline {
    background: linear-gradient(135deg, rgba(255, 171, 0, 0.2), rgba(224, 144, 0, 0.35));
    border: 1px solid #FFAB00; color: #FFAB00;
    box-shadow: 0 0 12px rgba(255, 171, 0, 0.25);
}
.badge-blend {
    background: linear-gradient(135deg, rgba(155, 93, 229, 0.2), rgba(123, 61, 181, 0.35));
    border: 1px solid #9B5DE5; color: #c77dff;
    box-shadow: 0 0 12px rgba(155, 93, 229, 0.3);
}
.badge-drift {
    background: linear-gradient(135deg, rgba(255, 23, 68, 0.25), rgba(204, 34, 34, 0.4));
    border: 1px solid #FF1744; color: #FF5252;
    animation: drift-glow 1.2s infinite;
}
@keyframes drift-glow {
    0%, 100% { opacity: 1; box-shadow: 0 0 14px rgba(255, 23, 68, 0.5); }
    50% { opacity: 0.65; box-shadow: 0 0 6px rgba(255, 23, 68, 0.2); }
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #090e18 0%, #050810 100%) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.06);
}
[data-testid="stSidebar"] [data-testid="stRadio"] > div { gap: 4px; }
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid transparent;
    border-radius: 8px;
    padding: 6px 12px !important;
    transition: all 0.2s ease;
    cursor: pointer;
    color: #94a3b8 !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    width: 100%;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: rgba(0, 229, 255, 0.06) !important;
    border-color: rgba(0, 229, 255, 0.2) !important;
    color: #ffffff !important;
    transform: translateX(3px);
}
/* FIX: active item — previous data-checked selector doesn't exist in Streamlit */
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
    background: linear-gradient(90deg, rgba(0, 229, 255, 0.12), rgba(0, 229, 255, 0.03)) !important;
    border-left: 3px solid #00E5FF !important;
    color: #ffffff !important;
    font-weight: 700 !important;
}
/* hide the radio circle */
[data-testid="stSidebar"] [data-testid="stRadio"] label > div:first-child { display: none; }

/* ── Metric Cards (FIX: works for both old and new test-ids) ── */
[data-testid="metric-container"], [data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(16, 24, 40, 0.75) 0%, rgba(10, 16, 28, 0.9) 100%) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-top: 2px solid rgba(0, 229, 255, 0.4) !important;
    border-radius: 12px !important;
    padding: 16px 18px !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35) !important;
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
}
[data-testid="metric-container"]:hover, [data-testid="stMetric"]:hover {
    transform: translateY(-3px) !important;
    border-color: rgba(0, 229, 255, 0.6) !important;
    box-shadow: 0 8px 24px rgba(0, 229, 255, 0.15) !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.75rem !important; font-weight: 600 !important;
    letter-spacing: 0.5px !important; text-transform: uppercase !important;
    color: #8da4be !important;
}
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.55rem !important; font-weight: 700 !important; color: #ffffff !important;
}

/* ── Tabs ── */
[data-baseweb="tab-list"] {
    background: rgba(14, 20, 34, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 12px !important;
    padding: 4px !important; gap: 6px !important;
}
[data-baseweb="tab"] {
    border-radius: 8px !important; padding: 8px 18px !important;
    color: #94a3b8 !important; font-weight: 600 !important; font-size: 0.85rem !important;
    transition: all 0.2s ease !important; border: none !important;
}
[data-baseweb="tab"]:hover { color: #ffffff !important; background: rgba(255, 255, 255, 0.04) !important; }
[aria-selected="true"][data-baseweb="tab"] {
    background: linear-gradient(135deg, rgba(0, 229, 255, 0.18) 0%, rgba(0, 180, 216, 0.25) 100%) !important;
    color: #00E5FF !important;
    border: 1px solid rgba(0, 229, 255, 0.35) !important;
    box-shadow: 0 0 14px rgba(0, 229, 255, 0.15) !important;
}
/* hide the default underline highlight so it doesn't clash with the pill */
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] { display: none !important; }

/* ── Buttons (FIX: also match newer data-testid-based kinds) ── */
button[kind="primary"], button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #00B4D8 0%, #0077B6 100%) !important;
    border: none !important; border-radius: 8px !important;
    color: #ffffff !important; font-weight: 600 !important;
    box-shadow: 0 4px 14px rgba(0, 180, 216, 0.35) !important;
    transition: all 0.2s ease !important;
}
button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(0, 180, 216, 0.55) !important;
}
button[kind="secondary"], button[data-testid="stBaseButton-secondary"] {
    background: rgba(255, 255, 255, 0.04) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 8px !important; color: #cbd5e1 !important;
    transition: all 0.2s ease !important;
}
button[kind="secondary"]:hover, button[data-testid="stBaseButton-secondary"]:hover {
    background: rgba(255, 255, 255, 0.08) !important;
    border-color: rgba(0, 229, 255, 0.4) !important; color: #ffffff !important;
}

/* ── Expanders ── */
[data-testid="stExpander"] {
    background: rgba(14, 20, 33, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.07) !important;
    border-radius: 10px !important; margin-bottom: 8px !important;
}
[data-testid="stExpander"]:hover { border-color: rgba(0, 229, 255, 0.25) !important; }

/* Smooth out fragment refreshes (no grey-out flicker) */
[data-stale="true"] { opacity: 1 !important; }

/* Mobile */
@media (max-width: 768px) {
    .block-container { max-width: 100%; padding-left: 0.8rem; padding-right: 0.8rem; }
    .ews-header { padding: 16px; }
    .ews-header h1 { font-size: 1.15em; }
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

    try:
        sim = StreamSimulator()
        sim.load_data(n_patients=30)
        p.load_labels(sim.get_patient_labels())

        # Fast in-memory warm-up of OnlineModel on real PhysioNet + high-risk trajectories
        # so Rolling AUROC, Tree Structure, and Online Score graphs show real values immediately.
        warm_samples: list[tuple[dict, int]] = []
        for idx, event in enumerate(sim.stream(n_patients=15, realtime=False)):
            f_raw = p.aggregator.update(event)
            if f_raw and idx % 3 == 0:
                feats = p.imputer.impute_and_update(f_raw)
                warm_samples.append((feats, int(event.get("sepsis_label") or 0)))
                if len(warm_samples) >= 80:
                    break
        for idx, event in enumerate(sim.stream_sepsis_patients(max_patients=3)):
            f_raw = p.aggregator.update(event)
            if f_raw and idx % 2 == 0:
                feats = p.imputer.impute_and_update(f_raw)
                warm_samples.append((feats, int(event.get("sepsis_label") or 0)))
                if len(warm_samples) >= 150:
                    break
        if warm_samples:
            p.online_model.update_from_batch(warm_samples)

        n = 0
        for event in sim.stream(n_patients=30, realtime=False):
            p.process_event(event)
            n += 1
            if n >= 60:
                break

        # Pre-populate confirmed real PhysioNet sepsis stays + high-risk synthetic stays (fast startup)
        for event in sim.stream_sepsis_patients(max_patients=3):
            p.process_event(event)

        for event in sim.generate_random_events(n_patients=12, high_risk_count=5):
            p.process_event(event)
    except Exception as exc:
        print(f"Pre-population warning: {exc}")

    return p


@st.cache_resource(show_spinner="Loading stream simulator …")
def get_simulator():
    from src.ingestion.stream_simulator import StreamSimulator
    return StreamSimulator()


# ── Background stream controller ───────────────────────────────────────────
# FIX: the old worker thread read st.session_state from a non-Streamlit thread,
# which has no script context — so the stream silently died / never reflected
# Start/Stop. State now lives in a process-wide object using threading.Event.
class StreamController:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and not self._stop.is_set()

    def start(self, pipeline, simulator, n_stays: int) -> bool:
        with self._lock:
            if self.running:
                return False
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run, args=(pipeline, simulator, n_stays), daemon=True
            )
            self._thread.start()
            return True

    def stop(self) -> None:
        self._stop.set()

    def _run(self, pipeline, simulator, n_stays: int) -> None:
        try:
            pipeline.load_labels(simulator.get_patient_labels())
        except Exception:
            pass

        dataset_stream = simulator.stream(n_patients=n_stays, realtime=False)

        while not self._stop.is_set():
            try:
                for event in simulator.generate_random_events(n_patients=20, high_risk_count=8):
                    if self._stop.is_set():
                        break
                    pipeline.process_event(event)

                for _ in range(30):
                    if self._stop.is_set():
                        break
                    try:
                        pipeline.process_event(next(dataset_stream))
                    except StopIteration:
                        dataset_stream = simulator.stream(n_patients=n_stays, realtime=False)
            except Exception as exc:  # keep the stream alive on a bad event
                print(f"Stream warning: {exc}")

            # interruptible 4 s wait (exits immediately on Stop)
            self._stop.wait(4.0)


@st.cache_resource
def get_stream_controller() -> StreamController:
    return StreamController()


# ── Session state defaults ─────────────────────────────────────────────────
st.session_state.setdefault("n_stays_override", 80)
st.session_state.setdefault("user_stopped", False)

pipeline = get_pipeline()
simulator = get_simulator()
stream = get_stream_controller()

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:18px 8px 12px; background:linear-gradient(180deg, rgba(0, 229, 255, 0.08) 0%, transparent 100%); border-radius:12px; border:1px solid rgba(0, 229, 255, 0.15); margin-bottom:12px;'>
      <div style='font-size:2.4em; filter: drop-shadow(0 0 10px rgba(0, 229, 255, 0.4));'>🏥</div>
      <div style='font-size:1.25em; font-weight:800; color:#ffffff; letter-spacing:-0.3px; margin-top:4px;'>SepsisGuard</div>
      <div style='font-size:0.75em; color:#00E5FF; font-weight:600; letter-spacing:0.5px; text-transform:uppercase;'>Clinical EWS Intelligence</div>
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
            "⚡ Online Model Metrics",
            "🃏 Model Card",
            "💬 AI Chatbot",
            "🧠 AI Assist",
            "⚙️ Admin",
        ],
        label_visibility="collapsed",
    )
    st.divider()

    st.markdown("**Stream Control**")
    n_stays = st.number_input("Simulate N ICU stays", min_value=10, max_value=5000, value=80, step=10)
    st.session_state["n_stays_override"] = n_stays

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("▶ Start", width="stretch", type="primary"):
            st.session_state["user_stopped"] = False
            if stream.running:
                stream.stop()  # restart with the new N
            simulator.load_data(n_patients=n_stays)
            stream.start(pipeline, simulator, n_stays)
            st.toast("Stream started", icon="🚀")
    with col_b:
        if st.button("⏹ Stop", width="stretch"):
            st.session_state["user_stopped"] = True
            stream.stop()
            st.toast("Stream stopped", icon="⏹")

    # Auto-start once on first load, but never fight an explicit Stop
    if not stream.running and not st.session_state["user_stopped"]:
        simulator.load_data(n_patients=st.session_state["n_stays_override"])
        stream.start(pipeline, simulator, st.session_state["n_stays_override"])

    st.caption(f"Stream: {'🟢 Running' if stream.running else '🔴 Stopped'}")

    st.divider()
    refresh = st.slider("Auto-refresh (s)", 3, 30, 4)
    st.caption("⚠️ Research demo only — not for clinical use")


# ── Live view (FIX: fragment auto-refresh instead of blocking sleep + rerun,
#    so sidebar clicks respond instantly and the sidebar never freezes) ─────
def render_live_view(page: str) -> None:
    status = pipeline.status()
    active = status["active_model"]
    badge_class = (
        "badge-online" if active == "online"
        else "badge-blend" if active == "blend"
        else "badge-baseline"
    )
    drift_badge = (
        f'<span class="badge-drift">⚠️ DRIFT ({status["total_drift_events"]})</span>'
        if status["total_drift_events"] > 0 else ""
    )
    now_str = datetime.now().strftime("%H:%M:%S")

    online_auroc = status.get("online_auroc", float("nan"))
    online_auroc_str = f"{online_auroc:.3f}" if online_auroc == online_auroc else "—"

    st.markdown(f"""
<div class="ews-header">
  <h1>
    <span class="pulse-dot"></span>
    SepsisGuard — Real-Time Deterioration EWS
    <span class="{badge_class}">{active.upper()} MODEL</span>
    {drift_badge}
  </h1>
  <div class="stats">
    <span class="stat-chip">🕐 {now_str}</span>
    <span class="stat-chip alert">🚨 Alerts: <strong>{status['n_alerts']:,}</strong></span>
    <span class="stat-chip">👥 Census: <strong>{status['n_patients_tracked']:,}</strong></span>
    <span class="stat-chip accent">⚡ Online AUROC: <strong>{online_auroc_str}</strong></span>
    <span class="stat-chip">Threshold: <strong>{status.get('alert_threshold', 0.65):.0%}</strong></span>
  </div>
</div>
""", unsafe_allow_html=True)

    if page == "🚨 Patient Alerts":
        from dashboard.components.alert_panel import render_alert_panel
        render_alert_panel(pipeline.high_risk_patients(), pipeline.audit, pipeline=pipeline)

    elif page == "🏥 Ward Census":
        from dashboard.components.ward_census import render_ward_census
        render_ward_census(pipeline)

    elif page == "📈 Patient Timeline":
        from dashboard.components.patient_timeline import render_patient_timeline
        render_patient_timeline(pipeline.recent_results(400))

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

    elif page == "⚡ Online Model Metrics":
        from dashboard.components.online_metrics import render_online_metrics
        render_online_metrics(pipeline)

    elif page == "🃏 Model Card":
        from dashboard.components.model_card import render_model_card
        render_model_card(pipeline)

    elif page == "💬 AI Chatbot":
        from dashboard.components.ai_chatbot import render_ai_chatbot
        render_ai_chatbot(pipeline)

    elif page == "🧠 AI Assist":
        from dashboard.components.ai_assist import render_ai_assist
        render_ai_assist(pipeline)

    elif page == "⚙️ Admin":
        render_admin(status)


def render_admin(status: dict) -> None:
    st.title("⚙️ System Administration")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model Controls")

        strategy_options = ["auto_rollback", "weighted_blend", "online_only", "baseline_only"]
        current = pipeline.selector.strategy
        new_strategy = st.selectbox(
            "Ensemble strategy",
            strategy_options,
            index=strategy_options.index(current) if current in strategy_options else 0,
            help="auto_rollback: use online, rollback when AUROC drops | weighted_blend: 0.6×online + 0.4×baseline | online_only | baseline_only",
        )
        if new_strategy != pipeline.selector.strategy:
            pipeline.selector.strategy = new_strategy
            st.info(f"Ensemble strategy set to: {new_strategy}")

        if pipeline.selector.strategy == "weighted_blend":
            blend_w = st.slider(
                "Online model blend weight", 0.0, 1.0,
                float(pipeline.selector.blend_online_weight), step=0.05,
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
                    "timestamp": datetime.now(timezone.utc).isoformat(),
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
            "Risk threshold", 0.30, 0.90,
            float(pipeline.selector.alert_threshold), step=0.05,
        )
        if abs(new_thr - pipeline.selector.alert_threshold) > 1e-9:
            pipeline.set_alert_threshold(new_thr)
            st.info(f"Alert threshold set to {new_thr:.0%}")

    with col2:
        st.subheader("Audit Export")
        if st.button("📥 Export Predictions CSV"):
            Path("logs").mkdir(exist_ok=True)
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


# Chat / AI / Admin / Online-metrics pages keep user input state, so no auto-refresh there.
NO_REFRESH = {"💬 AI Chatbot", "🧠 AI Assist", "⚡ Online Model Metrics", "⚙️ Admin"}
run_every = None if page in NO_REFRESH else refresh

st.fragment(run_every=run_every)(render_live_view)(page)