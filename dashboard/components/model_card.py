"""Model Card — displays baseline model metadata, performance, and feature importances."""

from __future__ import annotations

import json
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st
import yaml


def render_model_card(pipeline) -> None:
    st.title("🃏 Model Card")
    st.caption(
        "Transparency report for the deployed models. "
        "All metrics are from the held-out test set used during offline training."
    )

    # ── Load config + meta ────────────────────────────────────────────────
    try:
        with open("config/settings.yaml") as f:
            cfg = yaml.safe_load(f)
    except Exception:
        cfg = {}

    card_cfg = cfg.get("model_card", {})
    baseline = pipeline.baseline_model
    meta = baseline._meta if hasattr(baseline, "_meta") else {}

    # ── Model identity ────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div style='background:linear-gradient(135deg,#1a1f2e,#0f1923);
                    border-left:4px solid #00B4D8;padding:20px;border-radius:10px;
                    margin-bottom:20px;'>
          <h2 style='margin:0;color:#fff;'>{card_cfg.get("name","EWS Baseline Model")}</h2>
          <p style='margin:6px 0 0;color:#aaa;font-size:0.9em;'>
            v{meta.get("version","—")} &nbsp;·&nbsp;
            Trained: {str(meta.get("trained_at","—"))[:19]} &nbsp;·&nbsp;
            Dataset: {card_cfg.get("dataset","PhysioNet CinC 2019")}
          </p>
          <p style='margin:8px 0 0;color:#ccc;font-size:0.9em;'>
            {card_cfg.get("description","")}
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Intended use / limitations ─────────────────────────────────────────
    use_col, lim_col = st.columns(2)
    with use_col:
        st.markdown("#### ✅ Intended Use")
        st.info(card_cfg.get("intended_use", "Research and education only."))
        scores = card_cfg.get("clinical_scores", [])
        if scores:
            st.markdown("**Clinical scores alongside ML:**")
            for s in scores:
                st.markdown(f"- {s}")

    with lim_col:
        st.markdown("#### ⚠️ Known Limitations")
        limits = card_cfg.get("limitations", [])
        for lim in limits:
            st.warning(lim)

    st.divider()

    # ── Performance metrics ───────────────────────────────────────────────
    st.subheader("📈 Baseline Model Performance (Test Set)")
    m1, m2, m3, m4, m5 = st.columns(5)

    def _fmt(val, pct=False):
        if val is None or (isinstance(val, float) and val != val):
            return "—"
        return f"{val:.1%}" if pct else f"{val:.4f}"

    m1.metric("AUROC", _fmt(meta.get("test_auroc")))
    m2.metric("AUPRC", _fmt(meta.get("test_auprc")))
    m3.metric("Sensitivity", _fmt(meta.get("test_sensitivity"), pct=True))
    m4.metric("Specificity", _fmt(meta.get("test_specificity"), pct=True))
    m5.metric("Precision", _fmt(meta.get("test_precision"), pct=True))

    tr1, tr2, tr3 = st.columns(3)
    tr1.metric("Train samples", f"{meta.get('n_train', 0):,}")
    tr2.metric("Test samples", f"{meta.get('n_test', 0):,}")
    tr3.metric("Sepsis prevalence (train)", _fmt(meta.get("sepsis_prevalence_train"), pct=True))

    # ── Online model live status ──────────────────────────────────────────
    st.divider()
    st.subheader("🔄 Online Model Live Status")
    status = pipeline.status()
    o1, o2, o3, o4 = st.columns(4)
    o1.metric("Rolling AUROC", f"{status.get('online_auroc', 0):.4f}")
    o2.metric("Updates", f"{status.get('n_online_updates', 0):,}")
    o3.metric("Predictions", f"{status.get('n_predictions', 0):,}")
    o4.metric("Active Model", status.get("active_model", "—").title())

    # Ensemble strategy
    st.markdown(
        f"**Ensemble strategy:** `{pipeline.selector.strategy}`  &nbsp;·&nbsp;  "
        f"**Rollbacks:** `{status.get('rollback_count', 0)}`  &nbsp;·&nbsp;  "
        f"**Drift events:** `{status.get('total_drift_events', 0)}`"
    )

    # ── Feature importance ────────────────────────────────────────────────
    st.divider()
    st.subheader("🔬 Feature Importances (Baseline Model)")

    # Try loading from JSON file first, then compute from model
    fi_path = Path(cfg.get("paths", {}).get("feature_importance", "models/feature_importance.json"))
    importances: dict[str, float] = {}

    if fi_path.exists():
        try:
            with open(fi_path) as f:
                importances = json.load(f)
        except Exception:
            pass

    if not importances:
        importances = baseline.get_feature_importance()

    if importances:
        # Top 25 absolute
        sorted_fi = sorted(importances.items(), key=lambda x: abs(x[1]), reverse=True)[:25]
        feat_names = [f[0] for f in sorted_fi]
        feat_vals = [f[1] for f in sorted_fi]
        colors = ["#e63946" if v > 0 else "#457b9d" for v in feat_vals]

        fig = go.Figure(go.Bar(
            x=feat_vals,
            y=feat_names,
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.4f}" for v in feat_vals],
            textposition="outside",
        ))
        fig.update_layout(
            height=max(400, 18 * len(feat_names)),
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font=dict(color="#fafafa", size=11),
            xaxis=dict(title="Coefficient / Importance", zeroline=True, zerolinecolor="#555"),
            yaxis=dict(autorange="reversed"),
            margin=dict(l=200, r=80, t=20, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "🔴 Red = increases sepsis risk prediction. 🔵 Blue = decreases risk prediction. "
            "Logistic regression coefficients; XGBoost shows gain-based feature importance."
        )
    else:
        st.info(
            "Feature importances not available. Train the model first: "
            "`py -3 scripts/train_baseline.py --n-patients 500`"
        )

    # ── Regulatory disclaimer ─────────────────────────────────────────────
    st.divider()
    st.error(
        "⚖️ **Regulatory Note:** This is a research prototype. "
        "It has NOT been validated for clinical use, is NOT a registered medical device, "
        "and MUST NOT be used to guide real patient care decisions. "
        "Any deployment in a clinical setting would require IRB review, "
        "prospective validation, HIPAA compliance, and regulatory clearance."
    )
