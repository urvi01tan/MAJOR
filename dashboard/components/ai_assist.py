"""Clinical AI Assist Workspace: SBAR Handoff, Deterioration Narratives, SSC Protocol, and What-If Simulator."""

from __future__ import annotations

import random
import streamlit as st

from src.ai.services import AIAssistService, ward_context
from src.explainability.explainer import CLINICAL_ABNORMALITY_JUSTIFICATIONS


def _service() -> AIAssistService:
    if "ai_assist" not in st.session_state:
        st.session_state["ai_assist"] = AIAssistService()
    return st.session_state["ai_assist"]


def render_ai_assist(pipeline) -> None:
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(20, 16, 36, 0.7) 0%, rgba(12, 10, 24, 0.9) 100%);
                padding: 14px 20px; border-radius: 10px; border-left: 4px solid #9B5DE5;
                margin-bottom: 16px; border: 1px solid rgba(255, 255, 255, 0.06);">
        <h2 style="margin: 0; color: #ffffff; font-size: 1.3rem;">🧠 Clinical Decision Support Suite</h2>
    </div>
    """, unsafe_allow_html=True)

    assist = _service()
    status = assist.status()
    patients = pipeline.census() if hasattr(pipeline, "census") else []
    ids = [str(p.get("patient_id")) for p in patients if p.get("patient_id")]

    tab_shift, tab_narr, tab_care, tab_whatif, tab_organs, tab_just = st.tabs([
        "📋 SBAR Shift Handoff",
        "🚨 Deterioration Narrative",
        "🩺 Sepsis Protocol Review",
        "🔮 What-If Simulator",
        "🫀 Organ Breakdown",
        "📖 Case Examples",
    ])

    # ── TAB 1: SBAR Shift Handoff ──────────────────────────────────────────
    with tab_shift:
        st.subheader("📋 SBAR Shift Handoff Note")
        st.caption("Generate a structured handoff note synthesized from the current census, risk scores, and model telemetry.")

        if st.button("✨ Generate SBAR Shift Summary", type="primary", key="btn_gen_sbar"):
            with st.spinner("Synthesizing ward telemetry and generating SBAR report…"):
                result = assist.shift.summarize(ward_context(pipeline))
            handoff_text = result.get("text") or "No text returned."
            st.markdown(handoff_text)
            st.download_button(
                "📥 Export SBAR Note",
                data=handoff_text,
                file_name="icu_sbar_shift_handoff.md",
                mime="text/markdown",
            )

    # ── TAB 2: Deterioration Narrative ─────────────────────────────────────
    with tab_narr:
        st.subheader("🚨 Patient Alert Narrative")
        st.caption("Converts high-dimensional sliding-window features and composite scores into a plain-English clinical summary.")

        if not ids:
            st.info("No patients currently in census. Please ensure the stream is running.")
        else:
            col_p, col_preview = st.columns([1.5, 2.5])
            with col_p:
                pid = st.selectbox("Select Patient:", ids, key="ai_narr_pid")
                patient = next((p for p in patients if str(p.get("patient_id")) == pid), {})
                risk = float(patient.get("risk_score") or 0.0)
                st.metric("Sepsis Deterioration Risk", f"{risk:.1%}")
                st.metric("NEWS2 / SOFA", f"{patient.get('news2', '—')} / {patient.get('sofa', '—')}")

            with col_preview:
                st.markdown("**Current Patient Telemetry Snapshot:**")
                st.json({
                    "patient_id": patient.get("patient_id"),
                    "risk_score": patient.get("risk_score"),
                    "severity": patient.get("severity"),
                    "news2": patient.get("news2"),
                    "qsofa": patient.get("qsofa"),
                    "sirs": patient.get("sirs"),
                    "sofa": patient.get("sofa"),
                    "model_used": patient.get("model_used"),
                })

            if st.button("✨ Generate Clinical Narrative", type="primary", key="btn_gen_narrative"):
                with st.spinner(f"Generating deterioration narrative for patient {pid}…"):
                    result = assist.narrative.narrate(patient)
                st.markdown(result.get("text") or "No narrative returned.")

    # ── TAB 3: Sepsis Protocol Review ──────────────────────────────────────
    with tab_care:
        st.subheader("🩺 Surviving Sepsis Campaign (SSC) 1-Hour Bundle Checklist")
        st.caption("Cross-references patient vitals, lactate, and mean arterial pressure against international guidelines.")

        if not ids:
            st.info("No patients available. Start the data stream first.")
        else:
            pid_care = st.selectbox("Evaluate Patient:", ids, key="ai_care_pid")
            patient_care = next((p for p in patients if str(p.get("patient_id")) == pid_care), {})

            ssc_eval = assist.client.clinical_engine.evaluate_ssc_bundle(patient_care)

            stat_c1, stat_c2, stat_c3 = st.columns(3)
            stat_c1.metric("Risk Score", f"{float(patient_care.get('risk_score', 0)):.1%}")
            stat_c2.metric("Hypotension Present", "YES (MAP < 65)" if ssc_eval["hypotensive"] else "NO")
            stat_c3.metric("Lactate Alert", "HIGH (≥ 2.0)" if ssc_eval["high_lactate"] else "Normal / Not Drawn")

            st.markdown("#### 1-Hour Bundle Action Items:")
            for item in ssc_eval["items"]:
                icon = "🔴" if item["status"] in ("URGENT", "IMMEDIATE", "REQUIRED") else "🟡"
                with st.expander(f"{icon} {item['step']} — {item['status']}", expanded=item["triggered"]):
                    st.write(item["detail"])

            if st.button("📋 Generate Full Review Suggestions", type="primary", key="btn_gen_care"):
                with st.spinner("Compiling protocol review checks…"):
                    care_result = assist.care.suggest(patient_care)
                st.markdown(care_result.get("text") or "No recommendations generated.")

    # ── TAB 4: Physiological What-If Simulator ─────────────────────────────
    with tab_whatif:
        st.subheader("🔮 Physiological Counterfactual 'What-If' Simulator")
        st.caption("Simulate hemodynamic interventions (e.g. fluid bolus, vasopressor response, oxygen therapy) to estimate predicted impact on sepsis risk.")

        if not ids:
            st.info("No patients available. Start the data stream first.")
        else:
            sim_pid = st.selectbox("Select Patient for Simulation:", ids, key="ai_sim_pid")
            sim_patient = next((p for p in patients if str(p.get("patient_id")) == sim_pid), {})

            base_risk = float(sim_patient.get("risk_score") or 0.5)

            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                hr_sim = st.slider("Target Heart Rate (bpm)", min_value=40, max_value=160, value=85, step=5)
                map_sim = st.slider("Target MAP (mmHg)", min_value=45, max_value=120, value=75, step=5)
            with col_s2:
                resp_sim = st.slider("Target Respiration (breaths/min)", min_value=10, max_value=45, value=16, step=1)
                temp_sim = st.slider("Target Temp (°C)", min_value=34.0, max_value=41.0, value=37.0, step=0.1)
            with col_s3:
                lactate_sim = st.slider("Target Serum Lactate (mmol/L)", min_value=0.5, max_value=10.0, value=1.2, step=0.2)

            if st.button("⚡ Run What-If Simulation", type="primary", key="btn_run_sim"):
                overrides = {
                    "HR": hr_sim,
                    "MAP": map_sim,
                    "Resp": resp_sim,
                    "Temp": temp_sim,
                    "Lactate": lactate_sim,
                }
                sim_res = assist.simulator.simulate(sim_patient, overrides)

                delta = sim_res["delta"]
                delta_color = "normal" if abs(delta) < 0.02 else ("inverse" if delta > 0 else "off")

                res_c1, res_c2, res_c3 = st.columns(3)
                res_c1.metric("Baseline ML Risk", f"{sim_res['baseline_risk']:.1%}")
                res_c2.metric("Simulated Risk", f"{sim_res['simulated_risk']:.1%}", delta=f"{delta:+.1%}")
                res_c3.metric("Projected Relative Change", f"{sim_res['percent_change']:+.1f}%")

                st.success(f"**Interpretation:** {sim_res['interpretation']}")

                if sim_res.get("details"):
                    st.markdown("**Physiological Factor Breakdown:**")
                    for d in sim_res["details"]:
                        st.markdown(f"- {d}")

    # ── TAB 5: Organ System Breakdown ──────────────────────────────────────
    with tab_organs:
        st.subheader("🫀 Multi-Organ System Perfusion & Dysfunction Breakdown")
        st.caption("Assesses failure risk across organ systems based on partial SOFA scoring and physiological indicators.")

        if not ids:
            st.info("No patients available.")
        else:
            organ_pid = st.selectbox("Select Patient to Inspect:", ids, key="ai_organ_pid")
            organ_patient = next((p for p in patients if str(p.get("patient_id")) == organ_pid), {})

            organ_eval = assist.organs.evaluate(organ_patient)

            st.metric("Total Acute Organ Dysfunction Score", f"{organ_eval['total_score']} / 20")
            st.info(f"**Clinical Summary:** {organ_eval['summary']}")

            cols = st.columns(len(organ_eval["systems"]))
            for idx, sys_item in enumerate(organ_eval["systems"]):
                with cols[idx]:
                    st.markdown(f"**{sys_item['system']}**")
                    score_val = sys_item["score"]
                    color = "#FF4B4B" if score_val >= 3 else "#FFA500" if score_val >= 2 else "#00CC44"
                    st.markdown(
                        f"<div style='font-size:1.4rem;font-weight:700;color:{color}'>{score_val} / {sys_item['max']}</div>"
                        f"<div style='font-size:0.8rem;color:#a0b0c0'>{sys_item['status']}</div>",
                        unsafe_allow_html=True,
                    )

    # ── TAB 6: Abnormality Justifications & Clinical Case Examples ─────────
    with tab_just:
        st.subheader("📖 Clinical Abnormality Justifications & Case Examples")
        st.caption(
            "Evidence-based pathophysiological justifications and realistic patient case examples "
            "explaining why each vital sign or lab abnormality triggers an elevated sepsis deterioration risk."
        )

        abnormal_keys = list(CLINICAL_ABNORMALITY_JUSTIFICATIONS.keys())
        labels = [CLINICAL_ABNORMALITY_JUSTIFICATIONS[k]["label"] for k in abnormal_keys]

        col_sel, col_dice = st.columns([3, 1])
        with col_sel:
            selected_label = st.selectbox("Select Vital / Lab Parameter to Justify:", labels, key="just_select_label")
            sel_idx = labels.index(selected_label)
            selected_key = abnormal_keys[sel_idx]
            info = CLINICAL_ABNORMALITY_JUSTIFICATIONS[selected_key]
        with col_dice:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if st.button("🎲 Random Example", use_container_width=True):
                rand_key = random.choice(abnormal_keys)
                st.session_state["just_rand_pick"] = rand_key

        if "just_rand_pick" in st.session_state:
            rand_key = st.session_state["just_rand_pick"]
            info = CLINICAL_ABNORMALITY_JUSTIFICATIONS[rand_key]
            st.info(f"🎲 Randomly selected parameter: **{info['label']}**")

        c_ref1, c_ref2 = st.columns(2)
        c_ref1.metric("Normal ICU Reference Range", info["normal_range"])
        c_ref2.metric("Sepsis Alert / Abnormality Cutoff", info["abnormal_criteria"])

        st.markdown("#### 🔬 Pathophysiological Justification (Why It Drives Deterioration):")
        st.info(info["pathophysiology"])

        st.markdown("#### 🩺 Clinical Case Examples Justifying This Abnormality:")
        for idx, eg in enumerate(info["examples"], 1):
            with st.expander(f"Case Example {idx} — {info['label']}", expanded=True):
                st.markdown(f"**{eg}**")