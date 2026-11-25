"""Ward-aware clinical chatbot page with multi-provider LLM support and local clinical intelligence."""

from __future__ import annotations

import streamlit as st

from src.ai.services import AIAssistService, ward_context


def _service() -> AIAssistService:
    if "ai_assist" not in st.session_state:
        st.session_state["ai_assist"] = AIAssistService()
    return st.session_state["ai_assist"]


def render_ai_chatbot(pipeline) -> None:
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(16, 28, 44, 0.7) 0%, rgba(10, 16, 28, 0.9) 100%);
                padding: 14px 20px; border-radius: 10px; border-left: 4px solid #00E5FF;
                margin-bottom: 16px; border: 1px solid rgba(255, 255, 255, 0.06);">
        <h2 style="margin: 0; color: #ffffff; font-size: 1.3rem;">💬 Clinical AI Assistant</h2>
    </div>
    """, unsafe_allow_html=True)

    assist = _service()
    status = assist.status()

    # Quick prompt actions
    st.markdown("**Quick Clinical Prompts:**")
    q_col1, q_col2, q_col3, q_col4, q_col5 = st.columns(5)

    prompt_to_submit = None

    with q_col1:
        if st.button("🚨 Priority Alerts", width="stretch"):
            prompt_to_submit = "Summarize the active high-risk patients and their deterioration drivers."
    with q_col2:
        if st.button("📋 SBAR Handoff", width="stretch"):
            prompt_to_submit = "Generate an ICU SBAR shift handoff summary for current ward rounds."
    with q_col3:
        if st.button("🩺 Explain Scores", width="stretch"):
            prompt_to_submit = "Explain the clinical scoring criteria for NEWS2, qSOFA, and SOFA in this system."
    with q_col4:
        if st.button("🤖 Dual Model", width="stretch"):
            prompt_to_submit = "Explain how the online Hoeffding Adaptive Tree and baseline model interact with auto-rollback."
    with q_col5:
        if st.button("💡 Abnormalities", width="stretch"):
            prompt_to_submit = "Provide a clinical justification and random patient case example for abnormal vitals (such as tachycardia, elevated lactate, or hypotension)."

    # Patient-specific focus selector
    census_patients = pipeline.census() if hasattr(pipeline, "census") else []
    patient_ids = [str(p.get("patient_id")) for p in census_patients if p.get("patient_id")]

    if patient_ids:
        col_focus, col_btn = st.columns([3, 1])
        with col_focus:
            selected_pid = st.selectbox(
                "Focus inquiry on specific patient:",
                ["None (General Ward)"] + patient_ids,
                index=0,
                key="chat_patient_focus",
            )
        with col_btn:
            if selected_pid != "None (General Ward)":
                if st.button("🔍 Explain Selected Patient", use_container_width=True):
                    prompt_to_submit = f"Provide a detailed clinical narrative and deterioration breakdown for patient {selected_pid}."

    # Feature-value interactive analyzer
    with st.expander("🧪 Enter Patient Feature Values (Analyze & Describe)", expanded=False):
        fc1, fc2, fc3, fc4, fc5 = st.columns(5)
        hr_v = fc1.number_input("HR (bpm)", 30.0, 220.0, 128.0, step=2.0, key="cb_hr")
        sbp_v = fc2.number_input("SBP (mmHg)", 40.0, 220.0, 84.0, step=2.0, key="cb_sbp")
        map_v = fc3.number_input("MAP (mmHg)", 30.0, 150.0, 58.0, step=2.0, key="cb_map")
        temp_v = fc4.number_input("Temp (°C)", 33.0, 42.0, 39.1, step=0.1, key="cb_temp")
        resp_v = fc5.number_input("Resp (/min)", 6.0, 50.0, 28.0, step=1.0, key="cb_resp")

        fc6, fc7, fc8, fc9, fc10 = st.columns(5)
        o2_v = fc6.number_input("SpO₂ (%)", 60.0, 100.0, 89.0, step=1.0, key="cb_o2")
        lac_v = fc7.number_input("Lactate (mmol/L)", 0.3, 20.0, 4.2, step=0.2, key="cb_lac")
        wbc_v = fc8.number_input("WBC (×10³/µL)", 0.5, 50.0, 17.5, step=0.5, key="cb_wbc")
        cr_v = fc9.number_input("Creatinine (mg/dL)", 0.2, 12.0, 2.4, step=0.1, key="cb_cr")
        plt_v = fc10.number_input("Platelets (×10³/µL)", 10.0, 600.0, 95.0, step=5.0, key="cb_plt")

        if st.button("🩺 Analyze Features & Describe", type="primary", use_container_width=True):
            prompt_to_submit = (
                f"Analyze these patient feature values and describe their clinical meaning: "
                f"HR={hr_v:.0f}, SBP={sbp_v:.0f}, MAP={map_v:.0f}, Temp={temp_v:.1f}, "
                f"Resp={resp_v:.0f}, O2Sat={o2_v:.0f}, Lactate={lac_v:.1f}, "
                f"WBC={wbc_v:.1f}, Creatinine={cr_v:.1f}, Platelets={plt_v:.0f}"
            )

    st.divider()

    # Chat history display
    if "ai_chat_history" not in st.session_state:
        st.session_state["ai_chat_history"] = []

    if not st.session_state["ai_chat_history"]:
        with st.chat_message("assistant"):
            st.markdown("👋 How can I assist with ward telemetry, patient alerts, or protocols today?")

    for turn in st.session_state["ai_chat_history"]:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

    # Chat input
    user_input = st.chat_input("Ask about ward status, patient ID, or enter vitals (e.g. HR=130, SBP=82, Temp=39.2, Lactate=4.5)…")
    final_prompt = prompt_to_submit or user_input

    if final_prompt:
        st.session_state["ai_chat_history"].append({"role": "user", "content": final_prompt})
        with st.chat_message("user"):
            st.markdown(final_prompt)

        context = ward_context(pipeline)
        with st.spinner("Analyzing ward hemodynamics & clinical context…"):
            result = assist.chat.reply(
                st.session_state["ai_chat_history"][:-1],
                final_prompt,
                context=context,
            )

        reply = result.get("text") or "(No response returned)"
        st.session_state["ai_chat_history"].append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)

    # Chat management footer
    col_clear, col_export, _ = st.columns([1, 1, 3])
    with col_clear:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state["ai_chat_history"] = []
            st.rerun()
    with col_export:
        if st.session_state.get("ai_chat_history"):
            chat_text = "\n\n".join(
                f"**{m['role'].upper()}**: {m['content']}"
                for m in st.session_state["ai_chat_history"]
            )
            st.download_button(
                "📥 Export Transcript",
                data=chat_text,
                file_name="sepsisguard_chat_transcript.md",
                mime="text/markdown",
                use_container_width=True,
            )