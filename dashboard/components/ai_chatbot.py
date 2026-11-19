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
    <div style="background: linear-gradient(135deg, #101c2c 0%, #17223b 100%); padding: 18px 22px; border-radius: 10px; border-left: 4px solid #00B4D8; margin-bottom: 20px;">
        <h2 style="margin: 0; color: #ffffff; font-size: 1.4rem;">💬 AI Clinical Chatbot & Assistant</h2>
        <div style="color: #8da4be; font-size: 0.85rem; margin-top: 4px;">
            Ward-aware ICU reasoning engine powered by streaming clinical vitals, NEWS2/SOFA scores, and dual-model ML predictions.
        </div>
    </div>
    """, unsafe_allow_html=True)

    assist = _service()
    status = assist.status()

    # Provider & Engine Status Bar
    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 1.4])
    c1.metric("Provider", str(status.get("provider", "—")).upper())
    c2.metric("Model", str(status.get("model", "—")))
    mode_label = "🟢 Live LLM" if status.get("credentials_ready") else "⚡ Local Clinical Engine"
    c3.metric("Engine Mode", mode_label)
    tracked_count = len(pipeline.census()) if hasattr(pipeline, "census") else 0
    c4.metric("Patients in Context", f"{tracked_count} tracked")

    if status.get("using_placeholders"):
        st.info(
            "ℹ️ **Operating in Local Clinical Intelligence mode.** The assistant is using built-in ICU clinical reasoning "
            "and ward context. To route queries to an external provider (OpenAI, Gemini, Anthropic, Azure, or Ollama), "
            "set your API credentials in `config/settings.yaml`."
        )

    # Quick prompt actions
    st.markdown("**Quick Clinical Inquiries:**")
    q_col1, q_col2, q_col3, q_col4, q_col5 = st.columns(5)

    prompt_to_submit = None

    with q_col1:
        if st.button("🚨 Active High-Risk Alerts", use_container_width=True):
            prompt_to_submit = "Summarize the active high-risk patients and their deterioration drivers."
    with q_col2:
        if st.button("📋 SBAR Shift Handoff", use_container_width=True):
            prompt_to_submit = "Generate an ICU SBAR shift handoff summary for current ward rounds."
    with q_col3:
        if st.button("🩺 Explain NEWS2 & SOFA", use_container_width=True):
            prompt_to_submit = "Explain the clinical scoring criteria for NEWS2, qSOFA, and SOFA in this system."
    with q_col4:
        if st.button("🤖 Dual-Model Architecture", use_container_width=True):
            prompt_to_submit = "Explain how the online Hoeffding Adaptive Tree and baseline model interact with auto-rollback."
    with q_col5:
        if st.button("💡 Justify Abnormalities", use_container_width=True):
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

    st.divider()

    # Chat history display
    if "ai_chat_history" not in st.session_state:
        st.session_state["ai_chat_history"] = []

    if not st.session_state["ai_chat_history"]:
        with st.chat_message("assistant"):
            st.markdown(
                "👋 **Welcome to SepsisGuard Clinical Assistant.**\n\n"
                "I am integrated directly with the real-time ICU data stream and can answer questions regarding:\n"
                "- Active deterioration alerts & priority patients\n"
                "- SBAR shift handoff summaries\n"
                "- Specific patient deterioration trajectories and feature attributions\n"
                "- Clinical scoring rules (NEWS2, qSOFA, SIRS, SOFA)\n"
                "- Surviving Sepsis Campaign (SSC) 1-Hour Bundle evaluation\n\n"
                "*Select a quick prompt above or type a question below.*"
            )

    for turn in st.session_state["ai_chat_history"]:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

    # Chat input
    user_input = st.chat_input("Ask about ward status, a patient ID (e.g. P001), or sepsis protocols…")
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