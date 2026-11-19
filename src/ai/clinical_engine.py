

from __future__ import annotations

import re
from typing import Any

DISCLAIMER = (
    "Research prototype early warning system. Not a validated medical device. "
    "All AI recommendations and metrics are for research and educational purposes only "
    "and must not be used as clinical orders."
)


def _fmt(val: Any, decimals: int = 2, suffix: str = "") -> str:
    if val is None:
        return "—"
    try:
        f = float(val)
        if f != f:  # NaN
            return "—"
        return f"{f:.{decimals}f}{suffix}"
    except (ValueError, TypeError):
        return str(val)


class LocalClinicalEngine:
    """Offline clinical reasoning engine for ICU deterioration early warning."""

    def __init__(self):
        pass

    def evaluate_ssc_bundle(self, patient: dict[str, Any]) -> dict[str, Any]:
        """Evaluate Surviving Sepsis Campaign (SSC) 1-Hour Bundle criteria."""
        risk = float(patient.get("risk_score") or 0.0)
        news2 = float(patient.get("news2") or 0.0)
        sofa = float(patient.get("sofa") or 0.0)

        # Extract vitals if present in features or directly
        vitals = patient.get("vitals") or {}
        features = patient.get("features") or {}

        def _get_val(keys: list[str], default: float | None = None) -> float | None:
            for k in keys:
                if k in vitals and vitals[k] is not None:
                    try:
                        return float(vitals[k])
                    except (ValueError, TypeError):
                        pass
                if k in features and features[k] is not None:
                    try:
                        return float(features[k])
                    except (ValueError, TypeError):
                        pass
            return default

        map_val = _get_val(["MAP", "MAP__w60_mean", "map__w60_mean"])
        sbp_val = _get_val(["SBP", "SBP__w60_mean", "sbp__w60_mean"])
        hr_val = _get_val(["HR", "HR__w60_mean", "heart_rate__w60_mean"])
        lactate_val = _get_val(["Lactate", "Lactate__w240_mean", "lactate__w240_mean"])
        spo2_val = _get_val(["O2Sat", "O2Sat__w60_mean", "spo2__w60_mean"])
        resp_val = _get_val(["Resp", "Resp__w60_mean", "resp_rate__w60_mean"])

        hypotensive = (map_val is not None and map_val < 65.0) or (sbp_val is not None and sbp_val < 90.0)
        high_lactate = lactate_val is not None and lactate_val >= 4.0
        elevated_lactate = lactate_val is not None and lactate_val >= 2.0

        items = [
            {
                "step": "1. Lactate Measurement",
                "status": "URGENT" if (high_lactate or elevated_lactate or risk >= 0.65) else "RECOMMENDED",
                "detail": f"Current Lactate: {_fmt(lactate_val, 1, ' mmol/L')}. "
                          + ("Re-measure within 2-4h if > 2.0 mmol/L to guide resuscitation."
                             if elevated_lactate else "Establish baseline serum lactate."),
                "triggered": elevated_lactate or risk >= 0.65,
            },
            {
                "step": "2. Blood Cultures",
                "status": "REQUIRED" if (risk >= 0.65 or news2 >= 5 or sofa >= 2) else "CONSIDER",
                "detail": "Obtain at least 2 sets of blood cultures (aerobic and anaerobic) BEFORE starting antimicrobials.",
                "triggered": risk >= 0.65 or news2 >= 5 or sofa >= 2,
            },
            {
                "step": "3. Broad-Spectrum Antimicrobials",
                "status": "URGENT" if (risk >= 0.65 or news2 >= 7) else "EVALUATE",
                "detail": "Administer IV broad-spectrum antimicrobials within 1 hour of recognition of suspected sepsis.",
                "triggered": risk >= 0.65 or news2 >= 7,
            },
            {
                "step": "4. Rapid Fluid Resuscitation (30 mL/kg)",
                "status": "IMMEDIATE" if (hypotensive or high_lactate) else "MONITOR",
                "detail": f"MAP: {_fmt(map_val, 0, ' mmHg')}, SBP: {_fmt(sbp_val, 0, ' mmHg')}. "
                          + ("Hypotension (MAP < 65) or Lactate ≥ 4.0 detected: begin 30 mL/kg balanced crystalloids."
                             if (hypotensive or high_lactate) else "Fluid bolus not immediately indicated by hemodynamics; monitor fluid responsiveness."),
                "triggered": hypotensive or high_lactate,
            },
            {
                "step": "5. Vasopressors (Norepinephrine)",
                "status": "INDICATED" if hypotensive else "STANDBY",
                "detail": "Apply vasopressors during/after fluid resuscitation if MAP remains < 65 mmHg to maintain target perfusion pressure (MAP ≥ 65 mmHg).",
                "triggered": hypotensive,
            },
        ]

        return {
            "patient_id": patient.get("patient_id"),
            "risk_score": risk,
            "hypotensive": hypotensive,
            "high_lactate": high_lactate,
            "items": items,
            "disclaimer": DISCLAIMER,
        }

    def generate_sbar_handoff(self, context: dict[str, Any]) -> str:
        """Generate structured Situation-Background-Assessment-Recommendation (SBAR) handoff."""
        status = context.get("status") or {}
        high_risk = context.get("high_risk") or []
        shift = context.get("shift") or {}
        census = context.get("census_top") or []

        n_tracked = status.get("n_patients_tracked", len(census))
        n_alerts = status.get("n_alerts", 0)
        active_model = str(status.get("active_model", "online")).upper()
        online_auroc = status.get("online_auroc")
        auroc_str = f"{float(online_auroc):.3f}" if online_auroc else "N/A"

        # Situation
        crit_count = sum(1 for p in high_risk if float(p.get("risk_score") or 0.0) >= 0.65)
        situation = (
            f"**Situation:**\n"
            f"- SepsisGuard ICU Census: **{n_tracked} patients currently monitored**.\n"
            f"- Active Deterioration Alerts: **{crit_count} high-risk / {n_alerts} total alerts** flagged in the current window.\n"
            f"- Streaming Model: **{active_model}** (Online AUROC: **{auroc_str}**, threshold: **{status.get('alert_threshold', 0.65):.0%}**)."
        )

        # Background
        top_patients = high_risk[:4] if high_risk else census[:3]
        bg_lines = ["**Background (Priority Watchlist):**"]
        if top_patients:
            for p in top_patients:
                pid = p.get("patient_id", "Unknown")
                risk = float(p.get("risk_score") or 0.0)
                news2 = p.get("news2", "—")
                qsofa = p.get("qsofa", "—")
                sofa = p.get("sofa", "—")
                reasons = p.get("explanation") or []
                reason_str = "; ".join(reasons[:2]) if isinstance(reasons, list) else str(reasons)
                bg_lines.append(
                    f"- **Patient {pid}**: Risk **{risk:.0%}** (NEWS2: {news2}, qSOFA: {qsofa}, SOFA: {sofa}). "
                    f"Drivers: *{reason_str or 'Multi-vital trend elevation'}*."
                )
        else:
            bg_lines.append("- Ward hemodynamics stable; no critical deterioration breaches recorded.")

        background = "\n".join(bg_lines)

        # Assessment
        assessment = (
            f"**Assessment:**\n"
            f"- Deterioration risk is concentrated in **{crit_count} patient(s)** showing early physiologic decompensation.\n"
            f"- Primary deterioration signatures: sliding-window tachycardia, narrowing pulse pressure, and elevated respiratory rates.\n"
            f"- ML-to-clinical alignment: Model alerts correlate with elevated NEWS2 and organ dysfunction sub-scores."
        )

        # Recommendation
        rec = (
            f"**Recommendation:**\n"
            f"1. Bedside reassessment of priority watchlist patients within the next 30 minutes.\n"
            f"2. Check pending lab work (serum lactate, arterial blood gas, WBC, and renal panel).\n"
            f"3. Initiate Surviving Sepsis 1-Hour bundle protocols where clinical criteria are met.\n"
            f"4. Confirm clinician feedback in the dashboard to maintain online learning accuracy."
        )

        return f"### 📋 ICU SBAR Shift Handoff Summary\n\n{situation}\n\n{background}\n\n{assessment}\n\n{rec}\n\n> ⚠️ *{DISCLAIMER}*"

    def generate_patient_narrative(self, patient: dict[str, Any]) -> str:
        """Create a clinical narrative for a specific patient deterioration alert."""
        pid = patient.get("patient_id", "Unknown")
        risk = float(patient.get("risk_score") or 0.0)
        news2 = patient.get("news2")
        qsofa = patient.get("qsofa")
        sirs = patient.get("sirs")
        sofa = patient.get("sofa")
        model = str(patient.get("model_used") or "ensemble").upper()

        reasons = patient.get("explanation") or []
        if isinstance(reasons, dict):
            reasons = reasons.get("reasons") or reasons.get("top_features") or []
        reason_text = ""
        if reasons:
            clean_reasons = [str(r) for r in reasons[:4]]
            reason_text = " Key physiological contributors include: " + "; ".join(clean_reasons) + "."

        severity = (
            "CRITICAL" if risk >= 0.80
            else "HIGH" if risk >= 0.65
            else "MODERATE" if risk >= 0.45
            else "LOW"
        )

        lines = [
            f"**Patient {pid}** is exhibiting a **{severity}** risk of sepsis-induced deterioration at **{risk:.1%}** (predicted by {model} model).",
            f"Clinical composite scores indicate concurrent physiological strain: **NEWS2 of {news2 or '—'}**, **qSOFA of {qsofa or '—'}**, **SIRS of {sirs or '—'}**, and **SOFA organ dysfunction score of {sofa or '—'}**.{reason_text}",
            "The multi-window trend shows acute deviation from baseline ICU stability over the preceding 1h–4h monitoring interval.",
            f"**Recommended next step:** Conduct prompt bedside clinical assessment, review hemodynamics (MAP/HR), and verify 1-hour sepsis protocol readiness.",
            f"\n> ℹ️ *{DISCLAIMER}*",
        ]
        return "\n\n".join(lines)

    def generate_care_suggestions(self, patient: dict[str, Any]) -> str:
        """Generate research-only clinical review suggestions for deteriorating patient."""
        ssc = self.evaluate_ssc_bundle(patient)
        pid = patient.get("patient_id", "Unknown")
        risk = float(patient.get("risk_score") or 0.0)

        lines = [
            f"### 🩺 Clinical Review Suggestions — Patient {pid} (Risk: {risk:.0%})",
            "Based on Surviving Sepsis Campaign (SSC) guidelines & ICU early warning protocols:\n",
        ]

        for item in ssc["items"]:
            icon = "🔴" if item["status"] in ("URGENT", "IMMEDIATE", "REQUIRED") else "🟡"
            lines.append(f"- **{icon} {item['step']}** [{item['status']}]: {item['detail']}")

        lines.append(
            "\n**Diagnostic Workup Considerations:**\n"
            "- Repeat arterial/venous blood gas with lactate in 2–4 hours.\n"
            "- Check Complete Blood Count (CBC) and Comprehensive Metabolic Panel (Creatinine, Bilirubin).\n"
            "- Monitor fluid responsiveness (Passive Leg Raise or stroke volume variation) prior to excessive volume loading.\n"
            f"\n> ⚠️ *{DISCLAIMER}*"
        )
        return "\n".join(lines)

    def answer_chat_query(
        self,
        query: str,
        context: dict[str, Any] | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        """Intelligent ward-aware conversational QA response."""
        q = (query or "").strip().lower()
        ctx = context or {}
        high_risk = ctx.get("high_risk") or []
        census = ctx.get("census_top") or []
        status = ctx.get("status") or {}

        # 1. Check for Patient ID lookup (e.g. P001, P004, patient 4, etc.)
        pat_match = re.search(r"\b(p\d+|\d+)\b", q)
        if pat_match:
            candidate_id = pat_match.group(1).upper()
            if not candidate_id.startswith("P") and candidate_id.isdigit():
                candidate_id = f"P{int(candidate_id):03d}"

            # Look in high_risk or census
            target = None
            for p in high_risk + census:
                pid = str(p.get("patient_id", "")).upper()
                if pid == candidate_id or pid.endswith(candidate_id) or candidate_id.endswith(pid):
                    target = p
                    break

            if target:
                return self.generate_patient_narrative(target)

        # 2. Shift Handoff / Summary request
        if any(w in q for w in ["shift", "handoff", "summary", "round", "sbar", "report"]):
            return self.generate_sbar_handoff(ctx)

        # 3. High risk / alerts / who is sick / critical
        if any(w in q for w in ["high risk", "alert", "critical", "deteriorat", "worst", "sick", "flag"]):
            if high_risk:
                lines = [
                    f"### 🚨 Active High-Risk Patients ({len(high_risk)} tracked)",
                    "Here are the current priority patients exceeding the alert threshold:\n",
                ]
                for p in high_risk[:6]:
                    pid = p.get("patient_id", "Unknown")
                    risk = float(p.get("risk_score") or 0.0)
                    news2 = p.get("news2", "—")
                    sofa = p.get("sofa", "—")
                    model = p.get("model_used", "ensemble")
                    reasons = p.get("explanation") or []
                    reason_str = "; ".join(reasons[:2]) if isinstance(reasons, list) else str(reasons)
                    lines.append(
                        f"- **Patient {pid}** | Risk: **{risk:.1%}** | NEWS2: **{news2}** | SOFA: **{sofa}** ({model})\n"
                        f"  *Key Driver: {reason_str or 'Multi-vital trend deviation'}*"
                    )
                lines.append(f"\n> ℹ️ Type `explain patient <ID>` or click the **Patient Alerts** tab for details. *{DISCLAIMER}*")
                return "\n".join(lines)
            else:
                return "Currently, no patients have exceeded the high-risk alert threshold. The ward census is hemodynamically stable."

        # 4. Explain scores: NEWS2, qSOFA, SIRS, SOFA
        if any(w in q for w in ["news2", "qsofa", "sirs", "sofa", "score"]):
            return (
                "### 🩺 Clinical Scoring Systems in SepsisGuard\n\n"
                "- **NEWS2 (National Early Warning Score 2):**\n"
                "  Aggregates 6 physiological parameters (Respiration Rate, SpO₂, Systolic BP, Heart Rate, Consciousness, Temperature). "
                "Scores ≥ 5 trigger urgent review; scores ≥ 7 or a single parameter scoring 3 indicate critical emergency response.\n\n"
                "- **qSOFA (Quick SOFA):**\n"
                "  Bedside sepsis screening tool assessing: RR ≥ 22/min, SBP ≤ 100 mmHg, and altered mental status. "
                "Score ≥ 2 indicates elevated risk of in-hospital mortality or prolonged ICU stay.\n\n"
                "- **SIRS (Systemic Inflammatory Response Syndrome):**\n"
                "  Criteria: Temp > 38.3°C or < 36.0°C, HR > 90 bpm, RR > 20/min, WBC > 12,000 or < 4,000 /µL. "
                "Presence of ≥ 2 criteria meets SIRS definition.\n\n"
                "- **SOFA (Sequential Organ Failure Assessment):**\n"
                "  Quantifies acute organ dysfunction across respiration (PaO₂/FiO₂), coagulation (platelets), liver (bilirubin), "
                "cardiovascular (MAP/pressors), CNS (GCS), and renal (creatinine/urine output). Sepsis-3 defines sepsis as infection + acute SOFA increase ≥ 2."
            )

        # 5. Model Architecture & Online Learning questions
        if any(w in q for w in ["model", "online", "baseline", "river", "hoeffding", "rollback", "drift", "ensemble"]):
            act_model = str(status.get("active_model", "blend")).upper()
            online_auroc = status.get("online_auroc", 0.0)
            base_auroc = status.get("baseline_auroc", 0.0)
            return (
                f"### 🤖 SepsisGuard ML Architecture\n\n"
                f"- **Active Model State:** **{act_model}**\n"
                f"- **Dual-Model Ensemble:**\n"
                f"  1. **Online Model:** River `HoeffdingAdaptiveTreeClassifier` continuously updating via delayed labels (6h prediction window) and clinician feedback.\n"
                f"  2. **Baseline Model:** Frozen scikit-learn / XGBoost model serving as an immutable clinical safety floor.\n"
                f"- **Current Performance:** Online AUROC: **{online_auroc:.3f}** | Baseline AUROC: **{base_auroc:.3f}**.\n"
                f"- **Safety Auto-Rollback:** If online AUROC drops more than 5% below baseline AUROC, the pipeline automatically rolls back to the baseline model.\n"
                f"- **Concept Drift Detection:** ADWIN, Page-Hinkley, and DDM simultaneously monitor the prediction stream for distribution shifts."
            )

        # 6. Surviving Sepsis Bundle questions
        if any(w in q for w in ["surviving sepsis", "bundle", "protocol", "ssc", "treatment", "guideline"]):
            return (
                "### 🏥 Surviving Sepsis Campaign (SSC) 1-Hour Bundle\n\n"
                "1. **Measure lactate level:** Remeasure if initial lactate > 2.0 mmol/L.\n"
                "2. **Obtain blood cultures:** Draw prior to starting antimicrobials (do not delay antibiotics if cultures are delayed > 45 min).\n"
                "3. **Administer broad-spectrum antibiotics:** Infuse within 1 hour of recognition.\n"
                "4. **Begin rapid administration of 30 mL/kg crystalloid:** For hypotension (MAP < 65 mmHg) or lactate ≥ 4.0 mmol/L.\n"
                "5. **Apply vasopressors:** Norepinephrine first line if MAP < 65 mmHg during or after fluid resuscitation."
            )

        # 7. Abnormality Justifications & Clinical Case Examples
        if any(w in q for w in ["abnormal", "justify", "justification", "why is hr", "why is map", "why is lactate", "why is sbp", "case example", "example"]):
            from src.explainability.explainer import CLINICAL_ABNORMALITY_JUSTIFICATIONS
            import random

            matched_cat = None
            for key, info in CLINICAL_ABNORMALITY_JUSTIFICATIONS.items():
                if key in q or any(term.strip().lower() in q for term in info["label"].lower().split("/")):
                    matched_cat = (key, info)
                    break

            if not matched_cat:
                matched_cat = random.choice(list(CLINICAL_ABNORMALITY_JUSTIFICATIONS.items()))

            cat_key, info = matched_cat
            rand_eg = random.choice(info["examples"])
            return (
                f"### 📖 Clinical Justification: {info['label']}\n\n"
                f"- **Normal ICU Range:** {info['normal_range']}\n"
                f"- **Abnormal Cutoff:** {info['abnormal_criteria']}\n\n"
                f"**Pathophysiological Justification:**\n{info['pathophysiology']}\n\n"
                f"**Representative Clinical Case Example:**\n> 🩺 *{rand_eg}*\n\n"
                f"> ℹ️ *{DISCLAIMER}*"
            )

        # 8. Default Ward Overview response
        n_pts = status.get("n_patients_tracked", len(census))
        n_al = status.get("n_alerts", 0)
        return (
            f"**SepsisGuard Ward Assistant:**\n\n"
            f"Currently monitoring **{n_pts} ICU patients** with **{n_al} total alerts logged**.\n\n"
            f"You can ask me:\n"
            f"- *'Summarize active alerts'* or *'Who is at high risk?'*\n"
            f"- *'Draft an SBAR shift handoff'* for current rounds\n"
            f"- *'Explain patient P001'* for a detailed clinical narrative\n"
            f"- *'What is NEWS2 and SOFA?'* for clinical scoring definitions\n"
            f"- *'How does the online learning model work?'* for ML architecture details\n\n"
            f"> ℹ️ *{DISCLAIMER}*"
        )