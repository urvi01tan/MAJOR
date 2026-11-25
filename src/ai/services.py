"""Clinical AI helpers built on the multi-provider LLM client and local clinical engine."""

from __future__ import annotations

from typing import Any

from src.ai.client import LLMClient
from src.ai.clinical_engine import LocalClinicalEngine, DISCLAIMER

_DISCLAIMER = DISCLAIMER

_SYSTEM_CHAT = (
    "You are SepsisGuard Assistant, an intelligent clinical AI for an ICU early-warning system. "
    "Provide clear, concise, direct answers based on the supplied ward context and clinical vitals. "
    "Keep answers actionable and structured without repetitive disclaimer boilerplate."
)

_SYSTEM_SUMMARY = (
    "Write a concise, professional SBAR shift-handoff summary for ICU charge nurses. "
    "Use bullet points. Mention risk bands, priority patients, active alerts, and model health."
)

_SYSTEM_NARRATIVE = (
    "Turn structured sepsis-risk features into a clear, professional clinical narrative (3-5 sentences). "
    "Cite NEWS2, qSOFA, SIRS, SOFA and primary physiological contributors."
)

_SYSTEM_SUGGEST = (
    "Suggest concise, evidence-based next checks for a deteriorating ICU patient "
    "(vital monitoring, priority labs, organ review)."
)


def _truncate(value: Any, n: int = 80) -> str:
    text = str(value)
    return text if len(text) <= n else text[: n - 3] + "..."


def compact_patient(patient: dict[str, Any]) -> dict[str, Any]:
    explanation = patient.get("explanation")
    if isinstance(explanation, dict):
        reasons = explanation.get("reasons") or explanation.get("top_features") or []
        explanation = [_truncate(r, 120) for r in reasons[:5]]
    return {
        "patient_id": patient.get("patient_id"),
        "risk_score": patient.get("risk_score"),
        "severity": patient.get("severity"),
        "news2": patient.get("news2"),
        "qsofa": patient.get("qsofa"),
        "sirs": patient.get("sirs"),
        "sofa": patient.get("sofa"),
        "alert": patient.get("alert"),
        "acknowledged": patient.get("acknowledged"),
        "model_used": patient.get("model_used"),
        "explanation": explanation,
    }


def ward_context(pipeline, max_patients: int = 12) -> dict[str, Any]:
    status = pipeline.status() if hasattr(pipeline, "status") else {}
    high_risk = []
    if hasattr(pipeline, "high_risk_patients"):
        high_risk = [compact_patient(p) for p in pipeline.high_risk_patients()[:max_patients]]
    census = []
    if hasattr(pipeline, "census"):
        census = [compact_patient(p) for p in pipeline.census()[:max_patients]]
    shift = pipeline.shift_summary() if hasattr(pipeline, "shift_summary") else {}
    return {
        "disclaimer": _DISCLAIMER,
        "status": {
            "n_patients_tracked": status.get("n_patients_tracked"),
            "n_alerts": status.get("n_alerts"),
            "n_predictions": status.get("n_predictions"),
            "active_model": status.get("active_model"),
            "online_auroc": status.get("online_auroc"),
            "baseline_auroc": status.get("baseline_auroc"),
            "alert_threshold": status.get("alert_threshold"),
        },
        "high_risk": high_risk,
        "census_top": census,
        "shift": shift,
    }


class ClinicalChatService:
    def __init__(self, client: LLMClient | None = None, cfg_path: str = "config/settings.yaml"):
        self.client = client or LLMClient(cfg_path)

    def reply(
        self,
        history: list[dict[str, str]],
        user_message: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        context_block = ""
        if context:
            context_block = (
                "Current ward context (JSON):\n"
                f"{_pretty(context)}\n\n"
            )
        messages = list(history[-12:])
        messages.append(
            {
                "role": "user",
                "content": f"{context_block}Clinician question: {user_message}",
            }
        )
        result = self.client.chat(messages, system=_SYSTEM_CHAT, context=context)
        result["user_message"] = user_message
        return result


class ShiftSummaryService:
    def __init__(self, client: LLMClient | None = None, cfg_path: str = "config/settings.yaml"):
        self.client = client or LLMClient(cfg_path)

    def summarize(self, context: dict[str, Any]) -> dict[str, Any]:
        if not self.client.credentials_ready():
            text = self.client.clinical_engine.generate_sbar_handoff(context)
            return {
                "ok": True,
                "placeholder": True,
                "text": text,
                "provider": self.client.provider,
                "model": self.client.model,
            }

        user = (
            "Draft a shift handoff from this snapshot:\n"
            f"{_pretty(context)}"
        )
        return self.client.chat(
            [{"role": "user", "content": user}],
            system=_SYSTEM_SUMMARY,
            context=context,
        )


class AlertNarrativeService:
    def __init__(self, client: LLMClient | None = None, cfg_path: str = "config/settings.yaml"):
        self.client = client or LLMClient(cfg_path)

    def narrate(self, patient: dict[str, Any]) -> dict[str, Any]:
        if not self.client.credentials_ready():
            text = self.client.clinical_engine.generate_patient_narrative(patient)
            return {
                "ok": True,
                "placeholder": True,
                "text": text,
                "provider": self.client.provider,
                "model": self.client.model,
            }

        user = (
            "Write an alert narrative for this patient snapshot:\n"
            f"{_pretty(compact_patient(patient))}"
        )
        return self.client.chat(
            [{"role": "user", "content": user}],
            system=_SYSTEM_NARRATIVE,
        )


class CareSuggestionService:
    def __init__(self, client: LLMClient | None = None, cfg_path: str = "config/settings.yaml"):
        self.client = client or LLMClient(cfg_path)

    def suggest(self, patient: dict[str, Any]) -> dict[str, Any]:
        if not self.client.credentials_ready():
            text = self.client.clinical_engine.generate_care_suggestions(patient)
            return {
                "ok": True,
                "placeholder": True,
                "text": text,
                "provider": self.client.provider,
                "model": self.client.model,
            }

        user = (
            "Propose research-only review checks for this patient:\n"
            f"{_pretty(compact_patient(patient))}"
        )
        return self.client.chat(
            [{"role": "user", "content": user}],
            system=_SYSTEM_SUGGEST,
        )


class CounterfactualSimulatorService:
    """Clinical What-If risk simulator based on physiological interventions."""

    def __init__(self, pipeline=None):
        self.pipeline = pipeline

    def simulate(
        self, patient: dict[str, Any], vital_overrides: dict[str, float]
    ) -> dict[str, Any]:
        baseline_risk = float(patient.get("risk_score") or 0.5)

        hr_val = vital_overrides.get("HR")
        map_val = vital_overrides.get("MAP")
        resp_val = vital_overrides.get("Resp")
        temp_val = vital_overrides.get("Temp")
        lactate_val = vital_overrides.get("Lactate")

        delta = 0.0
        details = []

        if hr_val is not None:
            if hr_val > 105:
                d = (hr_val - 100) * 0.003
                delta += d
                details.append(f"Tachycardia ({hr_val:.0f} bpm) adds +{d:.1%} to deterioration risk")
            elif hr_val <= 85:
                d = -0.07
                delta += d
                details.append(f"Normalized heart rate ({hr_val:.0f} bpm) reduces risk by {abs(d):.1%}")

        if map_val is not None:
            if map_val < 65:
                d = (65 - map_val) * 0.006
                delta += d
                details.append(f"Hypotension (MAP {map_val:.0f} mmHg) adds +{d:.1%} to risk")
            elif map_val >= 75:
                d = -0.11
                delta += d
                details.append(f"Restored perfusion (MAP {map_val:.0f} mmHg) reduces risk by {abs(d):.1%}")

        if resp_val is not None:
            if resp_val >= 22:
                d = (resp_val - 20) * 0.008
                delta += d
                details.append(f"Tachypnea (Resp {resp_val:.0f}/min) adds +{d:.1%} to risk")
            elif resp_val <= 18:
                d = -0.05
                delta += d
                details.append(f"Normalized respiration ({resp_val:.0f}/min) reduces risk by {abs(d):.1%}")

        if temp_val is not None:
            if temp_val >= 38.3:
                d = 0.05
                delta += d
                details.append(f"Fever ({temp_val:.1f}°C) adds +{d:.1%} to risk")
            elif temp_val <= 35.8:
                d = 0.08
                delta += d
                details.append(f"Hypothermia ({temp_val:.1f}°C) adds +{d:.1%} to risk")

        if lactate_val is not None:
            if lactate_val >= 2.0:
                d = min(0.25, (lactate_val - 2.0) * 0.05)
                delta += d
                details.append(f"Elevated lactate ({lactate_val:.1f} mmol/L) adds +{d:.1%} to risk")
            elif lactate_val < 1.5:
                d = -0.09
                delta += d
                details.append(f"Normal lactate ({lactate_val:.1f} mmol/L) reduces risk by {abs(d):.1%}")

        simulated_risk = max(0.01, min(0.99, baseline_risk + delta))
        pct_change = ((simulated_risk - baseline_risk) / max(0.01, baseline_risk)) * 100.0

        return {
            "patient_id": patient.get("patient_id"),
            "baseline_risk": baseline_risk,
            "simulated_risk": round(simulated_risk, 4),
            "delta": round(delta, 4),
            "percent_change": round(pct_change, 1),
            "details": details,
            "interpretation": (
                "Projected physiological stabilization substantially mitigates sepsis risk."
                if delta < -0.05
                else "Simulated hemodynamic deterioration triggers high probability of critical escalation."
                if delta > 0.05
                else "Minor projected variation in deterioration risk score."
            ),
        }


class OrganDysfunctionService:
    """Evaluates multi-organ system dysfunction for a patient."""

    def evaluate(self, patient: dict[str, Any]) -> dict[str, Any]:
        sofa = float(patient.get("sofa") or 0.0)
        news2 = float(patient.get("news2") or 0.0)
        risk = float(patient.get("risk_score") or 0.0)

        # Cardiovascular
        cv_status = "Elevated" if (news2 >= 4 or risk >= 0.65) else "Normal"
        cv_score = min(4, int(risk * 4))

        # Respiratory
        resp_status = "Impaired" if news2 >= 5 else "Normal"
        resp_score = min(4, int(sofa * 0.5))

        # Renal
        renal_status = "At Risk" if sofa >= 2 else "Preserved"
        renal_score = min(4, int(sofa * 0.4))

        # Coagulation
        coag_status = "Borderline" if sofa >= 3 else "Normal"
        coag_score = 1 if sofa >= 3 else 0

        # Hepatic
        hep_status = "Normal"
        hep_score = 0

        systems = [
            {"system": "Cardiovascular", "status": cv_status, "score": cv_score, "max": 4},
            {"system": "Respiratory", "status": resp_status, "score": resp_score, "max": 4},
            {"system": "Renal", "status": renal_status, "score": renal_score, "max": 4},
            {"system": "Coagulation", "status": coag_status, "score": coag_score, "max": 4},
            {"system": "Hepatic", "status": hep_status, "score": hep_score, "max": 4},
        ]

        total_dysfunction = sum(s["score"] for s in systems)
        return {
            "patient_id": patient.get("patient_id"),
            "total_score": total_dysfunction,
            "systems": systems,
            "summary": (
                "Multi-system organ failure pattern detected."
                if total_dysfunction >= 4
                else "Isolated organ strain observed."
                if total_dysfunction >= 2
                else "Organ perfusion largely preserved."
            ),
        }


class AIAssistService:
    """Master clinical AI facade used across the SepsisGuard dashboard."""

    def __init__(self, cfg_path: str = "config/settings.yaml", pipeline=None):
        self.client = LLMClient(cfg_path)
        self.chat = ClinicalChatService(self.client)
        self.shift = ShiftSummaryService(self.client)
        self.narrative = AlertNarrativeService(self.client)
        self.care = CareSuggestionService(self.client)
        self.simulator = CounterfactualSimulatorService(pipeline)
        self.organs = OrganDysfunctionService()

    def status(self) -> dict[str, Any]:
        return self.client.status()


def _pretty(payload: Any) -> str:
    import json
    return json.dumps(payload, default=str, indent=2)[:8000]