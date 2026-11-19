"""Smoke-run the AI bots with placeholder keys (no live LLM required)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ai.client import LLMClient
from src.ai.services import AIAssistService, ward_context


class _DemoPipe:
    def status(self):
        return {
            "n_patients_tracked": 12,
            "n_alerts": 4,
            "n_predictions": 120,
            "active_model": "blend",
            "online_auroc": 0.81,
            "baseline_auroc": 0.74,
            "alert_threshold": 0.65,
        }

    def high_risk_patients(self):
        return [
            {
                "patient_id": "P001",
                "risk_score": 0.91,
                "severity": "critical",
                "news2": 8,
                "qsofa": 2,
                "sirs": 3,
                "sofa": 4,
                "model_used": "blend",
                "explanation": {"reasons": ["HR avg last 1 hr: 128 bpm", "Lactate last 4 hr: 3.8 mmol/L"]},
                "vitals": {"HR": 128, "MAP": 58, "Resp": 26, "Lactate": 3.8},
            }
        ]

    def census(self):
        return self.high_risk_patients()

    def shift_summary(self):
        return {"n_patients": 12, "n_alerts": 4}


def _safe(text: str) -> str:
    return text.encode("ascii", "replace").decode("ascii")


def main() -> None:
    client = LLMClient("config/settings.yaml")
    status = client.status()
    print("=== LLM client ===")
    print(f"  provider:           {status['provider']}")
    print(f"  model:              {status['model']}")
    print(f"  credentials_ready:  {status['credentials_ready']}")
    print(f"  using_placeholders: {status['using_placeholders']}")
    print(f"  mode:               {status['mode_display']}")
    print()

    assist = AIAssistService("config/settings.yaml")
    pipe = _DemoPipe()
    ctx = ward_context(pipe)
    patient = pipe.high_risk_patients()[0]

    chat = assist.chat.reply([], "Who is at high risk?", context=ctx)
    print("=== Chatbot (ward alerts) ===")
    print(_safe(chat["text"][:600]))
    print()

    narr = assist.narrative.narrate(patient)
    print("=== Alert narrative bot ===")
    print(_safe(narr["text"][:500]))
    print()

    sbar = assist.shift.summarize(ctx)
    print("=== SBAR handoff bot ===")
    print(_safe(sbar["text"][:500]))
    print()

    sim = assist.simulator.simulate(patient, {"HR": 80, "MAP": 78, "Lactate": 1.2})
    print("=== What-if simulator ===")
    print(f"  baseline={sim['baseline_risk']:.1%}  simulated={sim['simulated_risk']:.1%}  delta={sim['delta']:+.1%}")
    print("OK — AI bots ran in local-engine mode.")


if __name__ == "__main__":
    main()
