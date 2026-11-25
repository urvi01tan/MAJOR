"""AI client and clinical chatbot services (multi-provider, local clinical engine, no network)."""

from __future__ import annotations

import pytest

from src.ai.client import LLMClient, is_placeholder_value
from src.ai.clinical_engine import LocalClinicalEngine
from src.ai.services import (
    AIAssistService,
    CounterfactualSimulatorService,
    OrganDysfunctionService,
    compact_patient,
    ward_context,
)


@pytest.fixture(autouse=True)
def _clean_test_env(monkeypatch):
    for k in ("GEMINI_API_KEY", "OPENAI_API_KEY", "SEPSISGUARD_LLM_API_KEY", "ANTHROPIC_API_KEY", "AZURE_OPENAI_API_KEY", "SEPSISGUARD_LLM_PROVIDER"):
        monkeypatch.delenv(k, raising=False)


def test_placeholder_detection():
    assert is_placeholder_value("YOUR_LLM_API_KEY_HERE")
    assert is_placeholder_value("https://YOUR_LLM_BASE_URL_HERE/v1")
    assert is_placeholder_value("")
    assert is_placeholder_value(None)
    assert not is_placeholder_value("sk-live-real-key")


def test_client_stays_in_placeholder_mode():
    client = LLMClient("config/settings.yaml")
    status = client.status()
    assert status["using_placeholders"] is True
    assert status["credentials_ready"] is False

    result = client.chat(
        [{"role": "user", "content": "Which patients are high risk?"}],
        system="test",
    )
    assert result["ok"] is True
    assert result["placeholder"] is True
    assert len(result["text"]) > 20
    assert "high risk" in result["text"].lower() or "ward" in result["text"].lower()


def test_chat_service_uses_ward_context_without_http(monkeypatch):
    assist = AIAssistService("config/settings.yaml")
    called = {"n": 0}

    def _fail_post(*_args, **_kwargs):
        called["n"] += 1
        raise AssertionError("placeholder mode must not call the network")

    monkeypatch.setattr("src.ai.client.requests.post", _fail_post)
    result = assist.chat.reply(
        [],
        "Summarise alerts",
        context={"high_risk": [{"patient_id": "P001", "risk_score": 0.9}]},
    )
    assert called["n"] == 0
    assert result["placeholder"] is True
    assert result["user_message"] == "Summarise alerts"


def test_compact_patient_and_ward_context():
    class _Pipe:
        def status(self):
            return {
                "n_patients_tracked": 2,
                "n_alerts": 1,
                "n_predictions": 10,
                "active_model": "blend",
                "online_auroc": 0.8,
                "baseline_auroc": 0.7,
                "alert_threshold": 0.65,
            }

        def high_risk_patients(self):
            return [
                {
                    "patient_id": "P001",
                    "risk_score": 0.91,
                    "severity": "critical",
                    "news2": 8,
                    "explanation": {"reasons": ["HR trend elevated"]},
                }
            ]

        def census(self):
            return self.high_risk_patients()

        def shift_summary(self):
            return {"n_patients": 1, "n_alerts": 1}

    ctx = ward_context(_Pipe())
    assert ctx["high_risk"][0]["patient_id"] == "P001"
    compact = compact_patient(_Pipe().high_risk_patients()[0])
    assert compact["explanation"] == ["HR trend elevated"]


def test_narrative_and_shift_placeholder():
    assist = AIAssistService("config/settings.yaml")
    narr = assist.narrative.narrate({"patient_id": "P002", "risk_score": 0.8, "news2": 7})
    summary = assist.shift.summarize({"n_alerts": 3})
    care = assist.care.suggest({"patient_id": "P002", "sirs": 3})
    assert narr["placeholder"] and summary["placeholder"] and care["placeholder"]


def test_local_clinical_engine_ssc_bundle():
    engine = LocalClinicalEngine()
    patient = {
        "patient_id": "P101",
        "risk_score": 0.85,
        "news2": 8,
        "sofa": 3,
        "vitals": {"MAP": 58, "HR": 125, "Lactate": 4.2},
    }
    ssc = engine.evaluate_ssc_bundle(patient)
    assert ssc["hypotensive"] is True
    assert ssc["high_lactate"] is True
    assert len(ssc["items"]) == 5
    # Step 4 (Fluid) should be IMMEDIATE
    assert ssc["items"][3]["status"] == "IMMEDIATE"


def test_local_clinical_engine_sbar_handoff():
    engine = LocalClinicalEngine()
    context = {
        "status": {"n_patients_tracked": 10, "n_alerts": 3, "active_model": "blend", "online_auroc": 0.82},
        "high_risk": [{"patient_id": "P005", "risk_score": 0.92, "news2": 9, "sofa": 4, "explanation": ["Tachycardia"]}],
        "census_top": [{"patient_id": "P005", "risk_score": 0.92}],
    }
    sbar = engine.generate_sbar_handoff(context)
    assert "Situation:" in sbar
    assert "Background" in sbar
    assert "Assessment:" in sbar
    assert "Recommendation:" in sbar
    assert "P005" in sbar


def test_counterfactual_simulator():
    sim = CounterfactualSimulatorService()
    patient = {"patient_id": "P009", "risk_score": 0.70}
    # Stabilizing HR and MAP should reduce risk
    res_better = sim.simulate(patient, {"HR": 75, "MAP": 80, "Lactate": 1.1})
    assert res_better["simulated_risk"] < res_better["baseline_risk"]
    assert res_better["delta"] < 0

    # Worsening HR and MAP should increase risk
    res_worse = sim.simulate(patient, {"HR": 130, "MAP": 55, "Lactate": 4.5})
    assert res_worse["simulated_risk"] > res_worse["baseline_risk"]
    assert res_worse["delta"] > 0


def test_organ_dysfunction_service():
    organs = OrganDysfunctionService()
    patient = {"patient_id": "P010", "risk_score": 0.85, "news2": 7, "sofa": 4}
    res = organs.evaluate(patient)
    assert res["patient_id"] == "P010"
    assert len(res["systems"]) == 5
    assert res["total_score"] >= 2


def test_provider_headers_and_urls():
    client = LLMClient("config/settings.yaml")
    client.provider = "azure_openai"
    client.model = "my-deployment"
    client.base_url = "https://my-res.openai.azure.com"
    client.api_key = "test-key"
    assert "deployments/my-deployment/chat/completions" in client._chat_url()
    assert client._headers()["api-key"] == "test-key"

    client.provider = "anthropic"
    client.base_url = "https://api.anthropic.com"
    assert client._chat_url() == "https://api.anthropic.com/v1/messages"
    assert client._headers()["x-api-key"] == "test-key"


def test_clinical_abnormality_justifications_and_examples():
    from src.explainability.explainer import (
        CLINICAL_ABNORMALITY_JUSTIFICATIONS,
        get_clinical_justification,
    )

    # Verify every clinical abnormality has normal range, pathophysiology, and at least 3 case examples
    for key, info in CLINICAL_ABNORMALITY_JUSTIFICATIONS.items():
        assert "normal_range" in info
        assert "abnormal_criteria" in info
        assert "pathophysiology" in info
        assert len(info["examples"]) >= 3

    # Test lookup function
    hr_just = get_clinical_justification("HR__w60_mean", 125.0)
    assert hr_just["category"] == "heart_rate"
    assert "tachycardia" in hr_just["pathophysiology"].lower()
    assert len(hr_just["random_example"]) > 10

    lactate_just = get_clinical_justification("Lactate__w240_mean", 4.5)
    assert lactate_just["category"] == "lactate"
    assert "anaerobic" in lactate_just["pathophysiology"].lower()
    assert len(lactate_just["random_example"]) > 10


def test_online_model_detailed_metrics():
    from src.models.online_model import OnlineModel
    online = OnlineModel("config/settings.yaml")
    m = online.get_detailed_metrics()
    assert "model_type" in m
    assert "rolling_auroc" in m
    assert "n_updates" in m
    assert "n_predictions" in m
    assert "queue_len" in m
    assert "tree_summary" in m
