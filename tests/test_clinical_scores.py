"""Clinical score unit tests (no dataset required)."""

from src.clinical.scores import compute_news2, compute_qsofa, compute_sirs, clinical_bundle


def test_news2_stable_vitals():
    out = compute_news2({"HR": 80, "SBP": 120, "Resp": 16, "O2Sat": 98, "Temp": 37.0})
    assert out["score"] == 0
    assert out["risk"] == "none"


def test_news2_critical_pattern():
    out = compute_news2({"HR": 140, "SBP": 85, "Resp": 28, "O2Sat": 88, "Temp": 39.5})
    assert out["score"] >= 7
    assert out["risk"] == "high"


def test_qsofa_positive():
    out = compute_qsofa({"Resp": 24, "SBP": 95})
    assert out["score"] == 2
    assert out["positive"] is True


def test_sirs_positive():
    out = compute_sirs({"Temp": 38.6, "HR": 110, "Resp": 22, "WBC": 14})
    assert out["score"] >= 2
    assert out["positive"] is True


def test_clinical_bundle_keys():
    bundle = clinical_bundle({"HR": 110, "SBP": 95, "Resp": 22, "O2Sat": 93, "Temp": 38.4, "WBC": 13})
    for key in ("news2", "news2_risk", "qsofa", "sirs", "vitals"):
        assert key in bundle
