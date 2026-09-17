"""Rule-based clinical early-warning scores computed from latest vitals.

These scores sit next to the ML risk so clinicians can compare a familiar
NEWS2 / qSOFA / SIRS number with the model's probability.
"""

from __future__ import annotations

import math
from typing import Any


def _f(vitals: dict[str, Any], key: str) -> float:
    val = vitals.get(key)
    if val is None:
        return float("nan")
    try:
        return float(val)
    except (TypeError, ValueError):
        return float("nan")


def compute_news2(vitals: dict[str, Any]) -> dict[str, Any]:
    """National Early Warning Score 2 (Scale 1 for SpO2). Mental status omitted."""
    resp = _f(vitals, "Resp")
    spo2 = _f(vitals, "O2Sat")
    temp = _f(vitals, "Temp")
    sbp = _f(vitals, "SBP")
    hr = _f(vitals, "HR")

    def band(value: float, table: list[tuple[float | None, float | None, int]]) -> int:
        if math.isnan(value):
            return 0
        for lo, hi, pts in table:
            if (lo is None or value >= lo) and (hi is None or value <= hi):
                return pts
        return 0

    components = {
        "resp": band(resp, [
            (None, 8, 3), (9, 11, 1), (12, 20, 0), (21, 24, 2), (25, None, 3),
        ]),
        "spo2": band(spo2, [
            (None, 91, 3), (92, 93, 2), (94, 95, 1), (96, None, 0),
        ]),
        "temp": band(temp, [
            (None, 35.0, 3), (35.1, 36.0, 1), (36.1, 38.0, 0),
            (38.1, 39.0, 1), (39.1, None, 2),
        ]),
        "sbp": band(sbp, [
            (None, 90, 3), (91, 100, 2), (101, 110, 1),
            (111, 219, 0), (220, None, 3),
        ]),
        "hr": band(hr, [
            (None, 40, 3), (41, 50, 1), (51, 90, 0),
            (91, 110, 1), (111, 130, 2), (131, None, 3),
        ]),
    }
    total = sum(components.values())
    if total >= 7:
        risk = "high"
    elif total >= 5:
        risk = "medium"
    elif total >= 1:
        risk = "low"
    else:
        risk = "none"
    return {"score": total, "risk": risk, "components": components}


def compute_qsofa(vitals: dict[str, Any]) -> dict[str, Any]:
    """qSOFA without mental-status (GCS not in CinC 2019). Max score = 2."""
    resp = _f(vitals, "Resp")
    sbp = _f(vitals, "SBP")
    flags = {
        "rr_ge_22": (not math.isnan(resp)) and resp >= 22,
        "sbp_le_100": (not math.isnan(sbp)) and sbp <= 100,
    }
    score = int(flags["rr_ge_22"]) + int(flags["sbp_le_100"])
    return {"score": score, "positive": score >= 2, "components": flags}


def compute_sirs(vitals: dict[str, Any]) -> dict[str, Any]:
    """SIRS criteria from temperature, HR, RR, WBC."""
    temp = _f(vitals, "Temp")
    hr = _f(vitals, "HR")
    resp = _f(vitals, "Resp")
    wbc = _f(vitals, "WBC")
    flags = {
        "temp_abnormal": (not math.isnan(temp)) and (temp < 36.0 or temp > 38.0),
        "hr_gt_90": (not math.isnan(hr)) and hr > 90,
        "rr_gt_20": (not math.isnan(resp)) and resp > 20,
        "wbc_abnormal": (not math.isnan(wbc)) and (wbc < 4.0 or wbc > 12.0),
    }
    score = sum(int(v) for v in flags.values())
    return {"score": score, "positive": score >= 2, "components": flags}


def compute_sofa(vitals: dict[str, Any]) -> dict[str, Any]:
    """
    Simplified SOFA (Sequential Organ Failure Assessment) score.
    Uses available CinC 2019 fields. Full SOFA has 6 organ systems;
    we compute the subset available without GCS/urine output/vasopressors.

    Sub-scores included:
      - Respiration: SaO2/FiO2 ratio (surrogate for PaO2/FiO2)
      - Coagulation: Platelets
      - Liver: Bilirubin_total
      - Renal: Creatinine

    Returns score 0-12 (max 3 per system x 4 systems available).
    """
    # Respiration: SaO2/FiO2 ratio as surrogate
    fio2 = _f(vitals, "FiO2")
    sao2 = _f(vitals, "SaO2")
    resp_score = 0
    if not math.isnan(fio2) and fio2 > 0 and not math.isnan(sao2):
        ratio = sao2 / fio2
        if ratio < 67:
            resp_score = 3
        elif ratio < 80:
            resp_score = 2
        elif ratio < 90:
            resp_score = 1

    # Coagulation: Platelets (x10^3/uL)
    plt = _f(vitals, "Platelets")
    plt_score = 0
    if not math.isnan(plt):
        if plt < 50:
            plt_score = 3
        elif plt < 100:
            plt_score = 2
        elif plt < 150:
            plt_score = 1

    # Liver: Bilirubin total (mg/dL)
    bili = _f(vitals, "Bilirubin_total")
    bili_score = 0
    if not math.isnan(bili):
        if bili >= 12.0:
            bili_score = 3
        elif bili >= 6.0:
            bili_score = 2
        elif bili >= 2.0:
            bili_score = 1

    # Renal: Creatinine (mg/dL)
    creat = _f(vitals, "Creatinine")
    renal_score = 0
    if not math.isnan(creat):
        if creat >= 5.0:
            renal_score = 3
        elif creat >= 3.5:
            renal_score = 2
        elif creat >= 2.0:
            renal_score = 1

    total = resp_score + plt_score + bili_score + renal_score
    if total >= 9:
        risk = "critical"
    elif total >= 6:
        risk = "high"
    elif total >= 3:
        risk = "moderate"
    elif total >= 1:
        risk = "low"
    else:
        risk = "none"

    return {
        "score": total,
        "risk": risk,
        "components": {
            "respiration": resp_score,
            "coagulation": plt_score,
            "liver": bili_score,
            "renal": renal_score,
        },
    }


def clinical_bundle(vitals: dict[str, Any]) -> dict[str, Any]:
    news2 = compute_news2(vitals)
    qsofa = compute_qsofa(vitals)
    sirs = compute_sirs(vitals)
    sofa = compute_sofa(vitals)
    return {
        "news2": news2["score"],
        "news2_risk": news2["risk"],
        "news2_components": news2["components"],
        "qsofa": qsofa["score"],
        "qsofa_positive": qsofa["positive"],
        "sirs": sirs["score"],
        "sirs_positive": sirs["positive"],
        "sofa": sofa["score"],
        "sofa_risk": sofa["risk"],
        "sofa_components": sofa["components"],
        "vitals": {k: vitals.get(k) for k in (
            "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "WBC", "Lactate",
            "Creatinine", "Bilirubin_total", "Platelets", "FiO2", "SaO2",
        )},
    }
