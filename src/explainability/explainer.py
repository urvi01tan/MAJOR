from __future__ import annotations

import math
from typing import Any
import yaml



def _load_cfg(cfg_path: str = "config/settings.yaml") -> dict:
    with open(cfg_path) as f:
        return yaml.safe_load(f)

_PLAIN_ENGLISH: dict[str, str] = {
    "heart_rate__w15_mean": "HR avg last 15 min: {value:.0f} bpm",
    "heart_rate__w60_mean": "HR avg last 1 hr: {value:.0f} bpm",
    "heart_rate__w60_slope": "HR trend last 1 hr: {value:+.1f} bpm/min",
    "HR__w60_mean": "HR avg last 1 hr: {value:.0f} bpm",
    "HR__w60_slope": "HR trend last 1 hr: {value:+.2f} bpm/min",
    "sbp__w60_mean": "Systolic BP avg last 1 hr: {value:.0f} mmHg",
    "sbp__w60_slope": "Systolic BP trend: {value:+.1f} mmHg/min",
    "SBP__w60_mean": "Systolic BP avg last 1 hr: {value:.0f} mmHg",
    "SBP__w60_slope": "Systolic BP trend: {value:+.2f} mmHg/min",
    "map__w60_mean": "MAP avg last 1 hr: {value:.0f} mmHg",
    "MAP__w60_mean": "MAP avg last 1 hr: {value:.0f} mmHg",
    "resp_rate__w15_mean": "Respiratory rate avg last 15 min: {value:.0f} breaths/min",
    "resp_rate__w60_mean": "Respiratory rate avg last 1 hr: {value:.0f} breaths/min",
    "Resp__w60_mean": "Respiratory rate avg last 1 hr: {value:.0f} breaths/min",
    "spo2__w15_mean": "SpO₂ avg last 15 min: {value:.1f}%",
    "spo2__w60_mean": "SpO₂ avg last 1 hr: {value:.1f}%",
    "O2Sat__w60_mean": "SpO₂ avg last 1 hr: {value:.1f}%",
    "temperature__w60_mean": "Temperature avg last 1 hr: {value:.1f}°C",
    "Temp__w60_mean": "Temperature avg last 1 hr: {value:.1f}°C",
    "gcs_total__w60_mean": "GCS avg last 1 hr: {value:.0f}",
    "urine_output__w240_mean": "Urine output avg last 4 hr: {value:.0f} mL",
    "lactate__w240_mean": "Lactate last 4 hr: {value:.1f} mmol/L",
    "Lactate__w240_mean": "Lactate last 4 hr: {value:.1f} mmol/L",
    "wbc__w240_mean": "WBC last 4 hr: {value:.0f} ×10³/µL",
    "WBC__w240_mean": "WBC last 4 hr: {value:.0f} ×10³/µL",
    "creatinine__w240_mean": "Creatinine last 4 hr: {value:.2f} mg/dL",
    "Creatinine__w240_mean": "Creatinine last 4 hr: {value:.2f} mg/dL",
    "shock_index_60": "Shock index (HR/SBP) last 1 hr: {value:.2f}",
    "pulse_pressure_60": "Pulse pressure last 1 hr: {value:.0f} mmHg",
    "pf_ratio_proxy": "P/F ratio proxy: {value:.0f}",
    "lactate_rising": "Lactate rising flag: {value:.0f}",
}

_RISK_DIRECTION: dict[str, str] = {
    "heart_rate": "increases_risk",    # tachycardia = bad
    "sbp": "increases_risk",           # low SBP increases risk, but coef captures sign
    "resp_rate": "increases_risk",
    "spo2": "decreases_risk",          # high SpO2 is protective
    "temperature": "increases_risk",
    "gcs_total": "decreases_risk",
    "urine_output": "decreases_risk",
    "lactate": "increases_risk",
    "wbc": "increases_risk",
    "creatinine": "increases_risk",
    "shock_index_60": "increases_risk",
    "pulse_pressure_60": "decreases_risk",
    "pf_ratio_proxy": "decreases_risk",
}

# ---------------------------------------------------------------------------
# Clinical Abnormality Pathophysiological Justifications & Case Examples
# ---------------------------------------------------------------------------

CLINICAL_ABNORMALITY_JUSTIFICATIONS: dict[str, dict[str, Any]] = {
    "heart_rate": {
        "label": "Heart Rate / Tachycardia",
        "normal_range": "60–100 bpm",
        "abnormal_criteria": "HR > 100 bpm or acutely rising slope (+10 bpm/hr)",
        "pathophysiology": (
            "Systemic vasodilation, peripheral pooling, and pyrogenic cytokine release stimulate "
            "compensatory adrenergic tone. Sinus tachycardia maintains cardiac output in the face of "
            "reduced stroke volume and microvascular leakage."
        ),
        "examples": [
            "Case 1 (Post-Op Sepsis): 68yo male 3 days post-colectomy develops progressive tachycardia from 74 to 124 bpm over 3 hours. Occult anastomotic leak leads to septic vasodilation 4 hours prior to overt hypotension.",
            "Case 2 (Urosepsis): 79yo female with indwelling catheter presents with HR 118 bpm and rigors; compensatory tachycardia preserved MAP at 72 mmHg temporarily before decompensating into septic shock.",
            "Case 3 (Pneumosepsis): 54yo male with bacterial pneumonia shows persistent tachycardia (132 bpm) unresponsive to fluid boluses, heralding high cardiac output hyperdynamic septic state.",
        ],
    },
    "sbp": {
        "label": "Systolic Blood Pressure / Hypotension",
        "normal_range": "100–140 mmHg",
        "abnormal_criteria": "SBP < 90 mmHg or acute decrease > 40 mmHg from baseline",
        "pathophysiology": (
            "Endothelial dysfunction and inducible nitric oxide synthase (iNOS) release trigger "
            "profound arterial and arteriolar vasodilation, precipitating acute circulatory collapse."
        ),
        "examples": [
            "Case 1 (Septic Shock): 72yo female with cholangitis experiences sudden SBP collapse from 128 to 82 mmHg within 45 minutes; requires emergent norepinephrine infusion.",
            "Case 2 (Occult Sepsis): 61yo male post-chemotherapy exhibits gradual SBP decline from 115 to 88 mmHg over 4 hours, marking transition to decompensated distributive shock.",
            "Case 3 (Bacteremia): 83yo male with Staph aureus bacteremia presents with SBP 84 mmHg and cold extremities, reflecting refractory peripheral vascular tone failure.",
        ],
    },
    "map": {
        "label": "Mean Arterial Pressure (MAP)",
        "normal_range": "70–105 mmHg",
        "abnormal_criteria": "MAP < 65 mmHg",
        "pathophysiology": (
            "A MAP below 65 mmHg breaches the autoregulatory threshold for vital organ beds (renal cortex, "
            "splanchnic microcirculation), causing acute tubular necrosis, gut ischemia, and multi-organ failure."
        ),
        "examples": [
            "Case 1 (Renal Hypoperfusion): 66yo female with severe pyelonephritis shows MAP dropping to 54 mmHg; acute oliguria (<15 mL/hr) develops within 90 minutes due to loss of glomerular filtration pressure.",
            "Case 2 (Distributive Shock): 58yo male with necrotizing pancreatitis maintains MAP 58 mmHg despite 2L crystalloids, fulfilling Sepsis-3 definition for vasopressor initiation.",
            "Case 3 (Peritonitis): 74yo post-op patient drops MAP from 82 to 59 mmHg overnight; early intervention with norepinephrine restores perfusion pressure and halts rising lactate.",
        ],
    },
    "resp_rate": {
        "label": "Respiratory Rate / Tachypnea",
        "normal_range": "12–20 breaths/min",
        "abnormal_criteria": "Resp ≥ 22 breaths/min (qSOFA positive criterion)",
        "pathophysiology": (
            "Tachypnea is the earliest and most sensitive vital sign of clinical deterioration. Medullary chemoreceptors "
            "drive rapid breathing to expel CO₂ (respiratory alkalosis) and compensate for metabolic lactic acidosis, "
            "while capillary leak induces subclinical pulmonary edema."
        ),
        "examples": [
            "Case 1 (Metabolic Compensation): 71yo female with perforated diverticulitis breathes at 28 breaths/min to compensate for severe metabolic acidosis (serum bicarbonate 14 mEq/L, lactate 3.9 mmol/L).",
            "Case 2 (Early Deterioration): 80yo male with subtle aspiration exhibits tachypnea (26 breaths/min) 6 hours before fever or desaturation appears, illustrating tachypnea as the earliest EWS warning sign.",
            "Case 3 (ARDS Onset): 49yo female with influenza-associated secondary bacterial pneumonia exhibits respiratory rate accelerating to 34 breaths/min with accessory muscle use, preceding intubation.",
        ],
    },
    "spo2": {
        "label": "Oxygen Saturation (SpO₂)",
        "normal_range": "95–100%",
        "abnormal_criteria": "SpO₂ < 92% on room air or dropping > 3% acutely",
        "pathophysiology": (
            "Sepsis-induced systemic inflammation damages the alveolar-capillary membrane, causing microvascular "
            "thrombosis, surfactant breakdown, and ventilation-perfusion mismatch characteristic of early acute lung injury."
        ),
        "examples": [
            "Case 1 (ARDS Transition): 63yo male with abdominal sepsis desaturates from 96% to 88% on 2L nasal cannula; chest X-ray confirms bilateral non-cardiogenic pulmonary infiltrates.",
            "Case 2 (Severe Hypoxemia): 75yo female with hospital-acquired pneumonia develops sudden SpO₂ drop to 89% requiring high-flow nasal cannula (HFNC) and ICU transfer.",
            "Case 3 (Microvascular Shunting): 57yo male with meningococcal sepsis maintains SpO₂ 90% despite high FiO₂, driven by severe pulmonary shunting and septic microthrombi.",
        ],
    },
    "temperature": {
        "label": "Body Temperature",
        "normal_range": "36.5–37.5°C",
        "abnormal_criteria": "Temp > 38.3°C (fever) or < 36.0°C (hypothermia)",
        "pathophysiology": (
            "Pyrogens (IL-1, IL-6, TNF-α) reset hypothalamic thermoregulatory set-point triggering hyperthermia. "
            "Hypothermia (< 36.0°C) signals impaired thermogenesis, autonomic exhaustion, and is clinically associated "
            "with twice the mortality rate of febrile sepsis."
        ),
        "examples": [
            "Case 1 (Hypothermic Sepsis): 84yo frail nursing-home resident presents with core temperature 35.1°C, lethargy, and bradycardia. Blood cultures grow Klebsiella; hypothermia justifies high-risk escalation despite absence of fever.",
            "Case 2 (High Hyperpyrexia): 44yo male with acute pyelonephritis spikes temperature to 39.7°C with rigors, precipitating extreme sinus tachycardia (140 bpm) and hypermetabolic oxygen consumption.",
            "Case 3 (Post-Op Bacteremia): 69yo female post-orthopedic replacement develops sudden fever 38.9°C with shivering 48 hours post-op, leading to prompt blood culture identification of MRSA bacteremia.",
        ],
    },
    "lactate": {
        "label": "Serum Lactate",
        "normal_range": "0.5–1.6 mmol/L",
        "abnormal_criteria": "Lactate ≥ 2.0 mmol/L (hypoperfusion) or ≥ 4.0 mmol/L (severe cellular dysoxia)",
        "pathophysiology": (
            "Inadequate tissue microperfusion impairs mitochondrial oxidative phosphorylation, shifting cells to "
            "anaerobic glycolysis. Concurrently, hepatic lactate clearance is impaired and adrenergic stimulation of "
            "beta-2 receptors accelerates aerobic glycolysis."
        ),
        "examples": [
            "Case 1 (Occult Hypoperfusion / Cryptic Shock): 67yo male with normal BP (118/72) has screening lactate of 4.4 mmol/L; occult tissue hypoxia triggers immediate 30 mL/kg fluid protocol, preventing arrest.",
            "Case 2 (Refractory Shock): 73yo female with ischemic bowel exhibits lactate escalating from 2.1 to 6.2 mmol/L over 6 hours despite fluid resuscitation, heralding bowel necrosis.",
            "Case 3 (Resuscitation Clearance): 52yo male with pneumonia presents with lactate 3.5 mmol/L; clearance of 25% after 2 hours of fluid resuscitation confirms therapeutic response.",
        ],
    },
    "wbc": {
        "label": "White Blood Cell Count (WBC)",
        "normal_range": "4.5–11.0 ×10³/µL",
        "abnormal_criteria": "WBC > 12.0 ×10³/µL (leukocytosis) or < 4.0 ×10³/µL (leukopenia) or > 10% bands",
        "pathophysiology": (
            "Infection stimulates emergency granulopoiesis releasing mature and immature neutrophils (bandemia). "
            "Severe leukopenia (< 4.0 ×10³/µL) reflects widespread leukocyte margination to inflamed endothelial beds "
            "and marrow exhaustion, portending worse outcomes."
        ),
        "examples": [
            "Case 1 (Severe Leukocytosis): 59yo female with acute appendiceal perforation exhibits WBC 26.8 ×10³/µL with 15% bands, representing profound bone marrow inflammatory response.",
            "Case 2 (Fulminant Leukopenia): 77yo immunocompromised patient with pseudomonas bacteremia exhibits WBC drop to 1.8 ×10³/µL; extreme leukopenia justifies immediate broad-spectrum coverage.",
            "Case 3 (Left Shift): 63yo male with biliary sepsis presents with borderline WBC (11.2) but 22% immature band forms ('left shift'), unmasking active severe infection.",
        ],
    },
    "creatinine": {
        "label": "Serum Creatinine / Renal Function",
        "normal_range": "0.7–1.3 mg/dL",
        "abnormal_criteria": "Creatinine > 1.5 mg/dL or acute increase ≥ 0.3 mg/dL within 48h (KDIGO Stage 1+ AKI)",
        "pathophysiology": (
            "Renal microvascular vasoconstriction, microthrombi formation, and endothelial swelling reduce "
            "glomerular capillary hydrostatic pressure and filtration rate, leading to sepsis-induced Acute Kidney Injury."
        ),
        "examples": [
            "Case 1 (Septic AKI Stage 2): 70yo male with urosepsis sees baseline creatinine rise from 1.0 to 2.4 mg/dL over 24 hours with oliguria, qualifying for 2 SOFA renal failure points.",
            "Case 2 (Pre-Renal Azotemia): 82yo female with severe dehydration and pneumonia presents with creatinine 3.1 mg/dL and BUN 62 mg/dL; responsive to rapid crystalloid volume resuscitation.",
            "Case 3 (Multi-Organ Failure): 55yo male in septic shock develops anuric renal failure with creatinine 4.2 mg/dL, requiring continuous venovenous hemodiafiltration (CVVH).",
        ],
    },
    "platelets": {
        "label": "Platelet Count / Coagulation",
        "normal_range": "150–400 ×10³/µL",
        "abnormal_criteria": "Platelets < 100 ×10³/µL (thrombocytopenia) or dropping > 50% from baseline",
        "pathophysiology": (
            "Endothelial damage exposes tissue factor, triggering widespread microvascular thrombosis that consumes "
            "circulating platelets. Platelets also bind to activated neutrophils and endothelial cells, forming microaggregates."
        ),
        "examples": [
            "Case 1 (Consumptive Coagulopathy / DIC): 64yo female with meningococcemia sees platelet count plunge from 210k to 52k in 8 hours with petechiae, indicating Disseminated Intravascular Coagulation.",
            "Case 2 (SOFA Coagulation Score): 76yo male with peritonitis exhibits platelet count of 88k, contributing 2 points to the SOFA score and signaling advanced systemic inflammation.",
            "Case 3 (Sepsis-Induced Thrombocytopenia): 51yo male with necrotizing fasciitis drops platelets to 38k, increasing risk of catastrophic spontaneous hemorrhage.",
        ],
    },
    "shock_index": {
        "label": "Shock Index (HR / SBP)",
        "normal_range": "0.5–0.7",
        "abnormal_criteria": "Shock Index > 0.8 (moderate shock) or > 1.0 (severe occult circulatory collapse)",
        "pathophysiology": (
            "Shock Index detects occult hypovolemia before blood pressure fails. Normal compensatory vasoconstriction "
            "maintains SBP while heart rate accelerates; an elevated ratio exposes left ventricular stroke volume deficit."
        ),
        "examples": [
            "Case 1 (Cryptic Decompensation): 68yo female with HR 112 bpm and SBP 98 mmHg has Shock Index 1.14 (> 1.0); detects impending circulatory collapse 3 hours before overt hypotension.",
            "Case 2 (Fluid Responsiveness): 75yo male with SI 1.28 experiences drop to 0.76 after 1.5L crystalloid bolus, confirming stroke volume restoration and fluid responsiveness.",
            "Case 3 (ICU Triage Warning): 59yo post-op patient shows SI rising from 0.65 to 0.95 over 2 hours, signaling concealed septic vasodilation.",
        ],
    },
}


def _match_abnormal_category(feature: str) -> str | None:
    f = feature.lower()
    if any(k in f for k in ["heart_rate", "hr_"]):
        return "heart_rate"
    if "map" in f:
        return "map"
    if any(k in f for k in ["sbp", "systolic"]):
        return "sbp"
    if any(k in f for k in ["resp", "respiratory"]):
        return "resp_rate"
    if any(k in f for k in ["spo2", "o2sat"]):
        return "spo2"
    if any(k in f for k in ["temp", "temperature"]):
        return "temperature"
    if "lactate" in f:
        return "lactate"
    if "wbc" in f:
        return "wbc"
    if any(k in f for k in ["creatinine", "bun"]):
        return "creatinine"
    if "platelet" in f:
        return "platelets"
    if "shock_index" in f:
        return "shock_index"
    return None


def get_clinical_justification(feature: str, value: float | None = None) -> dict[str, Any]:
    """
    Returns clinical justification, normal reference range, and a representative
    randomized clinical case example justifying why this abnormal parameter correlates
    with sepsis-induced deterioration.
    """
    import random
    cat_key = _match_abnormal_category(feature)
    if not cat_key or cat_key not in CLINICAL_ABNORMALITY_JUSTIFICATIONS:
        return {
            "feature": feature,
            "label": feature.replace("__", " ").replace("_", " ").title(),
            "normal_range": "N/A",
            "pathophysiology": "Multi-vital composite interaction contributing to early warning score.",
            "random_example": "Case Example: Patient with multi-vital deviations showing systemic inflammatory activation.",
            "all_examples": [],
        }

    info = CLINICAL_ABNORMALITY_JUSTIFICATIONS[cat_key]
    selected_example = random.choice(info["examples"])
    return {
        "feature": feature,
        "category": cat_key,
        "label": info["label"],
        "normal_range": info["normal_range"],
        "abnormal_criteria": info["abnormal_criteria"],
        "pathophysiology": info["pathophysiology"],
        "random_example": selected_example,
        "all_examples": info["examples"],
    }


def _direction_from_contribution(contribution: float) -> str:
    return "increases_risk" if contribution > 0 else "decreases_risk"


def _plain_english(feature: str, value: float) -> str:
    template = _PLAIN_ENGLISH.get(feature)
    if template and not math.isnan(value):
        try:
            return template.format(value=value)
        except (KeyError, ValueError):
            pass
    # Fallback: prettify feature name
    name = feature.replace("__", " ").replace("_", " ").title()
    return f"{name}: {value:.2f}" if not math.isnan(value) else name


def explain_logistic_regression(
    features: dict[str, float],
    weights: dict[str, float],
    intercept: float = 0.0,
    top_k: int = 5,
) -> list[dict]:
    """
    Compute signed contributions: contribution_i = w_i * x_i.
    Returns top-k features by absolute contribution.
    """
    contributions = []
    for feat, val in features.items():
        if feat not in weights:
            continue
        if math.isnan(val):
            continue
        w = weights[feat]
        contrib = w * val
        just_info = get_clinical_justification(feat, val)
        contributions.append({
            "feature": feat,
            "value": round(val, 4),
            "weight": round(w, 4),
            "contribution": round(contrib, 4),
            "direction": _direction_from_contribution(contrib),
            "plain_english": _plain_english(feat, val),
            "reference_range": just_info.get("normal_range", "N/A"),
            "clinical_justification": just_info.get("pathophysiology", ""),
            "case_example": just_info.get("random_example", ""),
        })

    contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
    return contributions[:top_k]


# ---------------------------------------------------------------------------
# Hoeffding Tree explainer (decision path)
# ---------------------------------------------------------------------------

def explain_hoeffding_tree(
    model_pipeline,
    features: dict[str, float],
    top_k: int = 5,
) -> list[dict]:
    """
    Fall back to feature importances from the Hoeffding tree.
    River's HoeffdingAdaptiveTreeClassifier exposes feature importances.
    """
    inner = model_pipeline[-1] if hasattr(model_pipeline, "__getitem__") else model_pipeline
    results = []

    if hasattr(inner, "feature_importances"):
        importances = inner.feature_importances  # dict: feature → importance
        for feat, imp in sorted(importances.items(), key=lambda x: -x[1])[:top_k]:
            val = features.get(feat, float("nan"))
            just_info = get_clinical_justification(feat, val)
            results.append({
                "feature": feat,
                "value": round(val, 4) if not math.isnan(val) else None,
                "weight": round(imp, 4),
                "contribution": round(imp, 4),
                "direction": _direction_from_contribution(imp),
                "plain_english": _plain_english(feat, val) if not math.isnan(val) else feat,
                "reference_range": just_info.get("normal_range", "N/A"),
                "clinical_justification": just_info.get("pathophysiology", ""),
                "case_example": just_info.get("random_example", ""),
            })
    return results


# ---------------------------------------------------------------------------
# Baseline XGBoost / sklearn explainer
# ---------------------------------------------------------------------------

def explain_sklearn(
    model,
    feature_names: list[str],
    features: dict[str, float],
    top_k: int = 5,
) -> list[dict]:
    """
    Use model's coefficients or feature_importances_ for explanation.
    """
    inner = model
    if hasattr(model, "steps"):
        inner = model.steps[-1][1]

    contributions = []
    if hasattr(inner, "coef_"):
        coefs = inner.coef_[0]
        for feat, coef in zip(feature_names, coefs):
            val = features.get(feat, float("nan"))
            if math.isnan(val):
                continue
            contrib = coef * val
            just_info = get_clinical_justification(feat, val)
            contributions.append({
                "feature": feat,
                "value": round(val, 4),
                "weight": round(coef, 4),
                "contribution": round(contrib, 4),
                "direction": _direction_from_contribution(contrib),
                "plain_english": _plain_english(feat, val),
                "reference_range": just_info.get("normal_range", "N/A"),
                "clinical_justification": just_info.get("pathophysiology", ""),
                "case_example": just_info.get("random_example", ""),
            })
    elif hasattr(inner, "feature_importances_"):
        imps = inner.feature_importances_
        for feat, imp in zip(feature_names, imps):
            val = features.get(feat, float("nan"))
            if math.isnan(val):
                continue
            just_info = get_clinical_justification(feat, val)
            contributions.append({
                "feature": feat,
                "value": round(val, 4),
                "weight": round(imp, 4),
                "contribution": round(imp, 4),
                "direction": _direction_from_contribution(imp),
                "plain_english": _plain_english(feat, val),
                "reference_range": just_info.get("normal_range", "N/A"),
                "clinical_justification": just_info.get("pathophysiology", ""),
                "case_example": just_info.get("random_example", ""),
            })

    contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
    return contributions[:top_k]


# ---------------------------------------------------------------------------
# Summary sentence generator
# ---------------------------------------------------------------------------

def generate_summary(top_features: list[dict], risk_score: float) -> str:
    if not top_features:
        return f"Risk score: {risk_score:.0%}. Insufficient feature data for explanation."

    risk_level = "High" if risk_score >= 0.65 else "Moderate" if risk_score >= 0.4 else "Low"
    primary = top_features[0]
    secondary = top_features[1] if len(top_features) > 1 else None

    summary = f"{risk_level} deterioration risk ({risk_score:.0%}). "
    summary += f"Primary driver: {primary['plain_english']}."
    if secondary:
        summary += f" Also notable: {secondary['plain_english']}."
    if primary.get("clinical_justification"):
        summary += f" Clinical justification: {primary['clinical_justification']}"
    if primary.get("case_example"):
        summary += f" [{primary['case_example']}]"
    return summary


# ---------------------------------------------------------------------------
# Main Explainer class
# ---------------------------------------------------------------------------

class Explainer:
    """
    Unified explainability interface.

    Usage:
        explainer = Explainer(online_model, baseline_model)
        explanation = explainer.explain(features, risk_score, model_used="online")
    """

    def __init__(
        self,
        online_model=None,
        baseline_model=None,
        top_k: int = 5,
        cfg_path: str = "config/settings.yaml",
    ):
        self.online = online_model
        self.baseline = baseline_model
        self.top_k = top_k

    def explain(
        self,
        features: dict[str, float],
        risk_score: float,
        model_used: str = "online",
    ) -> dict[str, Any]:
        """
        Generate explanation for a single prediction.

        Returns:
            {
              top_features: [...],
              summary: str,
              model_used: str
            }
        """
        top_features: list[dict] = []

        if model_used == "online" and self.online is not None:
            mt = self.online.model_type
            if mt == "logistic_regression":
                weights = self.online.get_feature_weights()
                top_features = explain_logistic_regression(
                    features, weights, top_k=self.top_k
                )
            elif mt == "hoeffding_adaptive_tree":
                top_features = explain_hoeffding_tree(
                    self.online._model, features, top_k=self.top_k
                )

        elif model_used == "baseline" and self.baseline is not None:
            feat_imp = self.baseline.get_feature_importance()
            if feat_imp:
                contributions = []
                for feat, coef_or_imp in feat_imp.items():
                    val = features.get(feat, float("nan"))
                    if not math.isnan(val):
                        contrib = coef_or_imp * val
                        just_info = get_clinical_justification(feat, val)
                        contributions.append({
                            "feature": feat,
                            "value": round(val, 4),
                            "weight": round(coef_or_imp, 4),
                            "contribution": round(contrib, 4),
                            "direction": _direction_from_contribution(contrib),
                            "plain_english": _plain_english(feat, val),
                            "reference_range": just_info.get("normal_range", "N/A"),
                            "clinical_justification": just_info.get("pathophysiology", ""),
                            "case_example": just_info.get("random_example", ""),
                        })
                contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
                top_features = contributions[: self.top_k]

        summary = generate_summary(top_features, risk_score)

        return {
            "top_features": top_features,
            "summary": summary,
            "model_used": model_used,
            "risk_score": round(risk_score, 4),
        }
