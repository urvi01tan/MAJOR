"""
Per-Prediction Explainability
==============================
Generates clinician-readable explanations for each risk prediction.

Strategies by model type:
  1. Logistic Regression (online or baseline):
     contribution_i = weight_i × feature_value_i
     Signed contributions show direction and magnitude.

  2. Hoeffding Adaptive Tree:
     Extracts the decision path taken by the current feature vector
     and reports the key splitting conditions.

  3. Baseline XGBoost (if used):
     Uses SHAP TreeExplainer (or approximate feature importance × value).

Output format per prediction:
  {
    "top_features": [
      {"feature": "heart_rate__w60_mean", "value": 118.5, "contribution": 0.24,
       "direction": "increases_risk", "plain_english": "HR trending up (118 bpm avg over 1hr)"},
      ...
    ],
    "summary": "High risk driven primarily by elevated heart rate and low blood pressure.",
    "model_type": "online_logistic_regression"
  }
"""

from __future__ import annotations

import math
from typing import Any

import yaml


def _load_cfg(cfg_path: str = "config/settings.yaml") -> dict:
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Feature → plain-English template
# ---------------------------------------------------------------------------

_PLAIN_ENGLISH: dict[str, str] = {
    "heart_rate__w15_mean": "HR avg last 15 min: {value:.0f} bpm",
    "heart_rate__w60_mean": "HR avg last 1 hr: {value:.0f} bpm",
    "heart_rate__w60_slope": "HR trend last 1 hr: {value:+.1f} bpm/min",
    "sbp__w60_mean": "Systolic BP avg last 1 hr: {value:.0f} mmHg",
    "sbp__w60_slope": "Systolic BP trend: {value:+.1f} mmHg/min",
    "map__w60_mean": "MAP avg last 1 hr: {value:.0f} mmHg",
    "resp_rate__w15_mean": "Respiratory rate avg last 15 min: {value:.0f} breaths/min",
    "resp_rate__w60_mean": "Respiratory rate avg last 1 hr: {value:.0f} breaths/min",
    "spo2__w15_mean": "SpO₂ avg last 15 min: {value:.1f}%",
    "spo2__w60_mean": "SpO₂ avg last 1 hr: {value:.1f}%",
    "temperature__w60_mean": "Temperature avg last 1 hr: {value:.1f}°C",
    "gcs_total__w60_mean": "GCS avg last 1 hr: {value:.0f}",
    "urine_output__w240_mean": "Urine output avg last 4 hr: {value:.0f} mL",
    "lactate__w240_mean": "Lactate last 4 hr: {value:.1f} mmol/L",
    "wbc__w240_mean": "WBC last 4 hr: {value:.0f} ×10³/µL",
    "creatinine__w240_mean": "Creatinine last 4 hr: {value:.2f} mg/dL",
    "shock_index_60": "Shock index (HR/SBP) last 1 hr: {value:.2f}",
    "pulse_pressure_60": "Pulse pressure last 1 hr: {value:.0f} mmHg",
    "pf_ratio_proxy": "P/F ratio proxy: {value:.0f}",
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


# ---------------------------------------------------------------------------
# Logistic Regression explainer (online model weights)
# ---------------------------------------------------------------------------

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
        contributions.append({
            "feature": feat,
            "value": round(val, 4),
            "weight": round(w, 4),
            "contribution": round(contrib, 4),
            "direction": _direction_from_contribution(contrib),
            "plain_english": _plain_english(feat, val),
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
            results.append({
                "feature": feat,
                "value": round(val, 4) if not math.isnan(val) else None,
                "weight": round(imp, 4),
                "contribution": round(imp, 4),
                "direction": _direction_from_contribution(imp),
                "plain_english": _plain_english(feat, val) if not math.isnan(val) else feat,
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
            contributions.append({
                "feature": feat,
                "value": round(val, 4),
                "weight": round(coef, 4),
                "contribution": round(contrib, 4),
                "direction": _direction_from_contribution(contrib),
                "plain_english": _plain_english(feat, val),
            })
    elif hasattr(inner, "feature_importances_"):
        imps = inner.feature_importances_
        for feat, imp in zip(feature_names, imps):
            val = features.get(feat, float("nan"))
            if math.isnan(val):
                continue
            contributions.append({
                "feature": feat,
                "value": round(val, 4),
                "weight": round(imp, 4),
                "contribution": round(imp, 4),
                "direction": _direction_from_contribution(imp),
                "plain_english": _plain_english(feat, val),
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
                        contributions.append({
                            "feature": feat,
                            "value": round(val, 4),
                            "weight": round(coef_or_imp, 4),
                            "contribution": round(contrib, 4),
                            "direction": _direction_from_contribution(contrib),
                            "plain_english": _plain_english(feat, val),
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
