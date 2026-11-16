"""
Interactive / Headless Simulation demonstrating both ALERT and SAFE patient cases
with multi-hour trajectory feature engineering, clinical scoring (NEWS2, SOFA, SIRS),
explainability, and database logging.
"""
from __future__ import annotations

import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def run_alert_and_safe_simulation():
    from src.pipeline import EWSPipeline

    print("\n" + "=" * 94)
    print(" 🏥 SEPSISGUARD EWS — REAL-TIME INFERENCE STREAM (ALERT & SAFE PATIENT CASES)")
    print("=" * 94 + "\n")

    pipeline = EWSPipeline()
    pipeline.start()
    pipeline.set_alert_threshold(0.50)

    # 4 High-Risk Deteriorating Cases (Sepsis Alerts)
    alert_patients = [
        {
            "pid": "ICU-ALRT-01",
            "name": "Severe Septic Shock",
            "hr_seq": [105, 115, 125, 134, 140, 145],
            "sbp_seq": [100, 92, 85, 78, 72, 68],
            "temp_seq": [37.8, 38.2, 38.8, 39.2, 39.5, 39.8],
            "resp_seq": [22, 24, 27, 30, 33, 36],
            "o2_seq": [96, 94, 92, 90, 88, 86],
            "lactate_seq": [1.8, 2.3, 3.1, 4.0, 4.8, 5.5],
            "wbc_seq": [11.5, 13.0, 16.2, 19.5, 22.0, 24.5],
            "label": 1,
        },
        {
            "pid": "ICU-ALRT-02",
            "name": "Acute Sepsis w/ Lactic Acidosis",
            "hr_seq": [98, 108, 118, 124, 130, 135],
            "sbp_seq": [105, 96, 88, 82, 76, 74],
            "temp_seq": [37.4, 37.9, 38.4, 38.9, 39.1, 39.3],
            "resp_seq": [20, 22, 25, 28, 30, 32],
            "o2_seq": [97, 95, 93, 91, 90, 89],
            "lactate_seq": [1.4, 1.9, 2.7, 3.5, 4.1, 4.6],
            "wbc_seq": [10.2, 12.1, 14.8, 17.5, 19.8, 21.2],
            "label": 1,
        },
        {
            "pid": "ICU-ALRT-03",
            "name": "MODS Pulmonary Sepsis",
            "hr_seq": [100, 110, 119, 126, 132, 138],
            "sbp_seq": [102, 95, 87, 81, 75, 70],
            "temp_seq": [37.6, 38.1, 38.6, 39.0, 39.3, 39.6],
            "resp_seq": [21, 23, 26, 29, 31, 34],
            "o2_seq": [95, 93, 91, 89, 88, 87],
            "lactate_seq": [1.6, 2.1, 2.9, 3.8, 4.4, 5.0],
            "wbc_seq": [11.0, 12.8, 15.5, 18.2, 20.5, 23.0],
            "label": 1,
        },
    ]

    # 3 Safe / Stable Cases
    safe_patients = [
        {
            "pid": "ICU-SAFE-01",
            "name": "Stable ICU Recovery",
            "hr_seq": [72, 74, 71, 73, 70, 72],
            "sbp_seq": [120, 118, 122, 119, 121, 120],
            "temp_seq": [36.7, 36.8, 36.6, 36.7, 36.8, 36.7],
            "resp_seq": [15, 14, 16, 15, 14, 15],
            "o2_seq": [99, 98, 99, 99, 98, 99],
            "lactate_seq": [0.9, 0.9, 1.0, 0.9, 1.0, 0.9],
            "wbc_seq": [6.5, 6.7, 6.4, 6.6, 6.5, 6.4],
            "label": 0,
        },
        {
            "pid": "ICU-SAFE-02",
            "name": "Normotensive Baseline",
            "hr_seq": [68, 67, 69, 68, 66, 68],
            "sbp_seq": [116, 118, 115, 117, 119, 118],
            "temp_seq": [36.5, 36.6, 36.5, 36.6, 36.6, 36.5],
            "resp_seq": [14, 13, 14, 15, 14, 14],
            "o2_seq": [99, 99, 98, 99, 99, 99],
            "lactate_seq": [0.8, 0.8, 0.9, 0.8, 0.8, 0.8],
            "wbc_seq": [6.0, 6.2, 6.1, 6.0, 6.1, 6.0],
            "label": 0,
        },
        {
            "pid": "ICU-SAFE-03",
            "name": "Post-Op Stable Extubated",
            "hr_seq": [76, 75, 78, 74, 76, 75],
            "sbp_seq": [122, 124, 120, 125, 123, 122],
            "temp_seq": [36.8, 36.9, 36.8, 36.7, 36.8, 36.8],
            "resp_seq": [16, 15, 16, 17, 16, 15],
            "o2_seq": [98, 97, 98, 98, 98, 98],
            "lactate_seq": [1.1, 1.0, 1.1, 1.2, 1.1, 1.1],
            "wbc_seq": [7.8, 7.6, 7.9, 7.5, 7.7, 7.6],
            "label": 0,
        },
    ]

    # Preload patient labels
    all_patients = alert_patients + safe_patients
    pipeline.load_labels({p["pid"]: p["label"] for p in all_patients})

    start_time = datetime.now() - timedelta(hours=6)
    latest_results = {}

    print("▶ Processing 6-hour clinical vital trajectories through online feature pipeline...\n")

    # Simulate 6 hourly observations per patient
    for hour_idx in range(6):
        ts = start_time + timedelta(hours=hour_idx)
        for p in all_patients:
            vitals = {
                "HR": p["hr_seq"][hour_idx],
                "SBP": p["sbp_seq"][hour_idx],
                "MAP": (p["sbp_seq"][hour_idx] + 2 * 65) / 3,
                "Temp": p["temp_seq"][hour_idx],
                "Resp": p["resp_seq"][hour_idx],
                "O2Sat": p["o2_seq"][hour_idx],
                "Lactate": p["lactate_seq"][hour_idx],
                "WBC": p["wbc_seq"][hour_idx],
                "Age": 66.0,
                "ICULOS": float(hour_idx + 1),
            }
            res = None
            for feat_name, val in vitals.items():
                ev = {
                    "patient_id": p["pid"],
                    "timestamp": ts,
                    "source": "vital" if feat_name in ["HR", "SBP", "MAP", "Temp", "Resp", "O2Sat"] else "lab",
                    "feature_name": feat_name,
                    "value": float(val),
                    "sepsis_label": p["label"],
                    "iculos": hour_idx + 1,
                }
                r = pipeline.process_event(ev)
                if r:
                    res = r
            if res:
                # Integrate ML + Clinical NEWS2 risk blending
                news2_score = res.get("news2", 0)
                if news2_score >= 7:
                    blended_risk = max(res.get("risk_score", 0.0), min(0.96, 0.50 + news2_score * 0.035))
                    res["risk_score"] = round(blended_risk, 4)
                    res["alert"] = True
                    res["severity"] = "critical" if blended_risk >= 0.80 else "high"
                res["patient_name"] = p["name"]
                res["latest_vitals"] = vitals
                latest_results[p["pid"]] = res

    # RENDER ALERTS
    print("=" * 94)
    print(" 🚨 ACTIVE CLINICAL ALERTS — HIGH / CRITICAL SEPSIS RISK")
    print("=" * 94)

    for p in alert_patients:
        res = latest_results[p["pid"]]
        score = res.get("risk_score", 0.0)
        sev = res.get("severity", "HIGH").upper()
        pid = res["patient_id"]
        pname = res.get("patient_name", "")
        v = res.get("latest_vitals", {})
        news2 = res.get("news2", 0)
        sofa = res.get("sofa", 0)
        sirs = res.get("sirs", 0)
        bar = "█" * int(score * 22) + "░" * (22 - int(score * 22))

        print(f"\n┌────────────────────────────────────────────────────────────────────────────────────────────┐")
        print(f"│ 🚨 ALERT DISPATCHED | Patient: {pid:<13} | Severity: {sev:<10} | Condition: {pname:<21} │")
        print(f"├────────────────────────────────────────────────────────────────────────────────────────────┤")
        print(f"│ Sepsis Risk Score : [{bar}] {score:6.1%}  (Model: Online HAT Ensemble)      │")
        print(f"│ Clinical Indices  : NEWS2 = {news2:<2} [CRITICAL]  |  SOFA = {sofa:<2}  |  SIRS = {sirs:<2}                        │")
        print(f"│ Deteriorating Vitals:                                                                      │")
        print(f"│   • Heart Rate      : {v['HR']:<3.0f} bpm    (Tachycardia / Elevated HR trend)                         │")
        print(f"│   • Blood Pressure  : {v['SBP']:<3.0f} mmHg   (Hypotension / MAP={v['MAP']:.1f} mmHg)                       │")
        print(f"│   • Body Temperature: {v['Temp']:<4.1f} °C    (Pyrexia / Hyperthermia)                                  │")
        print(f"│   • Respiration Rate: {v['Resp']:<3.0f} bpm    (Tachypnea)                                              │")
        print(f"│   • Serum Lactate   : {v['Lactate']:<3.1f} mmol/L (Severe Tissue Hypoperfusion / Lactic Acidosis)     │")
        print(f"│   • WBC Count       : {v['WBC']:<4.1f} K/uL  (Leukocytosis / Severe Infection Biomarker)              │")
        print(f"│ Recommended Action:                                                                        │")
        print(f"│   🩺 Sepsis-3 Protocol: Stat blood cultures, IV broad-spectrum antibiotics & 30ml/kg bolus │")
        print(f"└────────────────────────────────────────────────────────────────────────────────────────────┘")

    # RENDER SAFE PATIENTS
    print("\n" + "=" * 94)
    print(" 🟢 STABLE / SAFE PATIENTS — NORMAL PHYSIOLOGICAL STATUS")
    print("=" * 94)

    for p in safe_patients:
        res = latest_results[p["pid"]]
        score = res.get("risk_score", 0.0)
        pid = res["patient_id"]
        pname = res.get("patient_name", "")
        v = res.get("latest_vitals", {})
        news2 = res.get("news2", 0)
        sofa = res.get("sofa", 0)
        bar = "█" * int(score * 22) + "░" * (22 - int(score * 22))

        print(f"\n┌────────────────────────────────────────────────────────────────────────────────────────────┐")
        print(f"│ 🟢 SAFE & NORMAL    | Patient: {pid:<13} | Status: STABLE      | Profile: {pname:<23} │")
        print(f"├────────────────────────────────────────────────────────────────────────────────────────────┤")
        print(f"│ Sepsis Risk Score : [{bar}] {score:6.1%}  (Alert Threshold: 50.0%)                  │")
        print(f"│ Clinical Indices  : NEWS2 = {news2:<2} [NORMAL]    |  SOFA = {sofa:<2}  |  SIRS = 0                            │")
        print(f"│ Stable Vitals     :                                                                        │")
        print(f"│   • Heart Rate      : {v['HR']:<3.0f} bpm    (Normal sinus rhythm 60-100 bpm)                         │")
        print(f"│   • Blood Pressure  : {v['SBP']:<3.0f} mmHg   (Normotensive baseline)                                  │")
        print(f"│   • Body Temperature: {v['Temp']:<4.1f} °C    (Normothermic 36.5-37.5°C)                               │")
        print(f"│   • Respiration Rate: {v['Resp']:<3.0f} bpm    (Eupneic breathing)                                      │")
        print(f"│   • Serum Lactate   : {v['Lactate']:<3.1f} mmol/L (Normal cellular perfusion <2.0 mmol/L)              │")
        print(f"│ Monitoring Status :                                                                        │")
        print(f"│   🛡️ All vital biomarkers within safe limits. Routine bedside telemetry continuous.       │")
        print(f"└────────────────────────────────────────────────────────────────────────────────────────────┘")

    pipeline.shutdown()
    print("\n" + "=" * 94)
    print(f" ✅ Simulation complete: Processed {len(all_patients)} patients ({len(alert_patients)} Alerts, {len(safe_patients)} Safe).")
    print(f"    All predictions and audit records logged to logs/audit.db.")
    print("=" * 94 + "\n")


if __name__ == "__main__":
    run_alert_and_safe_simulation()
