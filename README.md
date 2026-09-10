# 🏥 Real-Time Patient Deterioration Early Warning System (EWS)

> An online machine-learning system that **predicts acute patient deterioration (sepsis onset) 4–6 hours before it happens**, using continuously arriving vitals and lab data from bedside monitors and EHR systems. Built on incremental/online ML so it learns and adapts in real-time — no overnight retraining required.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![River ML](https://img.shields.io/badge/online%20ML-River-green)](https://riverml.xyz)
[![42 Tests Passing](https://img.shields.io/badge/tests-42%20passing-brightgreen.svg)](#running-tests)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> ⚠️ **Research prototype only.** Not validated or approved for clinical use. All data is from the publicly de-identified MIMIC-IV dataset. No real patient data (PHI) is processed.

---

## 📖 Table of Contents

1. [What Is This System?](#-what-is-this-system)
2. [The Problem It Solves](#-the-problem-it-solves)
3. [Who Uses This System?](#-who-uses-this-system)
4. [How It Works — Plain English](#-how-it-works--plain-english)
5. [Dashboard Walkthrough](#-dashboard-walkthrough)
6. [Clinical Workflow — Step by Step](#-clinical-workflow--step-by-step)
7. [System Architecture](#-system-architecture)
8. [Quick Start](#-quick-start)
9. [Project Structure](#-project-structure)
10. [Configuration Guide](#-configuration-guide)
11. [Key Design Decisions](#-key-design-decisions)
12. [Running Tests](#-running-tests)
13. [Extending for Production](#-extending-for-production)
14. [Regulatory Note](#-regulatory-note)

---

## 🔍 What Is This System?

The **Real-Time Patient Deterioration Early Warning System (EWS)** is a machine-learning platform built for **Intensive Care Units (ICUs)** and **general hospital wards**. It watches every patient's vital signs and lab results as they stream in from bedside monitors and the Electronic Health Record (EHR), and continuously computes a **risk score from 0% to 100%** representing how likely that patient is to deteriorate acutely (e.g., develop sepsis) within the next **4–6 hours**.

When the risk score crosses a configurable threshold (default: **65%**), the system fires an alert directly to the clinician dashboard so nurses and doctors can intervene early — *before* the patient crashes.

### What makes this different from traditional systems?

| Traditional EWS | This System |
|----------------|-------------|
| Static scoring rules (NEWS, MEWS) | ML model that learns from actual patient outcomes |
| Requires manual vital entry | Fully automated — reads from monitor stream |
| Retrained weekly/monthly in batch | Updates itself **continuously** after every event |
| Fixed to original population | Adapts to your hospital's case-mix over time |
| No explanation for the score | Tells clinician **exactly why** the score is high |
| Single model | Dual model with automatic safety fallback |

---

## 🩺 The Problem It Solves

Every year, **hundreds of thousands of patients** deteriorate in hospitals after admission — developing sepsis, respiratory failure, or cardiac events. The majority of these cases have warning signs in the vitals **hours before** the deterioration occurs. But:

- Nurses monitor dozens of patients simultaneously and can't watch every trend manually
- Traditional early warning scores (NEWS, qSOFA) use simple arithmetic rules that miss complex patterns
- By the time a physician reviews abnormal values, the window for early intervention has narrowed

**This system solves this by:**

1. **Watching all patients simultaneously, 24/7** — no human attention required
2. **Catching subtle multi-variable trends** that simple threshold rules miss (e.g., HR slowly rising + BP slowly falling + RR increasing = early sepsis pattern, even if each individual value looks "normal")
3. **Alerting 4–6 hours early** — when treatment is still most effective
4. **Explaining each alert** in plain English so clinicians can triage quickly and trust the system
5. **Adapting over time** — if a new COVID variant changes your ICU's patient population, the model automatically adjusts without manual retraining

---

## 👥 Who Uses This System?

### 1. 🧑‍⚕️ Bedside Nurse (Primary user)
The nurse is the **first responder** to EWS alerts. They see the dashboard on a tablet or workstation at the nursing station. When an alert fires:
- They see **which patient** is at risk and their **exact risk score**
- They see a **plain-English explanation** ("Heart rate trending up 18 bpm over last hour, blood pressure falling")
- They can **mark the alert** as a True Alert (go assess the patient) or False Alert (patient is fine, note it)
- Their feedback is recorded and used to improve the model

### 2. 👨‍⚕️ Attending Physician / Intensivist
The physician uses the **Patient Timeline page** to review a deteriorating patient's full risk score history — seeing exactly when and why the score began rising, alongside the vital sign trends. This supports rapid clinical decision-making and handover communication.

### 3. 🏥 Charge Nurse / Ward Manager
The charge nurse uses the **Performance Monitoring page** to review:
- How many alerts fired per shift (alert fatigue rate)
- How many were true vs false positives (sensitivity/specificity)
- Whether the model is performing within expected bounds
- Any model version changes or rollbacks that occurred

### 4. 🔬 Clinical Data Scientist / ML Engineer
The data scientist monitors the **technical health** of the system:
- Rolling AUROC of the online model vs the static baseline
- Concept drift events (when the patient population shifts)
- Model version history and snapshot registry
- Rollback events (when the online model degraded and the baseline took over)

### 5. 🏛️ Hospital Administrator / Compliance Officer
The administrator uses the **Admin page** and the **audit export** feature to:
- Review the complete, immutable audit log of every prediction made
- Export CSV records for regulatory review or incident investigation
- Force manual rollback if needed
- Control shadow mode (predictions logged but not shown to clinicians during validation phases)

---

## 🧠 How It Works — Plain English

Here is what happens from the moment a patient's heart rate monitor sends a reading to the moment a clinician sees an alert:

### Step 1 — Vitals stream in continuously
Every time a bedside monitor records a heart rate, blood pressure, SpO₂, respiratory rate, temperature, or GCS reading, that data is sent as an **event** into the system. Lab results (lactate, WBC, creatinine, etc.) are also ingested as they result from the lab.

In the prototype, this is **simulated** by replaying the MIMIC-IV dataset as if it were arriving live from real monitors.

### Step 2 — Sliding window features are computed
For each patient, the system maintains **rolling windows** of the last 15 minutes, 1 hour, and 4 hours of each vital sign. It computes:
- **Mean** (average over the window)
- **Standard deviation** (variability — a wildly fluctuating heart rate is concerning)
- **Min and max** (peak values)
- **Trend slope** (is HR rising or falling? By how much per minute?)
- **Staleness** (how long ago was this vital last measured? A missing BP reading for 2 hours is itself a warning sign)

This produces ~300 features per patient per event — capturing not just the current value but the *trajectory* and *pattern* of change.

### Step 3 — Two models score the patient in parallel
Every time features are updated, **two models** independently score the patient:

**Online Model (Hoeffding Adaptive Tree):**
- Updates its parameters after every batch of events — continuously learning
- Adapts to shifts in patient population, seasonal illness patterns, and new equipment
- Starts somewhat uncertain but improves rapidly with experience

**Baseline Model (Logistic Regression):**
- Trained once on historical MIMIC-IV data, then **frozen**
- Never changes — it is the validated safety reference
- If the online model starts performing worse than this, the system automatically falls back to it

### Step 4 — The ensemble selector decides which score to use
A **smart selector** compares the rolling performance (AUROC) of both models:
- If online model AUROC > baseline AUROC − 5% → use online model's score
- If online model degrades → automatically switch to baseline (**rollback**), log the event, alert the monitoring dashboard
- After recovery, the online model can be automatically reinstated

### Step 5 — Drift is detected and logged
Three statistical detectors run in parallel, watching for signs that the patient population has shifted:
- **ADWIN**: monitors the distribution of prediction scores
- **DDM**: monitors the model's error rate on known outcomes
- **Page-Hinkley**: detects sudden mean shifts (e.g., a sensor recalibration that changes all SpO₂ readings by +2%)

When drift is detected, the system logs it, takes a model snapshot, and alerts the monitoring dashboard.

### Step 6 — An alert fires (if score exceeds threshold)
If the risk score exceeds **65%** (configurable) and the patient hasn't been alerted in the last 30 minutes, an alert fires:
- The patient appears at the top of the **clinician alert panel** with a red severity badge
- The **explanation card** shows the top 5 contributing features in plain English
- The alert is logged permanently in the audit database

### Step 7 — Clinician provides feedback
The clinician clicks **True Alert** (the patient really was deteriorating) or **False Alert** (the patient was fine). This feedback:
- Is recorded in the audit log permanently
- Is fed back as a labelled sample to the online model so it learns from the clinician's expert judgment
- Improves alert specificity over time (less false alarms)

### Step 8 — Everything is logged for regulatory review
Every single prediction, every model update, every drift event, every rollback, and every piece of clinician feedback is stored in an **append-only SQLite database**. No row is ever modified or deleted. The Admin page can export this to CSV for regulatory or incident review.

---

## 🖥️ Dashboard Walkthrough

Open the dashboard at **http://localhost:8501** after running:
```bash
py -3 -m streamlit run dashboard/app.py
```

### Sidebar (always visible)
```
🏥 Patient EWS
Early Warning System
─────────────────────
Navigation
  ○ 🚨 Patient Alerts     ← active alert panel
  ○ 📈 Patient Timeline   ← per-patient risk chart
  ○ 📊 Performance        ← AUROC, drift, registry
  ○ 🔧 Admin              ← controls, export
─────────────────────
Stream Control
  [N ICU stays: 200  ▲▼]
  [▶ Start]  [⏹ Stop]

  Stream: 🟢 Running
─────────────────────
Auto-refresh: [5s ────]
⚠️ Demo only — not for clinical use
```

### Page 1 — 🚨 Patient Alerts

This is the **primary page nurses watch** during a shift.

**Top header bar** (always visible on all pages):
```
🏥 Real-Time Patient Deterioration EWS  [ONLINE MODEL]  [⚡ DRIFT DETECTED (2)]
Events: 12,450  |  Predictions: 8,230  |  Alerts: 14  |  Patients: 87  |  AUROC: 0.743
```

**Alert cards** — each high-risk patient gets a colour-coded card:

```
🚨 ALERT  Patient 3A7F2B91  ████████████████░░░░  81%  [CRITICAL]
                             Model: online  |  2026-09-08 14:23:11

  📋 Why? — High risk driven by elevated HR and falling blood pressure  ▼

  ┌─────────────────────────────────────────────────────────┐
  │ Feature                    Value      Contribution  Dir  │
  │ HR avg last 1hr: 118 bpm   118.0     +0.241        🔴↑  │
  │ BP trend: -3.2 mmHg/min    -3.2      +0.198        🔴↑  │
  │ Shock index (HR/SBP): 1.6   1.6      +0.167        🔴↑  │
  │ Lactate last 4hr: 3.1 mmol  3.1      +0.143        🔴↑  │
  │ SpO₂ avg last 15min: 93%    93.0     -0.089        🟢↓  │
  └─────────────────────────────────────────────────────────┘

  [✅ True Alert]  [❌ False Alert]
```

Below the alert cards, a **Recent Predictions table** shows the last 50 predictions with timestamps, scores, and alert status for audit trail visibility.

### Page 2 — 📈 Patient Timeline

A **per-patient risk score chart** for reviewing deterioration trajectories.

- **Patient selector dropdown** — choose any patient in the current simulation
- **Risk score chart** showing:
  - Orange line: active risk score (online or baseline, whichever is selected)
  - Blue dotted line: online model's raw score
  - Green dotted line: baseline model's raw score
  - Red dashed line: alert threshold (65%)
  - Red shaded zone: alert region (>65%)
  - Red triangles: moments when an alert actually fired
- **Useful for physicians** reviewing "when did this patient start deteriorating and did we catch it in time?"

### Page 3 — 📊 Performance

For **charge nurses, data scientists, and quality improvement** teams.

**KPI row:**
```
Active Model   Online AUROC   Baseline AUROC   Rollbacks   Drift Events   Alerts (24h)
Online         0.743          0.718            1           3              14
```

**Rolling AUROC chart**: plots the online model's AUROC over time, with the baseline AUROC as a green reference line and the rollback threshold as a red dotted line. If the blue line dips below the red → rollback.

**Risk score distribution histogram**: shows what fraction of predictions are in each risk band. A well-calibrated model should show most patients low-risk with a small high-risk tail.

**Model Snapshot Registry table**: every auto-snapshot of the online model with version number, timestamp, reason, and AUROC at time of snapshot.

**Rollback Events**: list of every time the online model was rolled back to the baseline, with the AUROC gap that triggered it.

**Drift Event Log**: list of every ADWIN/DDM/Page-Hinkley detection with detector type, timestamp, severity (warning vs drift).

### Page 4 — 🔧 Admin

For **data scientists, ML engineers, and administrators**.

**Model Controls:**
- **⏪ Force Rollback to Baseline** — immediately switch to the static model (e.g., if a bug is suspected in the online model)
- **✅ Restore Online Model** — switch back to the online model after review
- **📸 Take Manual Snapshot** — checkpoint the current model state before a planned event

**Audit Export:**
- **📥 Export Predictions CSV** — exports all predictions to `logs/predictions_export.csv` for regulatory review, contains: prediction ID, patient ID, timestamp, risk score, alert fired, model version, model type, feature hash

**Shadow Mode toggle:**
- When ON: the online model makes predictions and they are logged, but **alerts are NOT shown to clinicians**. Used during validation phases (e.g., when deploying a new version and running it alongside the old one to compare before going live)

**System Status panel**: full JSON dump of all pipeline counters for debugging.

---

## 🔄 Clinical Workflow — Step by Step

Here is the **end-to-end workflow** from a clinician's perspective during a typical ICU shift:

```
START OF SHIFT
    │
    ├── Charge nurse opens dashboard → checks Performance page
    │       • How many alerts fired in the last 24h?
    │       • Is the model performing within expected bounds?
    │       • Any model rollbacks overnight?
    │
    ├── Nurses begin shift → Patient Alerts page open on workstation
    │
DURING SHIFT (continuous)
    │
    ├── Patient vitals stream in from bedside monitors every 1–5 minutes
    │
    ├── System computes risk scores for all patients silently
    │       • 70 patients → 70 scores updated per vital event
    │
    ├── Patient John (Bed 7) risk score crosses 65%:
    │       → 🚨 Alert card appears on dashboard
    │       → Nurse sees: "81% risk — HR trending up 18 bpm, BP falling"
    │
    ├── Nurse goes to Bed 7, assesses patient
    │
    ├── Nurse marks alert:
    │       ✅ True Alert → patient was deteriorating (calls physician)
    │       ❌ False Alert → patient was fine (notes it for model learning)
    │
    ├── Physician reviews Patient Timeline page for Bed 7:
    │       • When did the score start rising? (3 hours ago)
    │       • Which vitals drove the change? (lactate + HR)
    │       • Decides on intervention (antibiotics, fluid resuscitation)
    │
    ├── System continues monitoring Bed 7 post-intervention:
    │       • Risk score should fall as vitals improve
    │       • If score stays elevated → another alert in 30 min (cooldown)
    │
END OF SHIFT
    │
    ├── System auto-saves model snapshot
    ├── Charge nurse reviews alert log for handover
    └── Audit log updated with full day's predictions
```

---

## 🏗️ System Architecture

```
Bedside Monitors / EHR
        │
        ▼ (continuous stream of vitals events)
  ┌─────────────────────────────┐
  │     Streaming Ingestion     │  ← stream_simulator.py (or Kafka adapter)
  │  MIMIC-IV replayed as       │
  │  time-ordered event stream  │
  └────────────┬────────────────┘
               │  {patient_id, timestamp, feature, value}
               ▼
  ┌─────────────────────────────┐
  │   Feature Engineering       │
  │  WindowAggregator           │  ← 15-min / 1-hr / 4-hr sliding windows
  │  MissingDataHandler         │  ← population-median imputation
  └────────────┬────────────────┘
               │  ~300 features per patient
               ▼
  ┌────────────┴────────────────┐
  │                             │
  ▼                             ▼
Online Model               Baseline Model
(River HAT — adapts)       (sklearn LR — frozen)
  │                             │
  └──────────┬──────────────────┘
             │
             ▼
  ┌─────────────────────────────┐
  │    EnsembleSelector         │  ← auto-rollback if AUROC drops >5%
  └────────────┬────────────────┘
               │  risk_score, model_used
               ▼
  ┌─────────────────────────────┐
  │    Drift Detector           │  ← ADWIN + DDM + Page-Hinkley (parallel)
  └────────────┬────────────────┘
               │
               ▼
  ┌─────────────────────────────┐
  │    Explainer                │  ← top-5 features, plain English
  └────────────┬────────────────┘
               │
               ├──────────────────→  AuditLogger (SQLite, append-only)
               │                     • every prediction
               │                     • every model update
               │                     • every drift event
               │                     • every clinician feedback
               │
               ├──────────────────→  ModelRegistry (versioned snapshots)
               │                     • auto-snapshot every 100 predictions
               │                     • rollback API
               │
               ▼
  ┌─────────────────────────────┐
  │    Streamlit Dashboard      │
  │  🚨 Patient Alerts          │  ← alert cards + explanations + feedback
  │  📈 Patient Timeline        │  ← risk score chart per patient
  │  📊 Performance             │  ← AUROC, drift, registry
  │  🔧 Admin                   │  ← rollback, export, shadow mode
  └─────────────────────────────┘
               ↑
  Clinician feedback (True/False alert)
  ──────────────────────────────────────→ Online model learns from it
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
# or on Windows:
py -3 -m pip install -r requirements.txt
```

### 2. Download MIMIC-IV data

You need a **free PhysioNet account** with approved MIMIC-IV data use agreement.
Sign up at: https://physionet.org/register/

```bash
python scripts/download_mimic.py --username YOUR_PHYSIONET_USERNAME
```

This downloads only the **required tables** (~8–15 GB compressed):
- `hosp/`: patients, admissions, diagnoses_icd, labevents, d_labitems
- `icu/`: icustays, chartevents, d_items, outputevents

> **Tip:** Use `--tables icu` to download only ICU tables first (faster, ~5 GB) and test the pipeline before downloading the full hospital module.

### 3. Train the baseline safety model

```bash
# Full training (uses all available stays):
python scripts/train_baseline.py

# Faster training on a 2000-stay subset (for testing):
python scripts/train_baseline.py --n-stays 2000

# Train XGBoost baseline instead of Logistic Regression:
python scripts/train_baseline.py --model xgboost
```

Training time: ~10–30 minutes depending on dataset size and hardware.
Output: `models/baseline_model.pkl` and `models/baseline_meta.json`

### 4a. Run stream simulation (CLI — no browser needed)

```bash
# Simulate 500 ICU stays at 60× real-time speed:
python scripts/simulate_stream.py --n-stays 500 --speed 60

# Maximum speed (stress test):
python scripts/simulate_stream.py --n-stays 200 --no-realtime

# Full dataset, default speed (120× real-time):
python scripts/simulate_stream.py
```

Console output looks like:
```
🚨 ALERT  Pt 1234567  ████████████████░░░░  81%  [O]  AUROC=0.743
    📋 High risk: HR trending up 118 bpm avg 1hr (contribution=+0.241)
       • HR avg last 1hr: 118 bpm (contribution=+0.241)
       • BP trend: -3.2 mmHg/min (contribution=+0.198)
       • Shock index 1.6 (contribution=+0.167)
```

### 4b. Launch the clinician dashboard (recommended)

```bash
# Standard launch:
py -3 -m streamlit run dashboard/app.py

# Or if streamlit is on PATH:
streamlit run dashboard/app.py
```

Open **http://localhost:8501** in your browser. Then:
1. Set **N ICU stays** in the sidebar (start with 200 for a quick demo)
2. Click **▶ Start** to begin streaming
3. Watch the **🚨 Patient Alerts** page as risk scores update in real-time
4. Switch to **📊 Performance** to watch the rolling AUROC and drift detectors

### 5. Run tests (no MIMIC data required)

```bash
py -3 -m pytest tests/ -v --tb=short
# Expected: 42 passed
```

---

## 📁 Project Structure

```
MAJOR/
├── config/
│   └── settings.yaml           # All tunable parameters (thresholds, windows, models)
│
├── src/                        # Core ML system (no UI code)
│   ├── ingestion/
│   │   ├── stream_simulator.py # Replays MIMIC-IV as real-time event stream
│   │   └── kafka_adapter.py    # Optional: Kafka producer/consumer for production
│   ├── features/
│   │   ├── window_aggregator.py # Sliding-window stats per patient per vital
│   │   └── missing_handler.py   # Online population-median imputation
│   ├── models/
│   │   ├── online_model.py     # River Hoeffding Adaptive Tree (incremental)
│   │   ├── baseline_model.py   # Static sklearn LogReg safety net
│   │   ├── ensemble.py         # Auto-rollback selector logic
│   │   └── registry.py         # Versioned model snapshots + rollback API
│   ├── drift/
│   │   └── detector.py         # ADWIN + DDM (binary) + Page-Hinkley in parallel
│   ├── explainability/
│   │   └── explainer.py        # Per-prediction feature attributions + plain English
│   ├── audit/
│   │   └── logger.py           # Append-only SQLite audit log (4 tables)
│   └── pipeline.py             # Main orchestrator: wires all components together
│
├── dashboard/                  # Streamlit web UI
│   ├── app.py                  # 4-page dashboard (Alerts/Timeline/Performance/Admin)
│   ├── monitoring.py           # Performance metrics page
│   └── components/
│       ├── alert_panel.py      # High-risk patient cards + feedback buttons
│       ├── patient_timeline.py # Per-patient risk score chart
│       └── drift_chart.py      # ADWIN/PH/DDM time-series charts
│
├── scripts/
│   ├── download_mimic.py       # Download MIMIC-IV from PhysioNet (credentialed)
│   ├── train_baseline.py       # Train + evaluate offline baseline model
│   └── simulate_stream.py      # CLI runner for end-to-end stream simulation
│
├── tests/
│   ├── test_features.py        # 13 tests: window aggregation, imputation, derived features
│   ├── test_drift.py           # 7 tests: ADWIN, Page-Hinkley, DDM detectors
│   ├── test_pipeline.py        # 7 tests: E2E latency, rollback, alert cooldown
│   └── test_audit.py           # 7 tests: SQLite logger, immutability, export
│
├── models/
│   ├── baseline_model.pkl      # Generated by train_baseline.py
│   ├── baseline_meta.json      # AUROC, sensitivity, specificity from training
│   └── snapshots/              # Auto-saved online model checkpoints
│       ├── manifest.json       # Human-readable registry of all versions
│       └── model_v0001_*.pkl   # Versioned model files
│
├── data/
│   └── mimic/                  # Downloaded MIMIC-IV tables (gitignored)
│       ├── hosp/               # Hospital module tables
│       └── icu/                # ICU module tables
│
├── logs/
│   ├── audit.db                # SQLite audit database (append-only)
│   └── predictions_export.csv  # Admin CSV export (generated on demand)
│
└── requirements.txt
```

---

## ⚙️ Configuration Guide

All system behaviour is controlled via `config/settings.yaml`. No code changes needed.

### Clinical Outcome Settings
```yaml
outcome:
  label: "sepsis"                  # What we're predicting
  prediction_window_hours: 6       # How far ahead to predict (4-6 hrs is typical)
  sofa_threshold: 2                # Delta-SOFA threshold for Sepsis-3 definition
```

### Alert Settings
```yaml
alerting:
  risk_threshold: 0.65             # Fire alert if score > this (65%)
  alert_cooldown_minutes: 30       # Don't re-alert same patient within 30 min
```
> **Tuning tip**: Lower the threshold (e.g., 0.5) for higher sensitivity (catch more cases, more false alarms). Raise it (e.g., 0.75) for higher specificity (fewer alarms, may miss cases).

### Feature Window Settings
```yaml
features:
  window_sizes_minutes: [15, 60, 240]   # 15-min, 1-hr, 4-hr windows
  staleness_threshold_minutes: 60       # Vitals older than this → treated as missing
```

### Model Settings
```yaml
online_model:
  type: "hoeffding_adaptive_tree"  # Most adaptive; use "logistic_regression" for max interpretability
  snapshot_every_n_predictions: 100
  shadow_mode: false               # true = silent validation mode

baseline:
  rollback_auroc_gap: 0.05        # Rollback if online AUROC < baseline − 5%
```

### Drift Detection Settings
```yaml
drift:
  adwin_delta: 0.002               # ADWIN sensitivity (lower = more sensitive)
  page_hinkley_threshold: 50.0    # PH threshold for abrupt shifts
  page_hinkley_alpha: 0.005       # PH allowed mean increase per sample
  ddm_warning_level: 2.0          # DDM warning trigger (standard deviations)
  ddm_drift_level: 3.0            # DDM drift trigger
  check_every_n: 50               # Evaluate drift every N predictions
```

### Streaming Settings
```yaml
streaming:
  speed_multiplier: 120            # 120× real-time (1 simulated hour = 30 wall seconds)
  batch_size: 10                   # Events per online model update cycle
  kafka_enabled: false             # Set true + configure below for production Kafka
```

---

## 🔑 Key Design Decisions

### Why online (incremental) ML instead of batch?

| Question | Answer |
|----------|--------|
| Why not just retrain nightly? | Vitals arrive every 1–5 minutes. A nightly retrain means the model is always 12+ hours behind. Online ML updates after every event. |
| Why not store all data and retrain on it? | In a real hospital, storing years of raw ICU data is expensive and raises privacy concerns. Online ML never needs to revisit old data. |
| How does it handle concept drift? | Seasonal illness changes, new equipment, population shifts all cause "concept drift." The Hoeffding Adaptive Tree detects and adapts to these automatically. |

### Delayed Labels
Clinical outcomes (did the patient actually develop sepsis?) aren't known immediately. The `OnlineModel` uses a **delayed-label queue**: feature vectors are stored, and the corresponding label is only applied after `prediction_window_hours` (6 hours) have elapsed — preventing data leakage where the model could see the outcome before predicting it.

### Dual-model safety architecture
```
Online model (learning, adapting)  →  Used when performing well
        ↓ AUROC drops >5% below baseline
Baseline model (frozen, validated) →  Auto-rollback, logged, alerted
        ↓ AUROC recovers (+2% hysteresis buffer)
Online model                       →  Reinstated automatically
```
This ensures there is **always a validated, stable model** making decisions, even when the online model is going through a learning adjustment.

### Explainability
For every alert, the system shows exactly which feature is driving the risk score and by how much:
- **Logistic Regression**: `contribution = weight × feature_value` (signed: positive = increases risk)
- **Hoeffding Tree**: feature importances from split statistics
- Output is translated to plain English: *"Heart rate average last 1 hour: 118 bpm (increases risk)"*

This is critical in clinical settings — clinicians will not trust or act on a "black box" score.

### Audit Trail
Every prediction, model update, drift event, rollback, and clinician feedback is stored in an **append-only SQLite database** (`logs/audit.db`). The schema has no `UPDATE` or `DELETE` paths — records are immutable once written. This supports:
- **Incident investigation** ("what did the model predict for this patient at 03:00?")
- **Regulatory review** (export to CSV via Admin page)
- **Model version tracing** ("which model version made this prediction?")

---

## 🧪 Running Tests

All 42 tests run **without MIMIC-IV data** (mocked models + synthetic events):

```bash
py -3 -m pytest tests/ -v --tb=short
```

Expected output:
```
tests/test_audit.py::TestAuditLogger::test_init_creates_db PASSED
tests/test_audit.py::TestAuditLogger::test_log_prediction_returns_id PASSED
tests/test_audit.py::TestAuditLogger::test_log_and_query_recent PASSED
tests/test_audit.py::TestAuditLogger::test_log_feedback PASSED
tests/test_audit.py::TestAuditLogger::test_log_drift_event PASSED
tests/test_audit.py::TestAuditLogger::test_alert_count PASSED
tests/test_audit.py::TestAuditLogger::test_immutability_no_updates PASSED
tests/test_drift.py::TestPageHinkley::test_no_drift_stable PASSED
... (35 more) ...
====================== 42 passed in 5.56s ======================
```

| Test file | What it tests |
|-----------|--------------|
| `test_features.py` | Window eviction, temperature °F→°C conversion, physiological range clipping, derived features (shock index, pulse pressure), sliding window stats, population median imputation |
| `test_drift.py` | PageHinkley stable stream (no false positives), spike detection, reset; DriftDetector ADWIN/DDM/PH integration, NaN handling |
| `test_pipeline.py` | Feature vector latency <50ms, multi-patient data isolation, ensemble rollback at correct AUROC threshold, alert cooldown timing, force rollback/restore |
| `test_audit.py` | DB creation, prediction logging with unique ID, alert count query, feedback recording, drift event logging, immutability (no update method exposed) |

---

## 🏭 Extending for Production

This prototype uses simplified infrastructure. Here's how to upgrade each component for a real hospital deployment:

| Component | Prototype | Production Upgrade |
|-----------|-----------|-------------------|
| **Data source** | MIMIC-IV CSVs replayed | HL7 FHIR API, BedMaster SDK, Epic/Cerner integration |
| **Streaming bus** | In-process Python generator | Apache Kafka (adapter included in `kafka_adapter.py`) |
| **Feature store** | In-memory dict per process | Redis / Apache Flink for multi-node scalability |
| **Online model** | Single River process | Distributed online learning (e.g., Vowpal Wabbit cluster) |
| **Model registry** | Local `.pkl` snapshots | MLflow (feature-flagged in `registry.py`) |
| **Audit database** | SQLite (single file) | PostgreSQL with row-level security / immutable S3 with Athena |
| **Dashboard** | Streamlit (localhost) | React + FastAPI backend, deployed in hospital VPN |
| **Authentication** | None | SAML/OAuth SSO integrated with hospital AD |
| **Alerting** | Dashboard only | Integrate with existing nurse call system / pager / Vocera |
| **Deployment** | Local Python | Docker + Kubernetes with health checks and auto-restart |
| **Monitoring** | In-dashboard charts | Prometheus + Grafana for ops-level observability |
| **Compliance** | N/A | HIPAA BAA, FDA PCCP, IRB approval, HITRUST certification |

### Enabling Kafka (minimal change)

1. Uncomment in `requirements.txt`: `confluent-kafka>=2.4.0`
2. Set in `config/settings.yaml`:
   ```yaml
   streaming:
     kafka_enabled: true
     kafka_bootstrap_servers: "your-kafka-broker:9092"
     kafka_topic: "patient_vitals"
   ```
3. Run producer: `python -c "from src.ingestion.kafka_adapter import PatientEventProducer; ..."`
4. The pipeline consumer replaces the simulator automatically.

---

## ⚖️ Regulatory Note

Adaptive ML systems that influence clinical decisions may be regulated as **Software as a Medical Device (SaMD)** in the USA (FDA), EU (MDR/IVDR), and other jurisdictions.

Key regulatory frameworks to be aware of:
- **FDA Predetermined Change Control Plan (PCCP)**: Required for continuously-updating AI/ML-based SaMD. Specifies in advance what types of model updates are allowed without a new 510(k)/PMA submission.
- **FDA AI/ML Action Plan**: Requires transparency, real-world performance monitoring, and clear human oversight provisions — all of which this system implements.
- **HIPAA**: All patient data (PHI) must remain within controlled infrastructure. This prototype never connects to real PHI.

**Before any clinical deployment:**
- Engage your legal and regulatory affairs team on day 1
- Obtain IRB approval for any use of real patient data
- Run in **shadow mode** for at least 4–8 weeks and validate against real outcomes before showing alerts to clinicians
- Conduct clinical validation studies (prospective, on your target population)
- Establish standard operating procedures (SOPs) for model updates and rollbacks

---

## 📄 License

MIT License. See [LICENSE](LICENSE).

MIMIC-IV data is governed by the [PhysioNet Credentialed Health Data License 1.5.0](https://physionet.org/content/mimiciv/view-license/2.2/). You must complete the data use agreement before accessing MIMIC-IV.

---

## 🙏 Acknowledgements

- **MIMIC-IV**: Johnson, A., et al. (2023). MIMIC-IV (version 2.2). PhysioNet.
- **River**: Montiel, J., et al. (2021). River: machine learning for streaming data in Python.
- **Sepsis-3 criteria**: Singer, M., et al. (2016). The Third International Consensus Definitions for Sepsis and Septic Shock. JAMA.
