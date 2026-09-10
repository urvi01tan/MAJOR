# 🏥 Real-Time Patient Deterioration Early Warning System (EWS)

> An online machine-learning system that **predicts acute patient deterioration (sepsis onset) 4–6 hours before it happens**, using continuously arriving vitals and lab data from bedside monitors and EHR systems. Built on incremental/online ML so it learns and adapts in real-time — no overnight retraining required.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![River ML](https://img.shields.io/badge/online%20ML-River-green)](https://riverml.xyz)
[![Dataset](https://img.shields.io/badge/dataset-Kaggle%20CinC%202019-orange)](https://www.kaggle.com/datasets/salikhussaini49/prediction-of-sepsis)
[![42 Tests Passing](https://img.shields.io/badge/tests-42%20passing-brightgreen.svg)](#running-tests)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> ⚠️ **Research prototype only.** Not validated or approved for clinical use. All data is from the publicly available PhysioNet/CinC Challenge 2019 dataset. No real patient data (PHI) is processed.

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
The system ingests data from the **PhysioNet/CinC Challenge 2019** dataset, replayed as if arriving live from real ICU monitors. Each hourly row from a patient's PSV file is emitted as a sequence of individual feature events — one per vital sign or lab value.

In a real hospital deployment, these events would come from bedside monitors via HL7 FHIR or BedMaster SDK.

### Step 2 — Sliding window features are computed
For each patient, the system maintains **rolling windows** of the last 1 hour, 4 hours, and 8 hours of each vital sign and lab value. It computes:
- **Mean** (average over the window)
- **Standard deviation** (variability — a wildly fluctuating heart rate is concerning)
- **Min and max** (peak values)
- **Trend slope** (is HR rising or falling? By how much per minute?)
- **Staleness** (how long ago was this vital last measured?)

This produces **~250+ features per patient per event** — capturing not just the current value but the *trajectory* and *pattern* of change.

### Step 3 — Two models score the patient in parallel
Every time features are updated, **two models** independently score the patient:

**Online Model (Hoeffding Adaptive Tree):**
- Updates its parameters after every batch of events — continuously learning
- Adapts to shifts in patient population, seasonal illness patterns, and new equipment
- Starts somewhat uncertain but improves rapidly with experience

**Baseline Model (Logistic Regression):**
- Trained once on historical PhysioNet CinC 2019 data, then **frozen**
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
- **Page-Hinkley**: detects sudden mean shifts

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
Every single prediction, every model update, every drift event, every rollback, and every piece of clinician feedback is stored in an **append-only SQLite database**. No row is ever modified or deleted.

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
  [N Patients: 200  ▲▼]
  [▶ Start]  [⏹ Stop]

  Stream: 🟢 Running
─────────────────────
Auto-refresh: [5s ────]
⚠️ Demo only — not for clinical use
```

### Page 1 — 🚨 Patient Alerts

This is the **primary page nurses watch** during a shift.

**Top header bar:**
```
🏥 Real-Time Patient Deterioration EWS  [ONLINE MODEL]  [⚡ DRIFT DETECTED (2)]
Events: 12,450  |  Predictions: 8,230  |  Alerts: 14  |  Patients: 87  |  AUROC: 0.743
```

**Alert cards** — each high-risk patient gets a colour-coded card:

```
🚨 ALERT  Patient p003A7F  ████████████████░░░░  81%  [CRITICAL]
                             Model: online  |  2026-09-10 14:23:11

  📋 Why? — High risk driven by elevated HR and rising Lactate  ▼

  ┌─────────────────────────────────────────────────────────┐
  │ Feature                      Value    Contribution  Dir  │
  │ HR avg last 1hr: 118 bpm     118.0   +0.241        🔴↑  │
  │ Lactate avg last 4hr: 3.1    3.1     +0.198        🔴↑  │
  │ Shock index (HR/SBP): 1.6   1.6     +0.167        🔴↑  │
  │ SBP trend: -3.2 mmHg/hr     -3.2    +0.143        🔴↑  │
  │ O2Sat avg last 1hr: 93%      93.0    -0.089        🟢↓  │
  └─────────────────────────────────────────────────────────┘

  [✅ True Alert]  [❌ False Alert]
```

### Page 2 — 📈 Patient Timeline
Per-patient risk score chart showing:
- Orange line: active risk score
- Blue dotted line: online model's raw score
- Green dotted line: baseline model's raw score
- Red dashed line: alert threshold (65%)
- Red triangles: moments when an alert actually fired

### Page 3 — 📊 Performance
KPI row, rolling AUROC chart, risk score distribution histogram, model snapshot registry, and drift event log.

### Page 4 — 🔧 Admin
Force rollback, restore online model, manual snapshot, CSV export, shadow mode toggle, system JSON.

---

## 🔄 Clinical Workflow — Step by Step

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
    ├── Patient data streams in (PhysioNet replay at 60× real-time)
    │
    ├── System computes risk scores for all patients silently
    │
    ├── Patient John (Bed 7) risk score crosses 65%:
    │       → 🚨 Alert card appears on dashboard
    │       → Nurse sees: "81% risk — HR trending up, Lactate rising"
    │
    ├── Nurse goes to Bed 7, assesses patient
    │
    ├── Nurse marks alert:
    │       ✅ True Alert → patient was deteriorating (calls physician)
    │       ❌ False Alert → patient was fine (notes it for model learning)
    │
    ├── System continues monitoring post-intervention
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
PhysioNet CinC 2019 PSV Files
        │
        ▼ (PSV rows replayed as hourly event stream)
  ┌─────────────────────────────┐
  │     Streaming Ingestion     │  ← stream_simulator.py (or Kafka adapter)
  │  ~40K patients replayed as  │
  │  time-ordered event stream  │
  └────────────┬────────────────┘
               │  {patient_id, timestamp, feature_name, value}
               ▼
  ┌─────────────────────────────┐
  │   Feature Engineering       │
  │  WindowAggregator           │  ← 1-hr / 4-hr / 8-hr sliding windows
  │  MissingDataHandler         │  ← online population-median imputation
  └────────────┬────────────────┘
               │  ~250 features per patient
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
               │
               ├──────────────────→  ModelRegistry (versioned snapshots)
               │
               ▼
  ┌─────────────────────────────┐
  │    Streamlit Dashboard      │
  │  🚨 Patient Alerts          │
  │  📈 Patient Timeline        │
  │  📊 Performance             │
  │  🔧 Admin                   │
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

### 2. Download the dataset (free — no credentials needed)

```bash
# Option A: Automatic via Kaggle API (recommended)
pip install kaggle
# Place your Kaggle API key at ~/.kaggle/kaggle.json
# Get it from: https://www.kaggle.com/account → "Create New Token"
python scripts/download_dataset.py

# Option B: Manual
python scripts/download_dataset.py --manual
# Follow the printed instructions to download from Kaggle
```

**Dataset:** [PhysioNet/CinC Challenge 2019 — Early Prediction of Sepsis](https://www.kaggle.com/datasets/salikhussaini49/prediction-of-sepsis)
- ~40,336 ICU patients, ~1.5M hourly rows
- 40 columns: 8 vitals + 26 labs + 6 demographics + SepsisLabel
- ~170 MB compressed (vs 13–15 GB for MIMIC-IV)
- **No PhysioNet account or data use agreement required**

### 3. Train the baseline safety model

```bash
# Full training (all ~40K patients):
python scripts/train_baseline.py

# Faster training on 2000 patients (for testing):
python scripts/train_baseline.py --n-patients 2000

# Train XGBoost baseline instead of Logistic Regression:
python scripts/train_baseline.py --model xgboost
```

Training time: ~5–15 minutes depending on patient count and hardware.
Output: `models/baseline_model.pkl` and `models/baseline_meta.json`

### 4a. Run stream simulation (CLI — no browser needed)

```bash
# Simulate 500 patients at 60× real-time speed:
python scripts/simulate_stream.py --n-patients 500 --speed 60

# Maximum speed (stress test):
python scripts/simulate_stream.py --n-patients 200 --no-realtime

# Full dataset, default speed (60× real-time):
python scripts/simulate_stream.py
```

Console output:
```
🚨 ALERT  Pt p003A7F  ████████████████░░░░  81%  [O]  AUROC=0.743
    📋 High risk: HR trending up 118 bpm avg 1hr (contribution=+0.241)
       • HR avg last 1hr: 118 bpm (contribution=+0.241)
       • Lactate avg 4hr: 3.1 mmol/L (contribution=+0.198)
       • Shock index 1.6 (contribution=+0.167)
```

### 4b. Launch the clinician dashboard (recommended)

```bash
py -3 -m streamlit run dashboard/app.py
```

Open **http://localhost:8501** in your browser. Then:
1. Set **N Patients** in the sidebar (start with 200 for a quick demo)
2. Click **▶ Start** to begin streaming
3. Watch the **🚨 Patient Alerts** page as risk scores update in real-time
4. Switch to **📊 Performance** to watch the rolling AUROC and drift detectors

### 5. Run tests (no dataset required)

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
│   │   ├── stream_simulator.py # Replays PhysioNet PSV files as real-time event stream
│   │   └── kafka_adapter.py    # Optional: Kafka producer/consumer for production
│   ├── features/
│   │   ├── window_aggregator.py # Sliding-window stats per patient per vital/lab
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
│   ├── download_dataset.py     # Download PhysioNet CinC 2019 from Kaggle (free)
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
│       ├── manifest.json
│       └── model_v0001_*.pkl
│
├── data/
│   └── physionet2019/          # PhysioNet CinC 2019 dataset (gitignored)
│       └── training/           # ~40,336 PSV files (one per patient)
│           ├── p000001.psv
│           ├── p000002.psv
│           └── ...
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
  label: "sepsis"
  prediction_window_hours: 6
```

### Alert Settings
```yaml
alerting:
  risk_threshold: 0.65             # Fire alert if score > this (65%)
  alert_cooldown_minutes: 30
```

### Feature Window Settings
```yaml
features:
  window_sizes_minutes: [60, 240, 480]   # 1-hr, 4-hr, 8-hr windows
  staleness_threshold_minutes: 120
```

### Model Settings
```yaml
online_model:
  type: "hoeffding_adaptive_tree"
  snapshot_every_n_predictions: 100
  shadow_mode: false

baseline:
  rollback_auroc_gap: 0.05        # Rollback if online AUROC < baseline − 5%
```

### Paths
```yaml
paths:
  data_dir: "data/physionet2019"
  training_dir: "data/physionet2019/training"   # PSV files live here
```

---

## 🔑 Key Design Decisions

### Why the PhysioNet CinC 2019 dataset?

| Question | Answer |
|----------|--------|
| Why not MIMIC-IV? | MIMIC-IV requires PhysioNet credentialing, 13–15 GB download, and complex multi-table joins. CinC 2019 is free, 170 MB, and pre-structured for streaming. |
| Why is it good for online ML? | Each patient's data is an hourly time series — perfect for simulating a live event stream. Labels are inline (SepsisLabel per row) — no separate derivation needed. |
| How many patients? | ~40,336 ICU patients, ~1.5M hourly rows, ~7% sepsis prevalence. |

### Delayed Labels
Clinical outcomes (did the patient actually develop sepsis?) aren't known immediately. The `OnlineModel` uses a **delayed-label queue**: feature vectors are stored, and the corresponding label is only applied after `prediction_window_hours` (6 hours) have elapsed.

### Dual-model safety architecture
```
Online model (learning, adapting)  →  Used when performing well
        ↓ AUROC drops >5% below baseline
Baseline model (frozen, validated) →  Auto-rollback, logged, alerted
        ↓ AUROC recovers (+2% hysteresis buffer)
Online model                       →  Reinstated automatically
```

### Explainability
For every alert, the system shows exactly which feature is driving the risk score:
- **Logistic Regression**: `contribution = weight × feature_value` (signed)
- **Hoeffding Tree**: feature importances from split statistics
- Output: *"Lactate average last 4 hours: 3.1 mmol/L (increases risk)"*

---

## 🧪 Running Tests

All 42 tests run **without the dataset** (mocked models + synthetic events):

```bash
py -3 -m pytest tests/ -v --tb=short
```

Expected output:
```
tests/test_audit.py::TestAuditLogger::test_init_creates_db PASSED
tests/test_audit.py::TestAuditLogger::test_log_prediction_returns_id PASSED
...
tests/test_features.py::TestWindowAggregator::test_single_observation_mean PASSED
tests/test_features.py::TestWindowAggregator::test_derived_shock_index PASSED
...
====================== 42 passed in 5.56s ======================
```

| Test file | What it tests |
|-----------|--------------|
| `test_features.py` | Window eviction, PhysioNet column names (HR/SBP/O2Sat), physiological clipping, derived features (shock index, P/F ratio, lactate_rising), sliding window stats, population median imputation |
| `test_drift.py` | PageHinkley stable stream, spike detection, reset; ADWIN/DDM/PH integration, NaN handling |
| `test_pipeline.py` | Feature vector latency <50ms, multi-patient isolation, ensemble rollback, alert cooldown, force rollback/restore |
| `test_audit.py` | DB creation, prediction logging, alert count, feedback recording, drift event logging, immutability |

---

## 🏭 Extending for Production

| Component | Prototype | Production Upgrade |
|-----------|-----------|-------------------|
| **Data source** | PhysioNet CinC 2019 PSV replay | HL7 FHIR API, BedMaster SDK, Epic/Cerner integration |
| **Streaming bus** | In-process Python generator | Apache Kafka (adapter included in `kafka_adapter.py`) |
| **Feature store** | In-memory dict per process | Redis / Apache Flink for multi-node scalability |
| **Online model** | Single River process | Distributed online learning (e.g., Vowpal Wabbit cluster) |
| **Model registry** | Local `.pkl` snapshots | MLflow |
| **Audit database** | SQLite (single file) | PostgreSQL with row-level security |
| **Dashboard** | Streamlit (localhost) | React + FastAPI backend, deployed in hospital VPN |
| **Authentication** | None | SAML/OAuth SSO integrated with hospital AD |
| **Alerting** | Dashboard only | Integrate with nurse call system / pager / Vocera |
| **Deployment** | Local Python | Docker + Kubernetes |
| **Monitoring** | In-dashboard charts | Prometheus + Grafana |
| **Compliance** | N/A | HIPAA BAA, FDA PCCP, IRB approval, HITRUST certification |

---

## ⚖️ Regulatory Note

Adaptive ML systems that influence clinical decisions may be regulated as **Software as a Medical Device (SaMD)**.

Key frameworks:
- **FDA Predetermined Change Control Plan (PCCP)**
- **FDA AI/ML Action Plan**
- **HIPAA**: All patient data (PHI) must remain within controlled infrastructure

**Before any clinical deployment:**
- Obtain IRB approval for any use of real patient data
- Run in **shadow mode** for at least 4–8 weeks before showing alerts to clinicians
- Conduct clinical validation studies on your target population
- Establish SOPs for model updates and rollbacks

---

## 📄 License

MIT License. See [LICENSE](LICENSE).

PhysioNet/CinC Challenge 2019 data is governed by the [PhysioNet Credentialed Health Data License 1.5.0](https://physionet.org/content/challenge-2019/1.0.0/). Download via Kaggle at: [kaggle.com/datasets/salikhussaini49/prediction-of-sepsis](https://www.kaggle.com/datasets/salikhussaini49/prediction-of-sepsis)

---

## 🙏 Acknowledgements

- **PhysioNet/CinC Challenge 2019**: Reyna, M., et al. (2019). Early Prediction of Sepsis from Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019.
- **River**: Montiel, J., et al. (2021). River: machine learning for streaming data in Python.
- **Sepsis-3 criteria**: Singer, M., et al. (2016). The Third International Consensus Definitions for Sepsis and Septic Shock. JAMA.
