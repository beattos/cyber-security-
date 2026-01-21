# 🛡️ Cyber Security Streaming Pipeline  
### Static vs Dynamic Malware Detection with Confidence-Based Decisions

This project implements a **streaming-style producer–consumer pipeline** for malware detection, comparing **static** and **dynamic** analysis using machine learning.

Instead of focusing only on offline metrics, the system demonstrates **how models behave operationally**:  
each sample is processed as an event, assigned a confidence score, and routed through a **SOC-style decision flow**.

---

## 🎯 Project Objectives

- Build a **realistic cybersecurity inference pipeline**
- Compare **static vs dynamic malware detection**
- Apply **confidence-based decisions** (`ALERT / REVIEW / PASS`)
- Simulate **SOC workflows** with human-in-the-loop logic
- Ensure **reproducibility** with Docker

---

## 🧠 Core Concept

Each input sample is treated as a **streaming event**:

CSV sample
↓
Producer
↓
Routing (Static / Dynamic)
↓
ML Inference
↓
Confidence Scoring
↓
Decision: ALERT / REVIEW / PASS

This design reflects how real security systems operate, rather than only reporting aggregate accuracy metrics.

---

## 📦 Project Structure

```text
Project Structure
cyber_project/
├── src/
│   ├── agents/          # Inference agents (static & dynamic)
│   ├── pipeline/        # Orchestrator, judge, enforcement
│   ├── streaming/       # Kafka consumer/producer helpers
│   └── common/          # Shared contracts and utilities
│
├── models/              # Trained & calibrated ML models
├── artifacts/           # Feature order definitions
├── config/
│   └── policy.json      # All thresholds & penalties
│
├── data/                # Clean CSV feature datasets
├── outputs/             # Generated reports (not committed)
│
├── run_system.py        # Single-sample demo
├── run_batch.py         # Batch inference
├── run_eval_split.py    # Test set evaluation
├── run_kafka_consumer.py
├── run_kafka_producer_demo.py
├── run_kafka_print_decisions.py
│
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```


---

## ⚙️ Models & Preprocessing

- **Static model**  
  Gradient Boosting classifier trained on file-level / static features

- **Dynamic model**  
  Gradient Boosting classifier trained on runtime behavioral features  
  (calibrated for better probability estimates)

- **Preprocessing**
  - Median imputation
  - Exact feature order preserved
  - Same artifacts used in training and inference

> Imputers and feature-column lists are **generated automatically** and are intentionally **not committed**.

---

## 🚦 Decision Logic (Confidence Gating)

Each event produces a probability `p(malware)`.

| Confidence Range | Decision |
|------------------|----------|
| `p ≥ 0.90`       | **ALERT** |
| `0.70 ≤ p < 0.90`| **REVIEW** |
| `p < 0.70`       | **PASS** |

This mirrors real SOC behavior:
- **ALERT** → immediate action  
- **REVIEW** → analyst inspection  
- **PASS** → no action  

---

## ▶️ Run Locally (CLI)

```bash
python -m pipeline.run_stream_demo \
  --static_csv data/static_clean.csv \
  --dynamic_csv data/dynamic_clean.csv \
  --static_model models/gb_static.pkl \
  --dynamic_model models/gb_dynamic_calibrated.pkl \
  --max_events 30 \
  --interactive_review \
  --suppress_sklearn_pickle_warnings
```
Example per-event output
[PRODUCER→PIPELINE] event=12 source=dynamic
[ROUTE] dynamic_model
[INFERENCE] p_malware=0.78
[DECISION] REVIEW

🐳 Run with Docker (Reproducible)
[docker compose up --build]
Docker runs the pipeline end-to-end, prints a summary, and exits cleanly.

📊 Example Output Summary
Total events: 300
static:  ALERT=125  REVIEW=0   PASS=25
dynamic: ALERT=50   REVIEW=100 PASS=0
Avg latency: ~1–7 ms

Interpretation
Static analysis is fast and decisive
Dynamic analysis captures nuanced behavior
Many dynamic samples fall into REVIEW, demonstrating the need for human-in-the-loop decision making

📁 Output Artifacts
outputs/stream_results.csv
Contains per-event predictions, confidence scores, latency, and final decisions.
This file is generated at runtime and intentionally not tracked in Git.

🧪 Course Context
This project aligns with AI in Cybersecurity / NVIDIA Morpheus-style concepts:
Streaming inference pipelines
Confidence-aware decision logic
SOC-oriented workflows
Reproducible experimentation
