cyber_project
Malware Detection System — Batch + Streaming (Kafka-enabled)
Overview
This project implements a malware detection decision engine that operates on already-extracted static and dynamic features.
The system supports:
Offline / batch analysis (CSV-based)
Online / streaming analysis using Apache Kafka
Multiple inference agents with confidence-aware judgement and enforcement
Policy-driven thresholds (no hard-coded values)
Reproducible execution via Docker
The core detection logic is identical across batch and streaming modes. Kafka is used only as a transport layer, not as part of the ML logic.
System Architecture (Conceptual)
[ Feature Extractors ]
   (static / dynamic)
           |
           v
      Kafka Topic
     "malware-input"
           |
           v
   Detection Engine
  (this project)
           |
           v
      Kafka Topic
   "malware-decisions"
           |
           v
 [ SOC / SIEM / Logs ]
Feature extraction is out of scope
This system consumes features, not raw binaries
Kafka enables real-time ingestion without modifying detection logic
Core Components
Detection Engine
Static agents:
AdaBoost
Gradient Boosting
Dynamic agents:
AdaBoost (calibrated)
Gradient Boosting (calibrated)
Judge:
Aggregates agent outputs
Resolves disagreements
Confidence Enforcer:
Applies policy thresholds
Produces final action:
ALLOW
REVIEW
NO_ACTION
Configuration
All thresholds and penalties are defined in:
config/policy.json
Examples:
Malicious probability thresholds
Confidence bounds
Disagreement penalties
Imputation penalties
No thresholds are hard-coded in code.
Batch / Evaluation Mode (Offline)
Used for reproducible analysis and grading.
Build environment
docker compose up --build
Run batch inference
docker compose run --rm cyber python run_batch.py
Processes full CSV datasets
Outputs reports to:
outputs/reports/
Run evaluation on stratified test split
docker compose run --rm cyber python run_eval_split.py
Evaluates on held-out test data
Produces:
static_test_report.csv
dynamic_test_report.csv
Prints classification reports to stdout
Kafka Streaming Mode (Online)
Kafka is optional and wraps the existing engine without changing its logic.
Topics
malware-input — incoming feature events
malware-decisions — detection results
Topics are auto-created by default Kafka configuration.
1. Start full stack (Zookeeper, Kafka, cyber)
docker compose up --build
Services started:
Zookeeper (zookeeper:2181)
Kafka broker (kafka:9092, host-accessible via localhost:9092)
Cyber container (runs run_system.py demo by default)
2. Run Kafka consumer (decision engine)
docker compose run --rm cyber python run_kafka_consumer.py
Behavior:
Consumes events from malware-input
Validates event schema
Converts event → Sample
Runs existing orchestrator (agents → judge → enforcement)
Publishes decision JSON to malware-decisions
Commits Kafka offsets only after successful publish
(at-least-once semantics)
Invalid messages are logged and skipped safely.
3. Produce demo events
Dynamic features
docker compose run --rm cyber python run_kafka_producer_demo.py \
  --source-type dynamic --rows 20
Static features
docker compose run --rm cyber python run_kafka_producer_demo.py \
  --source-type static --rows 20
Optional flags:
--sleep-ms 200 (message pacing)
The producer:
Reads rows from CSV
Builds Kafka events
Sends them to malware-input
4. (Optional) View decisions
docker compose run --rm cyber python run_kafka_print_decisions.py
Prints:
sample_id
action
effective_confidence
chosen_agent
Calibration
Dynamic models are probability-calibrated to avoid degenerate predictions.
Calibration script:
docker compose run --rm cyber python scripts/calibrate_dynamic.py
Outputs:
models/ada_dynamic_calibrated.pkl
models/gb_dynamic_calibrated.pkl
Calibration is compatible with sklearn 1.6.1 and warnings are suppressed cleanly.
Debugging
Inference debug logs are disabled by default.
Enable debug output:
DEBUG_INFER=1 docker compose run --rm cyber python run_batch.py
What We Learned
ML models are decision components, not full systems
Feature extraction, transport, inference, and enforcement must be decoupled
Kafka enables real-time ingestion without touching ML logic
Policy-driven thresholds are critical for SOC-style systems
The same engine can support:
Offline analysis
Online streaming
Batch evaluation
Real-time decisions
Submission Checklist
 Batch inference works
 Test split evaluation works
 Dynamic models calibrated
 Policy externalized to config
 Kafka streaming integrated
 Docker reproducible
 No schema changes
 Core logic unchanged by Kafka
Final Notes
This project demonstrates a production-oriented malware detection pipeline with both batch and streaming execution, clean separation of concerns, and SOC-style decision logic.
