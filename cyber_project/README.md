cyber_project
Malware Detection Decision Engine (Batch + Kafka Streaming)
Purpose
cyber_project contains a self-contained malware detection system built as part of the Cybersecurity course labs.
The project focuses on decision-level malware detection, operating on already-extracted static and dynamic features, and demonstrates how the same detection engine can be used in:
Offline / batch evaluation
Online / real-time streaming using Apache Kafka
The goal is to show system design, policy-driven decision making, and production-style deployment, not raw feature extraction.

What This Project Does
Core functionality
Runs multiple ML models (“agents”) for malware detection
Aggregates model outputs using a judge
Applies confidence-based enforcement rules
Produces final actions:
ALLOW
REVIEW
NO_ACTION
Supported execution modes
Batch mode (CSV files)
Evaluation mode (train/test split)
Streaming mode (Kafka)
All modes use the same detection logic.

High-Level Architecture
[ Static / Dynamic Features ]
            |
            v
     Detection Engine
 (agents → judge → enforcement)
            |
            v
      Final Decision

In streaming mode, Kafka is used only as a transport layer:
Producers → Kafka → Detection Engine → Kafka → Consumers

Detection Logic Overview
Inference agents
Static analysis
AdaBoost
Gradient Boosting
Dynamic analysis
AdaBoost (probability-calibrated)
Gradient Boosting (probability-calibrated)
Judge
Collects predictions from all relevant agents
Checks agreement / disagreement
Selects the most reliable agent
Enforcement
Applies confidence thresholds
Applies penalties for:
disagreement
missing / imputed features
Produces final action
All thresholds are defined in config/policy.json.

Batch / Evaluation Mode
Build environment
docker compose up --build

Run batch inference
docker compose run --rm cyber python run_batch.py

Processes full CSV datasets
Writes reports to outputs/reports/

Run evaluation on test split
docker compose run --rm cyber python run_eval_split.py
Evaluates on held-out test data
Prints classification reports
Writes CSV test reports

Kafka Streaming Mode
Kafka enables real-time ingestion of feature events.
Topics
malware-input — incoming feature events
malware-decisions — decision outputs
Start full stack
docker compose up --build
Starts:
Zookeeper
Kafka broker
Cyber container

Run detection engine as Kafka consumer
docker compose run --rm cyber python run_kafka_consumer.py
Behavior:
Consumes feature events from Kafka
Converts them to internal Sample objects
Runs detection engine
Publishes decisions to Kafka
Uses at-least-once semantics

Send demo events
docker compose run --rm cyber python run_kafka_producer_demo.py \
  --source-type dynamic --rows 20
or
docker compose run --rm cyber python run_kafka_producer_demo.py \
  --source-type static --rows 20

  View decisions
docker compose run --rm cyber python run_kafka_print_decisions.py
Calibration
Dynamic models are probability-calibrated to avoid over-confident predictions.
Calibration script:
docker compose run --rm cyber python scripts/calibrate_dynamic.py
Produces calibrated model files under models/.

Debugging
Inference debug output is disabled by default.
Enable debug logs:
DEBUG_INFER=1 docker compose run --rm cyber python run_batch.py

Debugging
Inference debug output is disabled by default.
Enable debug logs:
DEBUG_INFER=1 docker compose run --rm cyber python run_batch.py

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
