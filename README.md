<h1 align="center">🛡️ Cyber Security Course Repository</h1>

<p align="center">
  Hands-on Labs • Machine Learning • MITRE ATT&CK • Real-world datasets
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Cyber%20Security-Student%20Portfolio-blueviolet?style=for-the-badge">
  <img src="https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge">
  <img src="https://img.shields.io/badge/MITRE%20ATT%26CK-Mapping-red?style=for-the-badge">
  <img src="https://img.shields.io/badge/AI%2FML-Anomaly%20Detection-orange?style=for-the-badge">
</p>

---

## 📘 About This Repository
This repository contains all labs and the final project for the **Cyber Security course**.  
Each lab includes **real-world datasets**, **AI/ML techniques**, and where applicable is mapped to **MITRE ATT&CK tactics and techniques**.

The repository is structured as a **progressive learning path**, starting from CTI and anomaly detection and advancing toward **LLM-based agents and malware analysis**.

---

# 🧪 Labs Index

| Lab | Title | Folder | Description |
|----|------|--------|-------------|
| **1** | 🧩 Cyber Threat Intelligence – MITRE ATT&CK Mapping | [lab01-cti-mapping](lab01-cti-mapping/) | CTI report analyzing a real phishing campaign and mapping adversary behavior to ATT&CK |
| **2** | 🚨 Network Anomaly Detection (Isolation Forest) | [lab02-anomaly-detection](lab02-anomaly-detection/) | ML-based detection of anomalous network traffic using CICIDS dataset |
| **3** | 🔍 (Reserved) | — | Reserved for advanced detection / analysis lab |
| **4** | 🤖 LLM Agents & Tool Usage (Exploratory Analysis) | [lab04-exploratory-analysis](lab04-exploratory-analysis/) | Introduction to LLM-based agents, tool calling, and Dockerized agent environments |

---

## 🧪 Lab 4 – LLM Agents & Tool Usage

**Focus:**  
Agent mechanics, tool invocation, and controlled interaction with structured data and logs.

**Key components:**
- Docker-based development environment
- Agent Framework DevUI
- Multiple agents:
  - `hello_world_agent`
  - `dataset_eda_agent`
  - `log_explainer_agent`
- Deterministic Python tools exposed to LLMs
- Real-time inspection via Dev UI (events, traces, tools)

**Concepts covered:**
- Difference between chatbots and agents
- Tool calling and function execution
- Structured output generation
- Safe and explainable AI workflows

> This lab serves as the **technical foundation** for the final project.

---

# 🧠 Final Project – Static vs Dynamic Malware Analysis using AI Agents

## 🎯 Project Goal
Design and implement an **AI-assisted malware analysis system** that compares and explains **static** and **dynamic** analysis results using LLM-based agents.

The system assists a SOC analyst by:
- Parsing raw analysis artifacts
- Extracting indicators automatically
- Explaining behavior in clear natural language
- Highlighting differences between static and runtime behavior

---

## 🔬 Analysis Types

### 🧱 Static Analysis
- PE metadata
- Imports & strings
- Suspicious API usage
- Entropy / obfuscation indicators

### ⚙️ Dynamic Analysis
- Runtime behavior
- Network connections
- Process creation
- File & registry activity
- Execution-time IOCs

---

## 🤖 AI Agent Architecture

| Agent | Responsibility |
|-----|----------------|
| Static Analysis Agent | Interprets static malware artifacts |
| Dynamic Analysis Agent | Explains runtime logs and behaviors |
| Log Explainer Agent | Converts raw logs into analyst-friendly explanations |
| Evaluator Agent | Compares static vs dynamic findings |
| (Optional) Defender Agent | Provides mitigation or detection suggestions |

---

## 🧰 Technologies Used

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python">
  <img src="https://img.shields.io/badge/Docker-Containerized-blue?style=for-the-badge&logo=docker">
  <img src="https://img.shields.io/badge/LLM-Agents-purple?style=for-the-badge">
  <img src="https://img.shields.io/badge/MITRE-ATT%26CK-critical?style=for-the-badge">
  <img src="https://img.shields.io/badge/SOC-AI%20Assistant-green?style=for-the-badge">
</p>

---

## 📂 Repository Structure


cyber-security-/
│
├── lab01-cti-mapping/
├── lab02-anomaly-detection/
├── lab04-exploratory-analysis/
│   └── app/
│       ├── hello_world/
│       ├── dataset_eda/
│       ├── log_explainer_agent/
│       └── llm_defense/
│
├── Dockerfile
├── compose.yml
└── README.md

---

## 🚀 Course Learning Outcomes
- Cyber Threat Intelligence analysis  
- MITRE ATT&CK mapping  
- Network anomaly detection  
- Malware static & dynamic analysis  
- AI-assisted SOC workflows  
- Secure and explainable LLM agents  

---

📌 *This repository is both a course submission and a professional security portfolio.*
