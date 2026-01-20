# Lab 05 – Defensive LLM Workflow (Policy-Based Query Filtering)

## Objective
This lab demonstrates a **defensive LLM architecture** where user queries are **not answered directly**.
Instead, every query is first evaluated by a **policy (guard) agent**, and the final response is determined
by **workflow routing logic**, not by a single model call.

## Architecture Overview

User Query  
→ **Question Check Agent (Guard)**  
→ Allowed?  
- ✅ Yes → **Geography & Weather Agent**  
- ❌ No → **Refusal Agent**

Only queries related to **geography or weather** are allowed to reach the answering agent.

## Agents

### 1. Question Check Agent (Policy / Guard)
- Classifies the user query into one of:
  - `greeting`
  - `goodbye`
  - `weather`
  - `geography`
  - `other`
- Outputs **structured JSON only**
- Does **not** answer user questions
- Used exclusively for **routing decisions**

Fail-closed behavior:
- If the output is invalid or cannot be parsed, the query is treated as `other`
- This ensures unsafe or ambiguous queries are always rejected

### 2. Geography & Weather Agent
- Answers **only** geography and weather questions
- Receives input **only if** the guard allows it
- Has no knowledge of policy decisions

### 3. Refusal Agent
- Provides a polite refusal for disallowed queries
- Does not reveal internal logic, policies, or classifications

## Workflow Logic
1. User input is sent to the Question Check Agent
2. The intent classification is evaluated
3. If intent ∈ {greeting, goodbye, weather, geography} → route to Geography & Weather Agent
4. Otherwise → route to Refusal Agent

## Example Behavior

**Disallowed query**
