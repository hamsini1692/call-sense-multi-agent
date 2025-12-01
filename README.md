# CallSense – Multi-Agent Call-Center Intelligence

Automated call summarization, frustration detection, and action recommendation using a **multi-agent workflow** built with **Google ADK, Gemini, FastAPI, and Streamlit**.

---

## 1. Problem

Contact centers handle thousands of calls every day. Reviewing them is:

- Slow and manual  
- Subjective and inconsistent across quality analysts  
- Prone to missed emotional cues and late escalations  

This leads to poor customer experience and higher operational costs.

---

## 2. Solution Overview

**CallSense** is a multi-agent system that reads call transcripts and automatically:

1. **Summarizes** the call in a clear, structured way  
2. **Detects frustration** and emotional patterns  
3. **Recommends actions** (e.g., apologize, ask for more info, offer replacement, escalate)

The system uses three specialized LLM agents that communicate via **Agent-to-Agent (A2A)** messaging:

- **Summary Agent** – extracts reason for call, key details, and overall sentiment  
- **Frustration Agent** – identifies emotional intensity and triggers  
- **Action Agent** – recommends next steps & escalation based on the above

---

## 3. Architecture

High-level components:

- **Frontend:**  
  - `frontend/app.py` – Streamlit UI  
  - Upload / select sample transcript  
  - Button to run analysis and panels to display:
    - Call summary  
    - Frustration analysis  
    - Recommended actions & escalation flag

- **Backend:**  
  - `backend/main.py` – FastAPI app  
  - `/analyze` endpoint coordinates the workflow:
    1. Calls **Summary Agent**
    2. Passes summary (A2A) to **Frustration Agent**
    3. Passes both to **Action Agent**
    4. Returns a unified JSON response to the UI

- **Agents Layer (Google ADK):**
  - `agent_app/summary_agent.py`
  - `agent_app/frustration_agent.py`
  - `agent_app/action_agent.py`
  - `test_a2a.py` – simple script to validate A2A communication

- **Data:**
  - `data/` – sample CSV of PII-redacted call transcripts

> If you have an architecture diagram (PNG), save it as `docs/architecture.png` and reference it here.

---

## 4. Repository Structure

```text
call-sense-multi-agent/
├── agent_app/               # ADK/Gemini agents and prompts
├── backend/                 # FastAPI backend (orchestration + A2A glue)
├── data/                    # Sample call transcript data
├── frontend/                # Streamlit UI
├── remote_metrics_agent.py  # Example remote A2A metrics agent (optional)
├── test_a2a.py              # Quick test script for A2A
├── requirements.txt         # Python dependencies
└── README.md
