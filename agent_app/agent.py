# agent_app/agent.py

import os
import json
import asyncio
import uuid
import re
from typing import Dict, List, Any

from dotenv import load_dotenv

from google.genai import types
from google.adk.agents import LlmAgent
from google.adk.agents.remote_a2a_agent import (
    RemoteA2aAgent,
    AGENT_CARD_WELL_KNOWN_PATH,
)
from google.adk.runners import InMemoryRunner, Runner
from google.adk.sessions import InMemorySessionService

import warnings

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------
# 1. API key + basic identifiers
# --------------------------------------------------------------------
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY (or GOOGLE_API_KEY) is missing. Add it to your .env file."
    )

# google-genai looks for GOOGLE_API_KEY
os.environ["GOOGLE_API_KEY"] = api_key

APP_NAME = "callsense_adk"
USER_ID = "local_user"
MODEL_NAME = "gemini-2.5-flash-lite"

# Base URL of the remote metrics A2A agent (the one started via to_a2a)
REMOTE_METRICS_BASE_URL = os.getenv("DATA_A2A_URL", "http://localhost:8081")
REMOTE_METRICS_CARD_URL = f"{REMOTE_METRICS_BASE_URL}{AGENT_CARD_WELL_KNOWN_PATH}"

# --------------------------------------------------------------------
# 2. Controlled action catalog (only these are allowed)
# --------------------------------------------------------------------
ALLOWED_ACTIONS: Dict[str, str] = {
    "APOLOGIZE": "Apologize for the inconvenience and acknowledge the customer's feelings.",
    "ASK_MORE_INFO": "Ask the customer what specifically they are disappointed or upset about.",
    "CHECK_ORDER_STATUS": "Check internal systems for the order or ticket status and provide an update.",
    "OFFER_REPLACEMENT": "Offer a replacement if the product is damaged or defective and policy allows.",
    "OFFER_REFUND": "Offer a refund if within the refund policy window and appropriate.",
    "SET_EXPECTATION": "Set clear expectations on what will happen next and the expected timeline.",
    "ESCALATE": "Escalate the case to a supervisor or specialized team for further review.",
}


def action_selector(sentiment: str, transcript: str) -> Dict[str, Any]:
    """
    Safe Python logic that maps (sentiment, transcript) -> allowed actions.

    Returns:
        {
          "actions": [list of action codes],
          "escalate": bool
        }
    """
    actions: List[str] = []

    # Normalize inputs
    sent = (sentiment or "").strip().lower()
    text = (transcript or "").lower()

    # 1) Base on sentiment
    if sent in ["very_negative", "negative"]:
        actions += ["APOLOGIZE", "ASK_MORE_INFO"]
    elif sent == "neutral":
        actions.append("ASK_MORE_INFO")
    elif sent in ["positive", "very_positive"]:
        actions.append("SET_EXPECTATION")
    else:
        # Unknown sentiment → at least ask for more information
        actions.append("ASK_MORE_INFO")

    # 2) Pattern rules from transcript
    if any(w in text for w in ["broken", "damaged", "defective", "not working"]):
        actions.append("OFFER_REPLACEMENT")

    if any(w in text for w in ["refund", "money back", "return this"]):
        actions.append("OFFER_REFUND")

    if any(w in text for w in ["where is my order", "haven't received", "not delivered"]):
        actions.append("CHECK_ORDER_STATUS")

    if any(
        w in text
        for w in ["angry", "upset", "frustrated", "escalate", "speak to manager", "complaint"]
    ):
        actions.append("SET_EXPECTATION")

    # 3) De-duplicate while preserving order
    seen = set()
    deduped: List[str] = []
    for a in actions:
        if a not in seen:
            seen.add(a)
            deduped.append(a)

    # 4) Final escalation rule
    escalate = (
        sent == "very_negative"
        or "speak to manager" in text
        or "escalate" in text
        or "ESCALATE" in deduped
    )

    if escalate and "ESCALATE" not in deduped:
        deduped.append("ESCALATE")

    return {"actions": deduped, "escalate": escalate}


# --------------------------------------------------------------------
# 3. Agent #1 – Summary + Sentiment
# --------------------------------------------------------------------
SUMMARY_SYSTEM_PROMPT = """
You are CallSense, a call-center analysis assistant.

Given a call transcript, you MUST respond with valid JSON:
{
  "summary": "1–3 sentence summary of the customer's issue",
  "sentiment": "very_negative|negative|neutral|positive|very_positive"
}

Rules:
- Use ONLY JSON, no markdown.
- Do NOT include extra fields.
"""

summary_agent = LlmAgent(
    name="callsense_summary_agent",
    model=MODEL_NAME,
    instruction=SUMMARY_SYSTEM_PROMPT,
)

summary_runner = InMemoryRunner(agent=summary_agent, app_name=APP_NAME)
summary_session_service = summary_runner.session_service

# --------------------------------------------------------------------
# 4. Agent #2 – Frustration / Urgency Estimator
# --------------------------------------------------------------------
FRUSTRATION_SYSTEM_PROMPT = """
You are CallSense Frustration Estimator.

Given a call transcript, output ONLY valid JSON:
{
  "frustration_score": 0.0-1.0,
  "urgency": "low" | "medium" | "high"
}

Rules:
- frustration_score MUST be a number between 0.0 and 1.0.
- urgency MUST be exactly one of: "low", "medium", "high".
- Use ONLY JSON, no markdown, no code fences.
- Do NOT include extra fields.
"""

frustration_agent = LlmAgent(
    name="callsense_frustration_agent",
    model=MODEL_NAME,
    instruction=FRUSTRATION_SYSTEM_PROMPT,
)

frustration_runner = InMemoryRunner(agent=frustration_agent, app_name=APP_NAME)
frustration_session_service = frustration_runner.session_service

# --------------------------------------------------------------------
# 5. Agent #3 – Action Planner (LLM + custom tool)
# --------------------------------------------------------------------
ACTION_SYSTEM_PROMPT = """
You are CallSense Action Planner.

Your job:
- Read the call transcript and overall sentiment.
- Use ONLY the provided tool "action_selector" to choose next steps.
- NEVER invent your own actions; always call the tool.

Input you receive from the user is JSON:
{
  "transcript": "...",
  "sentiment": "very_negative|negative|neutral|positive|very_positive|unknown"
}

Steps:
1. Call action_selector(sentiment, transcript).
2. Return exactly this JSON:
{
  "actions": ["ACTION_CODE_1", "ACTION_CODE_2", ...],
  "escalate": true/false
}

Rules:
- Use ONLY JSON, no markdown.
- Do NOT add other keys.
"""

action_agent = LlmAgent(
    name="callsense_action_agent",
    model=MODEL_NAME,
    instruction=ACTION_SYSTEM_PROMPT,
    tools=[action_selector],  # ADK wraps this as a FunctionTool internally
)

action_runner = InMemoryRunner(agent=action_agent, app_name=APP_NAME)
action_session_service = action_runner.session_service

# --------------------------------------------------------------------
# 6. Agent #4 – Metrics Agent as REMOTE A2A agent
# --------------------------------------------------------------------
# This is the "heavy" agent, hosted separately and exposed via A2A.
# It has its own prompt & logic defined on the remote side using ADK's
# to_a2a(agent, ...). Here we just treat it as a RemoteA2aAgent.

metrics_agent = RemoteA2aAgent(
    name="callsense_remote_metrics_agent",
    description="Remote metrics analytics agent exposed via A2A.",
    agent_card=REMOTE_METRICS_CARD_URL,
) 

metrics_session_service = InMemorySessionService()
metrics_runner = Runner(
    app_name=APP_NAME,              # ✅ add this
    agent=metrics_agent,
    session_service=metrics_session_service,
)


# --------------------------------------------------------------------
# 7. Small helpers – session + one-shot run
# --------------------------------------------------------------------
def _init_session(service, session_id: str) -> None:
    async def _create():
        await service.create_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=session_id,
        )

    asyncio.run(_create())


def _run_agent_once(runner: Runner, session_id: str, content: types.Content):
    final_text = None
    debug_events: List[dict] = []
    for event in runner.run(
        user_id=USER_ID,
        session_id=session_id,
        new_message=content,
    ):
        debug_events.append(
            {
                "author": getattr(event, "author", None),
                "type": getattr(event, "type", None),
                "has_content": bool(getattr(event, "content", None)),
            }
        )

        if getattr(event, "content", None) and getattr(event.content, "parts", None):
            texts = [
                getattr(p, "text", "")
                for p in event.content.parts
                if getattr(p, "text", None)
            ]
            if texts:
                final_text = "".join(texts)

    return final_text, debug_events


# --------------------------------------------------------------------
# 8. Public API – main call analysis (summary + sentiment + frustration + actions)
# --------------------------------------------------------------------
def analyze_with_adk(transcript: str) -> dict:
    """
    Pipeline:
      1) Summary agent -> summary, sentiment
      2) Frustration agent -> frustration_score, urgency
      3) Python rule-based action_selector -> action codes, escalate
      4) Map action codes -> human descriptions for the UI
    """

    # ---------- 1) SUMMARY + SENTIMENT ----------
    sum_session_id = str(uuid.uuid4())
    _init_session(summary_session_service, sum_session_id)

    sum_content = types.Content(role="user", parts=[types.Part(text=transcript)])
    sum_text, sum_trace = _run_agent_once(summary_runner, sum_session_id, sum_content)

    summary = ""
    sentiment = "unknown"

    if not sum_text:
        summary = "No final ADK response from summary agent."
        sentiment = "unknown"
    else:
        try:
            payload = json.loads(sum_text)
            summary = payload.get("summary", "")
            sentiment = payload.get("sentiment", "unknown")
        except json.JSONDecodeError:
            # Fallback: try to regex from semi-structured output
            text = sum_text
            sum_match = re.search(r'"summary"\s*:\s*"([^"]+)"', text)
            sent_match = re.search(r'"sentiment"\s*:\s*"([^"]+)"', text)
            if sum_match:
                summary = sum_match.group(1)
            else:
                summary = text
            if sent_match:
                sentiment = sent_match.group(1)
            else:
                sentiment = "unknown"

    normalized_sentiment = (sentiment or "").strip().lower()
    print("DEBUG summary_agent:", {"summary": summary, "sentiment": sentiment})

    # ---------- 2) FRUSTRATION AGENT ----------
    fr_session_id = str(uuid.uuid4())
    _init_session(frustration_session_service, fr_session_id)

    fr_content = types.Content(role="user", parts=[types.Part(text=transcript)])
    fr_text, fr_trace = _run_agent_once(frustration_runner, fr_session_id, fr_content)

    print("RAW fr_text from frustration_agent:", repr(fr_text))

    frustration_score: float = 0.0
    urgency: str = "medium"

    if fr_text:
        try:
            cleaned = fr_text.strip()
            if cleaned.startswith("```"):
                # Strip ```json / ```python / ``` etc.
                cleaned = re.sub(r"^```[a-zA-Z0-9_]*\s*", "", cleaned)
                if cleaned.endswith("```"):
                    cleaned = cleaned[: cleaned.rfind("```")].strip()

            fr_payload = json.loads(cleaned)

            raw_score = fr_payload.get("frustration_score", 0.0)
            try:
                frustration_score = float(raw_score)
            except (TypeError, ValueError):
                frustration_score = 0.0

            urgency_raw = fr_payload.get("urgency", "medium")
            if isinstance(urgency_raw, str):
                urgency = urgency_raw.lower()
                if urgency not in {"low", "medium", "high"}:
                    urgency = "medium"
            else:
                urgency = "medium"

        except json.JSONDecodeError:
            frustration_score = 0.0
            urgency = "medium"

    print("DEBUG frustration_agent:", {"frustration_score": frustration_score, "urgency": urgency})

    # ---------- 3) ACTION AGENT (for trace) + PYTHON TOOL (for truth) ----------
    act_session_id = str(uuid.uuid4())
    _init_session(action_session_service, act_session_id)

    user_payload = {"transcript": transcript, "sentiment": sentiment or "unknown"}
    act_content = types.Content(
        role="user", parts=[types.Part(text=json.dumps(user_payload))]
    )

    act_text, act_trace = _run_agent_once(action_runner, act_session_id, act_content)

    # Source of truth: rule-based tool
    rule_out = action_selector(normalized_sentiment, transcript)
    print("DEBUG rule_out from action_selector:", rule_out)

    action_codes = rule_out.get("actions", []) or []
    if isinstance(action_codes, str):
        action_codes = [action_codes]

    raw_escalate = rule_out.get("escalate", False)
    print("DEBUG raw_escalate:", raw_escalate, "type:", type(raw_escalate))

    escalate = bool(raw_escalate) or normalized_sentiment == "very_negative"
    print("DEBUG final escalate:", escalate)

    actions_readable = [ALLOWED_ACTIONS.get(code, code) for code in action_codes]

    # ---------- 4) Final response ----------
    combined_trace: List[dict] = []

    for ev in sum_trace:
        ev2 = dict(ev)
        ev2["agent"] = "summary_agent"
        combined_trace.append(ev2)

    for ev in fr_trace:
        ev2 = dict(ev)
        ev2["agent"] = "frustration_agent"
        combined_trace.append(ev2)

    for ev in act_trace:
        ev2 = dict(ev)
        ev2["agent"] = "action_agent"
        combined_trace.append(ev2)

    return {
        "summary": summary,
        "sentiment": sentiment,
        "frustration_score": frustration_score,
        "urgency": urgency,
        "actions": actions_readable,
        "escalate": escalate,
        "trace": combined_trace,
    }


# --------------------------------------------------------------------
# 9. Public API – metrics / overall summary via REMOTE A2A agent
# --------------------------------------------------------------------
def analyze_metrics(call_records: List[Dict[str, Any]]) -> dict:
    """
    Metrics via remote A2A metrics_agent.

    call_records: [
      {"sentiment": "...", "frustration_score": 0.8},
      ...
    ]

    The remote agent is responsible for computing:
      - total_calls
      - pct_very_negative
      - avg_frustration
      - summary

    Returns:
        {
          "total_calls": int,
          "pct_very_negative": float,
          "avg_frustration": float,
          "summary": str,
          "trace": [...]
        }
    """
    metrics_session_id = str(uuid.uuid4())

    # For the remote agent we still create a session for consistency
    _init_session(metrics_session_service, metrics_session_id)

    payload = {"calls": call_records}
    metrics_content = types.Content(
        role="user", parts=[types.Part(text=json.dumps(payload))]
    )

    metrics_text, metrics_trace = _run_agent_once(
        metrics_runner, metrics_session_id, metrics_content
    )

    print("RAW metrics_text from remote metrics_agent:", repr(metrics_text))

    # Defaults if remote agent fails or output cannot be parsed
    total_calls = len(call_records)
    pct_very_negative = 0.0
    avg_frustration = 0.0
    summary_text = ""

    if metrics_text:
        try:
            cleaned = metrics_text.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```[a-zA-Z0-9_]*\s*", "", cleaned)
                if cleaned.endswith("```"):
                    cleaned = cleaned[: cleaned.rfind("```")].strip()

            m = json.loads(cleaned)

            if "total_calls" in m:
                total_calls = int(m["total_calls"])

            if "pct_very_negative" in m:
                pct_very_negative = float(m["pct_very_negative"])

            if "avg_frustration" in m:
                avg_frustration = float(m["avg_frustration"])

            if isinstance(m.get("summary"), str):
                summary_text = m["summary"].strip()

        except Exception as e:
            print("Error parsing metrics_text from remote agent:", e)

    # Annotate trace
    metrics_trace_annotated: List[dict] = []
    for ev in metrics_trace:
        ev2 = dict(ev)
        ev2["agent"] = "remote_metrics_agent"
        metrics_trace_annotated.append(ev2)

    return {
        "total_calls": total_calls,
        "pct_very_negative": pct_very_negative,
        "avg_frustration": avg_frustration,
        "summary": summary_text,
        "trace": metrics_trace_annotated,
    }
