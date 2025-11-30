from typing import List, Optional, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agent_app.agent import analyze_with_adk, analyze_metrics

from typing import List, Optional, Any, Dict  # add Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agent_app.agent import analyze_with_adk, analyze_metrics
from agent_app.eval import evaluate_sentiment_agent, evaluate_metrics_agent  # 👈 NEW


app = FastAPI(title="CallSense Backend")


# ---------------------------------------------
# Models for /analyze
# ---------------------------------------------
class AnalyzeRequest(BaseModel):
    transcript: str


class AnalyzeResponse(BaseModel):
    summary: str
    sentiment: str
    frustration_score: Optional[float] = None
    urgency: Optional[str] = None
    actions: List[Any]
    escalate: bool
    trace: List[Any]


# ---------------------------------------------
# Models for /metrics
# ---------------------------------------------
class MetricsRequest(BaseModel):
    transcripts: List[str]


class MetricsResponse(BaseModel):
    total_calls: int
    pct_very_negative: float
    avg_frustration: float
    summary: str
    trace: List[Any]

class EvalResponse(BaseModel):
    sentiment_eval: Dict[str, Any]
    metrics_eval: Dict[str, Any]



# ---------------------------------------------
# Health check
# ---------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------
# Single-call analysis endpoint
# ---------------------------------------------
@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):   # <-- sync, NOT async
    if not req.transcript.strip():
        raise HTTPException(400, "Transcript cannot be empty")

    result = analyze_with_adk(req.transcript)

    # result is expected to contain:
    #   summary, sentiment, frustration_score, urgency,
    #   actions, escalate, trace
    return AnalyzeResponse(
        summary=result.get("summary", ""),
        sentiment=result.get("sentiment", "unknown"),
        frustration_score=result.get("frustration_score"),
        urgency=result.get("urgency"),
        actions=result.get("actions", []),
        escalate=bool(result.get("escalate", False)),
        trace=result.get("trace", []),
    )


# ---------------------------------------------
# Batch metrics endpoint (Code Execution Agent)
# ---------------------------------------------
@app.post("/metrics", response_model=MetricsResponse)
def metrics(req: MetricsRequest):
    """
    Batch metrics endpoint.

    Input example:
        {
          "transcripts": ["call 1 text", "call 2 text", ...]
        }

    Steps:
      1) For each transcript, run analyze_with_adk (summary + sentiment + frustration).
      2) Build a list of {sentiment, frustration_score}.
      3) Call analyze_metrics(call_records) – the code-execution agent.
      4) Return its JSON (total_calls, pct_very_negative, avg_frustration, summary, trace).
    """
    # Filter out empty transcripts
    filtered = [t for t in req.transcripts if t and t.strip()]
    if not filtered:
        raise HTTPException(400, "At least one non-empty transcript is required")

    call_records = []
    for t in filtered:
        res = analyze_with_adk(t)
        call_records.append(
            {
                "sentiment": res.get("sentiment"),
                "frustration_score": res.get("frustration_score"),
            }
        )

    metrics_out = analyze_metrics(call_records)

    return MetricsResponse(
        total_calls=metrics_out.get("total_calls", len(filtered)),
        pct_very_negative=metrics_out.get("pct_very_negative", 0.0),
        avg_frustration=metrics_out.get("avg_frustration", 0.0),
        summary=metrics_out.get("summary", ""),
        trace=metrics_out.get("trace", []),
    )

# ---------------------------------------------
# Offline evaluation endpoint (uses eval.py)
# ---------------------------------------------
@app.get("/eval", response_model=EvalResponse)
def run_eval(n_samples: int = 30):
    """
    Run offline evaluation over the CSV using eval.py:

      - evaluate_sentiment_agent: compares call-level sentiment
        against the dataset labels.
      - evaluate_metrics_agent: compares A2A metrics against
        ground truth aggregate from labels.
    """
    sentiment_eval = evaluate_sentiment_agent(n_samples=n_samples)
    metrics_eval = evaluate_metrics_agent(n_samples=n_samples)

    return EvalResponse(
        sentiment_eval=sentiment_eval,
        metrics_eval=metrics_eval,
    )

