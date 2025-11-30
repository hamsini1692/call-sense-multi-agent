# remote_metrics_agent.py
import os

from dotenv import load_dotenv

from google.adk.agents import LlmAgent
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.code_executors import UnsafeLocalCodeExecutor  # or BuiltInCodeExecutor
from google.adk.models.google_llm import Gemini
import uvicorn
# --------------------------------------------------------------------
# 1. API key + model
# --------------------------------------------------------------------
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY (or GOOGLE_API_KEY) is missing. Add it to your .env file."
    )

# google-genai looks for GOOGLE_API_KEY
os.environ["GOOGLE_API_KEY"] = api_key

MODEL_NAME = "gemini-2.5-flash-lite"

# --------------------------------------------------------------------
# 2. Metrics system prompt (same logic as before)
# --------------------------------------------------------------------
METRICS_SYSTEM_PROMPT = """
You are CallSense Analytics Agent.

You will receive a JSON payload like:
{
  "calls": [
    {
      "sentiment": "very_negative|negative|neutral|positive|very_positive",
      "frustration_score": 0.0-1.0 or null
    },
    ...
  ]
}

Your job:

1) Use the Python code execution tool to compute these values from the data:
   - total_calls: integer, len(calls)
   - pct_very_negative: float in [0, 100], computed as
       100 * (# calls where sentiment == "very_negative") / total_calls
       If total_calls == 0, this must be 0.0.
   - avg_frustration: float in [0, 1], the mean of all non-null frustration_score values.
       If there are no non-null scores, this must be 0.0.

2) Using those computed numbers, write a short natural-language "summary"
   (2–3 sentences) that describes overall call health:
   - how many calls are very_negative
   - how high the average frustration is
   - anything notable for a call center lead.

IMPORTANT RULES:
- You MUST call the code execution tool at least once to do the math.
- Do NOT guess. Always base your numbers on the actual "calls" input.
- Do NOT return all zeros unless the true values are actually zero.
- Never ignore frustration_score if it is present.

Return ONLY valid JSON (no markdown, no code fences) with exactly this shape:

{
  "total_calls": <int>,
  "pct_very_negative": <float>,
  "avg_frustration": <float>,
  "summary": "<string>"
}
"""

# --------------------------------------------------------------------
# 3. Define the metrics agent
# --------------------------------------------------------------------
metrics_agent = LlmAgent(
    name="callsense_metrics_agent",
    model=Gemini(model=MODEL_NAME),
    instruction=METRICS_SYSTEM_PROMPT,
    # For local dev; in production prefer BuiltInCodeExecutor()
    code_executor=UnsafeLocalCodeExecutor(),
)

# --------------------------------------------------------------------
# 4. Expose it over A2A
# --------------------------------------------------------------------
if __name__ == "__main__":
    # You can change the port via DATA_A2A_PORT if needed
    port = int(os.getenv("DATA_A2A_PORT", "8081"))
    print(f"Starting remote metrics A2A agent on port {port} ...")

    # Wrap the ADK agent as an A2A app
    a2a_app = to_a2a(metrics_agent, port=port)

    # Actually start the HTTP server so it keeps running
    uvicorn.run(a2a_app, host="0.0.0.0", port=port)