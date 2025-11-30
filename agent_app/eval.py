# agent_app/eval.py

import os
import json
from collections import Counter
from typing import Dict, Any, List, Tuple

import pandas as pd

from .agent import analyze_with_adk, analyze_metrics

# Reuse the same CSV as the frontend
CALLS_CSV_PATH = os.getenv("CALLS_CSV_PATH", "data/call_recordings.csv")


def _normalize_sentiment(label: str) -> str:
    """
    Normalize sentiment labels from the dataset and from the agent.

    Examples:
      "Very negative" -> "very_negative"
      "negative"      -> "negative"
      "POSITIVE"      -> "positive"
    """
    if not isinstance(label, str):
        return "unknown"
    lab = label.strip().lower()
    lab = lab.replace(" ", "_")
    # Optional: map dataset-specific values if needed
    return lab


# -------------------------------------------------------------------
# 1. Evaluate call-level sentiment agent
# -------------------------------------------------------------------
def evaluate_sentiment_agent(
    n_samples: int = 30,
) -> Dict[str, Any]:
    """
    Evaluate the sentiment predictions from analyze_with_adk()
    against the 'Sentiment' column in call_recordings.csv.

    Returns a dict with:
      - num_examples
      - accuracy
      - confusion (nested dict)
      - examples (list of small error samples)
    """
    if not os.path.exists(CALLS_CSV_PATH):
        raise FileNotFoundError(f"CSV not found at {CALLS_CSV_PATH}")

    df = pd.read_csv(CALLS_CSV_PATH)

    if "Transcript" not in df.columns or "Sentiment" not in df.columns:
        raise ValueError(
            "CSV must contain 'Transcript' and 'Sentiment' columns for evaluation."
        )

    # Drop rows with missing values
    df = df.dropna(subset=["Transcript", "Sentiment"])
    if df.empty:
        raise ValueError("No non-empty rows with Transcript + Sentiment in CSV.")

    # Sample a subset to keep evaluation fast
    df_sample = df.sample(n=min(n_samples, len(df)), random_state=42)

    y_true: List[str] = []
    y_pred: List[str] = []

    error_examples: List[Dict[str, str]] = []

    for _, row in df_sample.iterrows():
        transcript = row["Transcript"]
        true_label = _normalize_sentiment(row["Sentiment"])

        try:
            out = analyze_with_adk(str(transcript))
            pred_label = _normalize_sentiment(out.get("sentiment", "unknown"))
        except Exception as e:
            pred_label = "error"
            error_examples.append(
                {
                    "transcript": str(transcript)[:300],
                    "true": true_label,
                    "pred": pred_label,
                    "error": str(e),
                }
            )

        y_true.append(true_label)
        y_pred.append(pred_label)

        if true_label != pred_label and pred_label != "error":
            error_examples.append(
                {
                    "transcript": str(transcript)[:300],
                    "true": true_label,
                    "pred": pred_label,
                }
            )

    # Compute accuracy
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    num = len(y_true)
    accuracy = correct / num if num > 0 else 0.0

    # Confusion matrix-style counts
    confusion: Dict[Tuple[str, str], int] = Counter(zip(y_true, y_pred))
    confusion_nested: Dict[str, Dict[str, int]] = {}
    for (t, p), c in confusion.items():
        confusion_nested.setdefault(t, {})[p] = c

    return {
        "num_examples": num,
        "accuracy": accuracy,
        "confusion": confusion_nested,
        "errors": error_examples[:20],  # cap for readability
    }


# -------------------------------------------------------------------
# 2. Evaluate dataset-level metrics agent (A2A)
# -------------------------------------------------------------------
def evaluate_metrics_agent(
    n_samples: int = 30,
) -> Dict[str, Any]:
    """
    Evaluate the remote metrics agent by:

      1) Taking N transcripts from the CSV.
      2) Computing ground-truth % very negative from the dataset labels.
      3) Running analyze_with_adk() on each transcript to get predicted
         sentiment + frustration_score.
      4) Calling analyze_metrics(call_records) to get the remote A2A summary.
      5) Comparing the predicted aggregate vs ground-truth aggregate.

    Returns a dict with:
      - num_examples
      - ground_truth_pct_very_negative
      - agent_pct_very_negative
      - delta_pct_very_negative
      - agent_avg_frustration
      - raw_metrics_output
    """
    if not os.path.exists(CALLS_CSV_PATH):
        raise FileNotFoundError(f"CSV not found at {CALLS_CSV_PATH}")

    df = pd.read_csv(CALLS_CSV_PATH)

    if "Transcript" not in df.columns or "Sentiment" not in df.columns:
        raise ValueError(
            "CSV must contain 'Transcript' and 'Sentiment' columns for metrics evaluation."
        )

    df = df.dropna(subset=["Transcript", "Sentiment"])
    if df.empty:
        raise ValueError("No non-empty rows with Transcript + Sentiment in CSV.")

    df_sample = df.sample(n=min(n_samples, len(df)), random_state=123)

    # Ground truth aggregate from dataset labels
    labels = [_normalize_sentiment(s) for s in df_sample["Sentiment"].tolist()]
    total = len(labels)
    true_very_neg = sum(1 for s in labels if s == "very_negative")
    gt_pct_very_negative = 100.0 * true_very_neg / total if total > 0 else 0.0

    # Build call_records using our *pipeline* (so we evaluate the end-to-end system)
    call_records: List[Dict[str, Any]] = []
    for _, row in df_sample.iterrows():
        t = row["Transcript"]
        try:
            out = analyze_with_adk(str(t))
        except Exception as e:
            # In case of error, treat as neutral with low frustration
            out = {"sentiment": "neutral", "frustration_score": 0.0}

        call_records.append(
            {
                "sentiment": _normalize_sentiment(out.get("sentiment", "unknown")),
                "frustration_score": float(out.get("frustration_score", 0.0) or 0.0),
            }
        )

    # Call the remote metrics agent via A2A
    metrics_out = analyze_metrics(call_records)

    agent_pct_very_negative = float(metrics_out.get("pct_very_negative", 0.0))
    agent_avg_frustration = float(metrics_out.get("avg_frustration", 0.0))

    delta_pct = agent_pct_very_negative - gt_pct_very_negative

    return {
        "num_examples": total,
        "ground_truth_pct_very_negative": gt_pct_very_negative,
        "agent_pct_very_negative": agent_pct_very_negative,
        "delta_pct_very_negative": delta_pct,
        "agent_avg_frustration": agent_avg_frustration,
        "raw_metrics_output": metrics_out,
    }


# -------------------------------------------------------------------
# 3. Simple CLI entrypoint for local testing
# -------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Evaluating sentiment agent on sample calls ===")
    sent_eval = evaluate_sentiment_agent(n_samples=3)
    print(json.dumps(sent_eval, indent=2))

    print("\n=== Evaluating metrics (A2A) agent on sample calls ===")
    metrics_eval = evaluate_metrics_agent(n_samples=3)
    print(json.dumps(metrics_eval, indent=2))
