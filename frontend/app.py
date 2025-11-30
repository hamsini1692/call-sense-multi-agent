import os
import requests
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

# Path to your CSV – adjust if different
CALLS_CSV_PATH = "data/call_recordings.csv"

# ---------------------------------------------------------------------
# Streamlit page config
# ---------------------------------------------------------------------
st.set_page_config(page_title="CallSense – ADK Call Analyzer", layout="wide")

# ---------------------------------------------------------------------
# Load dataset (cached)
# ---------------------------------------------------------------------
@st.cache_data
def load_calls_csv(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        st.sidebar.warning(f"Could not load CSV at {path}: {e}")
        return pd.DataFrame()


calls_df = load_calls_csv(CALLS_CSV_PATH)

# ---------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------
if "transcript_input" not in st.session_state:
    st.session_state["transcript_input"] = ""

if "sample_meta" not in st.session_state:
    st.session_state["sample_meta"] = None

# ---------------------------------------------------------------------
# Main title and description
# ---------------------------------------------------------------------
st.title("📞 CallSense")

st.write(
    "Paste a customer transcript or load a sample call from the dataset, "
    "then run analysis to get a summary, sentiment, frustration/urgency, "
    "and safe recommended actions."
)

# ---------------------------------------------------------------------
# Sidebar – sample calls from CSV
# ---------------------------------------------------------------------
st.sidebar.header("📂 Sample Call Recordings")

if calls_df is not None and not calls_df.empty:
    # Expected columns:
    # id, Type, Sentiment, Name, Order Number, Product Number, Transcript
    def make_label(row):
        base = f"{row.get('id', '')} • {row.get('Type', '')}"
        name = row.get("Name", "")
        if isinstance(name, str) and name:
            base += f" • {name}"
        return base

    calls_df["label"] = calls_df.apply(make_label, axis=1)

    selected_label = st.sidebar.selectbox(
        "Choose a sample call",
        options=calls_df["label"].tolist(),
    )

    if st.sidebar.button("Load sample into text box"):
        row = calls_df.loc[calls_df["label"] == selected_label].iloc[0]

        # Update transcript in state
        st.session_state["transcript_input"] = row.get("Transcript", "")

        # Store some metadata about the selected call
        st.session_state["sample_meta"] = {
            "id": row.get("id", ""),
            "Type": row.get("Type", ""),
            "Sentiment (label)": row.get("Sentiment", ""),
            "Name": row.get("Name", ""),
            "Order Number": row.get("Order Number", ""),
            "Product Number": row.get("Product Number", ""),
        }
else:
    st.sidebar.info("No CSV loaded or file is empty.")

# ---------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("📝 Call Transcript")

    transcript = st.text_area(
        "Paste the customer-side transcript here:",
        key="transcript_input",
        height=220,
    )

    run_clicked = st.button("🚀 Run Analysis")

with col_right:
    st.subheader("📌 Selected Sample (if any)")
    if st.session_state["sample_meta"]:
        meta = st.session_state["sample_meta"]
        st.markdown(f"**ID:** {meta.get('id')}")
        st.markdown(f"**Type:** {meta.get('Type')}")
        st.markdown(f"**Name:** {meta.get('Name')}")
        st.markdown(f"**Order #:** {meta.get('Order Number')}")
        st.markdown(f"**Product #:** {meta.get('Product Number')}")
        st.markdown(f"**Dataset Sentiment:** {meta.get('Sentiment (label)')}")
    else:
        st.write("No sample selected yet.")

st.markdown("---")

# ---------------------------------------------------------------------
# Call backend when Run Analysis is clicked
# ---------------------------------------------------------------------
if run_clicked:
    if not transcript.strip():
        st.error("Please paste a transcript or load a sample call first.")
    else:
        data = None
        with st.spinner("Contacting ADK backend…"):
            try:
                payload = {"transcript": transcript}
                r = requests.post(
                    f"{BACKEND_URL}/analyze",
                    json=payload,
                    timeout=60,
                )
                r.raise_for_status()
                data = r.json()  # dict from backend
            except Exception as e:
                st.error(f"Error calling backend: {e}")

        if data:
            # -----------------------------------------------------------------
            # Raw fields from backend
            # -----------------------------------------------------------------
            summary = data.get("summary", "")
            sentiment = data.get("sentiment", "unknown")
            actions = data.get("actions", [])
            escalate_backend = data.get("escalate", False)
            trace = data.get("trace", [])

            # New fields from multi-agent backend (may or may not be present)
            frustration_score_raw = data.get("frustration_score", None)
            urgency_raw = data.get("urgency", None)

            # -----------------------------------------------------------------
            # Sentiment normalization (used for fallback)
            # -----------------------------------------------------------------
            normalized_sentiment = (sentiment or "").strip().lower()
            escalate_ui = bool(escalate_backend) or normalized_sentiment == "very_negative"

            # -----------------------------------------------------------------
            # Parse frustration_score & urgency with fallback
            # -----------------------------------------------------------------
            frustration_score = None
            if frustration_score_raw is not None:
                try:
                    frustration_score = float(frustration_score_raw)
                except (TypeError, ValueError):
                    frustration_score = None

            urgency_display = None
            if isinstance(urgency_raw, str):
                urgency_display = urgency_raw.strip().lower()

            # Fallback: if backend didn't send them, derive from sentiment
            if frustration_score is None or not urgency_display:
                if normalized_sentiment in {"very_negative", "negative"}:
                    frustration_score = frustration_score or 0.8
                    urgency_display = urgency_display or "high"
                elif normalized_sentiment == "neutral":
                    frustration_score = frustration_score or 0.4
                    urgency_display = urgency_display or "medium"
                else:  # positive / very_positive / unknown
                    frustration_score = frustration_score or 0.2
                    urgency_display = urgency_display or "low"

            # Clamp score
            if frustration_score is not None:
                frustration_score = max(0.0, min(1.0, frustration_score))

            # Debug panel to see exactly what's going on
            with st.expander("🔧 Debug (raw backend values)"):
                st.json(
                    {
                        "backend_full_response": data,
                        "summary": summary,
                        "sentiment": sentiment,
                        "frustration_score_raw": frustration_score_raw,
                        "frustration_score_final": frustration_score,
                        "urgency_raw": urgency_raw,
                        "urgency_final": urgency_display,
                        "escalate_backend": escalate_backend,
                        "normalized_sentiment": normalized_sentiment,
                        "escalate_ui": escalate_ui,
                    }
                )

            # -----------------------------------------------------------------
            # Display results
            # -----------------------------------------------------------------
            st.subheader("✅ Analysis Complete")

            # Summary
            st.markdown("### 📝 Summary")
            st.write(summary or "_No summary returned._")

            # Sentiment
            st.markdown("### 🙂 Sentiment")
            st.write(sentiment)

            # Frustration & Urgency (from new parallel agent + fallback)
            st.markdown("### 🔥 Customer Frustration & Urgency")

            col_fs, col_urg = st.columns(2)

            with col_fs:
                if frustration_score is not None:
                    st.write(
                        f"**Frustration score:** {frustration_score:.2f} "
                        "(0 = calm, 1 = very angry)"
                    )
                    st.progress(frustration_score)
                else:
                    st.write("_Frustration score not available._")

            with col_urg:
                if urgency_display:
                    pretty_urg = urgency_display.capitalize()
                    st.write(f"**Urgency:** {pretty_urg}")
                    if urgency_display == "high":
                        st.warning("Customer reports high urgency.")
                    elif urgency_display == "medium":
                        st.info("Customer reports medium urgency.")
                    else:
                        st.success("Customer urgency appears low.")
                else:
                    st.write("_Urgency not available._")

            # Recommended actions
            st.markdown("### 🧩 Recommended Actions")
            if actions:
                for a in actions:
                    st.markdown(f"- {a}")
            else:
                st.write("_No actions suggested._")

            # Escalation
            st.markdown("### 🚨 Escalation Required?")
            st.write("Yes" if escalate_ui else "No")

            # Backend info
            st.markdown("### ⚙️ Backend")
            st.code(f"POST {BACKEND_URL}/analyze")

            with st.expander("🔍 Debug Trace (agent events)"):
                if isinstance(trace, list) and trace:
                    st.json(trace)
                else:
                    st.write("No trace events recorded.")

# ---------------------------------------------------------------------
# 📊 Dataset Analytics – Metrics Agent using Code Execution
# ---------------------------------------------------------------------
st.markdown("---")
st.subheader("📊 Dataset Analytics (Code Execution Agent)")

if calls_df is not None and not calls_df.empty:
    max_n = len(calls_df)
    n_calls = st.number_input(
        "Number of sample calls to analyze",
        min_value=1,
        max_value=max_n,
        value=min(10, max_n),
        step=1,
    )

    if st.button("📈 Run Analytics on Sample Calls"):
        # Take first N non-empty transcripts
        sample_df = calls_df.dropna(subset=["Transcript"]).head(int(n_calls))
        transcripts = sample_df["Transcript"].tolist()

        if not transcripts:
            st.error("No transcripts found in the sample.")
        else:
            metrics_data = None
            with st.spinner("Contacting /metrics endpoint…"):
                try:
                    payload = {"transcripts": transcripts}
                    r = requests.post(
                        f"{BACKEND_URL}/metrics",
                        json=payload,
                        timeout=120,
                    )
                    r.raise_for_status()
                    metrics_data = r.json()
                except Exception as e:
                    st.error(f"Error calling /metrics: {e}")

            if metrics_data:
                total_calls = metrics_data.get("total_calls", len(transcripts))
                pct_very_negative = metrics_data.get("pct_very_negative", 0.0)
                avg_frustration = metrics_data.get("avg_frustration", 0.0)
                overall_summary = metrics_data.get("summary", "")
                metrics_trace = metrics_data.get("trace", [])

                st.markdown("#### 📋 Overall Call Health Summary")
                st.write(overall_summary or "_No summary returned from metrics agent._")

                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    st.metric("Total Calls", total_calls)
                with col_m2:
                    st.metric("% Very Negative", f"{pct_very_negative:.1f}%")
                with col_m3:
                    st.metric("Avg Frustration", f"{avg_frustration:.2f}")

                with st.expander("🔧 Metrics Debug (raw JSON)"):
                    st.json(metrics_data)

                with st.expander("🔍 Metrics Agent Trace"):
                    if isinstance(metrics_trace, list) and metrics_trace:
                        st.json(metrics_trace)
                    else:
                        st.write("No trace events recorded for metrics agent.")
else:
    st.info("No dataset available to run analytics.")

# ---------------------------------------------------------------------
# 🧪 Offline Evaluation (eval.py via /eval)
# ---------------------------------------------------------------------
st.markdown("---")
st.subheader("🧪 Offline Evaluation vs Dataset Labels")

if calls_df is not None and not calls_df.empty:
    max_n_eval = len(calls_df)
    n_eval = st.number_input(
        "Number of samples for offline evaluation",
        min_value=1,
        max_value=max_n_eval,
        value=min(30, max_n_eval),
        step=1,
        key="n_eval_samples",
    )

    if st.button("🔍 Run Offline Eval"):
        eval_data = None
        with st.spinner("Running offline evaluation via /eval…"):
            try:
                r = requests.get(
                    f"{BACKEND_URL}/eval",
                    params={"n_samples": int(n_eval)},
                    timeout=180,
                )
                r.raise_for_status()
                eval_data = r.json()
            except Exception as e:
                st.error(f"Error calling /eval: {e}")

        if eval_data:
            sent_eval = eval_data.get("sentiment_eval", {})
            metrics_eval = eval_data.get("metrics_eval", {})

            # --- Sentiment agent metrics ---
            st.markdown("### 🙂 Sentiment Agent Evaluation")
            st.write(f"**# Examples:** {sent_eval.get('num_examples', 0)}")
            st.write(f"**Accuracy:** {sent_eval.get('accuracy', 0.0):.2f}")

            with st.expander("Confusion Matrix (truth → prediction counts)"):
                confusion = sent_eval.get("confusion", {})
                st.json(confusion)

            with st.expander("Example Errors"):
                errors = sent_eval.get("errors", [])
                if errors:
                    st.json(errors)
                else:
                    st.write("No error examples captured.")

            # --- Metrics agent (A2A) eval ---
            st.markdown("### 📊 Metrics Agent (A2A) Evaluation")
            st.write(
                f"**Ground-truth % Very Negative:** "
                f"{metrics_eval.get('ground_truth_pct_very_negative', 0.0):.1f}%"
            )
            st.write(
                f"**Agent % Very Negative:** "
                f"{metrics_eval.get('agent_pct_very_negative', 0.0):.1f}%"
            )
            st.write(
                f"**Δ % Very Negative (agent - ground truth):** "
                f"{metrics_eval.get('delta_pct_very_negative', 0.0):.1f}%"
            )
            st.write(
                f"**Agent Avg Frustration:** "
                f"{metrics_eval.get('agent_avg_frustration', 0.0):.2f}"
            )

            with st.expander("Raw metrics eval JSON"):
                st.json(metrics_eval)
else:
    st.info("No dataset available to run offline evaluation.")

