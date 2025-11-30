# test_a2a.py

import asyncio
import json

from google.adk.agents.remote_a2a_agent import RemoteA2aAgent, AGENT_CARD_WELL_KNOWN_PATH
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types

BASE_URL = "http://localhost:8081"
CARD_URL = f"{BASE_URL}{AGENT_CARD_WELL_KNOWN_PATH}"

remote_agent = RemoteA2aAgent(
    name="test_remote_metrics_agent",
    description="Tester for the remote metrics A2A agent.",
    agent_card=CARD_URL,
)

session_service = InMemorySessionService()

# ✅ NEED app_name when using Runner this way
runner = Runner(
    app_name="callsense_adk_test",
    agent=remote_agent,
    session_service=session_service,
)


async def main():
    session_id = "test-a2a-session"

    await session_service.create_session(
        app_name="callsense_adk_test",
        user_id="tester",
        session_id=session_id,
    )

    payload = {
        "calls": [
            {"sentiment": "very_negative", "frustration_score": 0.9},
            {"sentiment": "neutral", "frustration_score": 0.4},
            {"sentiment": "positive", "frustration_score": 0.1},
        ]
    }

    content = types.Content(
        role="user",
        parts=[types.Part(text=json.dumps(payload))],
    )

    print("➡️ Sending request to remote metrics agent over A2A…")

    final_text = None
    async for event in runner.run_async(
        user_id="tester",
        session_id=session_id,
        new_message=content,
    ):
        if getattr(event, "content", None) and event.content.parts:
            texts = [p.text for p in event.content.parts if getattr(p, "text", None)]
            if texts:
                final_text = "".join(texts)

    print("\n✅ RAW OUTPUT FROM REMOTE METRICS AGENT:")
    print(final_text)


if __name__ == "__main__":
    asyncio.run(main())
