"""
HTTP client for the AI Agent microservice (Bible Chapter 7.2, Boundary B).
If the AI Agent isn't running yet (Member 4 still building), calls fall back
to realistic mock data so backend/frontend integration isn't blocked.
"""
import os
import httpx

AGENT_BASE_URL = os.getenv("AGENT_BASE_URL", "http://127.0.0.1:8001")
TIMEOUT = 30.0


def _mock_segment(payload: dict) -> dict:
    return {
        "capture_id": payload.get("capture_id", "cap_mock"),
        "class_percentages": {
            "water": 34.2, "vegetation": 18.1,
            "bare_land": 21.6, "built_up": 26.1,
        },
        "mask_path": None,
        "model_version": "mock-v0",
    }


def _mock_anomaly(payload: dict) -> dict:
    return {
        "anomaly_score": 0.62,
        "change_type": "persistent_degradation",
        "expected_ndwi": 0.45, "observed_ndwi": 0.31,
    }


def _mock_changepoint(payload: dict) -> dict:
    return {
        "change_point_year": None,
        "before_mean": None, "after_mean": None,
        "narrative": "Insufficient historical data (mock response -- AI Agent not yet connected).",
    }


def _mock_structures(payload: dict) -> dict:
    return {"new_structures": None}


def _mock_cause(payload: dict) -> dict:
    return {"hypotheses": []}


def _mock_priority(payload: dict) -> dict:
    return {"priority_score": 50.0}


async def _call_agent(endpoint: str, payload: dict, mock_fn) -> dict:
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.post(f"{AGENT_BASE_URL}{endpoint}", json=payload)
            response.raise_for_status()
            return response.json()
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError):
        result = mock_fn(payload)
        result["_mock"] = True
        return result


async def segment(payload: dict) -> dict:
    return await _call_agent("/agent/segment", payload, _mock_segment)


async def anomaly(payload: dict) -> dict:
    return await _call_agent("/agent/anomaly", payload, _mock_anomaly)


async def changepoint(payload: dict) -> dict:
    return await _call_agent("/agent/changepoint", payload, _mock_changepoint)


async def structures(payload: dict) -> dict:
    return await _call_agent("/agent/structures", payload, _mock_structures)


async def cause(payload: dict) -> dict:
    return await _call_agent("/agent/cause", payload, _mock_cause)


async def priority(payload: dict) -> dict:
    return await _call_agent("/agent/priority", payload, _mock_priority)
