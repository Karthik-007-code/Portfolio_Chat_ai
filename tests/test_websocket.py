"""
tests/test_websocket.py

Integration tests for the Ask Karthik FastAPI application and WebSocket endpoint.
"""

from unittest.mock import patch
from fastapi.testclient import TestClient
from portfolio_chat_ai.main import app
from portfolio_chat_ai.schemas.chat import ControlMessageType


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "healthy"
        assert payload["service"] == "Ask Karthik Backend"


def test_websocket_connection_and_ready_message():
    with TestClient(app) as client:
        with client.websocket_connect("/ws/chat", headers={"origin": "http://localhost:5173"}) as ws:
            ready_frame = ws.receive_json()
            assert ready_frame["type"] == ControlMessageType.SESSION_READY.value
            assert "Connected to Ask Karthik" in ready_frame["message"]


def test_websocket_rejects_unauthorized_origin():
    with TestClient(app) as client:
        # Connect from an unlisted domain
        try:
            with client.websocket_connect("/ws/chat", headers={"origin": "https://malicious-site.com"}):
                assert False, "Should have rejected unauthorized origin"
        except Exception:
            # WebSocketDisconnect or handshake rejection expected
            pass


def test_websocket_invalid_json():
    with TestClient(app) as client:
        with client.websocket_connect("/ws/chat", headers={"origin": "http://localhost:5173"}) as ws:
            ws.receive_json()  # discard ready frame
            ws.send_text("not a valid json")
            error_frame = ws.receive_json()
            assert error_frame["type"] == ControlMessageType.ERROR.value
            assert "Invalid message format" in error_frame["message"]


def test_websocket_empty_message():
    with TestClient(app) as client:
        with client.websocket_connect("/ws/chat", headers={"origin": "http://localhost:5173"}) as ws:
            ws.receive_json()  # discard ready frame
            ws.send_json({"message": "   "})
            error_frame = ws.receive_json()
            assert error_frame["type"] == ControlMessageType.ERROR.value
            assert "Message cannot be empty" in error_frame["message"]


def test_websocket_chat_streaming():
    async def mock_stream_response(conversation, system_prompt):
        yield "Hello! "
        yield "I represent Karthik."

    with patch("portfolio_chat_ai.api.websocket.stream_response", side_effect=mock_stream_response):
        with TestClient(app) as client:
            with client.websocket_connect("/ws/chat", headers={"origin": "http://localhost:5173"}) as ws:
                ready = ws.receive_json()
                assert ready["type"] == ControlMessageType.SESSION_READY.value

                # Send visitor message
                ws.send_json({"message": "Who are you?"})

                # Expect chunk 1
                chunk1 = ws.receive_json()
                assert chunk1["type"] == "chunk"
                assert chunk1["content"] == "Hello! "

                # Expect chunk 2
                chunk2 = ws.receive_json()
                assert chunk2["type"] == "chunk"
                assert chunk2["content"] == "I represent Karthik."

                # Expect stream_end control message
                end_frame = ws.receive_json()
                assert end_frame["type"] == ControlMessageType.STREAM_END.value

def test_websocket_rate_limiting():
    with patch("portfolio_chat_ai.api.websocket.stream_response"):
        with TestClient(app) as client:
            with client.websocket_connect("/ws/chat", headers={"origin": "http://localhost:5173"}) as ws:
                ws.receive_json()  # discard ready

                # Settings default MAX_MESSAGES_PER_MINUTE is 15.
                # Send 16 quick messages
                for i in range(15):
                    ws.send_json({"message": f"Message {i}"})
                    # consume the stream chunks and stream_end
                    while True:
                        msg = ws.receive_json()
                        if msg.get("type") in (ControlMessageType.STREAM_END.value, ControlMessageType.ERROR.value):
                            break

                # 16th message triggers rate limit
                ws.send_json({"message": "Trigger rate limit"})
                rate_limit_frame = ws.receive_json()
                assert rate_limit_frame["type"] == ControlMessageType.RATE_LIMITED.value
                assert "too quickly" in rate_limit_frame["message"]
