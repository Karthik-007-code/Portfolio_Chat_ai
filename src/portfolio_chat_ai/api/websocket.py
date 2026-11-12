"""
api/websocket.py

WebSocket endpoint (/ws/chat) and in-memory session manager for Ask Karthik.

Design and Security Principles
-------------------------------
1. Zero Persistent Storage:
   Conversation history exists solely in-memory within the active WebSocket
   handler scope. No database, file, or cache persists conversation records.
   When the connection terminates, the session data is immediately released
   to garbage collection.

2. Strict Session Isolation:
   Every WebSocket connection instantiates an isolated Session object with a
   unique UUID. No session can inspect or mutate another visitor's conversation.

3. Privacy-First Logging:
   Complete visitor message bodies and LLM outputs are never written to application
   logs. Only connection state transitions, message lengths, and error classes are logged.

4. Defense-in-Depth Protection:
   - Origin verification: Validates the 'Origin' header against ALLOWED_ORIGINS.
   - Message size validation: Limits input character length.
   - Rate limiting: Sliding-window rate limit per session (MAX_MESSAGES_PER_MINUTE).
   - Turn capping: Caps total dialogue turns (MAX_CONVERSATION_TURNS) per session.
   - Timeout handling: Handled inside ai_service.py and surfaced safely.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status
from pydantic import ValidationError

from portfolio_chat_ai.core.config import get_settings
from portfolio_chat_ai.schemas.chat import (
    ControlMessage,
    ControlMessageType,
    IncomingMessage,
    StreamChunk,
)
from portfolio_chat_ai.services.ai_service import AIServiceError, stream_response

logger = logging.getLogger(__name__)

router = APIRouter()


@dataclass
class Session:
    """
    Temporary in-memory session state for an active visitor connection.

    Attributes
    ----------
    session_id : str
        Cryptographically secure UUID identifying the session.
    conversation : list[dict[str, str]]
        Active in-memory history: [{"role": "user"|"assistant", "content": "..."}]
    message_timestamps : list[float]
        Monotonic timestamps of messages received in the current minute window.
    turns_count : int
        Number of user questions processed in this session.
    """

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    conversation: list[dict[str, str]] = field(default_factory=list)
    message_timestamps: list[float] = field(default_factory=list)
    turns_count: int = 0

    def record_message_and_check_rate_limit(self, max_per_minute: int) -> bool:
        """
        Record a message arrival timestamp using a 60-second sliding window.
        Returns False if the rate limit is exceeded, True otherwise.
        """
        now = time.monotonic()
        # Discard timestamps older than 60 seconds
        self.message_timestamps = [
            ts for ts in self.message_timestamps if now - ts < 60.0
        ]
        if len(self.message_timestamps) >= max_per_minute:
            return False

        self.message_timestamps.append(now)
        return True


def _is_origin_allowed(websocket: WebSocket, allowed_origins: list[str]) -> bool:
    """
    Validate incoming WebSocket Origin header against configured allowed origins.
    Permits origin-less connections only when in development/testing (e.g. tools).
    """
    origin = websocket.headers.get("origin")
    if not origin:
        # Some non-browser clients or test runners omit Origin.
        # Check if wildcard or localhost is allowed
        allow_empty = any("localhost" in o or "127.0.0.1" in o for o in allowed_origins)
        return allow_empty

    # Normalize by stripping trailing slash
    norm_origin = origin.rstrip("/")
    for allowed in allowed_origins:
        if allowed == "*" or allowed.rstrip("/") == norm_origin:
            return True

    return False


@router.websocket("/ws/chat")
async def websocket_chat_endpoint(websocket: WebSocket) -> None:
    """
    Main WebSocket endpoint for Ask Karthik chat sessions.

    Lifecycle:
      1. Origin validation
      2. Accept connection
      3. Create temporary in-memory session
      4. Message processing loop (validate -> LLM stream -> stream chunk frames)
      5. Stream end or error control frame
      6. On disconnect/exit -> immediately delete session state from memory
    """
    settings = get_settings()

    # 1. Origin verification
    if not _is_origin_allowed(websocket, settings.ALLOWED_ORIGINS):
        origin = websocket.headers.get("origin")
        logger.warning("Rejected WebSocket connection from unauthorized origin: %s", origin)
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await websocket.accept()

    # 2. Initialize ephemeral in-memory session
    session = Session()
    logger.info("Session %s: Connected", session.session_id)

    # Inform client that session is ready
    ready_msg = ControlMessage(
        type=ControlMessageType.SESSION_READY,
        message="Connected to Ask Karthik.",
    )
    await websocket.send_text(ready_msg.model_dump_json())

    # Retrieve prebuilt system prompt from application state
    system_prompt: str = getattr(websocket.app.state, "system_prompt", "")

    try:
        while True:
            # 3. Receive message frame from client
            raw_text = await websocket.receive_text()

            # 4. Parse & Validate incoming JSON
            try:
                incoming = IncomingMessage.model_validate_json(raw_text)
            except (ValidationError, ValueError):
                error_msg = ControlMessage(
                    type=ControlMessageType.ERROR,
                    message="Invalid message format. Expected JSON payload: {'message': '...'}",
                )
                await websocket.send_text(error_msg.model_dump_json())
                continue

            cleaned_message = incoming.message.strip()
            if not cleaned_message:
                error_msg = ControlMessage(
                    type=ControlMessageType.ERROR,
                    message="Message cannot be empty.",
                )
                await websocket.send_text(error_msg.model_dump_json())
                continue

            # 5. Validate message length
            if len(cleaned_message) > settings.MAX_MESSAGE_LENGTH:
                error_msg = ControlMessage(
                    type=ControlMessageType.ERROR,
                    message=(
                        f"Message exceeds maximum allowed length of "
                        f"{settings.MAX_MESSAGE_LENGTH} characters."
                    ),
                )
                await websocket.send_text(error_msg.model_dump_json())
                continue

            # 6. Enforce per-session sliding window rate limit
            if not session.record_message_and_check_rate_limit(settings.MAX_MESSAGES_PER_MINUTE):
                logger.warning(
                    "Session %s: Rate limit triggered (%d msgs/min)",
                    session.session_id,
                    settings.MAX_MESSAGES_PER_MINUTE,
                )
                rate_msg = ControlMessage(
                    type=ControlMessageType.RATE_LIMITED,
                    message="You are sending messages too quickly. Please pause for a moment.",
                )
                await websocket.send_text(rate_msg.model_dump_json())
                continue

            # 7. Enforce max conversation turns
            if session.turns_count >= settings.MAX_CONVERSATION_TURNS:
                logger.warning(
                    "Session %s: Turn limit reached (%d turns)",
                    session.session_id,
                    settings.MAX_CONVERSATION_TURNS,
                )
                limit_msg = ControlMessage(
                    type=ControlMessageType.LIMIT_REACHED,
                    message=(
                        "Session conversation turn limit reached. Please refresh "
                        "the chat to start a new conversation."
                    ),
                )
                await websocket.send_text(limit_msg.model_dump_json())
                continue

            # 8. Add user message to active session history
            session.conversation.append({"role": "user", "content": cleaned_message})
            session.turns_count += 1

            logger.info(
                "Session %s: Processing turn %d (char_length=%d)",
                session.session_id,
                session.turns_count,
                len(cleaned_message),
            )

            # 9. Call AI Service and stream tokens to client
            accumulated_response: list[str] = []
            try:
                async for chunk_token in stream_response(session.conversation, system_prompt):
                    accumulated_response.append(chunk_token)
                    chunk_frame = StreamChunk(content=chunk_token)
                    await websocket.send_text(chunk_frame.model_dump_json())

                # Send stream completion signal
                end_frame = ControlMessage(type=ControlMessageType.STREAM_END)
                await websocket.send_text(end_frame.model_dump_json())

                # Record assistant response in in-memory conversation history
                full_assistant_reply = "".join(accumulated_response)
                session.conversation.append({"role": "assistant", "content": full_assistant_reply})

            except AIServiceError as ai_err:
                error_frame = ControlMessage(
                    type=ControlMessageType.ERROR,
                    message=ai_err.user_message,
                )
                await websocket.send_text(error_frame.model_dump_json())
            except Exception:
                logger.exception("Session %s: Unexpected failure during response streaming", session.session_id)
                error_frame = ControlMessage(
                    type=ControlMessageType.ERROR,
                    message="An unexpected error occurred while generating the response.",
                )
                await websocket.send_text(error_frame.model_dump_json())

    except WebSocketDisconnect:
        logger.info("Session %s: Disconnected by client", session.session_id)
    except Exception as exc:
        logger.warning("Session %s: Connection terminated with error: %s", session.session_id, type(exc).__name__)
    finally:
        # 10. Ephemeral Cleanup: Wipe all session memory immediately upon exit
        session.conversation.clear()
        session.message_timestamps.clear()
        del session
        logger.info("Session destroyed and memory released")

