"""
schemas/chat.py

Pydantic models for messages flowing over the WebSocket connection.

These models serve as the contract between the React frontend and the
FastAPI backend.  Every message in and out is validated against these
shapes, so malformed data is caught early with a clear error.

Message flow:
  Frontend  →  Backend : IncomingMessage  (JSON)
  Backend   →  Frontend: StreamChunk      (JSON, one per token)
  Backend   →  Frontend: ControlMessage   (JSON, lifecycle events)
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class IncomingMessage(BaseModel):
    """
    A message sent by the visitor to the chatbot.

    Fields
    ------
    message : str
        The visitor's text input.  Max length is enforced separately in
        the WebSocket handler (after parsing) using the configured limit.
    """

    message: str = Field(
        ...,
        min_length=1,
        description="The visitor's message text.",
    )


class StreamChunk(BaseModel):
    """
    A single streaming token chunk sent from the backend to the frontend.

    The frontend should append `content` to the in-progress assistant
    bubble as chunks arrive.

    Fields
    ------
    type : str
        Always "chunk" — lets the frontend distinguish chunk frames
        from control frames.
    content : str
        The token text to append.
    """

    type: str = "chunk"
    content: str


class ControlMessageType(str, Enum):
    """Enumeration of control message types."""

    STREAM_END = "stream_end"       # LLM response finished streaming
    ERROR = "error"                 # An error occurred
    SESSION_READY = "session_ready" # Session created, ready to chat
    RATE_LIMITED = "rate_limited"   # Too many messages
    LIMIT_REACHED = "limit_reached" # Conversation turn limit reached


class ControlMessage(BaseModel):
    """
    A control / lifecycle message sent from the backend to the frontend.

    These are distinct from streaming content and are used to signal
    state transitions (stream finished, errors, session events).

    Fields
    ------
    type : ControlMessageType
        The type of control event.
    message : Optional[str]
        A human-readable description of the event (shown to the visitor
        for errors, hidden otherwise).
    """

    type: ControlMessageType
    message: Optional[str] = None

