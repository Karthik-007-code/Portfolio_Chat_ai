"""
services/ai_service.py

Manages all communication with the LLM provider.

Design
------
- Wraps AsyncOpenAI (which is compatible with Groq's API).
- Exposes a single async generator:  stream_response(messages, system_prompt)
- Yields token chunks as they arrive — the WebSocket handler sends each
  chunk to the browser immediately, so the user sees text appear in real time.
- Handles common failure modes explicitly: timeout, auth error, rate limit,
  model not found, and generic API errors.
- Never logs the content of messages or responses.
- The client is initialised once at module import and reused for all requests
  (connection pooling — avoids opening a new HTTPS connection per message).

Usage
-----
    from portfolio_chat_ai.services.ai_service import stream_response

    async for chunk in stream_response(conversation_history, system_prompt):
        await websocket.send_text(chunk)
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncGenerator

from openai import AsyncOpenAI, APIConnectionError, APIStatusError, APITimeoutError

from portfolio_chat_ai.core.config import get_settings

logger = logging.getLogger(__name__)

# Initialise once at module load — AsyncOpenAI maintains an internal
# connection pool, so this is both efficient and thread-safe.
_settings = get_settings()
_client = AsyncOpenAI(
    api_key=_settings.GROQ_API_KEY,
    base_url=_settings.LLM_BASE_URL,
    # Pass a generous connect timeout; the per-request stream timeout is
    # enforced separately via asyncio.timeout() below.
    timeout=float(_settings.LLM_TIMEOUT_SECONDS) + 5,
)


class AIServiceError(Exception):
    """
    Raised when the LLM service encounters an unrecoverable error.

    Attributes
    ----------
    user_message : str
        A safe, friendly message that can be shown to the visitor.
    """

    def __init__(self, user_message: str) -> None:
        super().__init__(user_message)
        self.user_message = user_message


async def stream_response(
    conversation: list[dict[str, str]],
    system_prompt: str,
) -> AsyncGenerator[str, None]:
    """
    Stream the LLM response token by token.

    Parameters
    ----------
    conversation : list[dict[str, str]]
        The conversation history for this session, as a list of
        {"role": "user" | "assistant", "content": "..."} dicts.
        Does NOT include the system prompt — that is prepended here.

    system_prompt : str
        The prebuilt system prompt containing Karthik's knowledge.
        Generated once at startup by core/prompts.py.

    Yields
    ------
    str
        Individual token strings as they arrive from the LLM.

    Raises
    ------
    AIServiceError
        If the LLM call fails in a way that the caller should surface
        to the visitor (timeout, auth failure, etc.).
    """
    settings = get_settings()

    # Build the full message list: system prompt first, then conversation.
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        *conversation,
    ]

    logger.info(
        "Sending request to LLM. Model=%s, turns=%d",
        settings.LLM_MODEL,
        len(conversation),
    )

    try:
        async with asyncio.timeout(settings.LLM_TIMEOUT_SECONDS):
            stream = await _client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=messages,
                stream=True,
                # Leave max_tokens unset — let the model decide response length.
                # Adjust this if you need to control token costs.
                temperature=0.7,
            )

            async for chunk in stream:
                # Each chunk may or may not contain a delta
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta

    except asyncio.TimeoutError:
        logger.warning("LLM request timed out after %ds", settings.LLM_TIMEOUT_SECONDS)
        raise AIServiceError(
            "The response took too long. Please try again in a moment."
        )

    except APITimeoutError:
        logger.warning("OpenAI SDK reported a request timeout.")
        raise AIServiceError(
            "The response took too long. Please try again in a moment."
        )

    except APIStatusError as exc:
        status = exc.status_code

        if status == 401:
            # Auth failure — this is a configuration error, not visitor error
            logger.error("LLM auth failed (401). Check GROQ_API_KEY in .env.")
            raise AIServiceError(
                "The assistant is temporarily unavailable. Please try again later."
            )
        elif status == 429:
            logger.warning("LLM rate limit hit (429).")
            raise AIServiceError(
                "Too many requests right now. Please wait a moment and try again."
            )
        elif status == 404:
            logger.error(
                "LLM model not found (404). Check LLM_MODEL in .env. "
                "Current value: %s",
                settings.LLM_MODEL,
            )
            raise AIServiceError(
                "The assistant is temporarily unavailable. Please try again later."
            )
        else:
            logger.error("LLM API error (HTTP %d).", status)
            raise AIServiceError(
                "Something went wrong with the AI service. Please try again."
            )

    except APIConnectionError:
        logger.error("Could not connect to LLM provider.")
        raise AIServiceError(
            "Could not reach the AI service. Please check your connection and try again."
        )

    except Exception as exc:
        # Catch-all — log the exception type but NOT the message content
        logger.error("Unexpected error in ai_service: %s", type(exc).__name__)
        raise AIServiceError(
            "An unexpected error occurred. Please try again."
        ) from exc

    logger.info("LLM stream completed. Model=%s", settings.LLM_MODEL)


def build_conversation_messages(history: list[dict[str, str]]) -> list[dict[str, str]]:
    """
    Validate and return a copy of the conversation history suitable for
    sending to the LLM.

    Filters out any entries that don't have both 'role' and 'content' keys
    to prevent malformed API requests.

    Parameters
    ----------
    history : list[dict[str, str]]
        Raw conversation history from the in-memory session.

    Returns
    -------
    list[dict[str, str]]
        Sanitised copy of the history.
    """
    valid_roles = {"user", "assistant"}
    return [
        {"role": m["role"], "content": m["content"]}
        for m in history
        if m.get("role") in valid_roles and m.get("content")
    ]

