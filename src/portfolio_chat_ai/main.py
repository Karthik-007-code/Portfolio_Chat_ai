"""
main.py

FastAPI Application Factory and Server Entry Point for Ask Karthik.

Responsibilities:
-----------------
1. Lifespan context: Reads all knowledge JSON files once at server startup
   and compiles the system prompt into app.state.
2. Registers CORS middleware for the configured portfolio domains.
3. Exposes a lightweight /health endpoint for deployment health checks.
4. Mounts the WebSocket router (/ws/chat).
5. Provides a main() entry point to boot the Uvicorn server locally.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from portfolio_chat_ai.api.websocket import router as ws_router
from portfolio_chat_ai.core.config import get_settings
from portfolio_chat_ai.core.prompts import build_system_prompt
from portfolio_chat_ai.services.knowledge_service import load_all_knowledge

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ask_karthik")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.

    Executed on application startup and shutdown:
    - Startup: Loads structured JSON knowledge files and compiles the system prompt
      into application state memory so no per-request disk reads are needed.
    - Shutdown: Cleans up resources.
    """
    settings = get_settings()
    logger.info("Initializing Ask Karthik backend service...")

    # Load knowledge files once
    knowledge = load_all_knowledge(settings.KNOWLEDGE_DIR)
    app.state.knowledge = knowledge

    # Pre-build system prompt once
    app.state.system_prompt = build_system_prompt(knowledge)

    logger.info(
        "Ask Karthik ready. Knowledge sections: %s | System prompt size: %d chars",
        list(knowledge.keys()),
        len(app.state.system_prompt),
    )

    yield

    logger.info("Shutting down Ask Karthik backend service.")


def create_app() -> FastAPI:
    """
    Factory function creating and configuring the FastAPI application.
    """
    settings = get_settings()

    app = FastAPI(
        title="Ask Karthik – Personal AI Assistant",
        description="Backend API and WebSocket service representing Ankipalli Karthik.",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Configure CORS for portfolio website and local development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    @app.get("/")
    def root():
        return{
            "status":"success",
            "message":"this is the root route do /healthy for health"

        }
    # Health check endpoint for monitoring & Render deployment checks
    @app.get("/health", tags=["Health"])
    async def health_check() -> dict[str, str]:
        return {
            "status": "healthy",
            "service": "Ask Karthik Backend",
            "model": settings.LLM_MODEL,
        }

    # Mount WebSocket chat route
    app.include_router(ws_router)

    return app


app = create_app()


def main() -> None:
    """Entry point for local execution."""
    uvicorn.run(
        "portfolio_chat_ai.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()

