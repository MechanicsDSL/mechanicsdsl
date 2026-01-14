"""
MechanicsDSL Real-time Server

FastAPI-based server for real-time simulation and WebSocket streaming.

Usage:
    python -m mechanics_dsl.server
    uvicorn mechanics_dsl.server:app --reload
"""

from .app import create_app, app
from .routes import router
from .websocket import simulation_stream

__all__ = [
    'create_app',
    'app',
    'router',
    'simulation_stream',
]
