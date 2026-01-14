"""
FastAPI application for MechanicsDSL server.
"""
from typing import Optional
import os

try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    FastAPI = None


def create_app(
    title: str = "MechanicsDSL API",
    version: str = "1.0.0",
    enable_cors: bool = True,
    cors_origins: Optional[list] = None,
) -> 'FastAPI':
    """
    Create FastAPI application.
    
    Args:
        title: API title
        version: API version
        enable_cors: Enable CORS middleware
        cors_origins: Allowed origins (default: all)
        
    Returns:
        FastAPI application
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError(
            "FastAPI is not installed. Install with: pip install fastapi uvicorn"
        )
    
    app = FastAPI(
        title=title,
        version=version,
        description="Real-time physics simulation API",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    # CORS
    if enable_cors:
        origins = cors_origins or ["*"]
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    # Import and include routes
    from .routes import router
    app.include_router(router, prefix="/api")
    
    # WebSocket endpoint
    from .websocket import websocket_router
    app.include_router(websocket_router)
    
    # Health check
    @app.get("/health")
    async def health():
        return {"status": "healthy", "service": "mechanics_dsl"}
    
    return app


# Default app instance
if FASTAPI_AVAILABLE:
    app = create_app()
else:
    app = None


def main():
    """Run the server via uvicorn."""
    try:
        import uvicorn
    except ImportError:
        print("uvicorn not installed. Run: pip install uvicorn")
        return
    
    uvicorn.run(
        "mechanics_dsl.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    main()


__all__ = ['create_app', 'app', 'main']
