"""
REST API routes for MechanicsDSL server.
"""

import os
import tempfile
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

try:
    from fastapi import APIRouter, BackgroundTasks, HTTPException
    from fastapi.responses import FileResponse, JSONResponse

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    APIRouter = object

try:
    from mechanics_dsl import PhysicsCompiler
except ImportError:
    PhysicsCompiler = None

# Rate limiting
try:
    from ..utils.rate_limit import RateLimiter, SimulationRateLimiter

    rate_limiter = SimulationRateLimiter(simulations_per_minute=60, burst_limit=10)
except ImportError:
    rate_limiter = None


# Request/Response models
class CompileRequest(BaseModel):
    """Request to compile DSL code."""

    code: str = Field(..., description="MechanicsDSL code")
    validate_only: bool = Field(False, description="Only validate, don't prepare simulation")


class CompileResponse(BaseModel):
    """Response from compilation."""

    success: bool
    system_name: Optional[str] = None
    coordinates: List[str] = []
    parameters: Dict[str, float] = {}
    error: Optional[str] = None


class SimulateRequest(BaseModel):
    """Request to run simulation."""

    code: str = Field(..., description="MechanicsDSL code")
    t_start: float = Field(0, description="Start time")
    t_end: float = Field(10, description="End time")
    num_points: int = Field(1000, description="Number of output points")
    parameters: Optional[Dict[str, float]] = Field(None, description="Override parameters")


class SimulateResponse(BaseModel):
    """Response from simulation."""

    success: bool
    t: List[float] = []
    y: List[List[float]] = []  # [coord][time]
    coordinates: List[str] = []
    nfev: int = 0
    error: Optional[str] = None


class ExportRequest(BaseModel):
    """Request to export code."""

    code: str = Field(..., description="MechanicsDSL code")
    target: str = Field("cpp", description="Target language")


class ExportResponse(BaseModel):
    """Response from export."""

    success: bool
    code: Optional[str] = None
    language: str = ""
    error: Optional[str] = None


# Create router
if FASTAPI_AVAILABLE:
    router = APIRouter(tags=["simulation"])
else:
    router = None


# Session storage (in production, use Redis or similar)
_sessions: Dict[str, Any] = {}


def get_or_create_compiler(session_id: str = "default") -> "PhysicsCompiler":
    """Get or create a compiler for a session."""
    if session_id not in _sessions:
        if PhysicsCompiler is None:
            raise HTTPException(500, "PhysicsCompiler not available")
        _sessions[session_id] = PhysicsCompiler()
    return _sessions[session_id]


if FASTAPI_AVAILABLE:

    @router.post("/compile", response_model=CompileResponse)
    async def compile_dsl(request: CompileRequest, session_id: str = "default"):
        """
        Compile MechanicsDSL code.

        Returns parsed system info without running simulation.
        """
        try:
            compiler = get_or_create_compiler(session_id)
            result = compiler.compile_dsl(request.code)

            if not result["success"]:
                return CompileResponse(
                    success=False, error=result.get("error", "Compilation failed")
                )

            return CompileResponse(
                success=True,
                system_name=result.get("system_name"),
                coordinates=result.get("coordinates", []),
                parameters=dict(compiler.simulator.parameters),
            )
        except Exception as e:
            return CompileResponse(success=False, error=str(e))

    @router.post("/simulate", response_model=SimulateResponse)
    async def simulate(request: SimulateRequest, session_id: str = "default"):
        """
        Compile and run simulation.

        Returns time series data for all coordinates.
        """
        # Rate limiting
        if rate_limiter:
            if not rate_limiter.allow_simulation(
                session_id, num_points=request.num_points, time_span=request.t_end - request.t_start
            ):
                raise HTTPException(429, "Rate limit exceeded")

        try:
            compiler = get_or_create_compiler(session_id)

            # Compile
            result = compiler.compile_dsl(request.code)
            if not result["success"]:
                return SimulateResponse(
                    success=False, error=result.get("error", "Compilation failed")
                )

            # Override parameters
            if request.parameters:
                compiler.simulator.set_parameters(request.parameters)

            # Simulate
            solution = compiler.simulate(
                t_span=(request.t_start, request.t_end), num_points=request.num_points
            )

            if not solution["success"]:
                return SimulateResponse(
                    success=False, error=solution.get("error", "Simulation failed")
                )

            return SimulateResponse(
                success=True,
                t=solution["t"].tolist(),
                y=solution["y"].tolist(),
                coordinates=solution.get("coordinates", []),
                nfev=solution.get("nfev", 0),
            )
        except Exception as e:
            return SimulateResponse(success=False, error=str(e))

    @router.post("/export", response_model=ExportResponse)
    async def export_code(request: ExportRequest, session_id: str = "default"):
        """
        Export compiled system to target language.
        """
        try:
            compiler = get_or_create_compiler(session_id)

            # Compile first
            result = compiler.compile_dsl(request.code)
            if not result["success"]:
                return ExportResponse(
                    success=False, error=result.get("error"), language=request.target
                )

            # Export to temp file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=f".{request.target}", delete=False
            ) as f:
                temp_path = f.name

            compiler.export(request.target, temp_path)

            # Read generated code
            with open(temp_path, "r") as f:
                generated_code = f.read()

            os.unlink(temp_path)

            return ExportResponse(success=True, code=generated_code, language=request.target)
        except Exception as e:
            return ExportResponse(success=False, error=str(e), language=request.target)

    @router.get("/generators")
    async def list_generators():
        """List available code generators."""
        generators = [
            "cpp",
            "python",
            "rust",
            "julia",
            "matlab",
            "fortran",
            "javascript",
            "cuda",
            "openmp",
            "wasm",
            "arduino",
        ]
        return {"generators": generators}

    @router.delete("/session/{session_id}")
    async def clear_session(session_id: str):
        """Clear session state."""
        if session_id in _sessions:
            del _sessions[session_id]
            return {"cleared": True}
        return {"cleared": False}


__all__ = ["router"]
