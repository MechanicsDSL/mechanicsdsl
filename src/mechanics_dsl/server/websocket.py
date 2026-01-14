"""
WebSocket streaming for real-time simulation.
"""
from typing import Dict, Any, Optional
import asyncio
import json
import numpy as np

try:
    from fastapi import APIRouter, WebSocket, WebSocketDisconnect
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    APIRouter = object
    WebSocket = object

try:
    from mechanics_dsl import PhysicsCompiler
except ImportError:
    PhysicsCompiler = None


if FASTAPI_AVAILABLE:
    websocket_router = APIRouter()
else:
    websocket_router = None


class SimulationStreamer:
    """
    Streams simulation frames to WebSocket clients.
    
    Yields simulation state at regular intervals for
    real-time visualization.
    """
    
    def __init__(
        self,
        compiler: 'PhysicsCompiler',
        t_span: tuple = (0, 10),
        num_points: int = 1000,
        frame_rate: float = 60,
    ):
        self.compiler = compiler
        self.t_span = t_span
        self.num_points = num_points
        self.frame_rate = frame_rate
        
        self._solution = None
        self._current_frame = 0
        self._running = False
    
    async def run_simulation(self):
        """Run simulation (blocking computation in executor)."""
        loop = asyncio.get_event_loop()
        
        def _simulate():
            return self.compiler.simulate(
                t_span=self.t_span,
                num_points=self.num_points
            )
        
        self._solution = await loop.run_in_executor(None, _simulate)
        return self._solution
    
    async def stream_frames(self, websocket: 'WebSocket'):
        """
        Stream simulation frames to WebSocket.
        
        Yields frames at the specified frame rate.
        """
        if self._solution is None or not self._solution['success']:
            await websocket.send_json({
                "type": "error",
                "message": "Simulation not available"
            })
            return
        
        t = self._solution['t']
        y = self._solution['y']
        n_frames = len(t)
        
        frame_interval = 1.0 / self.frame_rate
        self._running = True
        self._current_frame = 0
        
        while self._running and self._current_frame < n_frames:
            # Build frame data
            frame = {
                "type": "frame",
                "frame": self._current_frame,
                "t": float(t[self._current_frame]),
                "state": y[:, self._current_frame].tolist(),
                "progress": self._current_frame / n_frames,
            }
            
            await websocket.send_json(frame)
            self._current_frame += 1
            
            await asyncio.sleep(frame_interval)
        
        # Send completion
        await websocket.send_json({
            "type": "complete",
            "total_frames": n_frames,
        })
    
    def pause(self):
        """Pause streaming."""
        self._running = False
    
    def resume(self):
        """Resume streaming."""
        self._running = True
    
    def seek(self, frame: int):
        """Seek to specific frame."""
        if self._solution:
            self._current_frame = max(0, min(frame, len(self._solution['t']) - 1))


# Active connections
_connections: Dict[str, 'WebSocket'] = {}
_streamers: Dict[str, SimulationStreamer] = {}


if FASTAPI_AVAILABLE:
    @websocket_router.websocket("/ws/simulation")
    async def simulation_stream(websocket: WebSocket):
        """
        WebSocket endpoint for streaming simulation.
        
        Protocol:
            Client sends:
                {"action": "compile", "code": "..."}
                {"action": "simulate", "t_start": 0, "t_end": 10}
                {"action": "pause"}
                {"action": "resume"}
                {"action": "seek", "frame": 100}
                {"action": "params", "values": {"m": 2.0}}
            
            Server sends:
                {"type": "compiled", "success": true, ...}
                {"type": "frame", "t": 0.5, "state": [...]}
                {"type": "complete", "total_frames": 1000}
                {"type": "error", "message": "..."}
        """
        await websocket.accept()
        
        session_id = str(id(websocket))
        _connections[session_id] = websocket
        
        compiler = None
        streamer = None
        
        try:
            while True:
                # Receive message
                data = await websocket.receive_json()
                action = data.get("action")
                
                if action == "compile":
                    # Compile DSL code
                    if PhysicsCompiler is None:
                        await websocket.send_json({
                            "type": "error",
                            "message": "PhysicsCompiler not available"
                        })
                        continue
                    
                    compiler = PhysicsCompiler()
                    result = compiler.compile_dsl(data.get("code", ""))
                    
                    await websocket.send_json({
                        "type": "compiled",
                        "success": result['success'],
                        "system_name": result.get('system_name'),
                        "coordinates": result.get('coordinates', []),
                        "error": result.get('error'),
                    })
                
                elif action == "simulate":
                    if compiler is None:
                        await websocket.send_json({
                            "type": "error",
                            "message": "No code compiled"
                        })
                        continue
                    
                    # Create streamer
                    t_start = data.get("t_start", 0)
                    t_end = data.get("t_end", 10)
                    num_points = data.get("num_points", 1000)
                    frame_rate = data.get("frame_rate", 60)
                    
                    streamer = SimulationStreamer(
                        compiler,
                        t_span=(t_start, t_end),
                        num_points=num_points,
                        frame_rate=frame_rate
                    )
                    _streamers[session_id] = streamer
                    
                    # Run simulation
                    await websocket.send_json({"type": "simulating"})
                    await streamer.run_simulation()
                    
                    # Stream frames
                    await websocket.send_json({"type": "streaming"})
                    await streamer.stream_frames(websocket)
                
                elif action == "pause":
                    if streamer:
                        streamer.pause()
                        await websocket.send_json({"type": "paused"})
                
                elif action == "resume":
                    if streamer:
                        streamer.resume()
                        # Continue streaming from current frame
                        await streamer.stream_frames(websocket)
                
                elif action == "seek":
                    if streamer:
                        frame = data.get("frame", 0)
                        streamer.seek(frame)
                        await websocket.send_json({
                            "type": "seeked",
                            "frame": frame
                        })
                
                elif action == "params":
                    if compiler:
                        params = data.get("values", {})
                        compiler.simulator.set_parameters(params)
                        await websocket.send_json({
                            "type": "params_updated",
                            "values": params
                        })
                
                elif action == "ping":
                    await websocket.send_json({"type": "pong"})
                
        except WebSocketDisconnect:
            pass
        finally:
            # Cleanup
            if session_id in _connections:
                del _connections[session_id]
            if session_id in _streamers:
                del _streamers[session_id]


__all__ = ['websocket_router', 'SimulationStreamer', 'simulation_stream']
