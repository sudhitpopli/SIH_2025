import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from .dual_runner import DualSimRunner

app = FastAPI(title="SIH 2025 Traffic Telemetry API")

# Configure CORS for your React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Traffic Telemetry API is online"}

@app.websocket("/ws/telemetry")
async def telemetry_stream(websocket: WebSocket):
    await websocket.accept()
    print("[API] Client connected via WebSocket.")
    runner = DualSimRunner()
    
    try:
        while True:
            # [Step 1] Wait for commands from frontend (non-blocking)
            try:
                # We use a very short timeout to keep the loop responsive to SIM steps
                data = await asyncio.wait_for(websocket.receive_json(), timeout=0.01)
                command = data.get("command")
                
                if command == "start":
                    if not runner.is_running:
                        await runner.setup()
                        print("[API] Dual-simulation initialized.")
                    else:
                        print("[API] Simulation already running.")
                
                elif command == "stop":
                    runner.stop()
                    print("[API] Simulation stopped by user.")
                    # We don't break, so the user can 'start' again on the same socket
            
            except asyncio.TimeoutError:
                # No new command, just continue to simulation step
                pass
            except Exception as e:
                print(f"[API] Error parsing command: {e}")
            
            # [Step 2] Advance the simulations and stream telemetry
            if runner.is_running:
                telemetry = await runner.step()
                if telemetry:
                    await websocket.send_json(telemetry)
                
                # Control the framerate for smooth visualization (10 FPS)
                await asyncio.sleep(0.1)
            else:
                # Idle loop to prevent constant CPU spinning
                await asyncio.sleep(0.5)

    except WebSocketDisconnect:
        print("[API] Frontend disconnected.")
    except Exception as e:
        print(f"[API] Error in stream loop: {e}")
    finally:
        runner.stop()
