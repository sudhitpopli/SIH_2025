import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import xml.etree.ElementTree as ET
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

@app.get("/api/map")
async def get_map():
    """Extracts raw polygon nodes from SUMO for absolute frontend Map accuracy"""
    net_file = "maps/connaught_place.net.xml"
    if not os.path.exists(net_file):
        return {"roads": []}
    
    tree = ET.parse(net_file)
    root = tree.getroot()
    roads = []
    
    for edge in root.findall('edge'):
        for lane in edge.findall('lane'):
            shape_str = lane.get('shape')
            if shape_str:
                points = [[float(p.split(',')[0]), float(p.split(',')[1])] for p in shape_str.split(' ')]
                roads.append(points)
    
    return {"roads": roads}

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
            except (WebSocketDisconnect, RuntimeError):
                break # the backend or frontend cleanly severed the socket 
            except Exception as e:
                print(f"[API] Error parsing command: {e}")
            
            # [Step 2] Advance the simulations and stream telemetry
            if runner.is_running:
                telemetry = await runner.step()
                if telemetry:
                    await websocket.send_json(telemetry)
                
                # Control the framerate for smooth visualization (~25 FPS)
                await asyncio.sleep(0.04)
            else:
                # Idle loop to prevent constant CPU spinning
                await asyncio.sleep(0.5)

    except WebSocketDisconnect:
        print("[API] Frontend disconnected.")
    except Exception as e:
        print(f"[API] Error in stream loop: {e}")
    finally:
        runner.stop()
