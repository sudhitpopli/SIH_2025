import os
import sys
import uvicorn

# This script ensures the 'src' directory is in the Python path
# so that internal imports like 'from core.envs.sumo_env' work correctly.

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(project_root, "src")
    sys.path.append(src_path)
    
    print("[SYSTEM] Starting SIH 2025 Backend Server...")
    print(f"[SYSTEM] Project Root: {project_root}")
    
    # Run the FastAPI app
    # Note: We use the string import so uvicorn can find the app correctly
    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True)
