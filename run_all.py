"""
D:\knee-ai-analyzer\run_all.py
==============================
Concurrent single-command launcher for both:
1. FastAPI Backend Server (http://localhost:8000)
2. React + Vite Frontend Dashboard (http://localhost:5173)
"""

import os
import sys
import subprocess
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT_DIR / "frontend"

def main():
    print("=" * 70)
    print("🚀 Starting KneeAI-Nexus Multi-Task Workstation...")
    print("=" * 70)

    # Set PYTHONPATH to include ROOT_DIR
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT_DIR)

    # 1. Start FastAPI Backend Process
    backend_cmd = [sys.executable, "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
    backend_proc = subprocess.Popen(
        backend_cmd,
        cwd=str(ROOT_DIR),
        env=env,
        shell=False,
    )
    print(" [1/2] FastAPI Backend started on http://localhost:8000")

    time.sleep(1.5)

    # 2. Start Vite Frontend Process
    npm_cmd = "npm.cmd" if os.name == "nt" else "npm"
    frontend_proc = subprocess.Popen(
        [npm_cmd, "run", "dev"],
        cwd=str(FRONTEND_DIR),
        shell=False,
    )
    print(" [2/2] React Vite Dashboard started on http://localhost:5173")

    print("\n" + "=" * 70)
    print(" ✅ Workstation Running Successfully:")
    print("    • Frontend UI:    http://localhost:5173")
    print("    • Backend Docs:   http://localhost:8000/docs")
    print("    • Press Ctrl + C to stop all services cleanly.")
    print("=" * 70 + "\n")

    try:
        backend_proc.wait()
        frontend_proc.wait()
    except KeyboardInterrupt:
        print("\n[STOPPING] Shutting down backend and frontend...")
        backend_proc.terminate()
        frontend_proc.terminate()
        backend_proc.wait()
        frontend_proc.wait()
        print("[SUCCESS] All processes terminated cleanly.")

if __name__ == "__main__":
    main()