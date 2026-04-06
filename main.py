import subprocess
import time
import sys
import os
import signal

def run_app():
    """
    Entry point to run both the FastAPI backend and Streamlit frontend together.
    """
    print("🚀 Starting Helmet Detection System...")

    # 1. Start the FastAPI backend
    # We use 'sys.executable -m uvicorn' to ensure we use the same python environment
    backend_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    print("✅ Backend starting on http://localhost:8000")

    # 2. Wait a moment for the backend to initialize
    time.sleep(2)

    # 3. Start the Streamlit frontend
    frontend_process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    print("✅ Frontend starting on http://localhost:8501")
    print("\n💡 Press Ctrl+C to stop both servers at once.\n")

    try:
        # Keep the script running while both processes are active
        while True:
            # Check if processes are still running
            if backend_process.poll() is not None:
                print("⚠️ Backend process stopped unexpectedly.")
                break
            if frontend_process.poll() is not None:
                print("⚠️ Frontend process stopped unexpectedly.")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down servers...")
    finally:
        # Gracefully terminate both processes on exit
        backend_process.terminate()
        frontend_process.terminate()
        print("👋 Both servers have been stopped.")

if __name__ == "__main__":
    run_app()
