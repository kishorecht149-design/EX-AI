import os
import sys

def main():
    print("\n" + "="*60)
    print("🚀 AI EXERCISE COACH - DEPLOYMENT COMPATIBILITY BRIDGE 🚀")
    print("="*60)
    print("System detected a legacy Streamlit start command from Render UI.")
    print("Intercepting command successfully! Rerouting to Flask backend...")
    
    # Extract the port from the arguments or default to $PORT or 8501
    port = os.environ.get("PORT", "8501")
    
    # Check sys.argv for '--server.port'
    for i in range(len(sys.argv)):
        if sys.argv[i] == "--server.port" and i + 1 < len(sys.argv):
            port = sys.argv[i + 1]
            break

    print(f"Target deployment port identified: {port}")
    
    # Construct the Gunicorn command to serve our Flask app
    # We use Gunicorn because it is installed, robust, and designed for production.
    gunicorn_cmd = ["gunicorn", "app:app", "--bind", f"0.0.0.0:{port}", "--timeout", "120"]
    
    print(f"Executing: {' '.join(gunicorn_cmd)}")
    print("="*60 + "\n")
    
    # Flush stdout to ensure logs are visible on Render before swapping
    sys.stdout.flush()
    sys.stderr.flush()
    
    # Hot-swap the current process with Gunicorn
    try:
        os.execvp(gunicorn_cmd[0], gunicorn_cmd)
    except Exception as e:
        print(f"Error executing Gunicorn: {e}")
        # Fallback to standard python runner if Gunicorn is not found
        print("Falling back to python runner...")
        os.environ["PORT"] = port
        python_cmd = [sys.executable, "app.py"]
        os.execvp(python_cmd[0], python_cmd)

if __name__ == "__main__":
    main()
