#!/bin/bash
# Move to the directory where this script is located
cd "$(dirname "$0")"

echo "=========================================================="
echo "          AI Exercise Coach - Desktop Launcher"
echo "=========================================================="
echo "Starting local environment and launching camera..."

# Check if .venv exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
else
    echo "Warning: .venv not found. Running with system python..."
fi

# Run the webcam demo
python demo/run_webcam_demo.py

# Keep terminal open if it exits
echo ""
echo "Press any key to close this window..."
read -n 1
