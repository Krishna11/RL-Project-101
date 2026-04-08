#!/bin/bash
set -e

# Start the FastAPI environment server in the background
echo "Starting CoolPilot environment server..."
uvicorn coolpilot.server.app:app --host 0.0.0.0 --port 7860 --workers 1 &
SERVER_PID=$!

# Wait for the server to be ready
echo "Waiting for server to become available..."
until curl -s http://127.0.0.1:7860/health > /dev/null; do
  sleep 1
done
echo "Server is up!"

# Run the inference agent
export ENV_BASE_URL="http://127.0.0.1:7860"
echo "Starting inference agent..."
python inference.py || echo "Inference script exited (normal for dummy LLM). Container staying alive..."

# Keep the container alive by waiting for the server
wait $SERVER_PID
