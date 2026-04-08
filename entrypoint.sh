#!/bin/bash
set -e

# Start the environment server in background
uvicorn coolpilot.server.app:app --host 0.0.0.0 --port 7860 --workers 1 &
SERVER_PID=$!

# Give the server time to start
sleep 5

# Run the inference agent (makes LLM calls through the hackathon proxy)
python app.py

# Keep server alive after agent finishes
wait $SERVER_PID
