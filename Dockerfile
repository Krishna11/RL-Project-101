FROM python:3.11-slim

# Install system dependencies including curl for health checks
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Non-root user (HF Spaces requirement)
RUN useradd -m -u 1000 appuser
ENV HOME=/home/appuser
WORKDIR $HOME/app

# Writable cache for HF libraries
ENV HF_HOME=/tmp/.cache/huggingface
ENV PYTHONUNBUFFERED=1

# Install deps first (layer caching)
COPY pyproject.toml ./
RUN pip install --no-cache-dir \
    "openenv-core[core]>=0.2.2" \
    "fastapi>=0.115.0" \
    "pydantic>=2.0.0" \
    "uvicorn[standard]>=0.30.0" \
    "httpx>=0.27.0" \
    "websockets>=15.0.1" \
    "openai>=1.50.0" \
    "requests>=2.31.0"

# Copy full project
COPY . .
RUN chmod +x run.sh

# Install the coolpilot package itself
RUN pip install --no-cache-dir .

# Switch to non-root
USER appuser


EXPOSE 7860

# Run the script that starts both server and inference agent
CMD ["./run.sh"]
