FROM python:3.11-slim

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
    "fastapi>=0.115.0" \
    "pydantic>=2.0.0" \
    "uvicorn[standard]>=0.30.0" \
    "httpx>=0.27.0" \
    "websockets>=15.0.1" \
    "openai>=1.50.0" \
    "requests>=2.31.0"

# Copy full project
COPY . .

# Switch to non-root
USER appuser

# HF Spaces expects port 7860
EXPOSE 7860

CMD ["uvicorn", "coolpilot.server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
