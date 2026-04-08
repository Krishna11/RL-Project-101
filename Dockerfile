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
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy full project
COPY . .
RUN chmod +x run.sh

# Install the coolpilot package itself
RUN pip install --no-cache-dir .

# Switch to non-root
USER appuser


EXPOSE 7860

# Run the inference entrypoint via bash script to ensure web server starts
CMD ["bash", "run.sh"]
