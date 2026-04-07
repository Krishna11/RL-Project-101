# 🚀 CoolPilot Deployment Guide (For Beginners)

This guide will walk you through deploying CoolPilot step-by-step.
No prior deployment experience needed — just follow along!

---

## Table of Contents

1. [What Does "Deployment" Mean?](#what-does-deployment-mean)
2. [Option 1: Run Locally (Your Computer)](#option-1-run-locally-your-computer)
3. [Option 2: Deploy to Hugging Face Spaces (Free Cloud)](#option-2-deploy-to-hugging-face-spaces-free-cloud)
4. [Option 3: Deploy with Docker (Advanced)](#option-3-deploy-with-docker-advanced)
5. [Troubleshooting](#troubleshooting)

---

## What Does "Deployment" Mean?

Right now, CoolPilot only runs on **your computer**. Deployment means putting it on the **internet** so:
- 🌐 Anyone can access it from anywhere
- 🤖 The hackathon judges can test it
- 💤 It keeps running even when your computer is off

Think of it like this:
```
Your Computer (local)     →     Cloud Server (deployed)
Only you can use it              Anyone with the link can use it
Stops when you close it          Runs 24/7
```

---

## Option 1: Run Locally (Your Computer)

**Best for:** Testing, development, debugging

### Prerequisites
- ✅ Python 3.10 or higher installed
- ✅ Git installed

### Steps

**1. Clone the repository**
```bash
git clone https://github.com/Krishna11/RL-Project-101.git
cd RL-Project-101
```

**2. Create a virtual environment**

> A virtual environment is like a clean room for your project — it keeps
> your project's packages separate from everything else on your computer.

```powershell
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
```

```bash
# Mac / Linux
python -m venv .venv
source .venv/bin/activate
```

> ✅ You'll see `(.venv)` at the start of your terminal — that means it's active!

**3. Install all dependencies**
```bash
pip install -e ".[dev]"
```

> This reads `pyproject.toml` and installs everything the project needs.

**4. Make sure it works — run the tests**
```bash
pytest tests/ -v
```

> You should see **73 passed** in green. If you see red errors, check
> the [Troubleshooting](#troubleshooting) section below.

**5. Start the server**
```bash
uvicorn coolpilot.server.app:app --port 7860
```

**6. Test it's working**

Open a new terminal (keep the server running!) and run:

```powershell
# Windows
Invoke-RestMethod http://localhost:7860/health
```

```bash
# Mac / Linux
curl http://localhost:7860/health
```

You should see:
```json
{"status": "ok"}
```

🎉 **Congratulations!** CoolPilot is running locally on `http://localhost:7860`

---

## Option 2: Deploy to Hugging Face Spaces (Free Cloud)

**Best for:** Hackathon submission, sharing with judges, free hosting

> **Hugging Face Spaces** is a free platform that runs your app in the cloud.
> It reads your `Dockerfile` and handles everything automatically.

### Prerequisites
- ✅ A free [Hugging Face account](https://huggingface.co/join)
- ✅ Git installed on your computer

### Step-by-Step

**1. Create a Hugging Face account**

Go to [huggingface.co/join](https://huggingface.co/join) and sign up (it's free).

**2. Create a new Space**

- Go to [huggingface.co/new-space](https://huggingface.co/new-space)
- Fill in:
  - **Space name:** `coolpilot`
  - **License:** BSD-3-Clause
  - **SDK:** Select **Docker**
  - **Visibility:** Public

- Click **Create Space**

**3. Copy the Space's git URL**

After creating, you'll see something like:
```
https://huggingface.co/spaces/YOUR_USERNAME/coolpilot
```

The git URL will be:
```
https://huggingface.co/spaces/YOUR_USERNAME/coolpilot
```

**4. Add Hugging Face as a remote**

In your project folder:
```bash
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/coolpilot
```

> Replace `YOUR_USERNAME` with your actual Hugging Face username!

**5. Push your code**
```bash
git push hf kunaldev:main
```

> This pushes your `kunaldev` branch to the `main` branch on Hugging Face.

It will ask for your Hugging Face credentials:
- **Username:** your HF username
- **Password:** use an [Access Token](https://huggingface.co/settings/tokens) (not your password!)

**6. Wait for the build**

- Go to your Space page: `https://huggingface.co/spaces/YOUR_USERNAME/coolpilot`
- Click the **"Logs"** tab at the top
- You'll see Docker building your app — this takes **2-5 minutes** the first time
- When you see `Application startup complete`, it's ready! 🎉

**7. Test your deployed app**

Your app is now live at:
```
https://YOUR_USERNAME-coolpilot.hf.space
```

Test the health endpoint:
```
https://YOUR_USERNAME-coolpilot.hf.space/health
```

### What Happens Behind the Scenes?

```
You push code → HF reads your Dockerfile → Builds a container
→ Installs Python + dependencies → Starts uvicorn on port 7860
→ Your app is live! 🌐
```

All of this is defined in your `Dockerfile`:
```dockerfile
FROM python:3.11-slim          ← Uses Python 3.11
COPY . .                       ← Copies your code
RUN pip install ...            ← Installs dependencies
EXPOSE 7860                    ← Opens port 7860
CMD ["uvicorn", "coolpilot.server.app:app", ...]  ← Starts server
```

---

## Option 3: Deploy with Docker (Advanced)

**Best for:** Running on your own server, team sharing, consistent environments

> **Docker** packages your entire app into a "container" — like a
> shipping container for software. It works the same everywhere.

### Prerequisites
- ✅ [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed

### Steps

**1. Build the Docker image**

Open a terminal in the project folder:
```bash
docker build -t coolpilot .
```

> This reads the `Dockerfile` and creates an image. Takes 2-3 minutes first time.

You'll see output like:
```
Step 1/9 : FROM python:3.11-slim
Step 2/9 : RUN useradd -m -u 1000 appuser
...
Successfully built abc123def456
Successfully tagged coolpilot:latest
```

**2. Run the container**
```bash
docker run -p 7860:7860 coolpilot
```

> `-p 7860:7860` means "connect port 7860 on your computer to port 7860 in the container"

**3. Test it**
```bash
curl http://localhost:7860/health
```

**4. Stop the container**

Press `Ctrl + C` in the terminal, or run:
```bash
docker ps                    # Find the container ID
docker stop <container_id>   # Stop it
```

### Useful Docker Commands
```bash
docker build -t coolpilot .           # Build image
docker run -p 7860:7860 coolpilot     # Run container
docker run -d -p 7860:7860 coolpilot  # Run in background
docker ps                             # List running containers
docker stop <id>                      # Stop a container
docker logs <id>                      # View logs
docker images                         # List all images
```

---

## Troubleshooting

### ❌ `python: command not found`
Python isn't installed or not in PATH.
- **Windows:** Download from [python.org](https://www.python.org/downloads/) — check "Add Python to PATH" during install
- **Mac:** `brew install python`
- **Linux:** `sudo apt install python3 python3-pip python3-venv`

### ❌ `pip install` fails with dependency errors
```bash
# Make sure you're in the virtual environment first
# You should see (.venv) in your prompt
pip install --upgrade pip
pip install -e ".[dev]"
```

### ❌ Tests fail
```bash
# Check Python version (needs 3.10+)
python --version

# Try reinstalling dependencies
pip install -e ".[dev]" --force-reinstall
```

### ❌ `Address already in use` when starting server
Another process is using port 7860.
```powershell
# Windows — find and kill the process
netstat -ano | findstr :7860
taskkill /PID <pid_number> /F

# Mac/Linux
lsof -i :7860
kill -9 <pid_number>
```

Or just use a different port:
```bash
uvicorn coolpilot.server.app:app --port 8000
```

### ❌ Docker build fails
```bash
# Make sure Docker Desktop is running
# Then try cleaning and rebuilding
docker system prune -f
docker build --no-cache -t coolpilot .
```

### ❌ Hugging Face Space shows "Build Failed"
1. Click the **Logs** tab on your Space page
2. Scroll to the red error message
3. Common issues:
   - Missing files → make sure you pushed all files with `git add .`
   - Port mismatch → the `Dockerfile` must use port **7860** (it already does ✅)

### ❌ `git push` asks for password and fails
Hugging Face requires an **Access Token**, not your password:
1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Click **New token** → set role to **Write**
3. Copy the token and use it as the password when `git push` asks

---

## Quick Cheat Sheet

| What you want | Command |
|---|---|
| Run locally | `uvicorn coolpilot.server.app:app --port 7860` |
| Run tests | `pytest tests/ -v` |
| Build Docker | `docker build -t coolpilot .` |
| Run Docker | `docker run -p 7860:7860 coolpilot` |
| Deploy to HF | `git push hf kunaldev:main` |
| Check health | `curl http://localhost:7860/health` |

---

## Need Help?

- **OpenEnv Docs:** [github.com/meta-pytorch/OpenEnv](https://github.com/meta-pytorch/OpenEnv)
- **Hugging Face Spaces Docs:** [huggingface.co/docs/hub/spaces](https://huggingface.co/docs/hub/spaces)
- **Docker Basics:** [docs.docker.com/get-started](https://docs.docker.com/get-started/)
- **Project README:** [README.md](./README.md)
- **How It Works:** [HOW_IT_WORKS.md](./HOW_IT_WORKS.md)
