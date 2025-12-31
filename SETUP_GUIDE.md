# Complete Setup Guide

**A step-by-step guide for running this project, even if you've never used Python before.**

This guide will walk you through everything you need to run the neural memory reproduction code on your computer.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Installation Methods](#2-installation-methods)
   - [Method A: Docker (Easiest)](#method-a-docker-easiest---recommended-for-beginners)
   - [Method B: UV Package Manager (Fastest)](#method-b-uv-package-manager-fastest)
   - [Method C: Traditional pip](#method-c-traditional-pip)
3. [Verifying Your Installation](#3-verifying-your-installation)
4. [Running the Code](#4-running-the-code)
5. [Troubleshooting](#5-troubleshooting)
6. [Frequently Asked Questions](#6-frequently-asked-questions)

---

## 1. Prerequisites

Before you begin, you'll need to install some software on your computer.

### Step 1.1: Check Your Operating System

This code works on:
- **macOS** (Apple computers)
- **Windows** 10 or 11
- **Linux** (Ubuntu, Fedora, etc.)

### Step 1.2: Install Git

Git is a tool for downloading code from the internet.

**macOS:**
```bash
# Open Terminal (press Cmd+Space, type "Terminal", press Enter)
# Then paste this command and press Enter:
xcode-select --install
```

**Windows:**
1. Go to https://git-scm.com/download/win
2. Download the installer
3. Run the installer, click "Next" on each screen (defaults are fine)
4. Restart your computer

**Linux (Ubuntu/Debian):**
```bash
sudo apt update && sudo apt install git
```

### Step 1.3: Verify Git Installation

Open your terminal (or Command Prompt on Windows) and type:
```bash
git --version
```

You should see something like: `git version 2.39.0`

---

## 2. Installation Methods

Choose ONE of the following methods:

---

### Method A: Docker (Easiest - Recommended for Beginners)

Docker runs the code in an isolated container, so you don't need to install Python or any dependencies.

#### Step A.1: Install Docker

**macOS:**
1. Go to https://www.docker.com/products/docker-desktop
2. Click "Download for Mac"
3. Open the downloaded `.dmg` file
4. Drag Docker to your Applications folder
5. Open Docker from Applications
6. Wait for Docker to start (you'll see a whale icon in your menu bar)

**Windows:**
1. Go to https://www.docker.com/products/docker-desktop
2. Click "Download for Windows"
3. Run the installer
4. Restart your computer when prompted
5. Open Docker Desktop from your Start menu
6. Wait for Docker to start

**Linux (Ubuntu):**
```bash
# Run these commands one by one:
sudo apt update
sudo apt install docker.io docker-compose
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER
# Log out and log back in for group changes to take effect
```

#### Step A.2: Verify Docker Installation

```bash
docker --version
```
You should see something like: `Docker version 24.0.0`

#### Step A.3: Download the Code

```bash
# Navigate to where you want to save the code
cd ~/Desktop

# Download the code
git clone https://github.com/aryateja2106/neural-memory-reproduction.git

# Go into the folder
cd neural-memory-reproduction
```

#### Step A.4: Run with Docker

```bash
# Run the tests (this will download everything automatically)
docker compose up test
```

**What you'll see:**
- Docker will download the base image (first time only, ~1-2 minutes)
- It will install dependencies (first time only, ~2-3 minutes)
- Then it will run 52 tests
- You should see: `52 passed` at the end

**To run interactively:**
```bash
# Start a development environment
docker compose run --rm dev bash

# Now you're inside the container! Try:
pytest tests/ -v
python -c "import torch; print('PyTorch works!')"

# Type 'exit' to leave
exit
```

**To start Jupyter notebook:**
```bash
docker compose up jupyter
```
Then open your web browser and go to: http://localhost:8888

---

### Method B: UV Package Manager (Fastest)

UV is a modern Python package manager that's 10-100x faster than pip.

#### Step B.1: Install Python

**macOS:**
```bash
# Install Homebrew if you don't have it:
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python:
brew install python@3.11
```

**Windows:**
1. Go to https://www.python.org/downloads/
2. Download Python 3.11 or 3.12
3. Run the installer
4. **IMPORTANT:** Check the box that says "Add Python to PATH"
5. Click "Install Now"

**Linux (Ubuntu):**
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip
```

#### Step B.2: Verify Python Installation

```bash
python --version
# or on some systems:
python3 --version
```
You should see: `Python 3.11.x` or `Python 3.12.x`

#### Step B.3: Install UV

```bash
pip install uv
```

#### Step B.4: Download the Code

```bash
# Navigate to where you want to save the code
cd ~/Desktop

# Download the code
git clone https://github.com/aryateja2106/neural-memory-reproduction.git

# Go into the folder
cd neural-memory-reproduction
```

#### Step B.5: Set Up the Environment

```bash
# Create a virtual environment
uv venv

# Activate it
# On macOS/Linux:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

**How to know it worked:** You should see `(.venv)` at the beginning of your command prompt.

#### Step B.6: Install Dependencies

```bash
uv pip install -e ".[dev]"
```

This will install:
- PyTorch (deep learning framework)
- NumPy (numerical computing)
- pytest (testing framework)
- And other required packages

---

### Method C: Traditional pip

If UV doesn't work for you, use the traditional pip method.

#### Step C.1: Install Python

(Same as Step B.1 above)

#### Step C.2: Download the Code

```bash
cd ~/Desktop
git clone https://github.com/aryateja2106/neural-memory-reproduction.git
cd neural-memory-reproduction
```

#### Step C.3: Create Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate it
# On macOS/Linux:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

#### Step C.4: Install Dependencies

```bash
pip install -e ".[dev]"
```

---

## 3. Verifying Your Installation

After installing with any method, verify everything works:

### Test 1: Check Python imports

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import src; print('Source code imported successfully!')"
```

### Test 2: Run all tests

```bash
pytest tests/ -v
```

**Expected output:**
```
======================== test session starts ========================
...
tests/test_equations/test_common_attention.py ........      [ 15%]
tests/test_equations/test_miras_memory.py ........................ [ 61%]
tests/test_equations/test_nl_optimizers.py ....              [ 69%]
tests/test_equations/test_titans_memory.py .....             [ 78%]
tests/test_integration/test_all_papers.py ...........        [100%]

======================== 52 passed in 0.69s =========================
```

### Test 3: Check code coverage

```bash
pytest tests/ --cov=src --cov-report=term-missing
```

You should see coverage of 87% or higher.

---

## 4. Running the Code

### Running Individual Paper Tests

```bash
# TITANS paper equations
pytest tests/test_equations/test_titans_memory.py -v

# MIRAS paper equations (Moneta, Yaad, Memora)
pytest tests/test_equations/test_miras_memory.py -v

# NL paper optimizers
pytest tests/test_equations/test_nl_optimizers.py -v

# Integration tests (all papers working together)
pytest tests/test_integration/ -v
```

### Using the Implementations in Python

```python
# Start Python interpreter
python

# Then type:
import torch
from src.titans.memory import MLPMemory
from src.miras.memory import MonetaMemory

# Create a TITANS memory module
memory = MLPMemory(input_dim=64, output_dim=128)

# Create random input
key = torch.randn(8, 64)  # 8 samples, 64 dimensions
value = torch.randn(8, 128)  # 8 samples, 128 dimensions

# Get output
output = memory(key)
print(f"Output shape: {output.shape}")  # Should be [8, 128]

# Compute loss
loss = memory.compute_loss(key, value)
print(f"Loss: {loss.item():.4f}")

# Exit Python
exit()
```

### Running Jupyter Notebook

```bash
# If not using Docker:
pip install jupyter
jupyter notebook notebooks/quickstart.ipynb

# Using Docker:
docker compose up jupyter
# Then open http://localhost:8888 in your browser
```

---

## 5. Troubleshooting

### Problem: "command not found: python"

**Solution:** Try `python3` instead of `python`:
```bash
python3 --version
python3 -m venv .venv
```

### Problem: "No module named 'torch'"

**Solution:** Make sure your virtual environment is activated:
```bash
# Look for (.venv) at the start of your prompt
# If not there, activate it:
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows

# Then reinstall:
pip install torch
```

### Problem: Docker says "Cannot connect to Docker daemon"

**Solution:**
1. Make sure Docker Desktop is running
2. Look for the whale icon in your menu bar (macOS) or system tray (Windows)
3. If not there, open Docker Desktop from your Applications

### Problem: "Permission denied" on Linux

**Solution:**
```bash
sudo chmod +x *.sh
# For Docker:
sudo usermod -aG docker $USER
# Then log out and log back in
```

### Problem: Tests fail with import errors

**Solution:** Make sure you're in the right directory:
```bash
pwd
# Should end with: neural-memory-reproduction

# If not, navigate there:
cd ~/Desktop/neural-memory-reproduction
```

### Problem: "No space left on device" with Docker

**Solution:** Clean up Docker:
```bash
docker system prune -a
```

---

## 6. Frequently Asked Questions

### Q: Do I need a GPU?

**A:** No! All tests run on CPU. A GPU would only help if you're training large models.

### Q: How long does installation take?

**A:**
- Docker: 5-10 minutes (first time)
- UV: 2-3 minutes
- pip: 5-10 minutes

### Q: Can I use this on Google Colab?

**A:** Yes! Create a new notebook and run:
```python
!git clone https://github.com/aryateja2106/neural-memory-reproduction.git
%cd neural-memory-reproduction
!pip install -e ".[dev]"
!pytest tests/ -v
```

### Q: What Python version do I need?

**A:** Python 3.10, 3.11, or 3.12. We recommend 3.11.

### Q: How do I update to the latest version?

**A:**
```bash
cd neural-memory-reproduction
git pull origin main
# If using Docker:
docker compose build --no-cache
# If using UV/pip:
uv pip install -e ".[dev]" --upgrade
```

### Q: Where can I get help?

**A:** Open an issue at: https://github.com/aryateja2106/neural-memory-reproduction/issues

---

## Quick Reference Card

| Task | Command |
|------|---------|
| Run all tests | `pytest tests/ -v` |
| Run with coverage | `pytest tests/ --cov=src` |
| Format code | `ruff format src/ tests/` |
| Check linting | `ruff check src/ tests/` |
| Start Jupyter | `jupyter notebook` |
| Docker tests | `docker compose up test` |
| Docker coverage | `docker compose up coverage` |
| Docker shell | `docker compose run --rm dev bash` |
| Activate venv | `source .venv/bin/activate` |
| Deactivate venv | `deactivate` |

---

**Need more help?** Open an issue on GitHub or check the README.md for more details.

**Author:** Arya Teja Rudraraju
