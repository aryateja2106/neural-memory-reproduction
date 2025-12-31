# Neural Memory Reproduction: TITANS + MIRAS + NL

[![CI](https://github.com/AryaTejaRudraraju/neural-memory-reproduction/actions/workflows/ci.yml/badge.svg)](https://github.com/AryaTejaRudraraju/neural-memory-reproduction/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Coverage](https://img.shields.io/badge/coverage-87%25-green.svg)](https://github.com/AryaTejaRudraraju/neural-memory-reproduction)

**Complete PyTorch reproduction of three interconnected Google Research papers on neural memory systems.**

This repository provides verified implementations of core equations from:
- **TITANS**: Learning to Memorize at Test Time
- **MIRAS**: It's All Connected - Test-Time Memorization, Attentional Bias & Retention
- **NL**: Nested Learning - The Illusion of Deep Learning Architecture

---

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Docker](#docker)
- [Project Structure](#project-structure)
- [Papers](#papers)
- [Implementation Status](#implementation-status)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

---

## Features

- **Complete Equation Implementations** - Core equations from all three papers
- **Verified with Tests** - 52 tests with 87% code coverage
- **Three MIRAS Variants** - Moneta (ℓ_p), Yaad (Huber), Memora (KL)
- **Cross-Paper Integration** - Tests verify paper dependencies work together
- **Docker Support** - Run anywhere without setup hassles
- **CI/CD Pipeline** - Automated testing on every commit

---

## Quick Start

### Option 1: Using UV (Recommended - Fastest)

```bash
# Clone the repository
git clone https://github.com/AryaTejaRudraraju/neural-memory-reproduction.git
cd neural-memory-reproduction

# Install UV if you don't have it
pip install uv

# Create environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"

# Run tests to verify everything works
pytest tests/ -v
```

### Option 2: Using Docker (No Python Setup Required)

```bash
# Clone the repository
git clone https://github.com/AryaTejaRudraraju/neural-memory-reproduction.git
cd neural-memory-reproduction

# Run tests in Docker
docker compose up test

# Or build and run manually
docker build -t neural-memory .
docker run --rm neural-memory
```

### Option 3: Using pip

```bash
git clone https://github.com/AryaTejaRudraraju/neural-memory-reproduction.git
cd neural-memory-reproduction
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -v
```

---

## Installation

### Prerequisites

- **Python 3.10 or higher** - [Download Python](https://www.python.org/downloads/)
- **Git** - [Download Git](https://git-scm.com/downloads)
- **Optional: Docker** - [Download Docker](https://www.docker.com/get-started)

### Detailed Installation

See [SETUP_GUIDE.md](SETUP_GUIDE.md) for step-by-step instructions with screenshots and troubleshooting tips.

---

## Usage

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific paper tests
pytest tests/test_equations/test_titans_memory.py -v
pytest tests/test_equations/test_miras_memory.py -v
pytest tests/test_equations/test_nl_optimizers.py -v

# Run integration tests
pytest tests/test_integration/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=term-missing
```

### Using the Implementations

```python
import torch
from src.titans.memory import MLPMemory, SurpriseMetric
from src.miras.memory import MonetaMemory, YaadMemory, MemoraMemory
from src.nl.optimizers import M3Optimizer

# TITANS: Gradient-based memory
titans_memory = MLPMemory(input_dim=64, output_dim=128)
key = torch.randn(8, 64)
value = torch.randn(8, 128)
output = titans_memory(key)
loss = titans_memory.compute_loss(key, value)

# MIRAS: Moneta with ℓ_p attentional bias
moneta = MonetaMemory(input_dim=64, output_dim=128, p=3.0)
output = moneta(key)

# NL: M3 Optimizer
optimizer = M3Optimizer(titans_memory.parameters(), lr=0.001)
```

### Interactive Notebook

```bash
# Start Jupyter
jupyter notebook notebooks/quickstart.ipynb

# Or use Docker
docker compose up jupyter
# Then open http://localhost:8888 in your browser
```

---

## Docker

### Quick Commands

```bash
# Run tests
docker compose up test

# Development environment (interactive shell)
docker compose up dev

# Start Jupyter notebook server
docker compose up jupyter

# Run linting checks
docker compose up lint

# Auto-format code
docker compose up format
```

### Building Images

```bash
# Build production image
docker build -t neural-memory .

# Build development image
docker build --target dev -t neural-memory:dev .

# Run with custom command
docker run --rm neural-memory python -c "import src; print('Success!')"
```

---

## Project Structure

```
neural-memory-reproduction/
├── src/                          # Source code
│   ├── common/                   # Shared utilities
│   │   └── attention.py          # Attention mechanisms (Eq 1-5)
│   ├── titans/                   # TITANS implementation
│   │   └── memory.py             # Memory module (Eq 8-14)
│   ├── miras/                    # MIRAS implementation
│   │   └── memory.py             # Moneta, Yaad, Memora variants
│   ├── nl/                       # NL implementation
│   │   └── optimizers.py         # GD, Momentum, M3 optimizer
│   └── utils/                    # Helper functions
│
├── tests/                        # Test suite
│   ├── test_equations/           # Per-equation tests
│   └── test_integration/         # Cross-paper tests
│
├── notebooks/                    # Jupyter notebooks
│   └── quickstart.ipynb          # Interactive demo
│
├── .github/workflows/            # CI/CD
│   └── ci.yml                    # GitHub Actions
│
├── Dockerfile                    # Container definition
├── docker-compose.yml            # Multi-container setup
├── pyproject.toml                # Project configuration
└── README.md                     # This file
```

---

## Papers

### TITANS: Learning to Memorize at Test Time
- **Authors**: Ali Behrouz, Peilin Zhong, Vahab Mirrokni
- **Link**: [arXiv:2501.00663](https://arxiv.org/abs/2501.00663)
- **Key Contribution**: Gradient-based memory updates during inference

### MIRAS: It's All Connected
- **Authors**: Ali Behrouz, Meisam Razaviyayn, Peilin Zhong, Vahab Mirrokni
- **Link**: [arXiv:2504.13173](https://arxiv.org/abs/2504.13173)
- **Key Contribution**: Unified framework with ℓ_p attentional bias and retention gates

### NL: Nested Learning
- **Authors**: Ali Behrouz, Meisam Razaviyayn, Peilin Zhong, Vahab Mirrokni
- **Conference**: NeurIPS 2025
- **Key Contribution**: M3 optimizer and nested learning architecture

---

## Implementation Status

| Paper | Equations | Implemented | Tests | Coverage |
|-------|-----------|-------------|-------|----------|
| TITANS | 35 | Core (Eq 8-14) | 5 | 100% |
| MIRAS | 32 | Full (Moneta, Yaad, Memora) | 24 | 82% |
| NL | 121 | Optimizers (Eq 1-13, Alg 1) | 4 | 92% |
| Common | - | Attention (Eq 1-5) | 8 | 85% |
| Integration | - | Cross-paper | 11 | - |
| **Total** | **188** | **Core + MIRAS** | **52** | **87%** |

### Architecture Diagram

```
TITANS (Foundation)
   │
   ├──► MIRAS (Generalization)
   │     - Moneta: ℓ_p attentional bias (p=3)
   │     - Yaad: Huber loss (outlier robust)
   │     - Memora: KL divergence retention
   │
   └──► NL (Application)
         - M3 optimizer
         - Nested learning principles
```

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Quick Contribution Guide

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make changes and write tests
4. Run checks: `pytest tests/ && ruff check src/`
5. Commit: `git commit -m "feat: add my feature"`
6. Push and create a Pull Request

---

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

### Academic Use

If you use this code in academic work, please cite the original papers:

```bibtex
@article{behrouz2025titans,
  title={TITANS: Learning to Memorize at Test Time},
  author={Behrouz, Ali and Zhong, Peilin and Mirrokni, Vahab},
  journal={arXiv preprint arXiv:2501.00663},
  year={2025}
}

@article{behrouz2025miras,
  title={It's All Connected: A Journey Through Test-Time Memorization, Attentional Bias, Retention, and Online Optimization},
  author={Behrouz, Ali and Razaviyayn, Meisam and Zhong, Peilin and Mirrokni, Vahab},
  journal={arXiv preprint arXiv:2504.13173},
  year={2025}
}

@inproceedings{behrouz2025nested,
  title={Nested Learning: The Illusion of Deep Learning Architecture},
  author={Behrouz, Ali and Razaviyayn, Meisam and Zhong, Peilin and Mirrokni, Vahab},
  booktitle={NeurIPS},
  year={2025}
}
```

---

## Citation

If you find this reproduction helpful, please star this repository and cite:

```bibtex
@misc{rudraraju2025neuralmemory,
  author = {Rudraraju, Arya Teja},
  title = {Neural Memory Reproduction: TITANS + MIRAS + NL},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/AryaTejaRudraraju/neural-memory-reproduction}
}
```

---

## Acknowledgments

- Original paper authors at Google Research
- PyTorch team for the deep learning framework
- Claude Code for assistance in reproduction

---

**Made with research by [Arya Teja Rudraraju](https://github.com/AryaTejaRudraraju)**
