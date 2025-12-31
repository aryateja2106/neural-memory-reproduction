# README Template

Use this template structure when generating project READMEs.

## Template Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `{PAPER_TITLE}` | Full paper title | "TITANS: Learning to Memorize at Test Time" |
| `{PAPER_ID}` | Short identifier | "TITANS" |
| `{AUTHORS}` | Author list | "Ali et al., 2025" |
| `{ARXIV_LINK}` | arXiv URL | "https://arxiv.org/abs/2501.00663" |
| `{REPO_NAME}` | Repository name | "titans-reproduction" |

## Template

```markdown
# {PAPER_TITLE} - Reproduction

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![uv](https://img.shields.io/badge/uv-managed-blueviolet.svg)](https://github.com/astral-sh/uv)
[![Tests](https://img.shields.io/badge/tests-{TEST_STATUS}-{TEST_COLOR}.svg)]()
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

> **Paper:** [{PAPER_TITLE}]({ARXIV_LINK})  
> **Authors:** {AUTHORS}  
> **Venue:** {VENUE}  
> **Reproduced:** {DATE} using [LeCoder Research Reproduction](https://github.com/lesearch-ai/research-reproduction)

---

## 🎯 Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (Python package manager)
- CUDA 11.8+ (for GPU training)

### Installation

```bash
# Clone repository
git clone https://github.com/{USER}/{REPO_NAME}.git
cd {REPO_NAME}

# Install dependencies with uv
uv sync

# Verify installation
uv run pytest tests/ -x -q
```

### Run Examples

```bash
# Quick inference demo
uv run python -m src.demo

# Train small model (CPU/single GPU)
uv run python -m src.train --config configs/small.yaml

# Evaluate pretrained checkpoint
uv run python -m src.evaluate --checkpoint checkpoints/best.pt
```

---

## 📊 Reproduction Results

### Main Results

| Dataset | Metric | Paper | Ours | Δ | Status |
|---------|--------|-------|------|---|--------|
{RESULTS_TABLE}

### Ablation Studies

| Configuration | {METRIC} | Paper | Ours |
|--------------|----------|-------|------|
{ABLATION_TABLE}

> **Note:** {RESULTS_NOTES}

---

## 🏗️ Architecture

```
{ARCHITECTURE_DIAGRAM}
```

### Key Components

| Component | Description | Equations |
|-----------|-------------|-----------|
{COMPONENTS_TABLE}

See [ARCHITECTURE.md](./ARCHITECTURE.md) for detailed technical documentation.

---

## 📁 Project Structure

```
{REPO_NAME}/
├── pyproject.toml        # Project configuration (uv)
├── uv.lock              # Locked dependencies
├── .python-version      # Python version (3.11)
│
├── src/                 # Source code
│   ├── __init__.py
│   ├── model.py         # Main model class
│   ├── layers/          # Layer implementations
│   │   ├── {LAYER_FILES}
│   ├── train.py         # Training script
│   ├── evaluate.py      # Evaluation script
│   └── utils/           # Utilities
│
├── tests/               # Test suite
│   ├── test_equations/  # Per-equation tests
│   ├── test_integration.py
│   └── conftest.py
│
├── configs/             # Configuration files
│   ├── small.yaml       # Quick experiments
│   ├── medium.yaml      # Development
│   └── paper.yaml       # Paper reproduction
│
├── notebooks/           # Jupyter notebooks
│   ├── quickstart.ipynb
│   └── analysis.ipynb
│
├── scripts/             # Utility scripts
│   ├── quality_check.py
│   └── verify_equations.py
│
├── papers/              # Paper context
│   └── {PAPER_ID}.context.md
│
└── docs/                # Additional docs
    ├── ARCHITECTURE.md
    └── equations.md
```

---

## 🔬 Equations Reference

{EQUATIONS_SECTION}

See [docs/equations.md](./docs/equations.md) for full equation documentation.

---

## 🚀 Training

### Local Training

```bash
# Small model (quick iteration)
uv run python -m src.train --config configs/small.yaml

# Paper configuration (requires GPU)
uv run python -m src.train --config configs/paper.yaml

# Resume from checkpoint
uv run python -m src.train --config configs/paper.yaml --resume checkpoints/last.pt
```

### Multi-GPU Training

```bash
# 4 GPUs with torchrun
uv run torchrun --nproc_per_node=4 -m src.train \
    --config configs/paper.yaml \
    --batch_size 8  # Per GPU
```

### Google Colab (via LeCoder-cgpu)

```bash
# Install LeCoder-cgpu CLI
npm install -g lecoder-cgpu

# Connect to Colab GPU
lecoder-cgpu connect --variant gpu

# Run training
lecoder-cgpu run "cd {REPO_NAME} && uv sync && uv run python -m src.train"
```

### Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
{CONFIG_OPTIONS_TABLE}

---

## 📈 Evaluation

```bash
# Single dataset
uv run python -m src.evaluate \
    --checkpoint checkpoints/best.pt \
    --dataset wikitext103

# All benchmarks
uv run python -m src.evaluate \
    --checkpoint checkpoints/best.pt \
    --all

# Custom dataset
uv run python -m src.evaluate \
    --checkpoint checkpoints/best.pt \
    --data_path /path/to/data
```

---

## 🧪 Testing

### Run All Tests

```bash
# Quick test
uv run pytest tests/ -x -q

# Verbose with coverage
uv run pytest tests/ -v --cov=src --cov-report=html

# Specific test file
uv run pytest tests/test_equations/test_eq1_memory.py -v
```

### Equation Verification

```bash
# List all equations and test status
uv run python scripts/verify_equations.py list

# Verify all equations
uv run python scripts/verify_equations.py verify

# Generate missing test stubs
uv run python scripts/verify_equations.py generate
```

### Code Quality

```bash
# Full quality check
uv run python scripts/quality_check.py

# Auto-fix issues
uv run python scripts/quality_check.py --fix

# Individual tools
uv run ruff format src/ tests/
uv run ruff check src/ tests/
uv run ty check src/
```

---

## 🔧 Configuration Reference

### Model Configuration (`configs/*.yaml`)

```yaml
model:
  d_model: 768           # Hidden dimension
  n_layers: 12           # Number of layers
  n_heads: 12            # Attention heads
  d_ff: 3072            # FFN dimension
  vocab_size: 50257     # Vocabulary size
  max_seq_len: 2048     # Maximum sequence length
  
  # Memory-specific
  memory_size: 64        # Memory slots per layer
  memory_lr: 0.01       # Memory update learning rate
  surprise_threshold: 0.5

training:
  batch_size: 32
  gradient_accumulation: 4
  learning_rate: 3.0e-4
  warmup_steps: 10000
  max_steps: 100000
  weight_decay: 0.1
  gradient_clip: 1.0
  
  # Optimizer
  optimizer: adamw
  betas: [0.9, 0.95]
  
  # Precision
  mixed_precision: bf16
  
data:
  dataset: wikitext103
  tokenizer: gpt2
  
logging:
  log_every: 100
  eval_every: 1000
  save_every: 5000
```

---

## 📝 Citation

If you use this reproduction, please cite the original paper:

```bibtex
{BIBTEX_CITATION}
```

---

## 🙏 Acknowledgments

- **Original Authors:** {AUTHORS} for the paper and research
- **Reproduction Tool:** [LeCoder Research Reproduction Skill](https://github.com/lesearch-ai)
- **Tooling:** [uv](https://github.com/astral-sh/uv), [ruff](https://github.com/astral-sh/ruff), [ty](https://github.com/astral-sh/ty)

---

## 📄 License

This reproduction is released under the MIT License. See [LICENSE](./LICENSE) for details.

The original paper may have different licensing terms.

---

## 🐛 Issues & Contributing

Found a bug or discrepancy with the paper? Please:

1. Check [existing issues](https://github.com/{USER}/{REPO_NAME}/issues)
2. Open a new issue with:
   - Expected behavior (from paper)
   - Actual behavior
   - Steps to reproduce
   - Relevant paper section/equation

Pull requests welcome!
```
