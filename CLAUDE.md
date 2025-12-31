# Claude Code Project Context

This file provides context for Claude Code (Anthropic's AI coding assistant) when working with this repository.

## Project Overview

**Neural Memory Reproduction** is a PyTorch implementation of three interconnected Google Research papers on neural memory systems:

1. **TITANS** - Learning to Memorize at Test Time
2. **MIRAS** - It's All Connected: Test-Time Memorization, Attentional Bias & Retention
3. **NL** - Nested Learning: The Illusion of Deep Learning Architecture

### Why This Matters: The Next Wave in AI

**Continual Learning** is emerging as the next major frontier in AI research. Current large language models have a fundamental limitation: they can't learn new information after training without expensive fine-tuning.

These papers address this by introducing **test-time memorization** - the ability to learn and adapt in real-time during inference, without modifying the base model weights. Key innovations:

- **Surprise-based memory**: Only memorize information that's "surprising" (high loss)
- **Attentional bias**: Use ℓ_p norms to control memory sparsity
- **Retention gates**: Balance new learning with memory preservation

This enables AI systems to:
- Adapt to new users/contexts instantly
- Handle extremely long sequences (100K+ tokens)
- Learn continuously without catastrophic forgetting

Read more: [Google Research Blog - TITANS & MIRAS](https://research.google/blog/titans-miras-helping-ai-have-long-term-memory/)

## How This Was Built

This reproduction was created using **Claude Code** with **Opus 4.5** and a custom **research-reproduction skill** (see `.claude/skills/`).

The workflow:
1. Paper analysis and equation extraction
2. Test-first development (TDD)
3. Implementation with comprehensive docstrings
4. Integration testing across papers
5. Docker containerization for reproducibility

**Current Status**: 52 tests passing, 87% code coverage

## Project Structure

```
neural-memory-reproduction/
├── src/                    # Source implementations
│   ├── common/             # Shared attention mechanisms
│   ├── titans/             # TITANS memory module
│   ├── miras/              # MIRAS variants (Moneta, Yaad, Memora)
│   └── nl/                 # NL optimizers (GD, Momentum, M3)
├── tests/                  # Test suite
│   ├── test_equations/     # Per-equation tests
│   └── test_integration/   # Cross-paper integration tests
├── notebooks/              # Jupyter notebooks for exploration
└── .claude/                # Claude Code configuration
    └── skills/             # Custom skills for this project
```

## Working with This Codebase

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific paper
pytest tests/test_equations/test_titans_memory.py -v
pytest tests/test_equations/test_miras_memory.py -v
pytest tests/test_equations/test_nl_optimizers.py -v

# With coverage
pytest tests/ --cov=src --cov-report=term-missing
```

### Using Docker

```bash
# Run tests
docker compose up test

# Run with coverage (requires root for write permissions)
docker compose up coverage

# Interactive development
docker compose run --rm dev bash

# Jupyter notebook
docker compose up jupyter
```

### Adding New Implementations

When implementing new equations:

1. **Write tests first** in `tests/test_equations/test_<paper>.py`
2. **Implement** in `src/<paper>/<module>.py`
3. **Document** with paper reference, equation number, LaTeX, plain English
4. **Verify** with `pytest -v`

Example docstring format:

```python
def equation_name(x: torch.Tensor) -> torch.Tensor:
    """
    Brief description.

    Paper: Paper Title
    arXiv: https://arxiv.org/abs/XXXX.XXXXX
    Equation: N (page M)

    LaTeX:
        y = f(x)

    Plain English:
        What this equation does in simple terms.

    Args:
        x: Input tensor [batch, dim]

    Returns:
        Output tensor [batch, output_dim]
    """
```

## Key Implementation Details

### TITANS (src/titans/memory.py)

- `MLPMemory`: Neural network-based associative memory
- `SurpriseMetric`: Measures information content for selective memorization
- Uses gradient descent to update memory at test time

### MIRAS (src/miras/memory.py)

Three variants with different loss functions:

- `MonetaMemory`: Uses ℓ_p norm (p=3) for sparse attentional bias
- `YaadMemory`: Uses Huber loss for outlier robustness
- `MemoraMemory`: Uses KL divergence for probabilistic retention

### NL (src/nl/optimizers.py)

- `GDOptimizer`: Basic gradient descent
- `MomentumOptimizer`: GD with momentum
- `M3Optimizer`: Memory-efficient momentum with gradient accumulation

## Common Tasks for Claude Code

### "Explain how X works"
- Read relevant source file in `src/`
- Cross-reference with test file in `tests/`
- Provide equation reference from paper

### "Add a new equation"
1. Identify paper and equation number
2. Create test in `tests/test_equations/`
3. Implement in appropriate `src/` module
4. Run `pytest -v` to verify

### "Debug failing test"
1. Run specific test with `-v --tb=long`
2. Check tensor shapes and types
3. Verify equation against paper
4. Check numerical stability

### "Improve coverage"
1. Run `pytest --cov=src --cov-report=term-missing`
2. Identify uncovered lines
3. Add tests for edge cases

## Paper Links

- **TITANS**: [arXiv:2501.00663](https://arxiv.org/abs/2501.00663)
- **MIRAS**: [arXiv:2504.13173](https://arxiv.org/abs/2504.13173)
- **NL**: [PDF](https://abehrouz.github.io/files/NL.pdf)
- **Google Blog**: [TITANS & MIRAS](https://research.google/blog/titans-miras-helping-ai-have-long-term-memory/)

## Author

**Arya Teja Rudraraju**

Built with Claude Code (Opus 4.5) using the research-reproduction skill workflow.
