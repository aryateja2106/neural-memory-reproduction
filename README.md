# Neural Memory Reproduction: TITANS + MIRAS + NL

Complete reproduction of three interconnected Google Research papers on neural memory systems:

- **TITANS**: Learning to Memorize at Test Time (arXiv:2501.00663v1)
- **MIRAS**: It's All Connected: A Journey Through Test-Time Memorization (arXiv:2504.13173v1)
- **NL**: Nested Learning: The Illusion of Deep Learning Architecture (NeurIPS 2025)

## Quick Start

```bash
# Install dependencies
uv venv
uv pip install -e ".[dev]"

# Run all tests
uv run pytest tests/ -v

# Run specific paper tests
./run_titans_test.sh   # TITANS equations
./run_miras_test.sh    # MIRAS equations (Moneta, Yaad, Memora)

# Check coverage
uv run pytest tests/ --cov=src --cov-report=term-missing
```

## Implementation Status

| Paper | Equations | Implemented | Tests | Coverage |
|-------|-----------|-------------|-------|----------|
| TITANS | 35 | Core (Eq 8-14) | 5 | 100% |
| MIRAS | 32 | Full (Moneta, Yaad, Memora) | 24 | 82% |
| NL | 121 | Optimizers (Eq 1-13, Alg 1) | 4 | 92% |
| Common | - | Attention (Eq 1-5) | 8 | 85% |
| Integration | - | Cross-paper tests | 11 | - |
| **Total** | **188** | **Core + MIRAS** | **52** | **87%** |

## Papers

1. **TITANS: Learning to Memorize at Test Time**
   - Authors: Ali Behrouz, Peilin Zhong, Vahab Mirrokni
   - arXiv: https://arxiv.org/abs/2501.00663
   - Core: Gradient-based memory updates, surprise-triggered learning

2. **MIRAS: It's All Connected**
   - Authors: Ali Behrouz, Meisam Razaviyayn, Peilin Zhong, Vahab Mirrokni
   - arXiv: https://arxiv.org/abs/2504.13173
   - Core: ℓ_p attentional bias, retention gates, three variants (Moneta, Yaad, Memora)

3. **Nested Learning: The Illusion of Deep Learning Architecture**
   - Authors: Ali Behrouz, Meisam Razaviyayn, Peilin Zhong, Vahab Mirrokni
   - Conference: NeurIPS 2025
   - Core: M3 optimizer, nested learning principles

## Project Structure

```
src/
├── common/         # Foundation (attention, embeddings)
│   └── attention.py  # Standard & linear attention (Eq 1-5)
├── titans/         # TITANS implementation
│   └── memory.py     # MLP memory, momentum, surprise (Eq 8-14)
├── miras/          # MIRAS implementation (FULL)
│   └── memory.py     # Moneta (ℓ_p), Yaad (Huber), Memora (KL)
├── nl/             # NL implementation
│   └── optimizers.py # GD, Momentum, M3 optimizer
└── utils/          # Utilities

tests/
├── test_equations/   # Equation-level tests (per paper)
│   ├── test_titans_memory.py
│   ├── test_miras_memory.py
│   └── test_nl_optimizers.py
└── test_integration/ # Cross-paper integration tests
    └── test_all_papers.py
```

## Key Implementations

### TITANS (src/titans/memory.py)
- `MLPMemory`: Gradient-based associative memory (Eq 8)
- `SurpriseMetric`: Computes surprise for memory updates (Eq 13-14)
- Momentum-based updates (Eq 9-10)

### MIRAS (src/miras/memory.py)
- `MonetaMemory`: ℓ_p attentional bias with p=3 (Eq 10-11)
- `YaadMemory`: Huber loss for outlier robustness (Eq 12)
- `MemoraMemory`: KL divergence retention with soft/hard forgetting (Eq 17)
- `LinearRNNMemory`: Base associative memory framework (Eq 3)
- `DeltaRuleMemory`: Gradient descent memory update (Eq 9)

### NL (src/nl/optimizers.py)
- `GradientDescent`: Standard GD (Eq 1-3)
- `MomentumGD`: GD with momentum (Eq 10-13)
- `M3Optimizer`: Multi-scale Momentum Muon (Algorithm 1)

## Running Tests

```bash
# All tests
uv run pytest tests/ -v

# With coverage
uv run pytest tests/ --cov=src --cov-report=html

# Specific test file
uv run pytest tests/test_equations/test_miras_memory.py -v

# Integration tests only
uv run pytest tests/test_integration/ -v
```

## Quality Checks

```bash
# Format code
uv run ruff format src/ tests/

# Lint code
uv run ruff check src/ tests/ --fix

# Type checking (if enabled)
uv run mypy src/
```

## Architecture Diagram

```
TITANS (Foundation)
   │
   ├──► MIRAS (Generalization)
   │     - ℓ_p attentional bias (Moneta)
   │     - Huber loss (Yaad)
   │     - KL retention (Memora)
   │
   └──► NL (Application)
         - M3 optimizer
         - Nested learning
         - Self-referential architecture
```

## License

Research reproduction for educational purposes.
