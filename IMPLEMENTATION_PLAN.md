# Implementation Plan: Neural Memory Reproduction (TITANS + MIRAS + NL)

**Generated:** 2025-12-30T22:47:00Z
**Papers:** TITANS, MIRAS, NL
**Intent:** Full Reproduction - Equation-verified implementation with tests
**Target Environment:** Local CPU (with GPU-ready code structure)

---

## Executive Summary

**What we're building:**
A complete reproduction of three interconnected Google neural memory papers: TITANS (foundation), MIRAS (generalization framework), and NL (nested learning paradigm with Hope architecture), implementing all equations with comprehensive test coverage.

**Paper dependency chain:**
```
TITANS ──────────┐
(Foundation:      │
 Test-time        ├──► NL ──► Hope Architecture
 memory with      │    (Nested    (Self-referential Titans
 gradient-based   │     Learning   + CMS for continual
 updates)         │     paradigm)   learning)
                  │
MIRAS ───────────┘
(Generalization:
 Associative memory
 framework unifying
 TITANS, Transformers,
 and RNNs)
```

**Estimated effort:**
- Equations to implement: ~188 (TITANS: 35, MIRAS: 32, NL: 121)
- Algorithms to implement: 4 (TITANS: 3, NL: 1 M3 optimizer)
- Total test files: ~50-60 (at least 3 tests per equation)
- Estimated time: 20-30 hours of implementation

---

## 1. DEPENDENCY GRAPH

### Paper Dependencies

```
┌─────────────────────────────────────────────────────────────────┐
│                         DEPENDENCY FLOW                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  LAYER 0: FOUNDATION                                            │
│  ┌──────────────────────────────────────┐                      │
│  │  TITANS (Behrouz et al. 2025c)       │                      │
│  │  - Test-time memory                   │                      │
│  │  - Gradient-based updates (Eq 8-14)  │                      │
│  │  - Surprise metric + momentum         │                      │
│  │  - Forgetting mechanism               │                      │
│  │  - Three arch variants: MAC/MAG/MAL   │                      │
│  └──────────────────────────────────────┘                      │
│            │                         │                           │
│            │                         │                           │
│            ▼                         ▼                           │
│  ┌─────────────────────┐   ┌──────────────────────────────┐   │
│  │ MIRAS (2025b)        │   │ NL (NeurIPS 2025)            │   │
│  │ - Generalizes TITANS │   │ - Uses TITANS equations      │   │
│  │ - Associative memory │   │ - Nested optimization        │   │
│  │ - General objectives │   │ - Self-referential Titans    │   │
│  │ - Novel variants:    │   │ - CMS (persistent memory)    │   │
│  │   Moneta, Yaad,      │   │ - Hope architecture          │   │
│  │   Memora             │   │ - Optimizers as memories     │   │
│  └─────────────────────┘   └──────────────────────────────┘   │
│            │                         │                           │
│            └─────────┬───────────────┘                           │
│                      ▼                                            │
│            ┌─────────────────────┐                              │
│            │   HOPE (NL Sec 6)    │                              │
│            │  Self-Ref Titans +   │                              │
│            │  CMS for continual   │                              │
│            │  learning            │                              │
│            └─────────────────────┘                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Critical Cross-Paper Relationships

**TITANS → MIRAS:**
- MIRAS generalizes TITANS-LMM (Table 1 in MIRAS)
- TITANS uses ℓ₂ objective; MIRAS extends to ℓₚ, Huber, etc.
- TITANS uses GD+momentum; MIRAS allows general optimizers
- MIRAS Table 1 row "Titans-LMM" must match TITANS implementation

**TITANS → NL:**
- NL Eq 65 (Delta rule) = core TITANS memory update
- Self-Referential Titans (NL Sec 5.3) extends TITANS architecture
- Hope uses TITANS memory as building block

**MIRAS → NL:**
- NL uses MIRAS Definition 1 (associative memory as optimization)
- NL Eq 6 from MIRAS
- Linear attention = MIRAS dot-product bias + GD (NL Eq 17-18)

### Equation Dependencies (High-Level)

| Equation | Paper | Depends On | Used By | Priority |
|----------|-------|------------|---------|----------|
| Eq 1-2: Standard Attention | TITANS | None | All papers (baseline) | P0 |
| Eq 3-5: Linear Attention | TITANS | None | MIRAS, NL | P0 |
| Eq 8-14: Memory Update | TITANS | None | MIRAS Moneta/Yaad, NL Titans | P0 |
| Eq 16-18: Parallelization | TITANS | Eq 8-14 | All efficient impls | P1 |
| Eq 21-31: MAC/MAG/MAL | TITANS | Eq 8-18 | Integration | P1 |
| MIRAS Eq 3: Linear RNN | MIRAS | TITANS baseline | Moneta/Yaad/Memora | P0 |
| MIRAS Eq 7-32: Variants | MIRAS | TITANS Eq 8-14 | NL comparisons | P1 |
| NL Eq 1-3: GD | NL | None | All NL optimizers | P0 |
| NL Eq 5: FWP | NL | None | Self-ref Titans | P0 |
| NL Eq 65: Delta rule | NL | TITANS Eq 8 | Self-ref Titans | P0 |
| NL Eq 70-71: CMS | NL | None | Hope | P1 |
| NL Eq 83-97: Self-ref Titans | NL | TITANS, NL Eq 65 | Hope | P1 |
| NL Alg 1: M3 Optimizer | NL | NL Eq 10-44 | Training | P0 |

### Module Dependencies

```
┌─────────────────────────────────────────────────────────────────────┐
│ LEVEL 0: FOUNDATION (No dependencies)                               │
│ ├── src/common/attention.py         # Standard scaled dot-product   │
│ ├── src/common/linear_attention.py  # TITANS Eq 3-5                 │
│ ├── src/common/embeddings.py        # Token + position              │
│ ├── src/common/mlp.py               # Standard feedforward          │
│ └── src/utils/config.py             # Configuration management      │
├─────────────────────────────────────────────────────────────────────┤
│ LEVEL 1: TITANS CORE (Requires Level 0)                             │
│ ├── src/titans/memory.py            # Eq 8-14: Memory update        │
│ ├── src/titans/surprise.py          # Eq 8-10: Surprise metric      │
│ ├── src/titans/momentum.py          # Eq 9-10: Momentum update      │
│ ├── src/titans/forgetting.py        # Eq 13-14: Forget gate         │
│ ├── src/titans/parallel.py          # Eq 16-18: Chunk parallelization│
│ ├── src/titans/mac.py               # Eq 21-23: Memory as Context   │
│ ├── src/titans/mag.py               # Eq 24-27: Memory as Gating    │
│ └── src/titans/mal.py               # Eq 28-31: Memory as Layer     │
├─────────────────────────────────────────────────────────────────────┤
│ LEVEL 2A: MIRAS CORE (Requires Level 0-1)                           │
│ ├── src/miras/associative_memory.py # General framework             │
│ ├── src/miras/objectives.py         # L_p, Huber, robust losses     │
│ ├── src/miras/retention.py          # General retention gates       │
│ ├── src/miras/moneta.py             # 2-layer MLP, ℓ₁ objective     │
│ ├── src/miras/yaad.py               # 2-layer MLP, ℓ₂ objective     │
│ └── src/miras/memora.py             # 2-layer MLP, Huber objective  │
├─────────────────────────────────────────────────────────────────────┤
│ LEVEL 2B: NL FOUNDATIONS (Requires Level 0)                         │
│ ├── src/nl/gradient_descent.py      # Eq 1-3: Standard GD           │
│ ├── src/nl/delta_gd.py              # Eq 113-121: DGD with normalization│
│ ├── src/nl/momentum.py              # Eq 10-13: GD with momentum    │
│ ├── src/nl/deep_momentum.py         # Eq 14-25: Nested momentum     │
│ ├── src/nl/adam_decomposition.py   # Eq 100-105: Adam as memory    │
│ └── src/nl/m3_optimizer.py          # Alg 1: Multi-scale Momentum   │
├─────────────────────────────────────────────────────────────────────┤
│ LEVEL 3: NL ARCHITECTURES (Requires Level 1, 2B)                    │
│ ├── src/nl/fast_weight_programmer.py # Eq 5: FWP update             │
│ ├── src/nl/self_ref_titans.py       # Eq 83-97: Titans gen own values│
│ ├── src/nl/cms.py                   # Eq 70-71: Continuum Memory    │
│ └── src/nl/hope.py                  # Eq 98-99: Hope = Self-Ref+CMS │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. IMPLEMENTATION PHASES

### Phase 1: Project Setup
**Duration:** 30-45 minutes

**Tasks:**
1. Create directory structure
2. Initialize UV project
3. Add dependencies (torch, numpy, einops, pytest, ruff)
4. Set up test framework
5. Create pyproject.toml

**Commands:**
```bash
cd /Users/aryateja/Desktop/Claude-WorkOnMac/Project-LeCoder/New-experiment-30dec/neural-memory-reproduction

# Create directory structure
mkdir -p src/{common,titans,miras,nl,utils}
mkdir -p tests/{test_equations,test_integration}
mkdir -p configs notebooks scripts

# Initialize UV project (create pyproject.toml)
# Add dependencies via pyproject.toml

# Install dependencies
uv sync
```

**Deliverables:**
- [ ] `pyproject.toml` configured with all dependencies
- [ ] Directory structure created
- [ ] `uv sync` succeeds
- [ ] `uv run pytest` runs (0 tests initially)
- [ ] `.gitignore` created

---

### Phase 2: Level 0 - Foundation Components
**Duration:** 2-3 hours

Implement basic building blocks needed by all papers.

#### Phase 2.1: Common Attention Mechanisms

| Task | Module | Equations | Test File | Priority |
|------|--------|-----------|-----------|----------|
| 2.1.1 | Standard Attention | TITANS Eq 1-2 | `test_standard_attention.py` | P0 |
| 2.1.2 | Linear Attention | TITANS Eq 3-5 | `test_linear_attention.py` | P0 |
| 2.1.3 | MLP/FFN | Standard | `test_mlp.py` | P0 |
| 2.1.4 | Embeddings | Standard | `test_embeddings.py` | P0 |

**Implementation Order:**
1. Write tests first (TDD approach)
2. Implement `src/common/attention.py`:
   - `scaled_dot_product_attention()` - TITANS Eq 2
   - Causal masking
   - Temperature scaling
3. Implement `src/common/linear_attention.py`:
   - Kernel-based attention - TITANS Eq 3
   - Linear complexity via associativity
4. Implement `src/common/mlp.py`
5. Implement `src/common/embeddings.py`

**Checkpoint:** All Level 0 tests pass
```bash
uv run pytest tests/test_equations/test_common_*.py -v
```

---

### Phase 3: Level 1 - TITANS Core Implementation
**Duration:** 6-8 hours

Implement TITANS foundation that MIRAS and NL depend on.

#### Phase 3.1: TITANS Memory Update (Critical Foundation)

| Task | Module | Equations | Test File | Priority |
|------|--------|-----------|-----------|----------|
| 3.1.1 | Memory Update | TITANS Eq 8 | `test_titans_eq8_memory_update.py` | P0 |
| 3.1.2 | Surprise Metric | TITANS Eq 8-9 | `test_titans_eq9_surprise.py` | P0 |
| 3.1.3 | Momentum | TITANS Eq 9-10 | `test_titans_eq10_momentum.py` | P0 |
| 3.1.4 | Forgetting | TITANS Eq 13-14 | `test_titans_eq14_forgetting.py` | P0 |

**Test-First Implementation:**
```python
# tests/test_equations/test_titans_eq8_memory_update.py
"""
TITANS Equation 8: M_{t+1} = M_t - η∇L(M_t; k_t, v_t)

Paper: TITANS (Section 3.1)
Description: Gradient-based memory update with ℓ₂ loss
"""

class TestTitansEq8:
    def test_output_shape(self):
        """Memory update preserves shape."""

    def test_gradient_flow(self):
        """Gradients flow through memory update."""

    def test_numerical_stability(self):
        """Update is numerically stable."""

    def test_loss_decreases(self):
        """Memory update decreases loss."""

    def test_deterministic(self):
        """Same inputs produce same outputs."""
```

**Implementation (`src/titans/memory.py`):**
```python
def memory_update(M_t, k_t, v_t, eta):
    """
    TITANS Equation 8: M_{t+1} = M_t - η∇L(M_t; k_t, v_t)

    Paper: TITANS (Section 3.1)
    Loss: L(M; k, v) = ||M(k) - v||²

    Args:
        M_t: Current memory (MLP or matrix)
        k_t: Key vector
        v_t: Value vector
        eta: Learning rate

    Returns:
        M_{t+1}: Updated memory
    """
    # Implementation here
```

#### Phase 3.2: TITANS Parallelization

| Task | Module | Equations | Test File | Priority |
|------|--------|-----------|-----------|----------|
| 3.2.1 | Chunk Processing | TITANS Eq 16 | `test_titans_eq16_chunks.py` | P1 |
| 3.2.2 | Tensorized Grad | TITANS Eq 17 | `test_titans_eq17_tensorized.py` | P1 |
| 3.2.3 | Momentum Scan | TITANS Eq 18 | `test_titans_eq18_momentum_scan.py` | P1 |

#### Phase 3.3: TITANS Architecture Variants

| Task | Module | Equations | Test File | Priority |
|------|--------|-----------|-----------|----------|
| 3.3.1 | MAC (Memory as Context) | TITANS Eq 21-23 | `test_titans_mac.py` | P1 |
| 3.3.2 | MAG (Memory as Gating) | TITANS Eq 24-27 | `test_titans_mag.py` | P1 |
| 3.3.3 | MAL (Memory as Layer) | TITANS Eq 28-31 | `test_titans_mal.py` | P1 |

**Checkpoint:** TITANS tests pass
```bash
uv run pytest tests/test_equations/test_titans_*.py -v
uv run pytest tests/test_integration/test_titans_model.py -v
```

---

### Phase 4: Level 2A - MIRAS Extensions
**Duration:** 4-5 hours

Implement MIRAS variants that generalize TITANS.

#### Phase 4.1: MIRAS Core Framework

| Task | Module | Equations | Test File | Priority |
|------|--------|-----------|-----------|----------|
| 4.1.1 | Associative Memory Base | MIRAS Eq 3 | `test_miras_eq3_base.py` | P0 |
| 4.1.2 | General Objectives | MIRAS Eq 7-15 | `test_miras_objectives.py` | P0 |
| 4.1.3 | Retention Gates | MIRAS Eq 16-20 | `test_miras_retention.py` | P0 |

#### Phase 4.2: MIRAS Variants

| Task | Module | Equations | Test File | Priority |
|------|--------|-----------|-----------|----------|
| 4.2.1 | Moneta (ℓ₁ objective) | MIRAS Eq 7-9 | `test_miras_moneta.py` | P1 |
| 4.2.2 | Yaad (ℓ₂ objective) | MIRAS Eq 10-12 | `test_miras_yaad.py` | P1 |
| 4.2.3 | Memora (Huber) | MIRAS Eq 13-15 | `test_miras_memora.py` | P1 |

**Critical Test:** MIRAS must reproduce TITANS-LMM
```python
def test_miras_reproduces_titans():
    """MIRAS with specific config = TITANS-LMM (Table 1)."""
    # Config: k-layer MLP, ℓ₂ bias, ℓ₂ retention, GD+momentum
    # Should produce identical results to TITANS
```

**Checkpoint:** MIRAS tests pass
```bash
uv run pytest tests/test_equations/test_miras_*.py -v
```

---

### Phase 5: Level 2B - NL Foundations
**Duration:** 5-6 hours

Implement NL's optimizer foundations.

#### Phase 5.1: Gradient Descent Variants

| Task | Module | Equations | Test File | Priority |
|------|--------|-----------|-----------|----------|
| 5.1.1 | Standard GD | NL Eq 1-3 | `test_nl_eq1_gd.py` | P0 |
| 5.1.2 | GD with Momentum | NL Eq 10-13 | `test_nl_eq10_momentum.py` | P0 |
| 5.1.3 | Delta GD (DGD) | NL Eq 113-121 | `test_nl_eq113_dgd.py` | P0 |
| 5.1.4 | Deep Momentum | NL Eq 14-25 | `test_nl_eq14_deep_momentum.py` | P1 |

**Critical Implementation Notes:**
- DGD requires normalization (NL Eq 115-116)
- Momentum uses orthogonalization (NL Eq 42-44 from Muon)
- Test against standard PyTorch optimizers where applicable

#### Phase 5.2: Adam Decomposition

| Task | Module | Equations | Test File | Priority |
|------|--------|-----------|-----------|----------|
| 5.2.1 | Adam as Memory | NL Eq 100-105 | `test_nl_eq100_adam.py` | P0 |
| 5.2.2 | M3 Optimizer | NL Alg 1 | `test_nl_m3_optimizer.py` | P0 |

**Checkpoint:** NL optimizer tests pass
```bash
uv run pytest tests/test_equations/test_nl_eq*.py -v
```

---

### Phase 6: Level 3 - NL Architectures
**Duration:** 5-6 hours

Implement NL's architectural innovations.

#### Phase 6.1: Self-Referential Titans

| Task | Module | Equations | Test File | Priority |
|------|--------|-----------|-----------|----------|
| 6.1.1 | FWP Update | NL Eq 5 | `test_nl_eq5_fwp.py` | P0 |
| 6.1.2 | Self-Ref Titans | NL Eq 83-97 | `test_nl_self_ref_titans.py` | P1 |

**Implementation Notes:**
- Self-referential: Titans generate their own target values
- Circular dependency: y_t depends on M_t which depends on y_t
- Requires iterative solution or fixed-point iteration

#### Phase 6.2: Continuum Memory System (CMS)

| Task | Module | Equations | Test File | Priority |
|------|--------|-----------|-----------|----------|
| 6.2.1 | CMS Core | NL Eq 70-71 | `test_nl_eq70_cms.py` | P1 |
| 6.2.2 | Multi-frequency MLP | NL Sec 5.4 | `test_nl_cms_frequencies.py` | P1 |

#### Phase 6.3: Hope Architecture

| Task | Module | Equations | Test File | Priority |
|------|--------|-----------|-----------|----------|
| 6.3.1 | Hope Integration | NL Eq 98-99 | `test_nl_hope.py` | P1 |

**Hope = Self-Referential Titans + CMS**

**Checkpoint:** All NL architecture tests pass
```bash
uv run pytest tests/test_equations/test_nl_*.py -v
uv run pytest tests/test_integration/test_nl_*.py -v
```

---

### Phase 7: Integration Testing
**Duration:** 2-3 hours

Test complete models end-to-end.

| Task | Model | Test File | Priority |
|------|-------|-----------|----------|
| 7.1 | TITANS-MAC Model | `test_titans_mac_model.py` | P0 |
| 7.2 | MIRAS-Yaad Model | `test_miras_yaad_model.py` | P0 |
| 7.3 | Hope Model | `test_hope_model.py` | P1 |

**Integration Tests:**
```python
def test_forward_pass():
    """Model completes forward pass without error."""

def test_backward_pass():
    """Gradients flow through entire model."""

def test_parameter_count():
    """Parameter count matches paper specification."""

def test_memory_shapes():
    """All memory states have correct shapes."""

def test_training_step():
    """Can perform one training step."""
```

---

### Phase 8: Code Quality & Documentation
**Duration:** 2-3 hours

#### Phase 8.1: Code Quality

```bash
# Format code
uv run ruff format src/ tests/

# Lint code
uv run ruff check src/ tests/ --fix

# Run all tests
uv run pytest tests/ -v --tb=short

# Coverage check
uv run pytest tests/ --cov=src --cov-report=term-missing
```

**Quality Gates:**
- [ ] `ruff format` - no changes needed
- [ ] `ruff check` - 0 errors
- [ ] `pytest` - 100% pass rate
- [ ] Coverage > 80% for equation modules

#### Phase 8.2: Documentation

| Task | Output | Template |
|------|--------|----------|
| 8.2.1 | README.md | Quick start, paper links, implementation status |
| 8.2.2 | ARCHITECTURE.md | Design decisions, dependency diagrams |
| 8.2.3 | Equation docstrings | Paper references in every function |

**Documentation Requirements:**
- Every equation implementation has LaTeX in docstring
- Every equation has paper section reference
- Every module has overview docstring
- README has implementation status table

---

### Phase 9: Git & Execution Prep
**Duration:** 1 hour

#### Phase 9.1: Git Initialization

```bash
cd /Users/aryateja/Desktop/Claude-WorkOnMac/Project-LeCoder/New-experiment-30dec/neural-memory-reproduction

git init
git add .
git commit -m "Complete reproduction: TITANS + MIRAS + NL

Multi-paper neural memory reproduction with equation verification
- TITANS: Test-time memory with gradient-based updates
- MIRAS: Associative memory framework generalizing TITANS
- NL: Nested learning with Hope architecture

Implemented:
- 188 equations with tests (TITANS: 35, MIRAS: 32, NL: 121)
- 4 algorithms (TITANS: 3, NL: M3 optimizer)
- Test coverage: TBD%

Papers:
- TITANS: arXiv:2501.00663v1
- MIRAS: arXiv:2504.13173v1
- NL: NeurIPS 2025"
```

#### Phase 9.2: Execution Scripts

| Task | Script | Purpose |
|------|--------|---------|
| 9.2.1 | `run_titans_test.sh` | Quick TITANS test (CPU, small model) |
| 9.2.2 | `run_miras_test.sh` | Quick MIRAS test (CPU, small model) |
| 9.2.3 | `notebooks/quickstart.ipynb` | Interactive demo of all papers |

---

## 3. FILE MANIFEST

### Source Files (Final Structure)

```
src/
├── __init__.py
│
├── common/                         # Level 0: Foundation
│   ├── __init__.py
│   ├── attention.py                # TITANS Eq 1-2: Standard attention
│   ├── linear_attention.py         # TITANS Eq 3-5: Linear attention
│   ├── mlp.py                      # Standard feedforward
│   └── embeddings.py               # Token + position embeddings
│
├── titans/                         # Level 1: TITANS
│   ├── __init__.py
│   ├── memory.py                   # Eq 8-14: Memory update
│   ├── surprise.py                 # Eq 8-9: Surprise metric
│   ├── momentum.py                 # Eq 9-10: Momentum
│   ├── forgetting.py               # Eq 13-14: Forget gate
│   ├── parallel.py                 # Eq 16-18: Parallelization
│   ├── mac.py                      # Eq 21-23: Memory as Context
│   ├── mag.py                      # Eq 24-27: Memory as Gating
│   └── mal.py                      # Eq 28-31: Memory as Layer
│
├── miras/                          # Level 2A: MIRAS
│   ├── __init__.py
│   ├── associative_memory.py       # Eq 3: Base framework
│   ├── objectives.py               # Eq 7-15: L_p, Huber, robust
│   ├── retention.py                # Eq 16-20: Retention gates
│   ├── moneta.py                   # 2-layer MLP, ℓ₁ objective
│   ├── yaad.py                     # 2-layer MLP, ℓ₂ objective
│   └── memora.py                   # 2-layer MLP, Huber objective
│
├── nl/                             # Level 2B + 3: NL
│   ├── __init__.py
│   ├── gradient_descent.py         # Eq 1-3: Standard GD
│   ├── delta_gd.py                 # Eq 113-121: DGD
│   ├── momentum.py                 # Eq 10-13: GD with momentum
│   ├── deep_momentum.py            # Eq 14-25: Nested momentum
│   ├── adam_decomposition.py       # Eq 100-105: Adam as memory
│   ├── m3_optimizer.py             # Alg 1: M3 optimizer
│   ├── fast_weight_programmer.py   # Eq 5: FWP
│   ├── self_ref_titans.py          # Eq 83-97: Self-referential
│   ├── cms.py                      # Eq 70-71: Continuum Memory
│   └── hope.py                     # Eq 98-99: Hope architecture
│
└── utils/
    ├── __init__.py
    ├── config.py                   # Configuration management
    └── logging.py                  # Logging utilities
```

### Test Files

```
tests/
├── __init__.py
├── conftest.py                     # Shared fixtures
│
├── test_equations/                 # Equation-level tests (TDD)
│   ├── __init__.py
│   │
│   ├── test_common_*.py            # Level 0 tests
│   │
│   ├── test_titans_eq*.py          # TITANS equation tests
│   │   ├── test_titans_eq1_attention.py
│   │   ├── test_titans_eq8_memory_update.py
│   │   ├── test_titans_eq9_surprise.py
│   │   └── ...
│   │
│   ├── test_miras_eq*.py           # MIRAS equation tests
│   │   ├── test_miras_eq3_base.py
│   │   ├── test_miras_objectives.py
│   │   └── ...
│   │
│   └── test_nl_eq*.py              # NL equation tests
│       ├── test_nl_eq1_gd.py
│       ├── test_nl_eq5_fwp.py
│       ├── test_nl_eq70_cms.py
│       ├── test_nl_eq113_dgd.py
│       └── ...
│
└── test_integration/               # Model-level tests
    ├── __init__.py
    ├── test_titans_mac_model.py    # TITANS-MAC full model
    ├── test_miras_yaad_model.py    # MIRAS-Yaad full model
    └── test_hope_model.py          # Hope full model
```

---

## 4. VERIFICATION CHECKPOINTS

### Checkpoint 0: Project Setup Complete
```bash
uv sync
uv run pytest --version
# Expected: UV environment works
```

### Checkpoint 1: Level 0 Complete
```bash
uv run pytest tests/test_equations/test_common_*.py -v
# Expected: All foundation tests pass
```

### Checkpoint 2: TITANS Complete
```bash
uv run pytest tests/test_equations/test_titans_*.py -v
# Expected: All TITANS equation tests pass
```

### Checkpoint 3: MIRAS Complete
```bash
uv run pytest tests/test_equations/test_miras_*.py -v
# Expected: All MIRAS tests pass
# CRITICAL: test_miras_reproduces_titans() must pass
```

### Checkpoint 4: NL Optimizers Complete
```bash
uv run pytest tests/test_equations/test_nl_eq*.py -v
# Expected: All NL optimizer tests pass
```

### Checkpoint 5: NL Architectures Complete
```bash
uv run pytest tests/test_nl_*.py -v
# Expected: All NL architecture tests pass
```

### Checkpoint 6: Integration Complete
```bash
uv run pytest tests/test_integration/*.py -v
# Expected: All full model tests pass
```

### Checkpoint 7: Quality Gates Pass
```bash
uv run ruff format --check src/ tests/
uv run ruff check src/ tests/
uv run pytest tests/ --cov=src --cov-fail-under=80
# Expected: All quality checks pass
```

---

## 5. RISK MITIGATION

### Identified Risks

| Risk | Impact | Mitigation | Status |
|------|--------|------------|--------|
| Circular dependencies (self-ref Titans) | High | Iterative solver, careful initialization | Monitor |
| MIRAS-TITANS compatibility | High | Explicit test for Table 1 equivalence | Must verify |
| 188 equations = many tests | Medium | Prioritize P0, parallelize test writing | Planned |
| CMS frequency confusion | Medium | Clear documentation of chunk vs frequency | Document |
| DGD normalization subtlety | Medium | Careful implementation of Eq 115-116 | Test thoroughly |
| M3 optimizer complexity | Medium | Break into sub-components, test each | Phase 5.2 |

### Contingency Plans

**If MIRAS doesn't match TITANS:**
1. Compare intermediate values (memory updates, gradients)
2. Check hyperparameter equivalence (Table 1)
3. Verify ℓ₂ objective implementation
4. Test with identical random seed

**If self-referential Titans don't converge:**
1. Try fixed-point iteration (NL suggests this works)
2. Check initialization from CMS
3. Reduce learning rate
4. Add damping term

**If tests take too long:**
1. Use small batch sizes
2. Reduce sequence lengths
3. Parallelize test execution
4. Skip P2 tests initially

---

## 6. SUCCESS CRITERIA

### Minimum Viable Success (Must Have)
- [ ] All P0 equations implemented with passing tests
- [ ] TITANS-MAC model forward/backward pass works
- [ ] MIRAS reproduces TITANS-LMM (Table 1 test passes)
- [ ] NL M3 optimizer implemented
- [ ] Code passes all quality checks (ruff, coverage >80%)
- [ ] Documentation complete (README, ARCHITECTURE, docstrings)
- [ ] Git repository initialized with meaningful commit

### Full Success (Should Have)
- [ ] All above criteria
- [ ] All P1 equations implemented with tests
- [ ] All three architecture variants work (MAC, MAG, MAL)
- [ ] All MIRAS variants implemented (Moneta, Yaad, Memora)
- [ ] Hope architecture complete (Self-Ref Titans + CMS)
- [ ] Quickstart notebook demonstrates all papers
- [ ] Run scripts work and produce expected outputs

### Stretch Goals (Nice to Have)
- [ ] Benchmark results on toy tasks match paper trends
- [ ] Parallelization optimizations (CUDA kernels, FlashAttention)
- [ ] Additional ablations beyond paper
- [ ] Hyperparameter sweep utilities

---

## 7. IMPLEMENTATION PRIORITY

### P0 (Critical Path - Must Complete)
1. Level 0: Foundation (attention, embeddings)
2. TITANS core memory (Eq 8-14)
3. MIRAS base framework + compatibility test
4. NL optimizers (GD, momentum, DGD, M3)
5. Integration tests for basic models
6. Quality gates + documentation

### P1 (Important - Complete If Time Allows)
7. TITANS parallelization (Eq 16-18)
8. TITANS architecture variants (MAC, MAG, MAL)
9. MIRAS variants (Moneta, Yaad, Memora)
10. NL Self-Referential Titans + CMS + Hope
11. Quickstart notebook

### P2 (Optional - Skip If Time Constrained)
12. Deep momentum (NL Eq 14-25)
13. Additional MIRAS objectives beyond ℓ₁, ℓ₂, Huber
14. Benchmark reproduction on real datasets

---

## METADATA

```yaml
plan_version: "1.0"
generated_at: "2025-12-30T22:47:00Z"
papers_included:
  - id: "TITANS"
    title: "Titans: Learning to Memorize at Test Time"
    authors: "Ali Behrouz, Peilin Zhong, Vahab Mirrokni"
    arxiv: "arXiv:2501.00663v1"
    context_file: "TITANS.context.md"
    equations_extracted: 35
    algorithms_extracted: 3
  - id: "MIRAS"
    title: "It's All Connected: A Journey Through Test-Time Memorization, Attentional Bias, Retention, and Online Optimization"
    authors: "Ali Behrouz, Meisam Razaviyayn, Peilin Zhong, Vahab Mirrokni"
    arxiv: "arXiv:2504.13173v1"
    context_file: "MIRAS.context.md"
    equations_extracted: 32
    algorithms_extracted: 0
  - id: "NL"
    title: "Nested Learning: The Illusion of Deep Learning Architecture"
    authors: "Ali Behrouz, Meisam Razaviyayn, Peilin Zhong, Vahab Mirrokni"
    conference: "NeurIPS 2025"
    context_file: "NL.context.md"
    equations_extracted: 121
    algorithms_extracted: 1
total_equations: 188
total_algorithms: 4
total_test_files_estimated: 50-60
estimated_hours: 20-30
dependency_chain: "TITANS → {MIRAS, NL} → Hope"
critical_dependency: "MIRAS must reproduce TITANS-LMM"
```
