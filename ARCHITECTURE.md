# Architecture Documentation

## Paper Dependency Graph

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         DEPENDENCY FLOW                                   │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  TITANS (Foundation - Test-Time Memory)                                   │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ - Gradient-based memory updates (Eq 8)                           │    │
│  │ - Surprise metric + momentum (Eq 9-10, 13-14)                    │    │
│  │ - Forgetting mechanism                                            │    │
│  │ - Core equations: 8-14                                            │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                    │                        │                             │
│                    ▼                        ▼                             │
│    ┌──────────────────────────┐  ┌───────────────────────────────┐     │
│    │ MIRAS (Generalization)   │  │ NL (Nested Learning)          │     │
│    │ ✅ FULLY IMPLEMENTED     │  │ - Uses TITANS Eq 8             │     │
│    │                          │  │ - Self-referential              │     │
│    │ - Moneta: ℓ_p bias (p=3)│  │ - M3 optimizer (Alg 1)         │     │
│    │   Equations 10-11        │  │ - Equations 1-13               │     │
│    │                          │  │                                 │     │
│    │ - Yaad: Huber loss       │  │                                 │     │
│    │   Equation 12            │  │                                 │     │
│    │                          │  │                                 │     │
│    │ - Memora: KL divergence  │  │                                 │     │
│    │   Equation 17            │  │                                 │     │
│    │                          │  │                                 │     │
│    │ - LinearRNN: Eq 3        │  │                                 │     │
│    │ - DeltaRule: Eq 9        │  │                                 │     │
│    └──────────────────────────┘  └───────────────────────────────┘     │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘
```

## Module Structure

```
src/
├── common/            # Level 0: Foundation (no dependencies)
│   ├── attention.py   # TITANS Eq 1-5: Standard & linear attention
│   │   ├── scaled_dot_product_attention()  # Eq 1-2
│   │   └── linear_attention()              # Eq 3-5
│   └── __init__.py
│
├── titans/            # Level 1: TITANS Core
│   ├── memory.py      # Eq 8-14: Gradient-based memory update
│   │   ├── MLPMemory          # Main memory module
│   │   ├── SurpriseMetric     # Eq 13-14: Surprise computation
│   │   └── momentum_update()  # Eq 9-10: Momentum
│   └── __init__.py
│
├── miras/             # Level 2A: MIRAS Framework (FULLY IMPLEMENTED)
│   ├── memory.py      # Complete MIRAS implementation
│   │   ├── LinearRNNMemory   # Eq 3: Base associative memory
│   │   ├── DeltaRuleMemory   # Eq 9: Gradient descent update
│   │   ├── MonetaMemory      # Eq 10-11: ℓ_p attentional bias
│   │   ├── YaadMemory        # Eq 12: Huber loss (outlier robust)
│   │   └── MemoraMemory      # Eq 17: KL divergence retention
│   └── __init__.py
│
├── nl/                # Level 2B: NL Optimizers
│   ├── optimizers.py  # Eq 1-3, 10-13, Algorithm 1
│   │   ├── GradientDescent   # Eq 1-3
│   │   ├── MomentumGD        # Eq 10-13
│   │   └── M3Optimizer       # Algorithm 1: Multi-scale Momentum Muon
│   └── __init__.py
│
└── utils/
    └── __init__.py
```

## Implementation Status

### TITANS ✅ Complete
- ✅ Standard Attention (Eq 1-2)
- ✅ Linear Attention (Eq 3-5)
- ✅ Memory Update (Eq 8)
- ✅ Momentum (Eq 9-10)
- ✅ Surprise Metric (Eq 13-14)
- ⏸️ Parallelization (Eq 16-18) - Future work
- ⏸️ Architecture Variants (MAC/MAG/MAL) - Future work

### MIRAS ✅ Complete
- ✅ LinearRNNMemory - Base associative memory (Eq 3)
- ✅ DeltaRuleMemory - Gradient descent update (Eq 9)
- ✅ MonetaMemory - ℓ_p attentional bias with p=3 (Eq 10-11)
- ✅ YaadMemory - Huber loss for outlier robustness (Eq 12)
- ✅ MemoraMemory - KL divergence retention with soft/hard modes (Eq 17)
- ✅ ℓ_p loss functions (Eq 10)
- ✅ Retention gates (Eq 9)

### NL ✅ Core Complete
- ✅ Standard Gradient Descent (Eq 1-3)
- ✅ Momentum Gradient Descent (Eq 10-13)
- ✅ M3 Optimizer (Algorithm 1)
- ⏸️ Delta Gradient Descent (Eq 113-121) - Future work
- ⏸️ Self-Referential Titans (Eq 83-97) - Future work
- ⏸️ CMS (Eq 70-71) - Future work
- ⏸️ Hope Architecture (Eq 98-99) - Future work

## Key Design Decisions

### 1. Test-First Development
All equation implementations follow TDD:
1. Write test with paper equation reference
2. Implement to pass test
3. Verify shapes, gradients, numerical stability
4. Run pytest after each implementation

### 2. Modular Architecture
Each paper has its own module, with clear dependencies:
- `common/` = shared foundations (Level 0)
- `titans/` = TITANS-specific (Level 1)
- `miras/` = MIRAS extensions (Level 2A)
- `nl/` = NL optimizers and architectures (Level 2B)

### 3. Paper Traceability
Every function includes:
- Paper reference (e.g., "MIRAS Eq 10-11")
- LaTeX equation in docstring
- Plain English description
- Variable definitions

### 4. Gradient-Verified Implementations
All implementations verified for:
- Correct output shapes
- Gradient flow (no NaNs/infinities)
- Loss decreases over updates
- Numerical stability with large/small values
- Deterministic behavior

## Testing Strategy

### Test Coverage: 87%
- Total tests: 52
- All passing: ✅

### Test Structure
```
tests/
├── conftest.py                        # Shared fixtures
├── test_equations/                    # Equation-level tests
│   ├── test_common_attention.py      # 8 tests
│   ├── test_titans_memory.py         # 5 tests
│   ├── test_miras_memory.py          # 24 tests
│   └── test_nl_optimizers.py         # 4 tests
└── test_integration/                  # Cross-paper tests
    └── test_all_papers.py            # 11 tests
```

### Integration Tests
The integration tests verify cross-paper dependencies:
- TITANS memory can be trained with NL's M3 optimizer
- MIRAS Moneta can be trained with M3 optimizer
- Memory layer pipeline works end-to-end
- Surprise-triggered memory switching works
- Multi-variant memory ensemble works
- Gradient flows through entire pipeline
- Numerical stability with extreme values

## MIRAS Variants Deep Dive

### Moneta (ℓ_p Attentional Bias)
```
Paper: MIRAS Equation 10-11
Loss: ℓ_p(W; k, v) = ||Wk - v||_p^p  where p=3
Retention: ℓ_q retention with q=4
Use case: Sharp attention patterns
```

### Yaad (Huber Loss)
```
Paper: MIRAS Equation 12
Loss: Huber(r) = {
  0.5 * r^2           if |r| ≤ δ
  δ * (|r| - 0.5*δ)   otherwise
}
Retention: ℓ_2 retention
Use case: Robustness to outliers
```

### Memora (KL Divergence Retention)
```
Paper: MIRAS Equation 17
Loss: ℓ_2 standard loss
Retention: KL(M_t || M_{t-1})
Modes: Hard forgetting (gate=0) or soft forgetting (0<gate<1)
Use case: Information-theoretic memory management
```

## Critical Relationships

### TITANS → MIRAS
MIRAS generalizes TITANS:
- TITANS uses ℓ₂ objective → MIRAS extends to ℓₚ, Huber, etc.
- TITANS uses GD+momentum → MIRAS allows general optimizers
- MIRAS Table 1 "Titans-LMM" must match TITANS implementation

### TITANS → NL
NL uses TITANS as building block:
- NL Eq 65 (Delta rule) = TITANS memory update
- Self-Referential Titans extend TITANS architecture
- Hope uses TITANS memory

### MIRAS → NL
NL uses MIRAS framework:
- NL Eq 6 from MIRAS (associative memory definition)
- Linear attention = MIRAS dot-product bias + GD

## References

1. **TITANS**: Learning to Memorize at Test Time
   - arXiv:2501.00663v1
   - Authors: Ali Behrouz, Peilin Zhong, Vahab Mirrokni

2. **MIRAS**: It's All Connected: A Journey Through Test-Time Memorization
   - arXiv:2504.13173v1
   - Authors: Ali Behrouz, Meisam Razaviyayn, Peilin Zhong, Vahab Mirrokni

3. **NL**: Nested Learning: The Illusion of Deep Learning Architecture
   - NeurIPS 2025
   - Authors: Ali Behrouz, Meisam Razaviyayn, Peilin Zhong, Vahab Mirrokni
