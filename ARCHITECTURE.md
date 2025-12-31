# Architecture Documentation

## Paper Dependency Graph

```
┌──────────────────────────────────────────────────────────┐
│                    DEPENDENCY FLOW                        │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  TITANS (Foundation - Test-Time Memory)                  │
│  ┌──────────────────────────────────────────┐           │
│  │ - Gradient-based memory updates          │           │
│  │ - Surprise metric + momentum              │           │
│  │ - Forgetting mechanism                    │           │
│  │ - Equations: 8-14 (core memory)           │           │
│  └──────────────────────────────────────────┘           │
│                  │                  │                     │
│                  ▼                  ▼                     │
│    ┌───────────────────┐  ┌────────────────────────┐   │
│    │ MIRAS             │  │ NL (Nested Learning)    │   │
│    │ (Generalization)  │  │ (Application)           │   │
│    │ - Extends TITANS  │  │ - Uses TITANS Eq 8      │   │
│    │ - General         │  │ - Self-referential      │   │
│    │   objectives      │  │ - CMS persistent memory │   │
│    │ - Novel variants  │  │ - Hope architecture     │   │
│    └───────────────────┘  └────────────────────────┘   │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

## Module Structure

```
src/
├── common/            # Level 0: Foundation (no dependencies)
│   ├── attention.py   # TITANS Eq 1-5: Standard & linear attention
│   └── __init__.py
│
├── titans/            # Level 1: TITANS Core
│   ├── memory.py      # Eq 8-14: Gradient-based memory update
│   └── __init__.py    # Momentum, surprise, forgetting
│
├── miras/             # Level 2A: MIRAS Extensions
│   └── __init__.py    # (Placeholder for future implementation)
│
├── nl/                # Level 2B + 3: NL Optimizers & Architectures
│   ├── optimizers.py  # Eq 1-3, 10-13, Alg 1: GD, Momentum, M3
│   └── __init__.py
│
└── utils/
    └── __init__.py
```

## Implementation Status

### TITANS
- ✅ Standard Attention (Eq 1-2)
- ✅ Linear Attention (Eq 3-5)
- ✅ Memory Update (Eq 8)
- ✅ Momentum (Eq 9-10)
- ✅ Surprise Metric
- ✅ Forgetting Gate (Eq 13-14)
- ⏸️ Parallelization (Eq 16-18) - Not yet implemented
- ⏸️ Architecture Variants (MAC/MAG/MAL) - Not yet implemented

### MIRAS
- ⏸️ Associative Memory Framework - Not yet implemented
- ⏸️ General Objectives - Not yet implemented
- ⏸️ Moneta/Yaad/Memora Variants - Not yet implemented

### NL
- ✅ Standard Gradient Descent (Eq 1-3)
- ✅ Momentum Gradient Descent (Eq 10-13)
- ✅ M3 Optimizer (Algorithm 1)
- ⏸️ Delta Gradient Descent (Eq 113-121) - Not yet implemented
- ⏸️ Self-Referential Titans (Eq 83-97) - Not yet implemented
- ⏸️ CMS (Eq 70-71) - Not yet implemented
- ⏸️ Hope Architecture (Eq 98-99) - Not yet implemented

## Key Design Decisions

### 1. Test-First Development
All equation implementations follow TDD:
1. Write test with paper equation reference
2. Implement to pass test
3. Verify shapes, gradients, numerical stability

### 2. Modular Architecture
Each paper has its own module, with clear dependencies:
- `common/` = shared foundations
- `titans/` = TITANS-specific
- `miras/` = MIRAS extensions
- `nl/` = NL optimizers and architectures

### 3. Paper Traceability
Every function includes:
- Paper reference (e.g., "TITANS Eq 8")
- LaTeX equation in docstring
- Plain English description

### 4. Gradient-Verified Implementations
All implementations verified for:
- Correct output shapes
- Gradient flow (no NaNs/infinities)
- Loss decreases over updates
- Deterministic behavior

## Testing Strategy

### Test Coverage
- Equation tests: 17 tests across 3 modules
- Code coverage: 91% (exceeds 80% target)
- All tests pass

### Test Structure
```
tests/
├── conftest.py                       # Shared fixtures
├── test_equations/                   # Equation-level tests
│   ├── test_common_attention.py     # 8 tests
│   ├── test_titans_memory.py        # 5 tests
│   └── test_nl_optimizers.py        # 4 tests
└── test_integration/                 # (Future) Model-level tests
```

## Future Work

### P1 Priority (Next to Implement)
1. TITANS parallelization (Eq 16-18) for efficiency
2. MIRAS framework and variants (Moneta, Yaad, Memora)
3. NL Self-Referential Titans + CMS + Hope

### P2 Priority (Optional)
1. TITANS architecture variants (MAC, MAG, MAL)
2. Additional MIRAS objectives beyond ℓ₁, ℓ₂, Huber
3. Benchmark reproduction on real datasets

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

2. **MIRAS**: It's All Connected
   - arXiv:2504.13173v1
   - Authors: Ali Behrouz, Meisam Razaviyayn, Peilin Zhong, Vahab Mirrokni

3. **NL**: Nested Learning
   - NeurIPS 2025
   - Authors: Ali Behrouz, Meisam Razaviyayn, Peilin Zhong, Vahab Mirrokni
