---
name: research-reproduction
description: Reproduce research papers into working code. Use when user wants to implement ML/AI papers, reproduce experiments, extract algorithms from PDFs, or convert research into executable code. Handles multiple interconnected papers with multi-agent extraction, equation verification, and benchmark validation.
---

# Research Reproduction Skill

This skill provides a systematic workflow for reproducing research papers into working, tested code.

## When to Use

USE WHEN:
- User wants to implement equations from ML/AI papers
- User needs to reproduce experimental results from papers
- User wants to convert research algorithms into executable code
- User is working with interconnected papers (like TITANS + MIRAS + NL)

## Workflow

### Phase 1: Paper Analysis
1. Read and understand the paper(s)
2. Identify core equations to implement
3. Map dependencies between equations
4. Note any ambiguities or missing details

### Phase 2: Test-First Implementation
1. Write tests BEFORE implementing each equation
2. Tests should verify:
   - Output shape correctness
   - Gradient flow (for backprop)
   - Numerical stability
   - Edge cases

### Phase 3: Implementation
1. Implement equations with clear docstrings
2. Include paper reference (title, arXiv, equation number)
3. Add LaTeX representation in docstring
4. Include plain English explanation

### Phase 4: Verification
1. Run all tests (target: 100% pass rate)
2. Check code coverage (target: >80%)
3. Verify against paper's reported results if available

## Example Usage

```python
# Example: Implementing MIRAS Delta Rule (Equation 9)
def delta_rule_update(M: torch.Tensor, k: torch.Tensor, v: torch.Tensor, eta: float) -> torch.Tensor:
    """
    Delta rule memory update.

    Paper: MIRAS - It's All Connected
    arXiv: https://arxiv.org/abs/2504.13173
    Equation: 9 (page 5)

    LaTeX:
        M_t = M_{t-1} - η(M_{t-1}k_t - v_t)k_t^T

    Plain English:
        Updates memory by moving towards storing the key-value pair,
        with learning rate η controlling the update magnitude.

    Args:
        M: Memory matrix [d_out, d_in]
        k: Key vector [batch, d_in]
        v: Value vector [batch, d_out]
        eta: Learning rate

    Returns:
        Updated memory matrix
    """
    retrieval_error = torch.matmul(M, k.T) - v.T
    gradient = torch.matmul(retrieval_error, k)
    return M - eta * gradient
```

## Project Structure for Reproductions

```
project-name/
├── src/
│   ├── paper1/           # Each paper gets its own module
│   │   └── module.py
│   ├── paper2/
│   │   └── module.py
│   └── common/           # Shared utilities
│       └── attention.py
├── tests/
│   ├── test_equations/   # Per-equation tests
│   └── test_integration/ # Cross-paper tests
├── notebooks/            # Interactive exploration
└── papers/               # Original PDFs (optional)
```

## Quality Checklist

- [ ] All equations have test coverage
- [ ] Docstrings include paper references
- [ ] Code runs without GPU requirement
- [ ] Tests pass in under 5 seconds
- [ ] Coverage exceeds 80%
- [ ] Docker support for reproducibility

## Tools Used

- **PyTorch**: Deep learning framework
- **pytest**: Testing framework
- **ruff**: Linting and formatting
- **Docker**: Containerization for reproducibility

## Related Papers in This Project

1. **TITANS**: Learning to Memorize at Test Time
   - Focus: Gradient-based memory updates
   - Key: Surprise metric for memory updates

2. **MIRAS**: It's All Connected
   - Focus: Unified framework for memory systems
   - Variants: Moneta (ℓ_p), Yaad (Huber), Memora (KL)

3. **NL**: Nested Learning
   - Focus: M3 optimizer and nested architecture
   - Key: Efficient optimization for deep networks
