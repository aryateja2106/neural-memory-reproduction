# Contributing to Neural Memory Reproduction

Thank you for your interest in contributing to this research reproduction project! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Submitting Changes](#submitting-changes)
- [Style Guidelines](#style-guidelines)
- [Testing](#testing)
- [Documentation](#documentation)

## Code of Conduct

This project follows a simple code of conduct:
- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Credit others' contributions appropriately

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- UV package manager (recommended) or pip

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/neural-memory-reproduction.git
   cd neural-memory-reproduction
   ```
3. Add upstream remote:
   ```bash
   git remote add upstream https://github.com/aryateja2106/neural-memory-reproduction.git
   ```

## Development Setup

### Using UV (Recommended)

```bash
# Create virtual environment
uv venv

# Activate it
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows

# Install dependencies with dev tools
uv pip install -e ".[dev]"
```

### Using pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Using Docker

```bash
docker build -t neural-memory-dev .
docker run -it -v $(pwd):/app neural-memory-dev bash
```

## Making Changes

### Branch Naming

Create a descriptive branch for your changes:

```bash
# For new features
git checkout -b feature/add-titans-parallelization

# For bug fixes
git checkout -b fix/memory-gradient-nan

# For documentation
git checkout -b docs/update-architecture

# For equation implementations
git checkout -b equation/miras-eq-15
```

### Commit Messages

Follow conventional commits format:

```
type(scope): brief description

Longer description if needed.

- Bullet points for details
- Reference paper equations: "Implements MIRAS Eq 15"

Refs: #issue-number
```

Types:
- `feat`: New feature or equation implementation
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding or updating tests
- `refactor`: Code refactoring
- `style`: Formatting changes
- `perf`: Performance improvements

### Equation Implementation Workflow

When implementing a new equation:

1. **Write the test first** (TDD approach):
   ```python
   # tests/test_equations/test_paper_name.py

   def test_equation_X_output_shape():
       """Test PAPER Equation X produces correct output shape."""
       # Test implementation

   def test_equation_X_gradient_flow():
       """Test PAPER Equation X has valid gradients."""
       # Test implementation

   def test_equation_X_numerical_stability():
       """Test PAPER Equation X handles edge cases."""
       # Test implementation
   ```

2. **Implement the equation**:
   ```python
   # src/paper_name/module.py

   def equation_x(input_tensor: torch.Tensor) -> torch.Tensor:
       """
       Implements PAPER Equation X.

       Paper: "Paper Title"
       Equation: X (page Y)

       LaTeX:
           output = f(input)

       Plain English:
           Description of what the equation does.

       Args:
           input_tensor: Description with shape [batch, dim]

       Returns:
           Output tensor with shape [batch, output_dim]
       """
       # Implementation
   ```

3. **Run tests**:
   ```bash
   uv run pytest tests/test_equations/test_paper_name.py -v
   ```

4. **Check coverage**:
   ```bash
   uv run pytest --cov=src --cov-report=term-missing
   ```

## Submitting Changes

### Before Submitting

1. **Update from upstream**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run all checks**:
   ```bash
   # Format code
   uv run ruff format src/ tests/

   # Lint code
   uv run ruff check src/ tests/ --fix

   # Run tests
   uv run pytest tests/ -v

   # Check coverage
   uv run pytest --cov=src --cov-report=term-missing
   ```

3. **Ensure coverage >= 80%** for new code

### Pull Request Process

1. Push your branch:
   ```bash
   git push origin your-branch-name
   ```

2. Create a Pull Request on GitHub

3. Fill out the PR template:
   - Describe what changes you made
   - Reference any related issues
   - List the equations implemented (if applicable)
   - Include test results

4. Wait for review and address feedback

## Style Guidelines

### Python Style

We use Ruff for formatting and linting. Configuration is in `pyproject.toml`.

Key points:
- Line length: 100 characters
- Use type hints for function signatures
- Use docstrings for all public functions
- Follow PEP 8 naming conventions

### Documentation Style

For equations, always include:
1. Paper reference (title, arXiv link)
2. Equation number and page
3. LaTeX representation
4. Plain English description
5. Variable definitions

Example:
```python
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

    Variables:
        M: Memory matrix [d_out, d_in]
        k: Key vector [batch, d_in]
        v: Value vector [batch, d_out]
        η: Learning rate scalar

    Args:
        M: Current memory state
        k: Key to store
        v: Value to associate with key
        eta: Learning rate

    Returns:
        Updated memory matrix
    """
```

## Testing

### Test Structure

```
tests/
├── conftest.py              # Shared fixtures
├── test_equations/          # Equation-level tests
│   ├── test_titans_*.py
│   ├── test_miras_*.py
│   └── test_nl_*.py
└── test_integration/        # Cross-paper tests
    └── test_all_papers.py
```

### Running Tests

```bash
# All tests
uv run pytest tests/ -v

# Specific paper
uv run pytest tests/test_equations/test_miras_*.py -v

# With coverage
uv run pytest --cov=src --cov-report=html

# Fast fail on first error
uv run pytest -x
```

### Test Requirements

Each equation must have at least 3 tests:
1. **Output shape test**: Verify dimensions are correct
2. **Gradient flow test**: Verify gradients propagate
3. **Numerical stability test**: Test edge cases

## Documentation

### Updating Documentation

When adding new features:
1. Update `README.md` implementation status
2. Update `ARCHITECTURE.md` if structure changes
3. Add docstrings to all new functions
4. Update `notebooks/quickstart.ipynb` with examples

### Building Documentation

```bash
# Generate API docs (if using Sphinx)
cd docs
make html
```

## Questions?

- Open an issue for bugs or feature requests
- Use discussions for questions
- Tag @aryateja2106 for urgent matters

---

Thank you for contributing to advancing research in neural memory systems!
