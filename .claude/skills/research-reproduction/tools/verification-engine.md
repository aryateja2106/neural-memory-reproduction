# Verification Engine Tool

## Purpose
Systematic verification of implemented equations and algorithms against paper specifications through automated test generation, shape checking, gradient verification, and numerical stability testing.

## Verification Philosophy

```
┌─────────────────────────────────────────────────────────────────┐
│                    EQUATION-FIRST VERIFICATION                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  PRINCIPLE: Test exists BEFORE implementation                    │
│                                                                  │
│  1. Extract equation from paper                                  │
│  2. Generate test skeleton with assertions                       │
│  3. Test FAILS (no implementation yet)                           │
│  4. Implement equation to pass test                              │
│  5. Verify against paper specifications                          │
│  6. Document any deviations                                      │
│                                                                  │
│  This ensures:                                                   │
│  ✓ Implementation matches paper exactly                          │
│  ✓ All edge cases considered upfront                             │
│  ✓ Numerical stability verified                                  │
│  ✓ Shape contracts enforced                                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Test Categories

### 1. Shape Verification
**Purpose:** Ensure tensor dimensions match paper specifications.

```python
class ShapeVerifier:
    """Verifies tensor shapes match paper specifications."""
    
    def __init__(self, equation_spec: dict):
        self.expected_shapes = equation_spec['expected_shapes']
    
    def verify_input_shapes(self, **inputs):
        """Check input tensor shapes."""
        for name, tensor in inputs.items():
            expected = self.expected_shapes.get(f"input_{name}")
            if expected:
                self.check_shape(tensor, expected, f"Input {name}")
    
    def verify_output_shape(self, output, name: str = "output"):
        """Check output tensor shape."""
        expected = self.expected_shapes.get(name)
        if expected:
            self.check_shape(output, expected, f"Output {name}")
    
    def check_shape(self, tensor, expected: str, context: str):
        """
        Check tensor shape against specification.
        
        Expected format: "(B, D)", "(N, N)", "(B, L, D)", etc.
        B = batch, L = sequence length, D = dimension
        """
        # Parse expected shape
        expected_dims = parse_shape_spec(expected)
        actual_dims = tensor.shape
        
        if len(expected_dims) != len(actual_dims):
            raise ShapeError(
                f"{context}: Expected {len(expected_dims)} dims, got {len(actual_dims)}. "
                f"Expected {expected}, got {tuple(actual_dims)}"
            )
        
        for i, (exp, act) in enumerate(zip(expected_dims, actual_dims)):
            if isinstance(exp, int) and exp != act:
                raise ShapeError(
                    f"{context}: Dim {i} mismatch. Expected {exp}, got {act}"
                )
```

### 2. Gradient Flow Verification
**Purpose:** Ensure gradients propagate correctly through operations.

```python
class GradientVerifier:
    """Verifies gradient flow through equations."""
    
    def verify_gradient_flow(
        self,
        forward_fn: Callable,
        inputs: dict[str, torch.Tensor],
        output_key: str = None
    ) -> dict[str, bool]:
        """
        Verify gradients flow to all inputs.
        
        Returns dict mapping input names to gradient existence.
        """
        # Enable gradients
        for tensor in inputs.values():
            tensor.requires_grad_(True)
        
        # Forward pass
        output = forward_fn(**inputs)
        if output_key:
            output = output[output_key]
        
        # Backward pass
        if output.dim() > 0:
            output.sum().backward()
        else:
            output.backward()
        
        # Check gradients
        results = {}
        for name, tensor in inputs.items():
            has_grad = tensor.grad is not None
            grad_nonzero = has_grad and tensor.grad.abs().sum() > 0
            results[name] = {
                'has_gradient': has_grad,
                'gradient_nonzero': grad_nonzero,
                'gradient_shape': tensor.grad.shape if has_grad else None,
            }
        
        return results
    
    def verify_no_grad_anomalies(self, inputs: dict[str, torch.Tensor]):
        """Check for NaN/Inf in gradients."""
        for name, tensor in inputs.items():
            if tensor.grad is not None:
                if torch.isnan(tensor.grad).any():
                    raise GradientError(f"NaN gradient in {name}")
                if torch.isinf(tensor.grad).any():
                    raise GradientError(f"Inf gradient in {name}")
```

### 3. Numerical Stability Verification
**Purpose:** Verify equations handle edge cases without numerical issues.

```python
class NumericalStabilityVerifier:
    """Verifies numerical stability of implementations."""
    
    STABILITY_TESTS = [
        {
            'name': 'zero_inputs',
            'generator': lambda shape: torch.zeros(shape),
            'description': 'All-zero input tensors',
        },
        {
            'name': 'large_values',
            'generator': lambda shape: torch.randn(shape) * 1e6,
            'description': 'Very large input values',
        },
        {
            'name': 'small_values',
            'generator': lambda shape: torch.randn(shape) * 1e-6,
            'description': 'Very small input values',
        },
        {
            'name': 'mixed_scale',
            'generator': lambda shape: torch.randn(shape) * torch.tensor([1e-6, 1e6]).view(-1, 1),
            'description': 'Mixed scale values',
        },
        {
            'name': 'negative_values',
            'generator': lambda shape: -torch.abs(torch.randn(shape)),
            'description': 'All negative values',
        },
    ]
    
    def verify_stability(
        self,
        forward_fn: Callable,
        input_shapes: dict[str, tuple],
        tests: list[str] = None
    ) -> dict[str, dict]:
        """
        Run stability tests on a function.
        
        Returns dict with test results.
        """
        tests = tests or [t['name'] for t in self.STABILITY_TESTS]
        results = {}
        
        for test_spec in self.STABILITY_TESTS:
            if test_spec['name'] not in tests:
                continue
            
            try:
                # Generate test inputs
                inputs = {
                    name: test_spec['generator'](shape)
                    for name, shape in input_shapes.items()
                }
                
                # Run forward pass
                output = forward_fn(**inputs)
                
                # Check output validity
                result = {
                    'passed': True,
                    'has_nan': torch.isnan(output).any().item(),
                    'has_inf': torch.isinf(output).any().item(),
                    'output_range': (output.min().item(), output.max().item()),
                }
                
                if result['has_nan'] or result['has_inf']:
                    result['passed'] = False
                    result['error'] = 'Output contains NaN or Inf'
                
            except Exception as e:
                result = {
                    'passed': False,
                    'error': str(e),
                }
            
            results[test_spec['name']] = result
        
        return results
```

### 4. Determinism Verification
**Purpose:** Ensure same inputs produce same outputs.

```python
class DeterminismVerifier:
    """Verifies deterministic behavior."""
    
    def verify_deterministic(
        self,
        forward_fn: Callable,
        inputs: dict[str, torch.Tensor],
        num_runs: int = 5
    ) -> bool:
        """
        Verify function produces consistent outputs.
        """
        outputs = []
        
        for _ in range(num_runs):
            # Reset any stateful components
            torch.manual_seed(42)
            
            # Clone inputs to avoid mutation
            cloned_inputs = {k: v.clone() for k, v in inputs.items()}
            
            # Run forward
            output = forward_fn(**cloned_inputs)
            outputs.append(output.clone())
        
        # Compare all outputs to first
        reference = outputs[0]
        for i, output in enumerate(outputs[1:], 1):
            if not torch.allclose(reference, output, rtol=1e-5, atol=1e-5):
                return False
        
        return True
```

### 5. Batch Independence Verification
**Purpose:** Ensure batch elements don't affect each other.

```python
class BatchIndependenceVerifier:
    """Verifies batch elements are processed independently."""
    
    def verify_batch_independence(
        self,
        forward_fn: Callable,
        inputs: dict[str, torch.Tensor],
        batch_dim: int = 0
    ) -> bool:
        """
        Verify that batch elements don't affect each other.
        
        Strategy: Run with full batch vs single elements, compare.
        """
        batch_size = list(inputs.values())[0].shape[batch_dim]
        
        # Run full batch
        full_output = forward_fn(**inputs)
        
        # Run each element separately
        single_outputs = []
        for i in range(batch_size):
            single_inputs = {
                name: tensor.select(batch_dim, i).unsqueeze(batch_dim)
                for name, tensor in inputs.items()
            }
            single_output = forward_fn(**single_inputs)
            single_outputs.append(single_output.squeeze(batch_dim))
        
        # Reconstruct and compare
        reconstructed = torch.stack(single_outputs, dim=batch_dim)
        
        return torch.allclose(full_output, reconstructed, rtol=1e-5, atol=1e-5)
```

## Test Generation

### Auto-Generated Test Template

```python
def generate_equation_test(equation: ExtractedEquation) -> str:
    """Generate pytest test for an equation."""
    
    template = '''
import pytest
import torch
import torch.nn.functional as F

from src.equations import {function_name}


class Test{class_name}:
    """Tests for Equation {eq_id}: {eq_name}
    
    LaTeX: {latex}
    Paper: {paper_name}, Section {section}
    """
    
    @pytest.fixture
    def sample_inputs(self):
        """Generate sample inputs matching paper specifications."""
        B, D = 4, 64  # Batch size, dimension
        return {{
{input_fixtures}
        }}
    
    def test_output_shape(self, sample_inputs):
        """Verify output shape matches paper: {expected_output_shape}"""
        output = {function_name}(**sample_inputs)
        
        expected_shape = {expected_shape_tuple}
        assert output.shape == expected_shape, \\
            f"Expected shape {{expected_shape}}, got {{output.shape}}"
    
    def test_gradient_flow(self, sample_inputs):
        """Verify gradients flow to all inputs."""
        for tensor in sample_inputs.values():
            tensor.requires_grad_(True)
        
        output = {function_name}(**sample_inputs)
        output.sum().backward()
        
{gradient_assertions}
    
    def test_numerical_stability_zeros(self, sample_inputs):
        """Verify handles zero inputs without NaN/Inf."""
        zero_inputs = {{k: torch.zeros_like(v) for k, v in sample_inputs.items()}}
        
        output = {function_name}(**zero_inputs)
        
        assert not torch.isnan(output).any(), "Output contains NaN with zero inputs"
        assert not torch.isinf(output).any(), "Output contains Inf with zero inputs"
    
    def test_numerical_stability_large(self, sample_inputs):
        """Verify handles large values without overflow."""
        large_inputs = {{k: v * 1e4 for k, v in sample_inputs.items()}}
        
        output = {function_name}(**large_inputs)
        
        assert not torch.isnan(output).any(), "Output contains NaN with large inputs"
        assert not torch.isinf(output).any(), "Output contains Inf with large inputs"
    
    def test_deterministic(self, sample_inputs):
        """Verify same inputs produce same outputs."""
        torch.manual_seed(42)
        output1 = {function_name}(**sample_inputs)
        
        torch.manual_seed(42)
        output2 = {function_name}(**sample_inputs)
        
        assert torch.allclose(output1, output2), "Non-deterministic behavior detected"
    
    def test_batch_independence(self, sample_inputs):
        """Verify batch elements don't affect each other."""
        # Run full batch
        full_output = {function_name}(**sample_inputs)
        
        # Run single elements
        B = list(sample_inputs.values())[0].shape[0]
        single_outputs = []
        for i in range(B):
            single_inputs = {{k: v[i:i+1] for k, v in sample_inputs.items()}}
            single_outputs.append({function_name}(**single_inputs))
        
        reconstructed = torch.cat(single_outputs, dim=0)
        
        assert torch.allclose(full_output, reconstructed, rtol=1e-5), \\
            "Batch elements affect each other"
{custom_tests}
'''
    
    return template.format(
        function_name=equation.name.lower().replace(' ', '_'),
        class_name=equation.name.replace(' ', ''),
        eq_id=equation.id,
        eq_name=equation.name,
        latex=equation.latex,
        paper_name=equation.paper_name,
        section=equation.section,
        input_fixtures=generate_input_fixtures(equation),
        expected_output_shape=equation.expected_shapes.get('output', '(B,)'),
        expected_shape_tuple=shape_to_tuple(equation.expected_shapes.get('output', '(B,)')),
        gradient_assertions=generate_gradient_assertions(equation),
        custom_tests=generate_custom_tests(equation),
    )
```

## Verification Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    VERIFICATION WORKFLOW                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  PHASE 1: Test Generation                                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  For each equation in paper.context.md:                   │   │
│  │  1. Parse equation specification                          │   │
│  │  2. Generate test_eq{N}_{name}.py                        │   │
│  │  3. Create fixtures for inputs/outputs                    │   │
│  │  4. Add shape, gradient, stability tests                  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  PHASE 2: Pre-Implementation Verification                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Run tests → ALL SHOULD FAIL (no implementation yet)      │   │
│  │  ✗ test_output_shape - ImportError (function not found)   │   │
│  │  ✗ test_gradient_flow - ImportError                       │   │
│  │  ✗ test_numerical_stability - ImportError                 │   │
│  │  This confirms tests are correctly targeting new code     │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  PHASE 3: Implementation                                         │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Implement equation following PyTorch hints               │   │
│  │  Run tests incrementally:                                 │   │
│  │  pytest tests/test_equations/test_eq1_*.py -v             │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  PHASE 4: Verification                                           │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  ✓ All equation tests pass                                │   │
│  │  ✓ Coverage report shows >80% on equation modules         │   │
│  │  ✓ No numerical warnings/errors                           │   │
│  │  ✓ Shapes match paper exactly                             │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  PHASE 5: Integration Verification                               │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Test equations work together:                            │   │
│  │  - Equation 1 output feeds Equation 2                     │   │
│  │  - Full forward pass through module                       │   │
│  │  - Compare against paper's toy example (if provided)      │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## CLI Commands

```bash
# Generate tests for all equations
uv run scripts/verify_equations.py generate

# Generate tests for specific paper
uv run scripts/verify_equations.py generate --paper titans

# Run verification
uv run scripts/verify_equations.py verify

# Run specific equation tests
uv run scripts/verify_equations.py verify --equation eq_1

# Check coverage
uv run scripts/verify_equations.py verify --coverage

# List equations and their test status
uv run scripts/verify_equations.py list
```

## Output Report

```
╔══════════════════════════════════════════════════════════════════╗
║                    EQUATION VERIFICATION REPORT                   ║
╠══════════════════════════════════════════════════════════════════╣
║  Paper: TITANS (arXiv:2501.00663)                                ║
║  Equations: 15 total, 12 verified, 3 pending                     ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  Eq 1: Surprise Computation                                       ║
║  ├─ Shape: ✓ PASS                                                ║
║  ├─ Gradient: ✓ PASS                                             ║
║  ├─ Stability: ✓ PASS (6/6 tests)                                ║
║  ├─ Determinism: ✓ PASS                                          ║
║  └─ Batch Independence: ✓ PASS                                   ║
║                                                                   ║
║  Eq 2: Memory Update Gate                                         ║
║  ├─ Shape: ✓ PASS                                                ║
║  ├─ Gradient: ✓ PASS                                             ║
║  ├─ Stability: ⚠ WARN (5/6 tests - zero_inputs has NaN)          ║
║  │             → Added eps=1e-8 to division                       ║
║  ├─ Determinism: ✓ PASS                                          ║
║  └─ Batch Independence: ✓ PASS                                   ║
║                                                                   ║
║  Eq 3: Persistent Memory                                          ║
║  └─ ○ PENDING (no implementation yet)                             ║
║                                                                   ║
╠══════════════════════════════════════════════════════════════════╣
║  Overall: 80% verified (12/15)                                    ║
║  Coverage: 85% of src/equations/                                  ║
║  Next: Implement Eq 3, 4, 5                                       ║
╚══════════════════════════════════════════════════════════════════╝
```

## Error Categories

```python
class VerificationError(Exception):
    """Base verification error."""
    pass

class ShapeError(VerificationError):
    """Tensor shape doesn't match paper."""
    pass

class GradientError(VerificationError):
    """Gradient computation issue."""
    pass

class StabilityError(VerificationError):
    """Numerical stability issue."""
    pass

class DeterminismError(VerificationError):
    """Non-deterministic behavior."""
    pass

class BatchDependenceError(VerificationError):
    """Batch elements affect each other."""
    pass
```
