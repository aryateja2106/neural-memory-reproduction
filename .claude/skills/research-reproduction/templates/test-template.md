# Equation Test Template

Standard test structure for verifying equation implementations.

## Template Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `{N}` | Equation number | `1`, `2`, `3` |
| `{NAME}` | Equation name | `memory_update`, `surprise_metric` |
| `{PAPER_ID}` | Paper identifier | `TITANS`, `MIRAS` |
| `{LATEX}` | LaTeX formula | `M_{t+1} = M_t + \eta \nabla_M \ell` |
| `{SECTION}` | Paper section | `Section 3.2` |

## Template

```python
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.0", "pytest>=8.0"]
# ///
"""
Tests for Equation {N}: {NAME}

Paper: {PAPER_ID}
Section: {SECTION}
LaTeX: ${LATEX}$

Variables:
{VARIABLE_DEFINITIONS}

Test Categories:
1. Shape verification - Output dimensions match paper
2. Gradient flow - Backward pass works correctly  
3. Numerical stability - Handles extreme values
4. Determinism - Reproducible outputs
5. Batch independence - Elements don't affect each other
6. Mathematical correctness - Matches hand-computed examples
"""
import torch
import pytest
from typing import Callable, Tuple

# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def default_dims() -> dict:
    """Standard dimensions for testing."""
    return {
        "batch_size": 4,
        "seq_len": 32,
        "d_model": 256,
        "n_heads": 8,
        "memory_size": 16,
    }


@pytest.fixture
def device() -> torch.device:
    """Get available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def equation_fn() -> Callable:
    """Import the equation implementation.
    
    Returns:
        The function/class implementing Equation {N}.
    """
    from src.layers.{MODULE} import {FUNCTION_NAME}
    return {FUNCTION_NAME}


# =============================================================================
# TEST 1: OUTPUT SHAPE
# =============================================================================

class TestOutputShape:
    """Verify output tensor shapes match paper specification.
    
    Paper states: {SHAPE_SPECIFICATION}
    """
    
    @pytest.mark.parametrize("batch_size,seq_len,d_model", [
        (1, 1, 64),       # Minimal case
        (4, 32, 256),     # Standard case
        (8, 128, 512),    # Large case
        (1, 1024, 256),   # Long sequence
        (16, 8, 128),     # Large batch
    ])
    def test_shape_variations(
        self, 
        equation_fn: Callable,
        batch_size: int, 
        seq_len: int, 
        d_model: int
    ):
        """Output shape correct across different input sizes."""
        x = torch.randn(batch_size, seq_len, d_model)
        result = equation_fn(x)
        
        # Expected shape from paper: {EXPECTED_SHAPE}
        expected = ({EXPECTED_SHAPE_TUPLE})
        assert result.shape == expected, \
            f"Expected {expected}, got {result.shape}"
    
    def test_dtype_preserved(self, equation_fn: Callable, default_dims: dict):
        """Output dtype matches input dtype."""
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            x = torch.randn(
                default_dims["batch_size"], 
                default_dims["seq_len"], 
                default_dims["d_model"],
                dtype=dtype
            )
            result = equation_fn(x)
            assert result.dtype == dtype, \
                f"Expected {dtype}, got {result.dtype}"


# =============================================================================
# TEST 2: GRADIENT FLOW
# =============================================================================

class TestGradientFlow:
    """Verify gradients flow correctly through the operation.
    
    Required for end-to-end training as described in {SECTION}.
    """
    
    def test_backward_pass(self, equation_fn: Callable, default_dims: dict):
        """Backward pass computes gradients without error."""
        x = torch.randn(
            default_dims["batch_size"],
            default_dims["seq_len"],
            default_dims["d_model"],
            requires_grad=True
        )
        
        result = equation_fn(x)
        loss = result.sum()
        
        # Should not raise
        loss.backward()
        
        assert x.grad is not None, "No gradient computed for input"
    
    def test_gradients_not_nan(self, equation_fn: Callable, default_dims: dict):
        """Gradients must not contain NaN values."""
        x = torch.randn(
            default_dims["batch_size"],
            default_dims["seq_len"],
            default_dims["d_model"],
            requires_grad=True
        )
        
        result = equation_fn(x)
        loss = result.sum()
        loss.backward()
        
        assert not torch.isnan(x.grad).any(), "NaN in gradients"
    
    def test_gradients_not_inf(self, equation_fn: Callable, default_dims: dict):
        """Gradients must not contain infinite values."""
        x = torch.randn(
            default_dims["batch_size"],
            default_dims["seq_len"],
            default_dims["d_model"],
            requires_grad=True
        )
        
        result = equation_fn(x)
        loss = result.sum()
        loss.backward()
        
        assert not torch.isinf(x.grad).any(), "Inf in gradients"
    
    def test_gradients_nonzero(self, equation_fn: Callable, default_dims: dict):
        """Gradients must be non-zero (operation is not dead)."""
        x = torch.randn(
            default_dims["batch_size"],
            default_dims["seq_len"],
            default_dims["d_model"],
            requires_grad=True
        )
        
        result = equation_fn(x)
        loss = result.sum()
        loss.backward()
        
        assert x.grad.abs().sum() > 0, "Zero gradients - dead operation"


# =============================================================================
# TEST 3: NUMERICAL STABILITY
# =============================================================================

class TestNumericalStability:
    """Verify operation handles extreme values correctly."""
    
    def test_large_inputs(self, equation_fn: Callable, default_dims: dict):
        """Operation handles large input values (overflow protection)."""
        x = torch.randn(
            default_dims["batch_size"],
            default_dims["seq_len"],
            default_dims["d_model"]
        ) * 1000  # Large values
        
        result = equation_fn(x)
        
        assert not torch.isnan(result).any(), \
            "NaN with large inputs - needs overflow protection"
        assert not torch.isinf(result).any(), \
            "Inf with large inputs - needs value clipping"
    
    def test_small_inputs(self, equation_fn: Callable, default_dims: dict):
        """Operation handles small input values (underflow protection)."""
        x = torch.randn(
            default_dims["batch_size"],
            default_dims["seq_len"],
            default_dims["d_model"]
        ) * 1e-8  # Small values
        
        result = equation_fn(x)
        
        assert not torch.isnan(result).any(), \
            "NaN with small inputs - needs epsilon in denominator"
        assert not torch.isinf(result).any(), \
            "Inf with small inputs - division by near-zero"
    
    def test_zero_inputs(self, equation_fn: Callable, default_dims: dict):
        """Operation handles zero inputs."""
        x = torch.zeros(
            default_dims["batch_size"],
            default_dims["seq_len"],
            default_dims["d_model"]
        )
        
        result = equation_fn(x)
        
        assert not torch.isnan(result).any(), "NaN with zero inputs"
        assert not torch.isinf(result).any(), "Inf with zero inputs"
    
    def test_mixed_signs(self, equation_fn: Callable, default_dims: dict):
        """Operation handles mixed positive/negative values."""
        x = torch.randn(
            default_dims["batch_size"],
            default_dims["seq_len"],
            default_dims["d_model"]
        )
        x[..., ::2] = x[..., ::2].abs()   # Positive
        x[..., 1::2] = -x[..., 1::2].abs() # Negative
        
        result = equation_fn(x)
        
        assert not torch.isnan(result).any(), "NaN with mixed signs"


# =============================================================================
# TEST 4: DETERMINISM
# =============================================================================

class TestDeterminism:
    """Verify operation produces reproducible results."""
    
    def test_same_input_same_output(
        self, 
        equation_fn: Callable, 
        default_dims: dict
    ):
        """Identical inputs produce identical outputs."""
        torch.manual_seed(42)
        x = torch.randn(
            default_dims["batch_size"],
            default_dims["seq_len"],
            default_dims["d_model"]
        )
        
        result1 = equation_fn(x.clone())
        result2 = equation_fn(x.clone())
        
        assert torch.allclose(result1, result2, rtol=1e-5, atol=1e-6), \
            "Non-deterministic behavior detected"
    
    def test_seed_reproducibility(
        self, 
        equation_fn: Callable, 
        default_dims: dict
    ):
        """Results reproducible with same random seed."""
        results = []
        
        for _ in range(3):
            torch.manual_seed(12345)
            x = torch.randn(
                default_dims["batch_size"],
                default_dims["seq_len"],
                default_dims["d_model"]
            )
            results.append(equation_fn(x))
        
        for i in range(1, len(results)):
            assert torch.allclose(results[0], results[i], rtol=1e-5), \
                f"Run {i} differs from run 0"


# =============================================================================
# TEST 5: BATCH INDEPENDENCE
# =============================================================================

class TestBatchIndependence:
    """Verify batch elements don't affect each other.
    
    Critical for correct training dynamics.
    """
    
    def test_single_vs_batched(
        self, 
        equation_fn: Callable, 
        default_dims: dict
    ):
        """Single element results match batched results."""
        torch.manual_seed(42)
        
        # Create individual inputs
        x1 = torch.randn(1, default_dims["seq_len"], default_dims["d_model"])
        x2 = torch.randn(1, default_dims["seq_len"], default_dims["d_model"])
        
        # Process individually
        r1_single = equation_fn(x1)
        r2_single = equation_fn(x2)
        
        # Process as batch
        x_batch = torch.cat([x1, x2], dim=0)
        r_batch = equation_fn(x_batch)
        
        assert torch.allclose(r1_single, r_batch[0:1], rtol=1e-5), \
            "Batch element 0 differs when batched vs single"
        assert torch.allclose(r2_single, r_batch[1:2], rtol=1e-5), \
            "Batch element 1 differs when batched vs single"
    
    def test_batch_order_independence(
        self, 
        equation_fn: Callable, 
        default_dims: dict
    ):
        """Batch order doesn't affect individual element results."""
        torch.manual_seed(42)
        
        x = torch.randn(
            default_dims["batch_size"],
            default_dims["seq_len"],
            default_dims["d_model"]
        )
        
        # Original order
        r_original = equation_fn(x)
        
        # Reversed order
        r_reversed = equation_fn(x.flip(0))
        
        # Compare (reversed back)
        assert torch.allclose(r_original, r_reversed.flip(0), rtol=1e-5), \
            "Results depend on batch order"


# =============================================================================
# TEST 6: MATHEMATICAL CORRECTNESS
# =============================================================================

class TestMathematicalCorrectness:
    """Verify against hand-computed examples from paper.
    
    Equation: {LATEX}
    """
    
    def test_simple_case(self, equation_fn: Callable):
        """Verify with simple hand-computable case.
        
        Input: {SIMPLE_INPUT}
        Expected: {EXPECTED_OUTPUT}
        Derivation: {DERIVATION_STEPS}
        """
        # TODO: Replace with actual test case from paper
        x = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        result = equation_fn(x)
        
        # Expected from manual computation
        # expected = torch.tensor([[[...], [...]]])
        # assert torch.allclose(result, expected, rtol=1e-4), \
        #     f"Expected {expected}, got {result}"
        
        pytest.skip("TODO: Add hand-computed verification case")
    
    def test_identity_case(self, equation_fn: Callable):
        """Verify identity/neutral element behavior if applicable.
        
        Some equations have special behavior with identity inputs.
        """
        # TODO: Add if equation has identity behavior
        pytest.skip("TODO: Add identity case if applicable")
    
    def test_boundary_case(self, equation_fn: Callable, default_dims: dict):
        """Verify behavior at equation boundaries.
        
        Paper mentions: {BOUNDARY_CONDITIONS}
        """
        # TODO: Add boundary condition tests
        pytest.skip("TODO: Add boundary condition tests")


# =============================================================================
# TEST 7: DEVICE COMPATIBILITY
# =============================================================================

class TestDeviceCompatibility:
    """Verify operation works across devices."""
    
    @pytest.mark.skipif(
        not torch.cuda.is_available(), 
        reason="CUDA not available"
    )
    def test_cuda(self, equation_fn: Callable, default_dims: dict):
        """Operation works on CUDA device."""
        x = torch.randn(
            default_dims["batch_size"],
            default_dims["seq_len"],
            default_dims["d_model"],
            device="cuda"
        )
        
        result = equation_fn(x)
        
        assert result.device.type == "cuda", "Output not on CUDA"
        assert not torch.isnan(result).any(), "NaN on CUDA"
    
    def test_cpu(self, equation_fn: Callable, default_dims: dict):
        """Operation works on CPU."""
        x = torch.randn(
            default_dims["batch_size"],
            default_dims["seq_len"],
            default_dims["d_model"],
            device="cpu"
        )
        
        result = equation_fn(x)
        
        assert result.device.type == "cpu", "Output not on CPU"
        assert not torch.isnan(result).any(), "NaN on CPU"


# =============================================================================
# TEST 8: EDGE CASES  
# =============================================================================

class TestEdgeCases:
    """Test boundary conditions and edge cases."""
    
    def test_single_token(self, equation_fn: Callable, default_dims: dict):
        """Operation handles single token sequences."""
        x = torch.randn(default_dims["batch_size"], 1, default_dims["d_model"])
        result = equation_fn(x)
        
        assert result.shape[1] == 1, "Single token output shape wrong"
        assert not torch.isnan(result).any(), "NaN with single token"
    
    def test_single_batch(self, equation_fn: Callable, default_dims: dict):
        """Operation handles batch size 1."""
        x = torch.randn(1, default_dims["seq_len"], default_dims["d_model"])
        result = equation_fn(x)
        
        assert result.shape[0] == 1, "Single batch output shape wrong"
        assert not torch.isnan(result).any(), "NaN with single batch"
    
    def test_very_long_sequence(self, equation_fn: Callable, default_dims: dict):
        """Operation handles very long sequences."""
        x = torch.randn(1, 4096, default_dims["d_model"])
        
        # Should not OOM or fail
        result = equation_fn(x)
        
        assert result.shape[1] == 4096, "Long sequence output shape wrong"
        assert not torch.isnan(result).any(), "NaN with long sequence"


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
```
