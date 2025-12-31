# Equation Verification Patterns

## Purpose
Reference guide for common equation patterns in ML papers and their verification strategies.

## Standard Test Suite

Every equation should have these tests:

```python
class EquationTestSuite:
    """Standard tests for any equation implementation."""
    
    REQUIRED_TESTS = [
        'test_output_shape',       # Tensor dimensions match paper
        'test_gradient_flow',      # Gradients propagate correctly
        'test_numerical_stability', # Handles edge cases
        'test_deterministic',      # Same input → same output
        'test_batch_independence', # Batch elements don't affect each other
    ]
    
    OPTIONAL_TESTS = [
        'test_dtype_preservation', # Output dtype matches input
        'test_device_compatibility', # Works on CPU/GPU
        'test_memory_efficiency',  # No memory leaks
        'test_jit_compatible',     # Works with torch.jit
        'test_known_values',       # Verify against hand-computed examples
    ]
```

## Pattern 1: Softmax Operations

### Paper Form
$$\text{softmax}(x)_i = \frac{\exp(x_i)}{\sum_j \exp(x_j)}$$

### Common Variants
```python
# Standard softmax
F.softmax(x, dim=-1)

# Scaled softmax (attention)
F.softmax(x / math.sqrt(d), dim=-1)

# Temperature softmax
F.softmax(x / temperature, dim=-1)

# Log-softmax (numerically stable)
F.log_softmax(x, dim=-1)
```

### Verification Tests
```python
def test_softmax_sums_to_one(self, x):
    """Softmax outputs should sum to 1 along softmax dim."""
    output = self.softmax(x)
    sums = output.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

def test_softmax_all_positive(self, x):
    """Softmax outputs should all be positive."""
    output = self.softmax(x)
    assert (output >= 0).all()

def test_softmax_numerical_stability(self):
    """Softmax should handle large values without overflow."""
    # Large values that could overflow exp()
    x = torch.tensor([1000.0, 1001.0, 1002.0])
    output = self.softmax(x)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()

def test_softmax_gradient_exists(self, x):
    """Gradients should flow through softmax."""
    x.requires_grad_(True)
    output = self.softmax(x)
    output.sum().backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
```

## Pattern 2: Attention Mechanisms

### Paper Form
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### Implementation
```python
def attention(Q, K, V, mask=None):
    """Scaled dot-product attention."""
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)
    
    return output, attn_weights
```

### Verification Tests
```python
def test_attention_output_shape(self):
    """Output shape: (B, L, D) same as V."""
    B, L, D = 2, 10, 64
    Q = torch.randn(B, L, D)
    K = torch.randn(B, L, D)
    V = torch.randn(B, L, D)
    
    output, _ = attention(Q, K, V)
    assert output.shape == (B, L, D)

def test_attention_weights_sum_to_one(self):
    """Attention weights should sum to 1 per query."""
    Q = torch.randn(2, 10, 64)
    K = torch.randn(2, 10, 64)
    V = torch.randn(2, 10, 64)
    
    _, weights = attention(Q, K, V)
    sums = weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

def test_attention_causal_mask(self):
    """Causal mask should prevent attending to future."""
    L = 5
    Q = torch.randn(1, L, 64)
    K = torch.randn(1, L, 64)
    V = torch.randn(1, L, 64)
    
    # Causal mask
    mask = torch.tril(torch.ones(L, L))
    
    _, weights = attention(Q, K, V, mask=mask)
    
    # Upper triangle should be zero
    upper = torch.triu(weights[0], diagonal=1)
    assert torch.allclose(upper, torch.zeros_like(upper), atol=1e-5)
```

## Pattern 3: Layer Normalization

### Paper Form
$$\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

### Verification Tests
```python
def test_layernorm_zero_mean(self):
    """Output should have approximately zero mean per sample."""
    x = torch.randn(4, 64)
    ln = nn.LayerNorm(64)
    
    output = ln(x)
    means = output.mean(dim=-1)
    
    # Not exactly zero due to learned shift
    # But should be close for random init
    assert torch.abs(means).max() < 0.5

def test_layernorm_unit_variance(self):
    """Output should have approximately unit variance."""
    x = torch.randn(4, 64)
    ln = nn.LayerNorm(64)
    
    output = ln(x)
    stds = output.std(dim=-1)
    
    assert torch.allclose(stds, torch.ones_like(stds), atol=0.2)

def test_layernorm_epsilon_stability(self):
    """Should handle zero-variance inputs."""
    x = torch.ones(4, 64)  # Zero variance
    ln = nn.LayerNorm(64)
    
    output = ln(x)
    assert not torch.isnan(output).any()
```

## Pattern 4: Gating Mechanisms

### Paper Form (GRU-style)
$$z_t = \sigma(W_z x_t + U_z h_{t-1})$$
$$r_t = \sigma(W_r x_t + U_r h_{t-1})$$
$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tanh(W_h x_t + U_h (r_t \odot h_{t-1}))$$

### Verification Tests
```python
def test_gate_values_in_range(self):
    """Gate values should be in [0, 1] for sigmoid gates."""
    x = torch.randn(4, 64)
    h = torch.randn(4, 128)
    
    z, r, h_new = self.gru_cell(x, h)
    
    # z and r are sigmoid outputs
    assert (z >= 0).all() and (z <= 1).all()
    assert (r >= 0).all() and (r <= 1).all()

def test_gate_gradient_flow(self):
    """Gradients should flow through all paths."""
    x = torch.randn(4, 64, requires_grad=True)
    h = torch.randn(4, 128, requires_grad=True)
    
    h_new = self.gru_cell(x, h)
    h_new.sum().backward()
    
    # Both inputs should receive gradients
    assert x.grad is not None
    assert h.grad is not None
    
    # Gradients should be non-zero (not vanished)
    assert x.grad.abs().sum() > 0
    assert h.grad.abs().sum() > 0

def test_gate_identity_initialization(self):
    """With identity init, should approximately pass through."""
    # Initialize gates to pass hidden state
    # z ≈ 0, r ≈ 1 means h_new ≈ h_old
    pass
```

## Pattern 5: Memory Operations

### Paper Form (TITANS surprise)
$$S_t = \|h_t - M_{t-1}(k_t)\|_2^2$$

### Verification Tests
```python
def test_surprise_non_negative(self):
    """Surprise (L2 squared) must be non-negative."""
    h = torch.randn(4, 64)
    k = torch.randn(4, 32)
    
    surprise = self.compute_surprise(h, k)
    
    assert (surprise >= 0).all()

def test_surprise_zero_for_exact_match(self):
    """Surprise should be zero when memory predicts exactly."""
    h = torch.randn(4, 64)
    k = torch.randn(4, 32)
    
    # Set memory to return exact h
    self.memory.set_exact_return(h)
    
    surprise = self.compute_surprise(h, k)
    
    assert torch.allclose(surprise, torch.zeros_like(surprise), atol=1e-5)

def test_surprise_increases_with_difference(self):
    """Larger differences should give larger surprise."""
    h = torch.randn(4, 64)
    k = torch.randn(4, 32)
    
    # Small perturbation
    h_small = h + 0.1 * torch.randn_like(h)
    # Large perturbation
    h_large = h + 1.0 * torch.randn_like(h)
    
    s_base = self.compute_surprise(h, k)
    s_small = self.compute_surprise(h_small, k)
    s_large = self.compute_surprise(h_large, k)
    
    # On average, larger perturbation → larger surprise
    # (may not hold for every sample due to randomness)
    assert s_large.mean() > s_small.mean()
```

## Pattern 6: Loss Functions

### Cross-Entropy
$$\mathcal{L} = -\sum_i y_i \log(\hat{y}_i)$$

### Verification Tests
```python
def test_loss_non_negative(self):
    """Cross-entropy loss must be non-negative."""
    logits = torch.randn(4, 10)
    targets = torch.randint(0, 10, (4,))
    
    loss = F.cross_entropy(logits, targets)
    
    assert loss >= 0

def test_loss_zero_for_perfect_prediction(self):
    """Loss should approach zero for perfect predictions."""
    # Create confident correct predictions
    targets = torch.tensor([0, 1, 2, 3])
    logits = torch.zeros(4, 10)
    logits[0, 0] = 100.0  # Very confident class 0
    logits[1, 1] = 100.0
    logits[2, 2] = 100.0
    logits[3, 3] = 100.0
    
    loss = F.cross_entropy(logits, targets)
    
    assert loss < 0.01

def test_loss_gradient_updates_wrong_classes(self):
    """Gradient should push wrong class logits down."""
    logits = torch.randn(1, 10, requires_grad=True)
    targets = torch.tensor([0])  # True class is 0
    
    loss = F.cross_entropy(logits, targets)
    loss.backward()
    
    # Gradient for true class should be negative (decrease loss by increasing)
    # Gradient for wrong classes should be positive (decrease by decreasing)
    grad = logits.grad[0]
    
    # After softmax, gradient of loss w.r.t. logit_i is (p_i - y_i)
    # So for true class (y=1): grad = p - 1 < 0 (negative)
    # For wrong class (y=0): grad = p > 0 (positive)
    assert grad[0] < 0  # True class
    assert grad[1:].mean() > 0  # Wrong classes on average
```

## Pattern 7: Residual Connections

### Paper Form
$$y = x + F(x)$$

### Verification Tests
```python
def test_residual_identity_at_init(self):
    """With proper init, residual should be near identity."""
    # Initialize F to output near-zero
    block = ResidualBlock(64, init_zero=True)
    x = torch.randn(4, 64)
    
    y = block(x)
    
    # Should be close to identity
    assert torch.allclose(y, x, atol=0.1)

def test_residual_gradient_shortcut(self):
    """Gradients should flow through shortcut connection."""
    block = ResidualBlock(64)
    x = torch.randn(4, 64, requires_grad=True)
    
    y = block(x)
    y.sum().backward()
    
    # Gradient should be non-vanishing due to shortcut
    assert x.grad.abs().mean() > 0.1

def test_residual_deep_gradient_flow(self):
    """Even with many residual blocks, gradients should flow."""
    blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(50)])
    x = torch.randn(4, 64, requires_grad=True)
    
    y = blocks(x)
    y.sum().backward()
    
    # Gradient should still be meaningful
    assert not torch.isnan(x.grad).any()
    assert x.grad.abs().mean() > 0.01
```

## Pattern 8: Positional Encoding

### Sinusoidal (Transformer)
$$PE_{pos, 2i} = \sin(pos / 10000^{2i/d})$$
$$PE_{pos, 2i+1} = \cos(pos / 10000^{2i/d})$$

### Verification Tests
```python
def test_positional_encoding_shape(self):
    """PE should have shape (max_len, d_model)."""
    pe = PositionalEncoding(d_model=64, max_len=100)
    
    assert pe.encoding.shape == (100, 64)

def test_positional_encoding_bounded(self):
    """Values should be in [-1, 1] for sin/cos."""
    pe = PositionalEncoding(d_model=64, max_len=100)
    
    assert pe.encoding.min() >= -1.0
    assert pe.encoding.max() <= 1.0

def test_positional_encoding_unique(self):
    """Each position should have unique encoding."""
    pe = PositionalEncoding(d_model=64, max_len=100)
    
    # No two positions should be identical
    for i in range(100):
        for j in range(i + 1, 100):
            diff = (pe.encoding[i] - pe.encoding[j]).abs().sum()
            assert diff > 0.01, f"Positions {i} and {j} too similar"

def test_positional_encoding_relative_property(self):
    """PE should encode relative positions via dot product."""
    pe = PositionalEncoding(d_model=64, max_len=100)
    
    # Dot product between positions should reflect relative distance
    # (not a strict test, but helpful sanity check)
    enc = pe.encoding
    
    # Nearby positions should be more similar
    close_sim = (enc[0] * enc[1]).sum()
    far_sim = (enc[0] * enc[50]).sum()
    
    assert close_sim > far_sim
```

## Quick Reference: Test Template

```python
import pytest
import torch

class TestEquation{N}:
    """Tests for Equation {N}: {Name}
    
    LaTeX: ${latex}$
    """
    
    @pytest.fixture
    def inputs(self):
        B, D = 4, 64
        return {
            'x': torch.randn(B, D),
            # Add other inputs
        }
    
    def test_output_shape(self, inputs):
        output = equation_n(**inputs)
        assert output.shape == (inputs['x'].shape[0], ...)
    
    def test_gradient_flow(self, inputs):
        for t in inputs.values():
            t.requires_grad_(True)
        
        output = equation_n(**inputs)
        output.sum().backward()
        
        for name, t in inputs.items():
            assert t.grad is not None, f"No gradient for {name}"
    
    def test_numerical_stability(self, inputs):
        # Test with zeros
        zero_inputs = {k: torch.zeros_like(v) for k, v in inputs.items()}
        output = equation_n(**zero_inputs)
        assert not torch.isnan(output).any()
        
        # Test with large values
        large_inputs = {k: v * 1e4 for k, v in inputs.items()}
        output = equation_n(**large_inputs)
        assert not torch.isinf(output).any()
    
    def test_deterministic(self, inputs):
        torch.manual_seed(42)
        out1 = equation_n(**inputs)
        torch.manual_seed(42)
        out2 = equation_n(**inputs)
        assert torch.allclose(out1, out2)
    
    def test_batch_independence(self, inputs):
        full_out = equation_n(**inputs)
        single_outs = []
        for i in range(inputs['x'].shape[0]):
            single = {k: v[i:i+1] for k, v in inputs.items()}
            single_outs.append(equation_n(**single))
        stacked = torch.cat(single_outs, dim=0)
        assert torch.allclose(full_out, stacked, rtol=1e-5)
```
