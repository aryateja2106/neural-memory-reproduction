# PyTorch Implementation Patterns

## Purpose
Reference guide for implementing research paper equations in PyTorch with best practices for reproducibility, efficiency, and correctness.

## Module Structure Pattern

### Standard Module Template

```python
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class PaperModule(nn.Module):
    """Implementation of [Module Name] from [Paper].
    
    Paper: [Title]
    ArXiv: [ID]
    Equation: [Number]
    
    Args:
        d_model: Model dimension (D in paper)
        n_heads: Number of attention heads (H in paper)
        dropout: Dropout probability
    
    Input:
        x: (B, L, D) - batch, sequence length, dimension
    
    Output:
        out: (B, L, D) - same shape as input
    
    Example:
        >>> module = PaperModule(d_model=512, n_heads=8)
        >>> x = torch.randn(2, 100, 512)
        >>> out = module(x)
        >>> assert out.shape == x.shape
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Validate arguments
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        # Store config
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        # Layers (following paper notation)
        self.W_q = nn.Linear(d_model, d_model, bias=False)  # Eq. (X)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Initialize weights (see paper Section X.X)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights following paper recommendations."""
        # Xavier/Glorot for linear layers
        for module in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(module.weight)
        
        # Or scaled init for residual paths
        # nn.init.normal_(self.W_o.weight, std=0.02 / math.sqrt(2 * n_layers))
    
    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x: (B, L, D) input tensor
            mask: (B, 1, L, L) or (L, L) attention mask
        
        Returns:
            out: (B, L, D) output tensor
        """
        B, L, D = x.shape
        
        # Eq. (1): Compute Q, K, V
        Q = self.W_q(x)  # (B, L, D)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Reshape for multi-head: (B, L, D) -> (B, H, L, D_h)
        Q = Q.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        
        # Eq. (2): Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.d_head)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # (B, H, L, L)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Eq. (3): Softmax and weighted sum
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, V)  # (B, H, L, D_h)
        
        # Reshape back: (B, H, L, D_h) -> (B, L, D)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        
        # Eq. (4): Output projection
        out = self.W_o(out)
        
        return out
```

## Common Patterns

### 1. Attention with Flash Attention Support

```python
def attention_with_flash(
    Q: Tensor,
    K: Tensor, 
    V: Tensor,
    mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
) -> Tensor:
    """Attention with automatic Flash Attention when available."""
    
    # Check if Flash Attention is available (PyTorch 2.0+)
    if hasattr(F, 'scaled_dot_product_attention'):
        return F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=mask,
            dropout_p=dropout_p if training else 0.0,
            is_causal=is_causal,
        )
    
    # Fallback to manual implementation
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    if is_causal:
        L = Q.size(-2)
        causal_mask = torch.triu(torch.ones(L, L, device=Q.device), diagonal=1).bool()
        scores = scores.masked_fill(causal_mask, float('-inf'))
    elif mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    attn = F.softmax(scores, dim=-1)
    
    if dropout_p > 0:
        attn = F.dropout(attn, p=dropout_p)
    
    return torch.matmul(attn, V)
```

### 2. Memory-Efficient Chunked Operations

```python
def chunked_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    chunk_size: int = 1024,
) -> Tensor:
    """Memory-efficient attention by processing in chunks.
    
    Useful for very long sequences where full attention doesn't fit.
    """
    B, H, L, D = Q.shape
    
    # Process queries in chunks
    outputs = []
    for i in range(0, L, chunk_size):
        chunk_end = min(i + chunk_size, L)
        Q_chunk = Q[:, :, i:chunk_end, :]
        
        # Compute attention for this chunk
        scores = torch.matmul(Q_chunk, K.transpose(-2, -1))
        scores = scores / math.sqrt(D)
        attn = F.softmax(scores, dim=-1)
        out_chunk = torch.matmul(attn, V)
        
        outputs.append(out_chunk)
    
    return torch.cat(outputs, dim=2)
```

### 3. Gradient Checkpointing

```python
class CheckpointedModule(nn.Module):
    """Module with gradient checkpointing for memory efficiency."""
    
    def __init__(self, layers: nn.ModuleList, checkpoint_every: int = 2):
        super().__init__()
        self.layers = layers
        self.checkpoint_every = checkpoint_every
    
    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            if self.training and i % self.checkpoint_every == 0:
                # Checkpoint this layer
                x = torch.utils.checkpoint.checkpoint(
                    layer,
                    x,
                    use_reentrant=False,
                )
            else:
                x = layer(x)
        return x
```

### 4. Custom Autograd Function

```python
class SurpriseFunction(torch.autograd.Function):
    """Custom autograd for surprise computation with numerical stability."""
    
    @staticmethod
    def forward(ctx, hidden: Tensor, memory_pred: Tensor, eps: float = 1e-8) -> Tensor:
        """Compute surprise: S = ||h - M(k)||_2^2"""
        diff = hidden - memory_pred
        surprise = (diff ** 2).sum(dim=-1)
        
        # Save for backward
        ctx.save_for_backward(diff)
        ctx.eps = eps
        
        return surprise
    
    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, Tensor, None]:
        """Gradient of L2 squared norm."""
        diff, = ctx.saved_tensors
        
        # d(||x||^2)/dx = 2x
        grad = 2 * diff * grad_output.unsqueeze(-1)
        
        return grad, -grad, None  # grad_hidden, grad_memory_pred, grad_eps


def compute_surprise(hidden: Tensor, memory_pred: Tensor) -> Tensor:
    """Wrapper for custom surprise function."""
    return SurpriseFunction.apply(hidden, memory_pred)
```

### 5. RMSNorm (Common in Recent Papers)

```python
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.
    
    Used in LLaMA, Gemma, and other modern architectures.
    More efficient than LayerNorm (no mean computation).
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: Tensor) -> Tensor:
        # RMS = sqrt(mean(x^2))
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)
```

### 6. Rotary Position Embedding (RoPE)

```python
class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding from RoFormer.
    
    Used in LLaMA, GPT-NeoX, and TITANS.
    """
    
    def __init__(self, dim: int, max_seq_len: int = 4096, base: int = 10000):
        super().__init__()
        self.dim = dim
        
        # Compute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute cos/sin
        pos = torch.arange(max_seq_len)
        freqs = torch.outer(pos, inv_freq)
        
        self.register_buffer('cos', freqs.cos())
        self.register_buffer('sin', freqs.sin())
    
    def forward(self, x: Tensor, seq_len: int) -> Tuple[Tensor, Tensor]:
        return self.cos[:seq_len], self.sin[:seq_len]


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply rotary embedding to input tensor."""
    # x: (B, H, L, D)
    # cos, sin: (L, D/2)
    
    x1, x2 = x.chunk(2, dim=-1)
    
    # Rotate
    x_rotated = torch.cat([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos,
    ], dim=-1)
    
    return x_rotated
```

### 7. KV Cache for Inference

```python
class KVCache:
    """Key-Value cache for efficient autoregressive inference."""
    
    def __init__(self, batch_size: int, max_seq_len: int, n_heads: int, head_dim: int, device: torch.device):
        self.max_seq_len = max_seq_len
        self.current_len = 0
        
        # Preallocate cache
        self.k_cache = torch.zeros(batch_size, n_heads, max_seq_len, head_dim, device=device)
        self.v_cache = torch.zeros(batch_size, n_heads, max_seq_len, head_dim, device=device)
    
    def update(self, k: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
        """Add new keys/values and return full cached tensors."""
        seq_len = k.size(2)
        
        self.k_cache[:, :, self.current_len:self.current_len + seq_len] = k
        self.v_cache[:, :, self.current_len:self.current_len + seq_len] = v
        self.current_len += seq_len
        
        return self.k_cache[:, :, :self.current_len], self.v_cache[:, :, :self.current_len]
    
    def reset(self):
        self.current_len = 0
```

### 8. Weight Initialization Patterns

```python
def init_weights(module: nn.Module, n_layers: int, method: str = "xavier"):
    """Initialize weights following common paper conventions.
    
    Args:
        module: Module to initialize
        n_layers: Number of layers (for residual scaling)
        method: 'xavier', 'normal', 'truncated_normal'
    """
    if isinstance(module, nn.Linear):
        if method == "xavier":
            nn.init.xavier_uniform_(module.weight)
        elif method == "normal":
            nn.init.normal_(module.weight, std=0.02)
        elif method == "truncated_normal":
            nn.init.trunc_normal_(module.weight, std=0.02)
        
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=0.02)
    
    elif isinstance(module, (nn.LayerNorm, RMSNorm)):
        nn.init.ones_(module.weight)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.zeros_(module.bias)


def init_residual_scaling(module: nn.Module, n_layers: int):
    """Scale output projections for residual connections.
    
    Following GPT-2/3 convention: scale by 1/sqrt(2*n_layers)
    """
    for name, param in module.named_parameters():
        if 'W_o' in name or 'out_proj' in name:
            param.data.mul_(1.0 / math.sqrt(2 * n_layers))
```

### 9. Mixed Precision Training Pattern

```python
from torch.cuda.amp import autocast, GradScaler


class Trainer:
    """Training loop with mixed precision support."""
    
    def __init__(self, model, optimizer, use_amp: bool = True):
        self.model = model
        self.optimizer = optimizer
        self.use_amp = use_amp
        self.scaler = GradScaler() if use_amp else None
    
    def train_step(self, batch: dict) -> float:
        self.optimizer.zero_grad()
        
        with autocast(enabled=self.use_amp):
            outputs = self.model(**batch)
            loss = outputs.loss
        
        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
        
        return loss.item()
```

### 10. Model Configuration Pattern

```python
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for model hyperparameters.
    
    Matches paper Table X notation.
    """
    # Architecture
    d_model: int = 512           # D in paper
    n_layers: int = 6            # L in paper  
    n_heads: int = 8             # H in paper
    d_ff: int = 2048             # D_ff in paper (usually 4*d_model)
    
    # Memory (TITANS-specific)
    memory_size: int = 1024      # M in paper
    memory_dim: int = 64         # D_m in paper
    
    # Training
    dropout: float = 0.1
    max_seq_len: int = 8192
    
    # Vocabulary
    vocab_size: int = 32000
    pad_token_id: int = 0
    
    @classmethod
    def from_paper(cls, variant: str = "base") -> "ModelConfig":
        """Load config matching paper specifications."""
        configs = {
            "small": cls(d_model=256, n_layers=4, n_heads=4),
            "base": cls(d_model=512, n_layers=6, n_heads=8),
            "large": cls(d_model=1024, n_layers=12, n_heads=16),
        }
        return configs[variant]
    
    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def build_model(config: ModelConfig) -> nn.Module:
    """Build model from config."""
    return TitansModel(
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        memory_size=config.memory_size,
        dropout=config.dropout,
        vocab_size=config.vocab_size,
    )
```

## Testing Utilities

```python
import pytest


@pytest.fixture
def device():
    """Get available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_batch(device):
    """Generate sample batch for testing."""
    B, L, D = 2, 10, 64
    return {
        'input_ids': torch.randint(0, 1000, (B, L), device=device),
        'attention_mask': torch.ones(B, L, device=device),
    }


def assert_tensor_equal(a: Tensor, b: Tensor, msg: str = "", rtol: float = 1e-5, atol: float = 1e-5):
    """Assert tensors are approximately equal."""
    assert torch.allclose(a, b, rtol=rtol, atol=atol), f"{msg}: max diff = {(a - b).abs().max()}"


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def check_gradient_flow(model: nn.Module, inputs: dict):
    """Verify gradients flow through model."""
    model.zero_grad()
    outputs = model(**inputs)
    loss = outputs.loss if hasattr(outputs, 'loss') else outputs.sum()
    loss.backward()
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
```

## Performance Tips

```markdown
1. **Use torch.compile (PyTorch 2.0+)**
   ```python
   model = torch.compile(model, mode="reduce-overhead")
   ```

2. **Prefer in-place operations when safe**
   ```python
   x.add_(y)  # instead of x = x + y
   ```

3. **Use contiguous tensors for matmul**
   ```python
   x = x.contiguous()  # before torch.matmul
   ```

4. **Fuse operations where possible**
   ```python
   # Instead of separate softmax + dropout
   x = F.dropout(F.softmax(x, dim=-1), p=0.1)
   ```

5. **Pre-allocate buffers**
   ```python
   self.register_buffer('mask', torch.triu(...))
   ```

6. **Use torch.no_grad() for inference**
   ```python
   with torch.no_grad():
       outputs = model(inputs)
   ```
```
