"""
Common attention mechanisms.

Implements:
- TITANS Eq 1-2: Standard scaled dot-product attention
- TITANS Eq 3-5: Linear attention with kernel
"""

import torch
import torch.nn.functional as F


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
) -> torch.Tensor:
    """
    TITANS Equation 2: Causal Scaled Dot-Product Attention

    y_i = sum_{j=1}^i softmax(q_i^T k_j / sqrt(d)) v_j

    Paper: TITANS (Section 2, Equation 2)
    Description: Standard causal attention with temperature scaling

    Args:
        q: Query tensor [batch, seq_len, d_model]
        k: Key tensor [batch, seq_len, d_model]
        v: Value tensor [batch, seq_len, d_model]
        causal: Whether to apply causal masking (default: True)

    Returns:
        Attention output [batch, seq_len, d_model]
    """
    batch_size, seq_len, d_model = q.shape

    # Compute attention scores: Q K^T / sqrt(d)
    # Shape: [batch, seq_len, seq_len]
    scores = torch.matmul(q, k.transpose(-2, -1)) / (d_model**0.5)

    if causal:
        # Create causal mask: upper triangle = -inf
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1).bool()
        scores = scores.masked_fill(causal_mask, float("-inf"))

    # Apply softmax: [batch, seq_len, seq_len]
    attn_weights = F.softmax(scores, dim=-1)

    # Compute output: [batch, seq_len, d_model]
    output = torch.matmul(attn_weights, v)

    return output


def linear_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_fn: str = "elu",
) -> torch.Tensor:
    """
    TITANS Equation 3-5: Linear Attention with Kernel

    y_i = phi(q_i)^T sum_{j=1}^i phi(k_j) v_j^T / phi(q_i)^T sum_{j=1}^i phi(k_j)

    Paper: TITANS (Section 2, Equations 3-5)
    Description: Linear complexity attention using kernel function phi

    Args:
        q: Query tensor [batch, seq_len, d_model]
        k: Key tensor [batch, seq_len, d_model]
        v: Value tensor [batch, seq_len, d_model]
        kernel_fn: Kernel function ('elu', 'relu', 'identity')

    Returns:
        Attention output [batch, seq_len, d_model]
    """
    batch_size, seq_len, d_model = q.shape

    # Apply kernel function phi
    if kernel_fn == "elu":
        # ELU(x) + 1 to ensure positivity
        phi_q = F.elu(q) + 1
        phi_k = F.elu(k) + 1
    elif kernel_fn == "relu":
        phi_q = F.relu(q)
        phi_k = F.relu(k)
    else:  # identity
        phi_q = q
        phi_k = k

    # Compute cumulative sums for causal linear attention
    # kv_state[i] = sum_{j=1}^i phi(k_j) v_j^T
    # Shape: [batch, d_model, d_model]
    kv_state = torch.zeros(batch_size, d_model, d_model, device=q.device)
    k_state = torch.zeros(batch_size, d_model, device=q.device)

    outputs = []

    for i in range(seq_len):
        # Update states with current position
        kv_state = kv_state + torch.einsum("bd,be->bde", phi_k[:, i], v[:, i])
        k_state = k_state + phi_k[:, i]

        # Compute output: phi(q_i)^T kv_state / phi(q_i)^T k_state
        numerator = torch.einsum("bd,bde->be", phi_q[:, i], kv_state)
        denominator = torch.einsum("bd,bd->b", phi_q[:, i], k_state).unsqueeze(-1) + 1e-6

        output_i = numerator / denominator
        outputs.append(output_i)

    # Stack outputs: [batch, seq_len, d_model]
    output = torch.stack(outputs, dim=1)

    return output
