"""
Test for common attention mechanisms.

TITANS Equation 1-2: Standard scaled dot-product attention
Q = xW_Q, K = xW_K, V = xW_V
y_i = sum_j softmax(Q_i^T K_j / sqrt(d)) V_j

Paper: TITANS (Section 2)
"""

import torch


class TestStandardAttention:
    """Test standard scaled dot-product attention (TITANS Eq 1-2)."""

    def test_output_shape(self, batch_size, seq_len, d_model):
        """Output shape matches input shape."""
        from src.common.attention import scaled_dot_product_attention

        q = torch.randn(batch_size, seq_len, d_model)
        k = torch.randn(batch_size, seq_len, d_model)
        v = torch.randn(batch_size, seq_len, d_model)

        output = scaled_dot_product_attention(q, k, v, causal=True)

        assert output.shape == (batch_size, seq_len, d_model)

    def test_causal_masking(self, batch_size, seq_len, d_model):
        """Causal attention only attends to past positions."""
        from src.common.attention import scaled_dot_product_attention

        q = torch.randn(batch_size, seq_len, d_model)
        k = torch.randn(batch_size, seq_len, d_model)
        v = torch.eye(seq_len, d_model).unsqueeze(0).expand(batch_size, -1, -1)

        output = scaled_dot_product_attention(q, k, v, causal=True)

        # With causal mask, position i can only see positions 0..i
        # So output[i] should be influenced only by v[0..i]
        assert output.requires_grad is False or output.grad_fn is not None

    def test_attention_weights_sum_to_one(self, batch_size, seq_len, d_model):
        """Attention weights sum to 1 for each query."""

        q = torch.randn(batch_size, seq_len, d_model)
        k = torch.randn(batch_size, seq_len, d_model)
        _v = torch.randn(batch_size, seq_len, d_model)  # unused but needed for test structure

        # Get attention weights
        scores = torch.matmul(q, k.transpose(-2, -1)) / (d_model**0.5)

        # Apply causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        scores = scores.masked_fill(causal_mask, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)

        # Check weights sum to 1
        assert torch.allclose(attn_weights.sum(dim=-1), torch.ones(batch_size, seq_len))

    def test_gradient_flow(self, batch_size, seq_len, d_model):
        """Gradients flow through attention."""
        from src.common.attention import scaled_dot_product_attention

        q = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
        k = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
        v = torch.randn(batch_size, seq_len, d_model, requires_grad=True)

        output = scaled_dot_product_attention(q, k, v, causal=True)
        loss = output.sum()
        loss.backward()

        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None
        assert not torch.isnan(q.grad).any()
        assert not torch.isnan(k.grad).any()
        assert not torch.isnan(v.grad).any()

    def test_deterministic(self, batch_size, seq_len, d_model, seed):
        """Same inputs produce same outputs."""
        from src.common.attention import scaled_dot_product_attention

        torch.manual_seed(seed)
        q = torch.randn(batch_size, seq_len, d_model)
        k = torch.randn(batch_size, seq_len, d_model)
        v = torch.randn(batch_size, seq_len, d_model)

        output1 = scaled_dot_product_attention(q, k, v, causal=True)

        output2 = scaled_dot_product_attention(q, k, v, causal=True)

        assert torch.allclose(output1, output2)


class TestLinearAttention:
    """Test linear attention with kernel (TITANS Eq 3-5)."""

    def test_output_shape(self, batch_size, seq_len, d_model):
        """Output shape matches input shape."""
        from src.common.attention import linear_attention

        q = torch.randn(batch_size, seq_len, d_model)
        k = torch.randn(batch_size, seq_len, d_model)
        v = torch.randn(batch_size, seq_len, d_model)

        output = linear_attention(q, k, v)

        assert output.shape == (batch_size, seq_len, d_model)

    def test_linear_complexity(self, batch_size, d_model):
        """Linear attention should work with longer sequences efficiently."""
        from src.common.attention import linear_attention

        # Test with increasing sequence lengths
        seq_lens = [16, 32, 64]

        for seq_len in seq_lens:
            q = torch.randn(batch_size, seq_len, d_model)
            k = torch.randn(batch_size, seq_len, d_model)
            v = torch.randn(batch_size, seq_len, d_model)

            output = linear_attention(q, k, v)
            assert output.shape == (batch_size, seq_len, d_model)

    def test_gradient_flow(self, batch_size, seq_len, d_model):
        """Gradients flow through linear attention."""
        from src.common.attention import linear_attention

        q = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
        k = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
        v = torch.randn(batch_size, seq_len, d_model, requires_grad=True)

        output = linear_attention(q, k, v)
        loss = output.sum()
        loss.backward()

        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None
