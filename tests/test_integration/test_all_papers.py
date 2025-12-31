"""
Integration tests for TITANS + MIRAS + NL paper implementations.

Tests that all components work together correctly and that
cross-paper dependencies are properly handled.

Papers:
- TITANS: Learning to Memorize at Test Time
- MIRAS: It's All Connected: A Journey Through Test-Time Memorization
- NL: Nested Learning: The Illusion of Deep Learning Architecture
"""

import torch

# Common imports
from src.common.attention import scaled_dot_product_attention

# MIRAS imports
from src.miras.memory import (
    LinearRNNMemory,
    MemoraMemory,
    MonetaMemory,
    YaadMemory,
    delta_rule_update,
    huber_loss,
    lp_loss,
)

# NL imports
from src.nl.optimizers import M3Optimizer

# TITANS imports
from src.titans.memory import (
    MLPMemory,
    compute_surprise,
    forgetting_gate,
    memory_update,
    momentum_update,
)

# Alias for clarity
standard_attention = scaled_dot_product_attention


class TestTitansMirasIntegration:
    """
    Test that TITANS and MIRAS share common foundations.

    MIRAS generalizes TITANS: TITANS is a special case of MIRAS with
    ℓ_2 attentional bias and ℓ_2 retention.
    """

    def test_memory_update_consistency(self):
        """
        TITANS gradient-based memory update should be equivalent to
        MIRAS with ℓ_2 attentional bias.
        """
        # Create equivalent memories
        titans_mem = MLPMemory(input_dim=32, output_dim=64, num_layers=2)
        miras_mem = MonetaMemory(input_dim=32, output_dim=64, p=2.0)

        # Copy parameters from TITANS to MIRAS
        with torch.no_grad():
            for (_n1, p1), (_n2, p2) in zip(
                titans_mem.named_parameters(), miras_mem.named_parameters(), strict=True
            ):
                p2.copy_(p1)

        # Same input
        k = torch.randn(4, 32)
        v = torch.randn(4, 64)

        # Compute losses (should be equivalent for p=2)
        titans_pred = titans_mem(k)
        titans_loss = ((titans_pred - v) ** 2).mean()

        miras_loss = lp_loss(miras_mem(k), v, p=2.0)

        # ℓ_2 loss with p=2 equals MSE
        assert torch.isclose(titans_loss, miras_loss, rtol=0.01), (
            "TITANS MSE should equal MIRAS ℓ_2 loss"
        )

    def test_delta_rule_vs_hebbian(self):
        """
        MIRAS delta rule with η=0 should reduce to Hebbian update.
        """
        M = torch.zeros(64, 32)
        k = torch.randn(1, 32)
        v = torch.randn(1, 64)

        # Delta rule with η=0 (no subtraction)
        M_delta = delta_rule_update(M.clone(), k, v, alpha=1.0, eta=0.0)

        # Hebbian update (direct outer product)
        M_hebbian = v.T @ k

        assert torch.allclose(M_delta, M_hebbian, atol=1e-5), (
            "Delta rule with η=0 should equal Hebbian"
        )


class TestTitansNlIntegration:
    """
    Test that TITANS memory update can use NL optimizers.

    NL shows that training algorithms are associative memories,
    so TITANS can use advanced optimizers like M3.
    """

    def test_titans_with_m3_optimizer(self):
        """
        TITANS memory can be updated with NL's M3 optimizer.
        """
        mem = MLPMemory(input_dim=32, output_dim=64)
        optimizer = M3Optimizer(mem.parameters(), lr=0.01, betas=(0.9, 0.999))

        k = torch.randn(8, 32)
        v = torch.randn(8, 64)

        # Training step using M3
        optimizer.zero_grad()
        pred = mem(k)
        loss = ((pred - v) ** 2).mean()
        loss.backward()
        optimizer.step()

        # Loss should decrease after update
        pred_after = mem(k)
        loss_after = ((pred_after - v) ** 2).mean()

        assert loss_after < loss, "M3 optimizer should decrease loss"


class TestMirasNlIntegration:
    """
    Test that MIRAS variants can use NL optimizer framework.
    """

    def test_moneta_with_m3(self):
        """
        Moneta memory trained with M3 optimizer.
        """
        mem = MonetaMemory(input_dim=32, output_dim=64, p=3.0)
        optimizer = M3Optimizer(mem.parameters(), lr=0.001, betas=(0.9, 0.999))

        k = torch.randn(4, 32)
        v = torch.randn(4, 64)

        initial_loss = mem.compute_loss(k, v).item()

        # Multiple training steps
        for _ in range(10):
            optimizer.zero_grad()
            loss = mem.compute_loss(k, v)
            loss.backward()
            optimizer.step()

        final_loss = mem.compute_loss(k, v).item()
        assert final_loss < initial_loss, "Training should reduce loss"


class TestAllPapersIntegration:
    """
    End-to-end integration test combining all three papers.
    """

    def test_memory_layer_pipeline(self):
        """
        Full pipeline: Attention -> Memory Update -> Retrieval
        Using components from all three papers.
        """
        batch_size = 4
        seq_len = 16
        d_model = 64

        # Input sequence
        x = torch.randn(batch_size, seq_len, d_model)

        # Step 1: Standard attention (baseline, shared across papers)
        attn_out = standard_attention(x, x, x, causal=True)
        assert attn_out.shape == x.shape, "Attention should preserve shape"

        # Step 2: TITANS memory update for each position
        titans_mem = MLPMemory(input_dim=d_model, output_dim=d_model)
        state = {}

        for t in range(seq_len):
            x_t = x[:, t, :]  # [batch, d_model]
            # Use momentum update from TITANS
            momentum_update(
                titans_mem,
                k_t=x_t,
                v_t=attn_out[:, t, :],
                state=state,
                eta_t=0.9,
                theta_t=0.01,
                beta_t=0.0,
            )

        # Step 3: Retrieve from memory using last query
        q_final = x[:, -1, :]
        mem_out = titans_mem(q_final)
        assert mem_out.shape == (batch_size, d_model)

        # Step 4: Apply MIRAS-style linear RNN for context aggregation
        linear_mem = LinearRNNMemory(d_key=d_model, d_value=d_model)
        for t in range(seq_len):
            k_t = x[:, t, :]
            v_t = attn_out[:, t, :]
            linear_mem.update(k_t, v_t)

        linear_out = linear_mem(q_final)
        assert linear_out.shape == (batch_size, d_model)

        # Pipeline completed successfully
        assert True, "Full pipeline executed successfully"

    def test_surprise_triggered_memory_switch(self):
        """
        Test surprise-based architecture selection.

        High surprise -> Use TITANS (learning-focused)
        Low surprise -> Use linear attention (efficient)
        """
        d_model = 64
        batch_size = 4

        # Create memories
        titans_mem = MLPMemory(input_dim=d_model, output_dim=d_model)

        # Normal input (low surprise expected)
        k_normal = torch.randn(batch_size, d_model) * 0.1
        v_normal = titans_mem(k_normal)  # Memory can already predict well

        surprise_normal = compute_surprise(titans_mem, k_normal, v_normal)

        # Novel input (high surprise expected)
        k_novel = torch.randn(batch_size, d_model) * 5.0  # Out of distribution
        v_novel = torch.randn(batch_size, d_model)  # Random target

        surprise_novel = compute_surprise(titans_mem, k_novel, v_novel)

        assert surprise_novel > surprise_normal, "Novel input should be more surprising"

        # Use forgetting gate to decide memory management
        should_forget = forgetting_gate(
            surprise=surprise_novel,
            gamma_local=1.0,
            gamma_global=5.0,
            local_threshold=1.0,
            global_threshold=5.0,
        )

        # With high surprise, forgetting might be triggered
        assert isinstance(should_forget, bool), "Forgetting gate should return bool"

    def test_multi_variant_memory_ensemble(self):
        """
        Combine Moneta, Yaad, and Memora for ensemble prediction.
        """
        d_in, d_out = 32, 64
        batch_size = 8

        # Create all three MIRAS variants
        moneta = MonetaMemory(d_in, d_out, p=3.0)
        yaad = YaadMemory(d_in, d_out, delta=1.0)
        memora = MemoraMemory(d_in, d_out, hard=False)

        k = torch.randn(batch_size, d_in)
        v = torch.randn(batch_size, d_out)

        # Train each variant
        for _ in range(5):
            moneta.update(k, v)
            yaad.update(k, v)
            memora.update(k, v)

        # Ensemble prediction (simple average)
        pred_moneta = moneta(k)
        pred_yaad = yaad(k)
        pred_memora = memora(k)

        ensemble_pred = (pred_moneta + pred_yaad + pred_memora) / 3

        # Ensemble should have same shape
        assert ensemble_pred.shape == (batch_size, d_out)

        # Compute ensemble loss (should be reasonable)
        ensemble_loss = ((ensemble_pred - v) ** 2).mean()
        assert ensemble_loss.item() < 100, "Ensemble loss should be bounded"


class TestGradientFlowAllPapers:
    """
    Test that gradients flow correctly through all components.
    """

    def test_end_to_end_gradient(self):
        """
        Gradients should flow from loss through all paper components.
        """
        d_model = 64
        batch_size = 4
        seq_len = 8

        # Input with gradients
        x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)

        # Attention layer
        attn_out = standard_attention(x, x, x, causal=True)

        # Moneta memory
        moneta = MonetaMemory(d_model, d_model)
        mem_out = moneta(attn_out.reshape(-1, d_model))
        mem_out = mem_out.reshape(batch_size, seq_len, d_model)

        # Final projection and loss
        output = mem_out[:, -1, :]
        target = torch.randn(batch_size, d_model)
        loss = ((output - target) ** 2).mean()

        # Backward pass
        loss.backward()

        # Check gradient flow
        assert x.grad is not None, "Gradients should flow to input"
        assert not torch.isnan(x.grad).any(), "No NaN gradients"
        assert x.grad.abs().max() > 0, "Gradients should be non-zero"


class TestNumericalStability:
    """
    Test numerical stability of all paper implementations.
    """

    def test_large_values(self):
        """Components should handle large values gracefully."""
        d_model = 64
        batch_size = 4

        # Large input values
        k = torch.randn(batch_size, d_model) * 100
        v = torch.randn(batch_size, d_model) * 100

        # TITANS memory
        titans_mem = MLPMemory(d_model, d_model)
        memory_update(titans_mem, k, v, eta=0.001)  # Small LR for stability
        out_titans = titans_mem(k)
        assert not torch.isnan(out_titans).any(), "TITANS should handle large values"

        # MIRAS Moneta
        moneta = MonetaMemory(d_model, d_model)
        loss = moneta.compute_loss(k, v)
        assert not torch.isnan(loss), "Moneta should handle large values"

        # MIRAS Huber (designed for robustness)
        huber = huber_loss(k, v, delta=1.0)
        assert not torch.isnan(huber), "Huber should handle large values"

    def test_small_values(self):
        """Components should handle small values gracefully."""
        d_model = 64
        batch_size = 4

        # Small input values
        k = torch.randn(batch_size, d_model) * 1e-6
        v = torch.randn(batch_size, d_model) * 1e-6

        # All losses should be computable
        lp = lp_loss(k, v, p=3.0)
        huber = huber_loss(k, v, delta=1.0)

        assert not torch.isnan(lp), "ℓ_p should handle small values"
        assert not torch.isnan(huber), "Huber should handle small values"
