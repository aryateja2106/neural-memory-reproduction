"""
Tests for MIRAS memory implementations.

Tests equations:
- Equation 10-11: ℓ_p attentional bias (Moneta)
- Equation 12: Huber loss (Yaad)
- Equation 17: KL divergence retention (Memora)
- Equation 9: Delta rule update
- Equation 3: Linear RNN memory

Paper: MIRAS (arXiv:2504.13173)
"""

import torch
import torch.nn.functional as F

from src.miras.memory import (
    LinearRNNMemory,
    MemoraMemory,
    MonetaMemory,
    YaadMemory,
    delta_rule_update,
    huber_loss,
    kl_divergence_retention,
    lp_loss,
    lq_retention,
)


class TestLpLoss:
    """
    MIRAS Equation 10: ℓ_p Attentional Bias Loss

    L(M(k_t); v_t) = ||M(k_t) - v_t||_p^p
    """

    def test_output_shape(self):
        """Loss should return scalar."""
        pred = torch.randn(8, 64)
        target = torch.randn(8, 64)
        loss = lp_loss(pred, target, p=3.0)
        assert loss.dim() == 0, "Loss should be scalar"

    def test_zero_difference(self):
        """Loss should be zero when pred == target."""
        pred = torch.randn(8, 64)
        loss = lp_loss(pred, pred, p=3.0)
        assert loss.item() < 1e-6, "Loss should be ~0 for identical inputs"

    def test_p_affects_magnitude(self):
        """Higher p should amplify large errors."""
        pred = torch.tensor([[0.0, 0.0, 2.0]])  # One large error
        target = torch.tensor([[0.0, 0.0, 0.0]])

        loss_p2 = lp_loss(pred, target, p=2.0)
        loss_p3 = lp_loss(pred, target, p=3.0)
        loss_p4 = lp_loss(pred, target, p=4.0)

        # Higher p means larger loss for same diff > 1
        assert loss_p3 > loss_p2, "ℓ_3 should be larger than ℓ_2 for diff > 1"
        assert loss_p4 > loss_p3, "ℓ_4 should be larger than ℓ_3 for diff > 1"

    def test_gradient_flow(self):
        """Gradients should flow through ℓ_p loss."""
        pred = torch.randn(8, 64, requires_grad=True)
        target = torch.randn(8, 64)
        loss = lp_loss(pred, target, p=3.0)
        loss.backward()
        assert pred.grad is not None, "Gradient should flow"
        assert not torch.isnan(pred.grad).any(), "No NaN gradients"


class TestHuberLoss:
    """
    MIRAS Equation 12: Huber Loss (Yaad)

    L_δ(e) = { 0.5 * e^2,           if |e| ≤ δ
             { δ * (|e| - 0.5*δ),   otherwise
    """

    def test_output_shape(self):
        """Loss should return scalar."""
        pred = torch.randn(8, 64)
        target = torch.randn(8, 64)
        loss = huber_loss(pred, target, delta=1.0)
        assert loss.dim() == 0, "Loss should be scalar"

    def test_quadratic_region(self):
        """Small errors should be quadratic."""
        pred = torch.tensor([[0.5]])
        target = torch.tensor([[0.0]])
        loss = huber_loss(pred, target, delta=1.0)
        expected = 0.5 * 0.5**2  # Quadratic
        assert abs(loss.item() - expected) < 1e-6

    def test_linear_region(self):
        """Large errors should be linear."""
        pred = torch.tensor([[2.0]])
        target = torch.tensor([[0.0]])
        delta = 1.0
        loss = huber_loss(pred, target, delta=delta)
        expected = delta * (2.0 - 0.5 * delta)  # Linear
        assert abs(loss.item() - expected) < 1e-6

    def test_robustness_to_outliers(self):
        """Huber should be less sensitive to outliers than MSE."""
        pred = torch.tensor([[0.0, 0.0, 10.0]])  # One outlier
        target = torch.tensor([[0.0, 0.0, 0.0]])

        mse_loss = F.mse_loss(pred, target)
        hub_loss = huber_loss(pred, target, delta=1.0)

        # Huber should be much smaller due to linear treatment of outlier
        assert hub_loss < mse_loss, "Huber should be more robust to outliers"


class TestRetention:
    """
    MIRAS Equation 14 and 17: Retention functions
    """

    def test_lq_retention_zero(self):
        """Zero change should give zero retention loss."""
        W = torch.randn(64, 64)
        ret = lq_retention(W, W.clone(), q=4.0)
        assert ret.item() < 1e-6, "Same params should give zero retention"

    def test_lq_retention_positive(self):
        """Different params should give positive retention loss."""
        W = torch.randn(64, 64)
        W_prev = torch.randn(64, 64)
        ret = lq_retention(W, W_prev, q=4.0)
        assert ret.item() > 0, "Different params should give positive retention"

    def test_kl_retention_soft(self):
        """KL divergence retention (soft mode)."""
        W = torch.randn(64)
        W_prev = torch.randn(64)
        kl = kl_divergence_retention(W, W_prev, temperature=1.0, hard=False)
        assert kl.item() >= 0, "KL divergence should be non-negative"
        assert not torch.isnan(kl), "No NaN in KL"

    def test_kl_retention_hard(self):
        """KL divergence retention (hard mode)."""
        W = torch.randn(64)
        W_prev = torch.randn(64)
        kl = kl_divergence_retention(W, W_prev, temperature=1.0, hard=True)
        assert not torch.isnan(kl), "No NaN in hard KL"


class TestDeltaRule:
    """
    MIRAS Equation 9: Delta Rule Update

    M_t = α (I - η k_t k_t^T) M_{t-1} + v_t k_t^T
    """

    def test_output_shape(self):
        """Memory should maintain shape after update."""
        M = torch.randn(64, 32)
        k = torch.randn(4, 32)  # batch of keys
        v = torch.randn(4, 64)  # batch of values
        M_new = delta_rule_update(M, k, v, alpha=0.99, eta=0.1)
        assert M_new.shape == M.shape, "Memory shape should be preserved"

    def test_alpha_decay(self):
        """Strong decay should shrink memory norm."""
        M = torch.randn(64, 32) * 10  # Large initial memory
        k = torch.zeros(1, 32)
        v = torch.zeros(1, 64)
        M_new = delta_rule_update(M, k, v, alpha=0.5, eta=0.0)
        assert M_new.norm() < M.norm(), "Memory should decay with α < 1"

    def test_association_added(self):
        """New k-v pair should be added to memory."""
        M = torch.zeros(64, 32)
        k = torch.randn(1, 32)
        v = torch.randn(1, 64)
        M_new = delta_rule_update(M, k, v, alpha=1.0, eta=0.0)
        # With no decay and no subtraction, should just add v @ k^T
        expected = v.T @ k
        assert torch.allclose(M_new, expected, atol=1e-5)


class TestMonetaMemory:
    """
    MIRAS Moneta: ℓ_p attentional bias + ℓ_q retention

    Paper: MIRAS Section 4.1
    """

    def test_forward_shape(self):
        """Forward pass should produce correct output shape."""
        mem = MonetaMemory(input_dim=32, output_dim=64)
        k = torch.randn(8, 32)
        out = mem(k)
        assert out.shape == (8, 64), "Output shape should match (batch, output_dim)"

    def test_loss_computation(self):
        """Should compute ℓ_p loss correctly."""
        mem = MonetaMemory(input_dim=32, output_dim=64, p=3.0)
        k = torch.randn(8, 32)
        v = torch.randn(8, 64)
        loss = mem.compute_loss(k, v)
        assert loss.item() > 0, "Loss should be positive for random inputs"

    def test_update_changes_params(self):
        """Update should modify memory parameters."""
        mem = MonetaMemory(input_dim=32, output_dim=64)
        k = torch.randn(8, 32)
        v = torch.randn(8, 64)

        params_before = {n: p.clone() for n, p in mem.named_parameters()}
        mem.update(k, v)
        params_after = {n: p.clone() for n, p in mem.named_parameters()}

        changed = False
        for name in params_before:
            if not torch.allclose(params_before[name], params_after[name]):
                changed = True
                break

        assert changed, "Parameters should change after update"


class TestYaadMemory:
    """
    MIRAS Yaad: Huber loss + ℓ_2 retention

    Paper: MIRAS Section 4.2
    """

    def test_forward_shape(self):
        """Forward pass should produce correct output shape."""
        mem = YaadMemory(input_dim=32, output_dim=64)
        k = torch.randn(8, 32)
        out = mem(k)
        assert out.shape == (8, 64)

    def test_loss_robustness(self):
        """Yaad should be robust to outliers."""
        mem = YaadMemory(input_dim=32, output_dim=64, delta=1.0)

        # Normal loss
        k_normal = torch.randn(8, 32)
        v_normal = torch.randn(8, 64)
        loss_normal = mem.compute_loss(k_normal, v_normal)

        # Loss with outlier
        k_outlier = k_normal.clone()
        v_outlier = v_normal.clone()
        v_outlier[0, 0] = 100.0  # Add outlier
        loss_outlier = mem.compute_loss(k_outlier, v_outlier)

        # Huber treats outlier linearly, so increase should be bounded
        # MSE would have (100)^2 = 10000 increase, Huber much less
        assert loss_outlier.item() < loss_normal.item() + 200, "Huber should be robust"


class TestMemoraMemory:
    """
    MIRAS Memora: ℓ_2 loss + KL divergence retention

    Paper: MIRAS Section 4.3
    """

    def test_forward_shape(self):
        """Forward pass should produce correct output shape."""
        mem = MemoraMemory(input_dim=32, output_dim=64)
        k = torch.randn(8, 32)
        out = mem(k)
        assert out.shape == (8, 64)

    def test_soft_vs_hard(self):
        """Soft and hard modes should both work."""
        mem_soft = MemoraMemory(input_dim=32, output_dim=64, hard=False)
        mem_hard = MemoraMemory(input_dim=32, output_dim=64, hard=True)

        k = torch.randn(4, 32)
        v = torch.randn(4, 64)

        loss_soft = mem_soft.compute_loss(k, v)
        loss_hard = mem_hard.compute_loss(k, v)

        assert not torch.isnan(loss_soft), "Soft mode should not produce NaN"
        assert not torch.isnan(loss_hard), "Hard mode should not produce NaN"


class TestLinearRNNMemory:
    """
    MIRAS Equation 3: Linear RNN Memory

    M_t = α * M_{t-1} + v_t k_t^T
    """

    def test_output_shape(self):
        """Forward should produce correct output shape."""
        mem = LinearRNNMemory(d_key=32, d_value=64)
        q = torch.randn(8, 32)
        out = mem(q)
        assert out.shape == (8, 64)

    def test_memory_accumulation(self):
        """Memory should accumulate associations."""
        mem = LinearRNNMemory(d_key=32, d_value=64, alpha=1.0)

        k1 = torch.randn(1, 32)
        v1 = torch.randn(1, 64)
        mem.update(k1, v1)

        k2 = torch.randn(1, 32)
        v2 = torch.randn(1, 64)
        mem.update(k2, v2)

        # Memory should have both associations
        assert mem.M.norm().item() > 0, "Memory should accumulate"

    def test_reset(self):
        """Reset should clear memory."""
        mem = LinearRNNMemory(d_key=32, d_value=64)
        k = torch.randn(1, 32)
        v = torch.randn(1, 64)
        mem.update(k, v)
        mem.reset()
        assert mem.M.norm().item() == 0, "Memory should be cleared after reset"
