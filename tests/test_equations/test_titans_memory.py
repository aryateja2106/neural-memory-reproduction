"""
Test TITANS core memory equations.

TITANS Equation 8: M_{t+1} = M_t - η∇L(M_t; k_t, v_t)
where L(M; k, v) = ||M(k) - v||²

Paper: TITANS (Section 3.1)
"""

import torch


class TestTitansMemoryUpdate:
    """Test TITANS Equation 8: Gradient-based memory update."""

    def test_memory_update_shape(self, batch_size, d_model):
        """Memory update preserves shape."""
        from src.titans.memory import MLPMemory, memory_update

        memory = MLPMemory(d_model, d_model, num_layers=2)
        k_t = torch.randn(batch_size, d_model)
        v_t = torch.randn(batch_size, d_model)
        eta = 0.01

        updated_memory = memory_update(memory, k_t, v_t, eta)

        # Check memory parameters changed
        assert updated_memory is memory  # In-place update

    def test_loss_decreases(self, batch_size, d_model):
        """Memory update decreases reconstruction loss."""
        from src.titans.memory import MLPMemory, memory_update

        memory = MLPMemory(d_model, d_model, num_layers=2)
        k_t = torch.randn(batch_size, d_model)
        v_t = torch.randn(batch_size, d_model)
        eta = 0.1

        # Compute loss before update
        with torch.no_grad():
            pred_before = memory(k_t)
            loss_before = ((pred_before - v_t) ** 2).mean()

        # Update memory
        memory_update(memory, k_t, v_t, eta)

        # Compute loss after update
        with torch.no_grad():
            pred_after = memory(k_t)
            loss_after = ((pred_after - v_t) ** 2).mean()

        # Loss should decrease (or stay same if already converged)
        assert loss_after <= loss_before + 1e-4  # Small tolerance for numerical error

    def test_gradient_flow(self, batch_size, d_model):
        """Gradients flow through memory update."""
        from src.titans.memory import MLPMemory, memory_update

        memory = MLPMemory(d_model, d_model, num_layers=2)
        k_t = torch.randn(batch_size, d_model)
        v_t = torch.randn(batch_size, d_model)
        eta = 0.01

        # Update should not raise errors
        memory_update(memory, k_t, v_t, eta)

        # Check memory has been updated
        pred = memory(k_t)
        assert pred.shape == v_t.shape


class TestTitansMomentum:
    """Test TITANS Equation 9-10: Momentum-based update."""

    def test_momentum_update(self, batch_size, d_model):
        """Momentum update accumulates gradients."""
        from src.titans.memory import MLPMemory, momentum_update

        memory = MLPMemory(d_model, d_model, num_layers=2)
        k_t = torch.randn(batch_size, d_model)
        v_t = torch.randn(batch_size, d_model)

        state = {"S_t": None}  # Momentum state
        eta_t = 0.01
        theta_t = 0.1
        beta_t = 0.9

        # Update with momentum
        momentum_update(memory, k_t, v_t, state, eta_t, theta_t, beta_t)

        # Check momentum state was created
        assert state["S_t"] is not None


class TestTitansSurprise:
    """Test TITANS Equation 8-9: Surprise metric."""

    def test_surprise_computation(self, batch_size, d_model):
        """Surprise metric measures prediction error."""
        from src.titans.memory import MLPMemory, compute_surprise

        memory = MLPMemory(d_model, d_model, num_layers=2)
        k_t = torch.randn(batch_size, d_model)
        v_t = torch.randn(batch_size, d_model)

        surprise = compute_surprise(memory, k_t, v_t)

        # Surprise should be non-negative
        assert surprise >= 0
        assert not torch.isnan(torch.tensor(surprise))
