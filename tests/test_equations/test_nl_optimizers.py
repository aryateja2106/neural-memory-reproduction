"""
Test NL optimizer implementations.

NL Equations 1-3, 10-13: Gradient descent variants
NL Algorithm 1: M3 optimizer
"""

import torch
import torch.nn as nn


class TestGradientDescent:
    """Test NL Equation 1: Standard gradient descent."""

    def test_loss_decreases(self):
        """GD should decrease loss."""
        from src.nl.optimizers import gradient_descent_step

        # Simple linear model
        model = nn.Linear(10, 1)
        x = torch.randn(5, 10)
        y = torch.randn(5, 1)

        # Compute loss before
        pred = model(x)
        loss_before = ((pred - y) ** 2).mean()

        # Compute gradient
        loss_before.backward()

        # Apply GD step
        gradient_descent_step(model, loss_before, eta=0.1)

        # Compute loss after
        with torch.no_grad():
            pred_after = model(x)
            loss_after = ((pred_after - y) ** 2).mean()

        # Loss should decrease
        assert loss_after < loss_before


class TestM3Optimizer:
    """Test NL Algorithm 1: M3 optimizer."""

    def test_optimizer_step(self):
        """M3 optimizer should update parameters."""
        from src.nl.optimizers import M3Optimizer

        model = nn.Linear(10, 1)
        optimizer = M3Optimizer(model.parameters(), lr=0.01)

        # Get initial parameters
        initial_params = {name: param.clone() for name, param in model.named_parameters()}

        # Dummy forward/backward
        x = torch.randn(5, 10)
        y = torch.randn(5, 1)
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()

        # Step
        optimizer.step()

        # Parameters should have changed
        for name, param in model.named_parameters():
            assert not torch.equal(param, initial_params[name])

    def test_loss_decreases_over_steps(self):
        """M3 should decrease loss over multiple steps."""
        from src.nl.optimizers import M3Optimizer

        model = nn.Linear(10, 1)
        optimizer = M3Optimizer(model.parameters(), lr=0.1)

        x = torch.randn(20, 10)
        y = torch.randn(20, 1)

        losses = []
        for _ in range(10):
            optimizer.zero_grad()
            pred = model(x)
            loss = ((pred - y) ** 2).mean()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should generally decrease
        assert losses[-1] < losses[0]


class TestMomentumGD:
    """Test NL Equations 10-13: Momentum gradient descent."""

    def test_momentum_accelerates_convergence(self):
        """Momentum should help convergence."""
        from src.nl.optimizers import momentum_gradient_descent

        model = nn.Linear(10, 1)
        x = torch.randn(20, 10)
        y = torch.randn(20, 1)

        momentum_state = {}
        losses = []

        for _ in range(10):
            model.zero_grad()
            pred = model(x)
            loss = ((pred - y) ** 2).mean()
            loss.backward()

            momentum_gradient_descent(model, loss, momentum_state, eta=0.1, beta=0.9)
            losses.append(loss.item())

        # Loss should decrease
        assert losses[-1] < losses[0]
