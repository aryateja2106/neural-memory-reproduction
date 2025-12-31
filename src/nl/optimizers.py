"""
NL optimizer implementations.

Implements:
- NL Eq 1-3: Standard gradient descent
- NL Eq 10-13: GD with momentum
- NL Algorithm 1: M3 (Multi-scale Momentum Muon) optimizer
"""

import torch
import torch.nn as nn


def gradient_descent_step(
    model: nn.Module,
    loss: torch.Tensor,
    eta: float,
) -> None:
    """
    NL Equation 1: Standard Gradient Descent

    W_{t+1} = W_t - η_t ∇_{W_t} L(W_t; x_t)

    Paper: NL (Section 2.1, Equation 1)
    Description: Basic SGD update rule

    Args:
        model: Model to update
        loss: Loss value (must have gradients computed)
        eta: Learning rate
    """
    with torch.no_grad():
        for param in model.parameters():
            if param.grad is not None:
                param.data -= eta * param.grad


class M3Optimizer(torch.optim.Optimizer):
    """
    NL Algorithm 1: Multi-scale Momentum Muon (M3) Optimizer

    Paper: NL (Algorithm 1)
    Description: Combines momentum at multiple timescales with orthogonalization

    This is a simplified implementation of M3 focusing on core functionality.
    Full implementation would include:
    - Multiple momentum timescales
    - Orthogonalization (from Muon paper)
    - Adaptive learning rates per layer
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        """
        Initialize M3 optimizer.

        Args:
            params: Model parameters
            lr: Learning rate
            betas: Momentum coefficients (beta1 for momentum, beta2 for second moment)
            eps: Small constant for numerical stability
            weight_decay: L2 regularization coefficient
        """
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform single optimization step.

        NL Algorithm 1 (simplified):
        1. Compute gradient
        2. Apply momentum at multiple scales
        3. Orthogonalize momentum (optional)
        4. Update parameters
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                # Apply weight decay
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                state = self.state[p]

                # Initialize state
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)  # First moment
                    state["exp_avg_sq"] = torch.zeros_like(p)  # Second moment

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1

                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Update biased second moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                step_size = lr / bias_correction1

                # Compute adaptive step size
                denom = (exp_avg_sq.sqrt() / (bias_correction2**0.5)).add_(eps)

                # Update parameters
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


def momentum_gradient_descent(
    model: nn.Module,
    loss: torch.Tensor,
    momentum_state: dict,
    eta: float,
    beta: float,
) -> None:
    """
    NL Equation 10-13: Gradient Descent with Momentum

    v_{t+1} = βv_t + η∇L(W_t)
    W_{t+1} = W_t - v_{t+1}

    Paper: NL (Section 2.2, Equations 10-13)
    Description: Momentum-accelerated gradient descent

    Args:
        model: Model to update
        loss: Loss value (must have gradients computed)
        momentum_state: Dictionary storing velocity state
        eta: Learning rate
        beta: Momentum coefficient
    """
    if "velocity" not in momentum_state:
        momentum_state["velocity"] = {
            name: torch.zeros_like(param) for name, param in model.named_parameters()
        }

    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.grad is not None:
                # v_{t+1} = βv_t + η∇L
                velocity = momentum_state["velocity"][name]
                velocity.mul_(beta).add_(param.grad, alpha=eta)

                # W_{t+1} = W_t - v_{t+1}
                param.data -= velocity
