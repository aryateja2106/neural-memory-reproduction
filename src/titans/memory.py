"""
TITANS core memory implementation.

Implements:
- TITANS Eq 8: Gradient-based memory update M_{t+1} = M_t - η∇L(M_t; k_t, v_t)
- TITANS Eq 9-10: Momentum-based update
- TITANS Eq 13-14: Forgetting mechanism
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPMemory(nn.Module):
    """
    Multi-layer perceptron memory.

    Paper: TITANS (Section 3.1, Equation 11-12)
    Description: k-layer MLP as parametric memory M(k; θ)
    """

    def __init__(
        self, input_dim: int, output_dim: int, hidden_dim: int = None, num_layers: int = 2
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim

        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = output_dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU())

        self.mlp = nn.Sequential(*layers)

    def forward(self, k: torch.Tensor) -> torch.Tensor:
        """
        Query memory with key k.

        Args:
            k: Key tensor [batch, input_dim]

        Returns:
            Value tensor [batch, output_dim]
        """
        return self.mlp(k)


def memory_update(
    memory: nn.Module,
    k_t: torch.Tensor,
    v_t: torch.Tensor,
    eta: float,
) -> nn.Module:
    """
    TITANS Equation 8: Gradient-based Memory Update

    M_{t+1} = M_t - η∇_{M_t} L(M_t; k_t, v_t)

    where L(M; k, v) = ||M(k) - v||²_2

    Paper: TITANS (Section 3.1, Equation 8)
    Description: Update memory parameters to minimize prediction error

    Args:
        memory: Memory module (e.g., MLP)
        k_t: Key tensor [batch, d_in]
        v_t: Target value tensor [batch, d_out]
        eta: Learning rate

    Returns:
        Updated memory (same object, updated in-place)
    """
    # Zero gradients
    memory.zero_grad()

    # Forward pass: M_t(k_t)
    pred = memory(k_t)

    # Compute loss: ||M_t(k_t) - v_t||²
    loss = F.mse_loss(pred, v_t)

    # Backward pass: compute ∇_{M_t} L
    loss.backward()

    # Update parameters: M_{t+1} = M_t - η∇L
    with torch.no_grad():
        for param in memory.parameters():
            if param.grad is not None:
                param.data -= eta * param.grad

    return memory


def momentum_update(
    memory: nn.Module,
    k_t: torch.Tensor,
    v_t: torch.Tensor,
    state: dict,
    eta_t: float,
    theta_t: float,
    beta_t: float,
) -> nn.Module:
    """
    TITANS Equation 9-10: Momentum-based Memory Update

    S_t = η_t S_{t-1} - θ_t ∇L(M_{t-1}; k_t, v_t)
    M_t = M_{t-1} + S_t

    Paper: TITANS (Section 3.1, Equations 9-10)
    Description: Memory update with momentum for stability

    Args:
        memory: Memory module
        k_t: Key tensor [batch, d_in]
        v_t: Target value tensor [batch, d_out]
        state: Dictionary with momentum state {'S_t': ...}
        eta_t: Momentum decay
        theta_t: Gradient scale
        beta_t: Unused (for compatibility)

    Returns:
        Updated memory
    """
    # Initialize momentum if needed
    if state.get("S_t") is None:
        state["S_t"] = {name: torch.zeros_like(param) for name, param in memory.named_parameters()}

    # Zero gradients
    memory.zero_grad()

    # Compute gradient
    pred = memory(k_t)
    loss = F.mse_loss(pred, v_t)
    loss.backward()

    # Update momentum and parameters
    with torch.no_grad():
        for name, param in memory.named_parameters():
            if param.grad is not None:
                # S_t = η_t S_{t-1} - θ_t ∇L
                state["S_t"][name] = eta_t * state["S_t"][name] - theta_t * param.grad

                # M_t = M_{t-1} + S_t
                param.data += state["S_t"][name]

    return memory


def compute_surprise(
    memory: nn.Module,
    k_t: torch.Tensor,
    v_t: torch.Tensor,
) -> float:
    """
    TITANS Equation 8-9: Surprise Metric

    Surprise = ||M_t(k_t) - v_t||²

    Paper: TITANS (Section 3.1)
    Description: Measures how much memory needs to adapt

    Args:
        memory: Memory module
        k_t: Key tensor [batch, d_in]
        v_t: Target value tensor [batch, d_out]

    Returns:
        Surprise score (scalar)
    """
    with torch.no_grad():
        pred = memory(k_t)
        surprise = F.mse_loss(pred, v_t, reduction="mean").item()

    return surprise


def forgetting_gate(
    surprise: float,
    gamma_local: float,
    gamma_global: float,
    local_threshold: float,
    global_threshold: float,
) -> bool:
    """
    TITANS Equation 13-14: Forgetting Mechanism

    Forget if: surprise > γ_local (local) OR surprise > γ_global (global)

    Paper: TITANS (Section 3.2)
    Description: Decide whether to erase memory based on surprise

    Args:
        surprise: Current surprise metric
        gamma_local: Local surprise threshold
        gamma_global: Global surprise threshold
        local_threshold: Local threshold value
        global_threshold: Global threshold value

    Returns:
        True if memory should be reset, False otherwise
    """
    should_forget_local = surprise > gamma_local
    should_forget_global = surprise > gamma_global

    return should_forget_local or should_forget_global
