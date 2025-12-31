"""
MIRAS memory implementations.

Implements core equations from the MIRAS paper:
- Equation 3: Linear RNN memory update M_t = A_t * M_{t-1} + v_t k_t^T
- Equation 5: Gradient descent update W_t = W_{t-1} - η_t ∇ℓ(W_{t-1}; k_t, v_t)
- Equation 9: Delta rule with retention
- Equation 10-11: ℓ_p attentional bias (Moneta)
- Equation 12: Huber loss (Yaad)
- Equation 17: KL divergence retention (Memora)

Paper: MIRAS (arXiv:2504.13173)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def lp_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    p: float = 3.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    MIRAS Equation 10: ℓ_p Attentional Bias Loss

    L(M(k_t); v_t) = ||M(k_t) - v_t||_p^p

    Paper: MIRAS Section 4.1, Equation 10
    Description: ℓ_p norm loss for attentional bias. p > 2 focuses more
                 on larger errors (recent/salient tokens).

    Args:
        pred: Predictions [batch, dim]
        target: Targets [batch, dim]
        p: Power for ℓ_p norm (default: 3.0 for Moneta)
        reduction: 'mean', 'sum', or 'none'

    Returns:
        ℓ_p loss value
    """
    diff = torch.abs(pred - target)
    loss = diff.pow(p)

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


def lp_gradient(
    pred: torch.Tensor,
    target: torch.Tensor,
    p: float = 3.0,
) -> torch.Tensor:
    """
    MIRAS Equation 11: Gradient of ℓ_p Loss

    ∇_{M(k_t)} L = p * |M(k_t) - v_t|^{p-1} * sign(M(k_t) - v_t)

    Paper: MIRAS Section 4.1, Equation 11
    Description: Gradient for ℓ_p attentional bias update

    Args:
        pred: Predictions [batch, dim]
        target: Targets [batch, dim]
        p: Power for ℓ_p norm

    Returns:
        Gradient tensor [batch, dim]
    """
    diff = pred - target
    return p * torch.abs(diff).pow(p - 1) * torch.sign(diff)


def huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    delta: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    MIRAS Equation 12: Huber Loss (Yaad)

    L_δ(e) = { 0.5 * e^2,           if |e| ≤ δ
             { δ * (|e| - 0.5*δ),   otherwise

    Paper: MIRAS Section 4.2, Equation 12
    Description: Huber loss for robust attentional bias. Reduces
                 influence of outlier tokens.

    Args:
        pred: Predictions [batch, dim]
        target: Targets [batch, dim]
        delta: Threshold for Huber loss (default: 1.0)
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Huber loss value
    """
    diff = pred - target
    abs_diff = torch.abs(diff)

    quadratic = 0.5 * diff.pow(2)
    linear = delta * (abs_diff - 0.5 * delta)

    loss = torch.where(abs_diff <= delta, quadratic, linear)

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


def huber_gradient(
    pred: torch.Tensor,
    target: torch.Tensor,
    delta: float = 1.0,
) -> torch.Tensor:
    """
    MIRAS Equation 12: Gradient of Huber Loss

    ∇L_δ(e) = { e,              if |e| ≤ δ
              { δ * sign(e),    otherwise

    Paper: MIRAS Section 4.2
    Description: Gradient clipped at ±δ for robustness

    Args:
        pred: Predictions [batch, dim]
        target: Targets [batch, dim]
        delta: Threshold for Huber loss

    Returns:
        Gradient tensor [batch, dim]
    """
    diff = pred - target
    return torch.clamp(diff, -delta, delta)


def lq_retention(
    W: torch.Tensor,
    W_prev: torch.Tensor,
    q: float = 4.0,
) -> torch.Tensor:
    """
    MIRAS Equation 14: ℓ_q Retention Gate

    D_t(W, W_{t-1}) = ||W - W_{t-1}||_q^q

    Paper: MIRAS Section 4.1
    Description: ℓ_q norm retention penalizes large changes more strongly
                 when q > 2, encouraging stability.

    Args:
        W: Current memory parameters
        W_prev: Previous memory parameters
        q: Power for ℓ_q norm (default: 4.0 for Moneta)

    Returns:
        Retention loss
    """
    diff = torch.abs(W - W_prev)
    return diff.pow(q).mean()


def lq_retention_gradient(
    W: torch.Tensor,
    W_prev: torch.Tensor,
    q: float = 4.0,
) -> torch.Tensor:
    """
    MIRAS: Gradient of ℓ_q Retention

    ∇_W D_t = q * |W - W_{t-1}|^{q-1} * sign(W - W_{t-1})

    Args:
        W: Current memory parameters
        W_prev: Previous memory parameters
        q: Power for ℓ_q norm

    Returns:
        Gradient tensor
    """
    diff = W - W_prev
    return q * torch.abs(diff).pow(q - 1) * torch.sign(diff)


def kl_divergence_retention(
    W: torch.Tensor,
    W_prev: torch.Tensor,
    temperature: float = 1.0,
    hard: bool = False,
) -> torch.Tensor:
    """
    MIRAS Equation 17: KL Divergence Retention (Memora)

    D_t(W, W_{t-1}) = KL(softmax(W/τ) || softmax(W_{t-1}/τ))

    Paper: MIRAS Section 4.3, Equation 17
    Description: KL divergence retention for probabilistic memory.
                 Hard forgetting (temperature→0) creates winner-take-all.

    Args:
        W: Current memory [dim] or [batch, dim]
        W_prev: Previous memory
        temperature: Temperature for softmax (τ)
        hard: If True, use hard (argmax) approximation

    Returns:
        KL divergence retention loss
    """
    if hard:
        # Hard forgetting: deterministic selection
        current_soft = F.softmax(W / temperature, dim=-1)
        prev_hard = F.one_hot(W_prev.argmax(dim=-1), num_classes=W.size(-1)).float()
        # Cross-entropy between current soft and previous hard
        return -(prev_hard * torch.log(current_soft + 1e-8)).sum(dim=-1).mean()
    else:
        # Soft forgetting: KL divergence
        current_log = F.log_softmax(W / temperature, dim=-1)
        prev_prob = F.softmax(W_prev / temperature, dim=-1)
        return F.kl_div(current_log, prev_prob, reduction="batchmean")


def delta_rule_update(
    M: torch.Tensor,
    k_t: torch.Tensor,
    v_t: torch.Tensor,
    alpha: float = 0.99,
    eta: float = 0.1,
) -> torch.Tensor:
    """
    MIRAS Equation 9: Delta Rule with Retention

    M_t = α (I - η_t k_t k_t^T) M_{t-1} + v_t k_t^T

    Paper: MIRAS Section 3.2, Equation 9
    Description: Delta rule removes old associations while adding new.
                 This is the core memory update for DeltaNet variants.

    Args:
        M: Memory matrix [d_out, d_in]
        k_t: Key vector [d_in] or [batch, d_in]
        v_t: Value vector [d_out] or [batch, d_out]
        alpha: Retention coefficient (0.99 = strong retention)
        eta: Learning rate for delta subtraction

    Returns:
        Updated memory matrix [d_out, d_in]
    """
    # Ensure proper shapes
    if k_t.dim() == 1:
        k_t = k_t.unsqueeze(0)
    if v_t.dim() == 1:
        v_t = v_t.unsqueeze(0)

    # Compute k_t k_t^T (outer product for projection)
    # k_t: [batch, d_in], kkt: [batch, d_in, d_in]
    kkt = torch.bmm(k_t.unsqueeze(2), k_t.unsqueeze(1))  # [batch, d_in, d_in]

    # Identity minus projection: I - η k_t k_t^T
    # Average over batch for stability
    kkt_mean = kkt.mean(dim=0)  # [d_in, d_in]
    identity = torch.eye(k_t.size(1), device=M.device, dtype=M.dtype)
    proj = identity - eta * kkt_mean

    # Apply retention and projection to previous memory
    # M_t = α * M_{t-1} @ (I - η k_t k_t^T)^T
    M_decayed = alpha * M @ proj.T

    # Add new association: + v_t k_t^T
    # v_t: [batch, d_out], k_t: [batch, d_in]
    # vkt: [batch, d_out, d_in]
    vkt = torch.bmm(v_t.unsqueeze(2), k_t.unsqueeze(1)).mean(dim=0)

    return M_decayed + vkt


class AssociativeMemory(nn.Module):
    """
    MIRAS Equation 4: Base Associative Memory

    M* = argmin_M L(M(K); V)

    Paper: MIRAS Section 3.1, Equation 4
    Description: Base class for all MIRAS memory variants.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int | None = None,
        num_layers: int = 2,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        # Build MLP memory (MIRAS uses 2-layer MLP by default)
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = output_dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU())

        self.memory = nn.Sequential(*layers)

    def forward(self, k: torch.Tensor) -> torch.Tensor:
        """Query memory with key k."""
        return self.memory(k)


class MonetaMemory(AssociativeMemory):
    """
    MIRAS Moneta: ℓ_p Attentional Bias + ℓ_q Retention

    Paper: MIRAS Section 4.1
    Description: Uses ℓ_3 loss (focuses on large errors) with
                 ℓ_4 retention (stable updates). Best for salient tokens.

    Default parameters from paper:
    - p = 3.0 (attentional bias)
    - q = 4.0 (retention)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        p: float = 3.0,
        q: float = 4.0,
        eta: float = 0.01,
        alpha: float = 0.1,
    ):
        super().__init__(input_dim, output_dim)
        self.p = p
        self.q = q
        self.eta = eta  # Learning rate
        self.alpha = alpha  # Retention weight

    def compute_loss(
        self,
        k_t: torch.Tensor,
        v_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        MIRAS Eq 10: ℓ_p attentional bias loss
        """
        pred = self.forward(k_t)
        return lp_loss(pred, v_t, p=self.p)

    def update(
        self,
        k_t: torch.Tensor,
        v_t: torch.Tensor,
        prev_params: dict | None = None,
    ):
        """
        Update memory with ℓ_p gradient and ℓ_q retention.

        Combined update minimizes:
        L_p(M(k_t); v_t) + (1/α) * ||W - W_{t-1}||_q^q
        """
        # Zero gradients
        self.zero_grad()

        # Compute ℓ_p loss
        pred = self.forward(k_t)
        loss = lp_loss(pred, v_t, p=self.p)

        # Add retention if we have previous parameters
        if prev_params is not None:
            for name, param in self.named_parameters():
                if name in prev_params:
                    ret_loss = lq_retention(param, prev_params[name], q=self.q)
                    loss = loss + (1.0 / self.alpha) * ret_loss

        # Backward and update
        loss.backward()

        with torch.no_grad():
            for param in self.parameters():
                if param.grad is not None:
                    param.data -= self.eta * param.grad

        return loss.item()


class YaadMemory(AssociativeMemory):
    """
    MIRAS Yaad: Huber Loss + ℓ_2 Retention

    Paper: MIRAS Section 4.2
    Description: Uses Huber loss (robust to outliers) with
                 ℓ_2 retention. Best for noisy data.

    Default parameters from paper:
    - delta = 1.0 (Huber threshold)
    - q = 2.0 (L2 retention)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        delta: float = 1.0,
        eta: float = 0.01,
        alpha: float = 0.1,
    ):
        super().__init__(input_dim, output_dim)
        self.delta = delta
        self.eta = eta
        self.alpha = alpha

    def compute_loss(
        self,
        k_t: torch.Tensor,
        v_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        MIRAS Eq 12: Huber loss
        """
        pred = self.forward(k_t)
        return huber_loss(pred, v_t, delta=self.delta)

    def update(
        self,
        k_t: torch.Tensor,
        v_t: torch.Tensor,
        prev_params: dict | None = None,
    ):
        """
        Update memory with Huber gradient and ℓ_2 retention.
        """
        self.zero_grad()

        pred = self.forward(k_t)
        loss = huber_loss(pred, v_t, delta=self.delta)

        # ℓ_2 retention
        if prev_params is not None:
            for name, param in self.named_parameters():
                if name in prev_params:
                    ret_loss = F.mse_loss(param, prev_params[name])
                    loss = loss + (1.0 / self.alpha) * ret_loss

        loss.backward()

        with torch.no_grad():
            for param in self.parameters():
                if param.grad is not None:
                    param.data -= self.eta * param.grad

        return loss.item()


class MemoraMemory(AssociativeMemory):
    """
    MIRAS Memora: ℓ_2 Loss + KL Divergence Retention

    Paper: MIRAS Section 4.3
    Description: Uses standard ℓ_2 loss with KL divergence retention
                 for probabilistic forgetting. Supports hard/soft modes.

    Default parameters:
    - temperature = 1.0 (softmax temperature)
    - hard = False (soft forgetting)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        temperature: float = 1.0,
        hard: bool = False,
        eta: float = 0.01,
        alpha: float = 0.1,
    ):
        super().__init__(input_dim, output_dim)
        self.temperature = temperature
        self.hard = hard
        self.eta = eta
        self.alpha = alpha

    def compute_loss(
        self,
        k_t: torch.Tensor,
        v_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        ℓ_2 MSE loss
        """
        pred = self.forward(k_t)
        return F.mse_loss(pred, v_t)

    def update(
        self,
        k_t: torch.Tensor,
        v_t: torch.Tensor,
        prev_params: dict | None = None,
    ):
        """
        Update memory with ℓ_2 gradient and KL divergence retention.
        """
        self.zero_grad()

        pred = self.forward(k_t)
        loss = F.mse_loss(pred, v_t)

        # KL divergence retention
        if prev_params is not None:
            for name, param in self.named_parameters():
                if name in prev_params:
                    # Flatten for KL computation
                    flat_param = param.flatten()
                    flat_prev = prev_params[name].flatten()
                    ret_loss = kl_divergence_retention(
                        flat_param, flat_prev, temperature=self.temperature, hard=self.hard
                    )
                    loss = loss + (1.0 / self.alpha) * ret_loss

        loss.backward()

        with torch.no_grad():
            for param in self.parameters():
                if param.grad is not None:
                    param.data -= self.eta * param.grad

        return loss.item()


class LinearRNNMemory(nn.Module):
    """
    MIRAS Equation 3: General Linear RNN Memory

    M_t = A_t * M_{t-1} + v_t k_t^T

    Paper: MIRAS Section 3.2, Equation 3
    Description: Matrix-valued memory with retention decay.
    """

    def __init__(
        self,
        d_key: int,
        d_value: int,
        alpha: float = 0.99,
    ):
        super().__init__()
        self.d_key = d_key
        self.d_value = d_value
        self.alpha = alpha

        # Initialize memory matrix
        self.register_buffer("M", torch.zeros(d_value, d_key))

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """
        Retrieve from memory: y = M @ q

        Args:
            q: Query vector [batch, d_key]

        Returns:
            Retrieved value [batch, d_value]
        """
        return F.linear(q, self.M)  # [batch, d_value]

    def update(self, k: torch.Tensor, v: torch.Tensor):
        """
        Update memory: M_t = α * M_{t-1} + v @ k^T

        Args:
            k: Key vector [batch, d_key]
            v: Value vector [batch, d_value]
        """
        with torch.no_grad():
            # Decay previous memory
            self.M.mul_(self.alpha)

            # Add new association (average over batch)
            if k.dim() == 1:
                k = k.unsqueeze(0)
            if v.dim() == 1:
                v = v.unsqueeze(0)

            # v @ k^T: [batch, d_value, d_key] -> mean -> [d_value, d_key]
            outer = torch.bmm(v.unsqueeze(2), k.unsqueeze(1)).mean(dim=0)
            self.M.add_(outer)

    def reset(self):
        """Reset memory to zeros."""
        self.M.zero_()
