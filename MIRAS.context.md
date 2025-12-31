# MIRAS - Implementation Context

**Paper ID:** MIRAS
**Full Title:** It's All Connected: A Journey Through Test-Time Memorization, Attentional Bias, Retention, and Online Optimization
**Authors:** Ali Behrouz, Meisam Razaviyayn, Peilin Zhong, and Vahab Mirrokni (Google Research)
**arXiv:** arXiv:2504.13173v1 [cs.LG]
**Year:** 2025

**Key Contribution (1 sentence):**
MIRAS is a general framework to design deep learning architectures based on four choices: (i) associative memory architecture, (ii) attentional bias objective, (iii) retention gate, and (iv) memory learning algorithm, unifying Transformers, Titans, and modern linear RNNs while introducing three novel sequence models (Moneta, Yaad, Memora).

---

## Quick Reference
| Aspect | Count | Key Items |
|--------|-------|-----------|
| Equations | 32+ | Attention (Eq 1-2), Memory Updates (Eq 3-32), Novel Variants |
| Algorithms | 3 | Moneta, Yaad, Memora with parallelization |
| Architectures | 3 | 2-layer MLP memory with different objectives/gates |
| Hyperparameters | 15+ | Learning rates, retention gates, chunk sizes, p/q values |

---

## 1. EQUATIONS

### 1.1 Core Attention Mechanism (Baseline)

**Equation 1-2: Transformer Attention**

**LaTeX:**
```latex
Q = xW_Q, \quad K = xW_K, \quad V = xW_V
```

**Plain Text:**
Query, Key, and Value matrices are computed as linear projections of input x.

**Variables:**
| Variable | Shape | Description |
|----------|-------|-------------|
| x | (N, d_in) | Input sequence |
| W_Q, W_K, W_V | (d_in, d_in) | Learnable projection matrices |
| Q, K, V | (N, d_in) | Query, Key, Value matrices |
| N | scalar | Sequence length |
| d_in | scalar | Input dimension |

**Equation 2: Causal Softmax Attention**

**LaTeX:**
```latex
y_i = \sum_{j=1}^{i} \frac{\exp(q_i^\top k_j / \sqrt{d_{in}}) v_j}{\sum_{\ell=1}^{i} \exp(q_i^\top k_\ell / \sqrt{d_{in}})}
```

**Plain Text:**
Output at position i is weighted sum of values, where weights are softmax of scaled dot-product similarities.

**Variables:**
| Variable | Shape | Description |
|----------|-------|-------------|
| y_i | (d_in,) | Output at position i |
| q_i | (d_in,) | Query vector at position i |
| k_j | (d_in,) | Key vector at position j |
| v_j | (d_in,) | Value vector at position j |

**Implementation Notes:**
- Temperature scaling: 1/√d_in
- Causal masking: only attend to positions j ≤ i
- This is non-parametric solution to ℓ2-MSE loss

**Dependencies:** None (baseline)

**Test Criteria:**
- Attention weights sum to 1 per position
- Output shape matches input shape
- Gradient flows through all components

---

### 1.2 Linear RNN Memory Update (Hebbian)

**Equation 3: General Linear RNN Memory**

**LaTeX:**
```latex
M_t = A_t * M_{t-1} + v_t k_t^\top
```

**Plain Text:**
Memory at time t equals previous memory scaled by A_t plus outer product of current value and key.

**Variables:**
| Variable | Shape | Description |
|----------|-------|-------------|
| M_t | (d, n) | Memory state at time t |
| A_t | (d, d) or scalar | Decay/retention parameter (data-dependent or fixed) |
| v_t | (d,) | Value vector at time t |
| k_t | (d,) | Key vector at time t |
| * | operator | Arbitrary associative operator |
| n | scalar | Memory width (1=vector, d=matrix) |

**Implementation Notes:**
- When n=1: vector-valued memory (RetNet)
- When n=d: matrix-valued memory (Linear Attention)
- A_t can be scalar (RetNet), diagonal matrix (GLA), or data-dependent (Mamba)

**Special Cases:**
- α=1: Linear Attention
- α learnable scalar: RetNet, Lightning Attention
- α_t data-dependent: Mamba2, GLA

**Dependencies:** Key/value projections from input

**Test Criteria:**
- Memory maintains shape (d, n)
- Causality: M_t only depends on inputs up to t
- Numerical stability for long sequences

---

### 1.3 Delta Rule Memory Update

**Equation 9: Delta Rule with Retention**

**LaTeX:**
```latex
M_t = \alpha (I - \eta_t k_t k_t^\top) M_{t-1} + v_t k_t^\top
```

**Plain Text:**
Memory updated by delta rule: decay previous memory, subtract key-projection, add new key-value pair.

**Variables:**
| Variable | Shape | Description |
|----------|-------|-------------|
| M_t | (d, n) | Memory at time t |
| α | scalar or (d,) | Retention coefficient |
| η_t | scalar | Learning rate at time t |
| I | (d, d) | Identity matrix |
| k_t | (d,) | Key vector |
| v_t | (d,) | Value vector |

**Implementation Notes:**
- Delta rule removes previous associations proportionally to k_t
- When α=1: DeltaNet
- When α_t data-dependent (scalar): Gated DeltaNet
- When α_t data-dependent (vector): RWKV-7

**Special Cases (Table 1):**
```
DeltaNet: α=1, η_t constant
Gated DeltaNet: α_t ∈ ℝ (scalar), data-dependent
RWKV-7: α_t ∈ ℝ^d (vector), data-dependent
```

**Dependencies:** Gradient of ℓ2 loss

**Test Criteria:**
- Memory rank can decrease (value replacement)
- Bounded memory norm for stable α < 1
- Orthogonal key updates don't interfere

---

### 1.4 Core MIRAS Definition

**Equation 4: Associative Memory with Attentional Bias**

**LaTeX:**
```latex
M^* = \arg\min_M \mathcal{L}(M(\mathcal{K}); \mathcal{V})
```

**Plain Text:**
Optimal memory minimizes attentional bias objective L over mapping from keys K to values V.

**Variables:**
| Variable | Shape | Description |
|----------|-------|-------------|
| M | function | Memory operator: K → V |
| K | set in ℝ^{d_k} | Set of keys |
| V | set in ℝ^{d_v} | Set of values |
| L | function | Attentional bias objective |

**Implementation Notes:**
- This is meta-learning: inner loop optimizes L, outer loop optimizes other parameters
- L determines similarity metric and prioritization
- Can use any optimizer (GD, momentum, Newton, non-parametric)

**Dependencies:** None (core definition)

---

### 1.5 Gradient Descent Update (General)

**Equation 5: Memory Update via Gradient Descent**

**LaTeX:**
```latex
W_t = W_{t-1} - \eta_t \nabla \ell(W_{t-1}; k_t, v_t)
```

**Plain Text:**
Memory parameters updated by gradient descent on loss for current key-value pair.

**Variables:**
| Variable | Shape | Description |
|----------|-------|-------------|
| W_t | (varies) | Memory parameters at time t |
| η_t | scalar | Learning rate at time t |
| ℓ(W; k, v) | scalar | Loss for single key-value pair |
| ∇ℓ | (same as W) | Gradient of loss |

**Implementation Notes:**
- W can parameterize linear map, MLP, or other architecture
- ℓ is attentional bias for single example: ℓ(W; k, v) = L(M(W, k), v)
- η_t can be data-dependent (learnable)

---

### 1.6 FTRL Viewpoint

**Equation 7: Follow-The-Regularized-Leader**

**LaTeX:**
```latex
W_t = \arg\min_W \sum_{i=1}^{t} \langle W - W_{i-1}, \nabla\ell(W_{i-1}; k_i, v_i) \rangle + \frac{1}{2\eta} \|W\|_2^2
```

**Plain Text:**
Memory minimizes sum of linearized losses plus quadratic regularization.

**Variables:**
| Variable | Shape | Description |
|----------|-------|-------------|
| W_t | (varies) | Memory at time t |
| η | scalar | Regularization strength (inverse learning rate) |
| ⟨·,·⟩ | scalar | Inner product |

**Implementation Notes:**
- First term: learn from all past tokens (linearized)
- Second term: regularize memory size
- Equivalent to online GD with specific initialization

**Dependencies:** Sequence of gradients

**Test Criteria:**
- Convex optimization per step
- Equivalent to Eq 5 when W_0 = 0

---

### 1.7 Learning-Retaining Viewpoint (General)

**LaTeX:**
```latex
W_t = \arg\min_{W \in \mathcal{W}} \tilde{\ell}_t(W; k_t, v_t) + \text{Ret}_t(W, W_{t-1})
```

**Plain Text:**
Memory minimizes loss on current token plus retention term that keeps memory close to previous state.

**Components:**
```latex
\text{Ret}_t(W, W_{t-1}) = \frac{1}{\eta_t} D_t(W, W_{t-1}) + \frac{1}{\alpha_t} G_t(W)
```

**Variables:**
| Variable | Shape | Description |
|----------|-------|-------------|
| W_t | (varies) | Memory at time t |
| ℓ̃_t | function | Approximation of loss (e.g., linearization) |
| Ret_t | function | Retention function (forget gate) |
| D_t | function | Local retention (premetric) |
| G_t | function | Global retention (regularization) |
| η_t | scalar | Learning rate (controls new learning) |
| α_t | scalar | Global retention rate |
| W | set | Constraint set for memory |

**Implementation Notes:**
- ℓ̃_t can be exact loss or linearization
- D_t measures deviation from W_{t-1} (retention of past)
- G_t controls memory size/norm
- η_t: larger = more learning, more forgetting
- α_t: controls global memory properties

**Relation to FTRL (Proposition 3.2):**
When h_t(W) = ∑_{i=1}^{t-1} ℓ̃_i(W; k_i, v_i) + (1/η)R(W) is strictly convex,
setting Ret_t(W, W') = D_h(W, W') (Bregman divergence) makes Learning-Retaining equivalent to FTRL.

---

## 2. NOVEL ATTENTIONAL BIASES

### 2.1 ℓ_p Attentional Bias (Moneta)

**Equation 10-11: ℓ_p Loss and Gradient**

**LaTeX:**
```latex
\mathcal{L}(M(W, k_t); v_t) = \|M(k_t) - v_t\|_p^p
```

**Gradient for matrix memory M(W, k) = Wk:**
```latex
W_t = W_t - \eta_t \cdot p \cdot (\text{Sign}(Wk_t - v_t) \odot |Wk_t - v_t|^{p-1}) k_t^\top
```

**Plain Text:**
ℓ_p norm objective measures distance; gradient involves element-wise sign and power p-1.

**Variables:**
| Variable | Shape | Description |
|----------|-------|-------------|
| p | scalar ≥ 1 | Norm parameter |
| ∥·∥_p | scalar | p-norm |
| Sign(·) | element-wise | Sign function (-1, 0, +1) |
| ⊙ | operator | Hadamard (element-wise) product |
| |·| | element-wise | Absolute value |

**Special Case p=1 (Equation 12):**
```latex
W_t = W_t - \eta_t \text{Sign}(W_t k_t - v_t) k_t^\top
```
"Value-less" memory: only stores signs (-1, +1), robust to value magnitudes.

**Smooth Approximations (Remark 5):**
```latex
\text{Sign}(x) \approx \tanh(\alpha x)
|x| = \sqrt{x^2 + \epsilon}, \quad \epsilon = 10^{-6}
```

**Implementation Notes:**
- p=2: Standard MSE (existing models)
- p=1: Robust to outliers, maps to {-1, +1}
- p=3: Paper uses for Moneta
- p=4: Worst performance in experiments
- Requires smooth approximations for backprop

**Dependencies:** Memory architecture

**Test Criteria:**
- Gradient finite and bounded
- p=2 recovers standard delta rule
- Sign approximation smooth

---

### 2.2 Huber Loss Attentional Bias (Yaad)

**Equation 13-16: Huber Loss Variants**

**Huber function:**
```latex
H(a) = \begin{cases}
\frac{1}{2} a^2 & \text{if } |a| \leq \delta \\
\delta (|a| - \frac{1}{2}\delta) & \text{if } |a| > \delta
\end{cases}
```

**Variant 1: Coordinate-wise (Eq 14)**
```latex
\ell(W; k_t, v_t) = \sum_j H(M(W, k_t)_j - v_{t,j})
```

**Gradient (matrix memory):**
```latex
W_t = W_{t-1} - \eta_t [(Wk_t - v_t)k_t^T \odot I(|Wk_t - v_t| \leq \delta_t)1^\top
                        + (\delta_t \text{Sign}(Wk_t - v_t)k_t^\top) \odot I(|Wk_t - v_t| > \delta_t)1^\top]
```

**Variant 2: ℓ2 norm-based (Eq 15)**
```latex
\ell(W; k_t, v_t) = H(\|M(W, k_t) - v_t\|_2)
```

**Gradient:**
```latex
W_t = W_{t-1} - \eta_t \begin{cases}
(M(W_{t-1}, k_t) - v_t) k_t^T & \text{if } \|M(W_{t-1}, k_t) - v_t\|_2 \leq \delta_t \\
\delta_t \frac{M(W_{t-1}, k_t) - v_t}{\|M(W_{t-1}, k_t) - v_t\|_2} k_t^T & \text{otherwise}
\end{cases}
```

**Variant 3: Smooth mixture (Eq 16, used in Yaad)**
```latex
W_t = W_{t-1} - \begin{cases}
\eta_t \nabla\ell_2(W_{t-1}; k_t, v_t) & \text{if } \|M(k_t) - v_t\| \leq \delta_t \\
\eta_t \delta_t \nabla\ell_1(W_{t-1}; k_t, v_t) & \text{otherwise}
\end{cases}
```

**Variables:**
| Variable | Shape | Description |
|----------|-------|-------------|
| δ_t | scalar or (d,) | Huber threshold (can be data-dependent) |
| I(·) | element-wise | Indicator function (0 or 1) |
| H(·) | scalar | Huber function |
| ∇ℓ_1, ∇ℓ_2 | (same as W) | Gradients of ℓ1 and ℓ2 losses |

**Implementation Notes:**
- δ_t controls transition between quadratic and linear
- Small errors: use ℓ2 (smooth, efficient)
- Large errors: use ℓ1 or normalized (robust)
- δ_t can be learnable, data-dependent
- Variant 3 is smoothest, used in Yaad

**Dependencies:** Memory architecture

**Test Criteria:**
- Continuous gradient at threshold
- Reduces to ℓ2 when all errors small
- Robust to outliers (verified on S-NIAH tasks)

---

### 2.3 Robust to Value Shifts (Equation 17)

**LaTeX:**
```latex
\mathcal{L}(M(W, k_t); v_t) = \max_{\|\delta v_t\|_2 \leq \Delta} \frac{1}{2} \|M(W, k_t) - (v_t + \delta v_t)\|_2^2
```

**Optimal perturbation:**
```latex
\delta v_t^* = \Delta \frac{-M(W, k_t) + v_t}{\|M(W, k_t) - v_t\|_2}
```

**Resulting loss:**
```latex
\mathcal{L}(M(W, k_t); v_t) = \frac{1}{2}\|M(W, k_t) - v_t\|_2^2 + \Delta\|M(W, k_t) - v_t\|_2 + \frac{1}{2}\Delta^2
```

**Gradient (linear memory):**
```latex
W_t = W_{t-1} - \eta [(M(W_{t-1}, k_t) - v_t)k_t^\top + \Delta \frac{M(W_{t-1}, k_t) - v_t}{\|M(W_{t-1}, k_t) - v_t\|_2} k_t^\top]
```

**Variables:**
| Variable | Shape | Description |
|----------|-------|-------------|
| Δ | scalar | Robustness radius |
| δv_t | (d,) | Perturbation to value |

**Implementation Notes:**
- Δ controls trade-off: fit nominal data vs. robustness
- Can make Δ learnable parameter
- Combines ℓ2 loss with ℓ2 norm of error (similar to Huber)

**Dependencies:** Memory architecture

---

## 3. NOVEL RETENTION GATES

### 3.1 f-Divergence Retention (KL Divergence for Memora)

**Equation 18: General f-divergence update**

**LaTeX:**
```latex
W_t = W_{t-1} \odot g(-\zeta_t - \eta_t \nabla\ell(W_{t-1}; k_t, v_t))
```

where g is inverse of f', and ζ_t chosen such that ∥W_t∥_1 = c.

**Constraint set:**
```latex
\mathcal{W} = \{W \mid \|W\|_1 = c, W_{jl} \geq 0, \forall j,l\}
```

**KL Divergence Specialization (Equation 19-21):**

**Retention function:**
```latex
\text{Ret}_t(W, W_{t-1}) = \frac{1}{\eta_t} \sum_{jl} W_{jl} \log\frac{W_{jl}}{(W_t)_{jl}} + \frac{1}{\alpha_t} \sum_{jl} W_{jl} \log(W_{jl})
```

**Update rule (Equation 21):**
```latex
W_t \leftarrow c \cdot \text{Softmax}((1-\lambda_t)\log(W_{t-1}) - \eta'_t \nabla\ell(W_{t-1}; k_t, v_t))
```

where:
```latex
\lambda_t = \frac{1/\alpha_t}{1/\alpha_t + 1/\eta_t}, \quad \eta'_t = \frac{1}{1/\alpha_t + 1/\eta_t}
```

**Variables:**
| Variable | Shape | Description |
|----------|-------|-------------|
| f(·) | ℝ+ → ℝ | Strictly convex function, f(1)=0 |
| g(·) | function | Inverse of f' |
| ζ_t | scalar | Lagrange multiplier for constraint |
| c | scalar | Total "probability mass" (scaling) |
| λ_t | scalar ∈ (0,1) | Retention coefficient |
| η'_t | scalar > 0 | Effective learning rate |

**Implementation Notes:**
- Softmax ensures W_t stays in probability simplex
- log-Softmax structure prevents numerical overflow
- λ_t interpolates between previous state and gradient update
- For f(τ) = τ log(τ): KL divergence
- Normalizes per-slice for neural networks

**Dependencies:** Non-negative weights

**Test Criteria:**
- W_t ≥ 0 element-wise
- ∥W_t∥_1 = c (probability simplex)
- Stable for large sequences

---

### 3.2 Elastic Net Retention (Equation 22-23)

**Learning-Retaining formulation (Eq 22):**

**Global retention:**
```latex
G_t(W) = \frac{1}{2\beta}\|W\|_2^2 + \frac{1}{\alpha}\|W\|_1
```

**Local retention:**
```latex
D_t(W, W_{t-1}) = \frac{1}{2}\|W - W_{t-1}\|_2^2
```

**Update with soft-thresholding:**
```latex
W_t = S_\gamma(\lambda W_{t-1} - \zeta \nabla\ell(W_{t-1}; k_t, v_t))
```

where:
```latex
\gamma = \frac{\eta\beta}{\alpha(\eta+\beta)}, \quad \lambda = \frac{\beta}{\beta+\eta}, \quad \zeta = \eta\lambda
```

**Soft-thresholding operator:**
```latex
S_\gamma(z) = \text{sign}(z) \max\{0, |z| - \gamma\}
```

**Smooth approximation:**
```latex
S_\gamma(z) \approx \frac{|z| \cdot \arctan(z/\gamma)}{\pi/2}
```

**FTRL formulation (Eq 23):**
```latex
A_t = A_{t-1} - \eta\nabla\ell(W_{t-1}; k_t, v_t)
W_t = S_{\eta/\alpha}(A_t)
```

**Variables:**
| Variable | Shape | Description |
|----------|-------|-------------|
| α, β | scalars | Regularization strengths (ℓ1, ℓ2) |
| γ | scalar | Soft-threshold amount |
| λ | scalar ∈ (0,1) | Soft forgetting |
| ζ | scalar | Effective learning rate |
| A_t | (same as W) | Accumulated gradients (FTRL) |

**Implementation Notes:**
- Soft forgetting: multiply by λ < 1
- Hard forgetting: S_γ sets small values to 0
- FTRL version cleaner for implementation
- Use smooth approximation for differentiability

**Dependencies:** None

**Test Criteria:**
- Sparsity increases with α
- Bounded norm with β
- S_γ(0) = 0

---

### 3.3 ℓ_q Memory Stability (Equation in Section 5.2, Variant 4)

**FTRL regularization:**
```latex
\frac{1}{\eta_t}R(W) = \frac{1}{2\eta(q-1)}\|W\|_q^2
```

**Update rule:**
```latex
A_t = A_{t-1} - \eta\nabla\ell(W_{i-1}; k_t, v_t)
W_t = \frac{A_t}{\|A_t\|_p^{p-2}}
```

where p = q/(q-1) (conjugate), 1 < q ≤ 2.

**Variables:**
| Variable | Shape | Description |
|----------|-------|-------------|
| q | scalar ∈ (1,2] | Regularization norm |
| p | scalar ≥ 2 | Conjugate exponent |
| A_t | (same as W) | Accumulated gradients |

**Implementation Notes:**
- Normalization by ℓ_p norm to power p-2
- q=2: reduces to standard ℓ2 regularization
- Different q values change memory dynamics

**Dependencies:** Accumulated gradients

**Test Criteria:**
- Normalization prevents explosion
- q=2 recovers standard GD

---

### 3.4 Bregman Divergence Retention (Equation in Section 5.2, Variant 5)

**Retention (premetric):**
```latex
D_t(W, W') = F(W) - F(W') - \langle\nabla F(W'), W - W'\rangle
```

where F(W) = ∑_{jl} f(W_{jl}) for strictly convex f.

**Update:**
```latex
W_t = g(-\eta\nabla\ell(W_{t-1}; k_t, v_t) + F'(W_{t-1}))
```

where g is inverse of F'.

**Sigmoid example:**
Choose f'(τ) = log(τ/(1-τ)) (logit function), then g(τ) = σ(τ) (sigmoid).

**Update:**
```latex
W_t = \sigma(\log(W_t/(1-W_t)) - \eta\nabla\ell(W_{t-1}; k_t, v_t))
```

**Variables:**
| Variable | Shape | Description |
|----------|-------|-------------|
| f(·) | ℝ+ → ℝ | Strictly convex function |
| F(W) | scalar | Sum of f applied element-wise |
| D_t | function | Bregman divergence |
| g(·) | function | Inverse of F' |
| σ(·) | sigmoid | 1/(1+exp(-·)) |

**Implementation Notes:**
- f(τ)=τ²/2: reduces to standard GD (Euclidean)
- Sigmoid: keeps W_t ∈ (0,1) element-wise
- Element-wise operations, division 𝑊_t/(1-𝑊_t)
- Adds non-linearity to memory dynamics

**Dependencies:** Choice of f

**Test Criteria:**
- W_t ∈ (0,1) for sigmoid
- Gradient finite
- f convex ensures Bregman divergence ≥ 0

---

## 4. MIRAS VARIANTS (MONETA, YAAD, MEMORA)

### 4.1 MONETA (p,q-Moneta)

**Architecture:**
- Memory: 2-layer MLP with expansion factor 4, GELU activation
- Formula: M(x) = x + LN(W₁σ(W₂x))

**Attentional Bias:** ℓ_p loss (Equation 10)
```latex
\mathcal{L} = \|M(k_t) - v_t\|_p^p
```

**Retention Gate:** Hybrid ℓ_q + ℓ_2
```latex
\text{Global: } G_t(W) = \frac{1}{2(q-1)}\|W\|_q^2 + \frac{1}{\beta}\|W\|_2^2
```

**Memory Algorithm:** Gradient Descent

**Update Rule (Equation 24):**
```latex
A_t = \alpha_t A_{t-1} - \eta_t \nabla\ell_p(W_{i-1}; k_t, v_t)
W_t = \frac{A_t}{\|A_t\|_q^{q-2}}
```

**Gradient (Equation 25):**
```latex
\nabla\ell(W_{t-1}; k_t, v_t) = p\eta_t (\text{Sign}(Wk_t - v_t) \odot |Wk_t - v_t|^{p-1}) k_t^\top
```

**Hyperparameters (from paper):**
- p = 3 (ℓ_p norm)
- q = 4 (ℓ_q retention)
- MLP expansion: 4
- Activation: GELU
- LayerNorm: applied after MLP

**Variables:**
| Variable | Shape | Description |
|----------|-------|-------------|
| W₁ | (d, 4d) | MLP first layer (expansion) |
| W₂ | (4d, d) | MLP second layer (projection) |
| A_t | (same as W) | Accumulated gradients |
| α_t | scalar or (d,) | Retention coefficient (data-dependent) |
| η_t | scalar or (d,) | Learning rate (data-dependent) |
| σ | function | GELU activation |
| LN | function | Layer normalization |

**Implementation Notes:**
- Residual connection: x + LN(...)
- Smooth approximations for Sign and |·| (see Eq 11 notes)
- α_t and η_t are channel-wise, use low-rank projections (rank 32 or 64)
- Normalization by ℓ_q^{q-2} norm applied at end of each chunk (training)

**Dependencies:** None

**Test Criteria:**
- p=3 performs best (ablation study)
- Memory stable with α < 1
- Better on noisy synthetic tasks (S-NIAH-PK)

---

### 4.2 YAAD

**Architecture:**
- Memory: 2-layer MLP (same as Moneta)
- Formula: M(x) = x + LN(W₁σ(W₂x))

**Attentional Bias:** Huber loss (Equation 16, Variant 3)
```latex
\ell(W; k_t, v_t) = \begin{cases}
\text{ℓ2 loss} & \text{if } \|M(k_t) - v_t\| \leq \delta_t \\
\text{ℓ1 loss (scaled)} & \text{otherwise}
\end{cases}
```

**Retention Gate:** ℓ_2 local + ℓ_2 global (Titans-style)
```latex
\text{Ret}_t(W, W_{t-1}) = \frac{1}{2\theta_t}\|W - W_{t-1}\|_F^2 + \frac{1}{\beta_t}\|W\|_2^2
```

Equivalent to "forget gate" mechanism from Titans.

**Memory Algorithm:** Gradient Descent

**Update Rule (Equation 26):**
```latex
W_t = \alpha_t W_{t-1} - \begin{cases}
\eta_t \nabla\ell_2(W_{t-1}; k_t, v_t) & \text{if } \|M(k_t) - v_t\| \leq \delta_t \\
\eta_t \delta_t \nabla\ell_1(W_{t-1}; k_t, v_t) & \text{otherwise}
\end{cases}
```

where:
```latex
\alpha_t = \frac{\beta_t}{\beta_t + \theta_t}
```

**Hyperparameters:**
- δ_t: data-dependent threshold (learnable)
- α_t ∈ [0,1]^d: channel-wise retention
- η_t: channel-wise learning rate
- β_t: decoupled from η_t (independent parameter)

**Variables:**
| Variable | Shape | Description |
|----------|-------|-------------|
| δ_t | scalar or (d,) | Huber threshold (data-dependent) |
| α_t | (d,) | Retention gate (channel-wise) |
| η_t | (d,) | Learning rate (channel-wise) |
| β_t | (d,) | Global retention parameter |
| θ_t | (d,) | Local retention parameter |

**Implementation Notes:**
- Adaptive: switches between ℓ_2 and ℓ_1 based on error magnitude
- More robust to outliers than pure ℓ_2
- "Coping mechanism": protects memory from extreme events
- Decoupling η and α improves expressivity

**Dependencies:** Error magnitude ∥M(k_t) - v_t∥

**Test Criteria:**
- Smooth transition at threshold
- Robust to outliers (verified experimentally)
- Comparable to Moneta and Memora on most tasks

---

### 4.3 MEMORA

**Architecture:**
- Memory: 2-layer MLP (same as Moneta, Yaad)
- Formula: M(x) = x + LN(W₁σ(W₂x))

**Attentional Bias:** ℓ_2 regression
```latex
\mathcal{L} = \|M(k_t) - v_t\|_2^2
```

**Retention Gate:** KL divergence (Equation 21)
```latex
\text{Ret}_t(W, W_{t-1}) = \frac{1}{\eta_t}\sum_{jl} W_{jl}\log\frac{W_{jl}}{(W_{t-1})_{jl}} + \frac{1}{\alpha_t}\sum_{jl} W_{jl}\log(W_{jl})
```

**Memory Algorithm:** Gradient Descent

**Update Rule (Equation 27):**
```latex
W_t = \text{Softmax}(\alpha_t \log(W_{t-1}) - \eta_t \nabla\ell_2(W_{t-1}; k_t, v_t))
```

**Hyperparameters:**
- α_t: retention coefficient (data-dependent)
- η_t: learning rate (data-dependent)
- Softmax scaling: c (constant, usually 1)

**Variables:**
| Variable | Shape | Description |
|----------|-------|-------------|
| α_t | scalar or (d,) | Retention coefficient |
| η_t | scalar or (d,) | Learning rate |
| log(W) | element-wise | Natural logarithm |
| Softmax | per-slice | Applied per slice for neural networks |

**Implementation Notes:**
- Softmax ensures non-negativity and normalization
- Hard + soft forgetting: combines KL divergence (soft) with thresholding behavior
- log-domain updates prevent numerical issues
- Apply Softmax per-slice (e.g., per column or specific dimension)
- Lag tokens: apply full nonlinearity (Softmax + log) at chunk boundaries (index kb+1)

**Parallelization (Section 5.4):**
Inside chunk (tokens kb+2 to (k+1)b):
- Use linear approximation (skip log and Softmax)
- Calculate: W_t ≈ W_t-1 - η_t∇ℓ_2(W_1; k_t, v_t) where W_1 is from lag token

At chunk boundary (token kb+1):
```latex
M_1 = \text{Softmax}(\alpha_1 \log(M_0) - \eta_1 \nabla\ell_2(M_0; k_1, v_1))
```

**Dependencies:** Non-negative weights

**Test Criteria:**
- W_t ≥ 0 (enforced by Softmax)
- Sum constraints maintained
- Stable for long sequences

---

## 5. PARALLELIZATION (CRITICAL FOR TRAINING)

### 5.1 Chunk-based Parallelization (Section 5.4)

**Core Idea:**
- Divide sequence into chunks of size b (typically 16 or 64)
- Calculate gradients for chunk w.r.t. last state of previous chunk
- Use linearization + matrix operations within chunk

**Equation 28: Expanded Recurrence (Moneta example, q=2)**

**LaTeX:**
```latex
M_t = \alpha_t M_{t-1} - \eta_t \nabla\ell(M_{t-1}; k_t, v_t)
    = \beta_t M_0 - \sum_{i=1}^{t} \eta_i \frac{\beta_t}{\beta_i} \nabla\ell(M_{t'}; k_i, v_i)
```

where t' = t - mod(t, b) (last token of previous chunk), β_i = ∏_{j=1}^i α_j.

**For linear memory M(W, k) = Wk with ℓ_p loss:**

**Equation 29: Batch Gradient Computation**
```latex
\sum_{i=1}^{b} \eta_i \frac{\beta_b}{\beta_i} \nabla\ell(W_0; k_i, v_i) = p \mathbf{E}_b \odot \mathbf{B}_b \odot \text{Sign}(W_0K - V) \odot (|W_0K - V|^{p-1}) K^\top
```

where:
- K = [k₁, k₂, ..., k_b] ∈ ℝ^{d×b} (stacked keys)
- V = [v₁, v₂, ..., v_b] ∈ ℝ^{d×b} (stacked values)
- E_b = [η₁, η₂, ..., η_b] ∈ ℝ^{1×b}
- B_b = [β_b/β₁, β_b/β₂, ..., β_b/β_b] ∈ ℝ^{1×b}

**Variables:**
| Variable | Shape | Description |
|----------|-------|-------------|
| b | scalar | Chunk size (16 or 64) |
| t' | scalar | Last token of previous chunk |
| M_0 | (varies) | Memory at start of chunk |
| K, V | (d, b) | Batched keys/values |
| E_b, B_b | (1, b) | Learning rate and decay masks |
| β_i | scalar | Cumulative product of α |

**Smooth Approximations:**
```latex
\text{Sign}(x) \approx \tanh(\alpha x), \quad \alpha \text{ large (e.g., 10)}
|x| = \sqrt{x^2 + \epsilon}, \quad \epsilon = 10^{-6}
```

**Implementation Notes:**
- All operations in Eq 29 are matrix multiplications or element-wise
- Can run on GPU/TPU efficiently
- Non-linearity (normalization for q≠2) applied only at chunk boundaries
- For 2-layer MLP: apply same process to each layer's parameters

**Test Criteria:**
- Chunk processing much faster than sequential
- Gradient approximation error small (controlled by chunk size)
- Memory overflow prevented by periodic normalization

---

### 5.2 Parallelization for Yaad

**Process:**
- Calculate gradients for both ℓ₁ and ℓ₂ losses
- Use masking based on error magnitude

**Pseudo-algorithm:**
```python
# Inside chunk
grad_l2 = compute_l2_gradient(W0, K, V)  # (W0*K - V) * K^T
grad_l1 = compute_l1_gradient(W0, K, V)  # sign(W0*K - V) * K^T

errors = ||W0*K - V||  # shape (b,)
mask = (errors <= delta_t)  # boolean mask

# Combine gradients
grad = mask * grad_l2 + (~mask) * (delta_t * grad_l1)

# Apply with learning rates
grad_weighted = E_b ⊙ B_b ⊙ grad
W_new = alpha_t * W0 - grad_weighted
```

**Variables:**
| Variable | Shape | Description |
|----------|-------|-------------|
| mask | (b,) | Boolean: True if error ≤ δ_t |
| grad_l1 | (d, d) or MLP | Gradient for ℓ₁ loss |
| grad_l2 | (d, d) or MLP | Gradient for ℓ₂ loss |

**Implementation Notes:**
- Two gradient computations per chunk
- Masking is element-wise per token
- Can still use batched matrix operations

---

### 5.3 Parallelization for Memora (Equations 30-32)

**Lag Token (Equation 30):**
At first token of chunk (i = kb + 1):
```latex
M_1 = \text{Softmax}(\alpha_1 \log(M_0) - \eta_1 \nabla\ell_2(M_0; k_1, v_1))
```

**Linear Approximation Inside Chunk:**
For i = kb+2 to (k+1)b, use M_1 instead of M_{i-1}:

**Equation 31-32: Batch Computation**
```latex
\nabla\ell(W_1; k_t, v_t) = (W_1 k_t - v_t) k_t^\top

\sum_{i=1}^{b} \eta_i \frac{\beta_b}{\beta_i} \nabla\ell(W_1; k_i, v_i) = \mathbf{E}_b \odot \mathbf{B}_b \odot (W_1 K - V) K^\top
```

**Implementation:**
```python
# Step 1: Lag token (exact update)
M_1 = softmax(alpha_1 * log(M_0) - eta_1 * grad_l2(M_0, k_1, v_1))

# Step 2: Rest of chunk (linear approximation)
errors = M_1 @ K - V  # shape (d, b)
grad_batch = (E_b ⊙ B_b ⊙ errors) @ K.T  # shape (d, d)
M_next = beta_b * M_1 - grad_batch

# Step 3: Apply Softmax at next boundary
M_new = softmax(log(M_next))  # or keep for next lag
```

**Variables:**
| Variable | Shape | Description |
|----------|-------|-------------|
| M_0 | (varies) | Memory from previous chunk |
| M_1 | (varies) | Memory after lag token |
| log(M_0) | element-wise | Natural logarithm |

**Implementation Notes:**
- Log and Softmax only at chunk boundaries
- Prevents compounding of non-linearity
- Linear approximation sufficient inside chunk
- Softmax can be per-slice for neural network parameters

---

## 6. ARCHITECTURE DETAILS

### 6.1 MIRAS Layer Block

**Components (from Figure 2 and Section 5.4):**

1. **Input Projections:**
   ```
   q = x @ W_q
   k = x @ W_k
   v = x @ W_v
   ```

2. **1D Depthwise-Separable Convolution:**
   ```
   q = conv1d(q, kernel_size=4)
   k = conv1d(k, kernel_size=4)
   v = conv1d(v, kernel_size=4)
   ```

3. **ℓ_2 Normalization:**
   ```
   q = q / ||q||_2
   k = k / ||k||_2
   ```

4. **Memory Module:**
   ```
   # One of: Moneta, Yaad, Memora
   output = MemoryModule(q, k, v)
   ```

5. **Output Processing:**
   ```
   output = LayerNorm(output)
   output = output * sigmoid(output @ W_gate)
   ```

**Channel-wise Parameters (Section 5.4):**
- η_t, δ_t, α_t ∈ ℝ^d (channel-wise)
- Use low-rank projection to reduce parameters:
  ```
  param_t = Linear_1(x)  # x → ℝ^k (k=32 or 64)
  param_t = Linear_2(param_t)  # ℝ^k → ℝ^d
  ```

**Full Architecture:**
```
Input (N, d)
  ↓
[Q, K, V Projections] (d → d each)
  ↓
[Conv1D (kernel=4)] for each
  ↓
[ℓ2 Norm] for Q, K
  ↓
[MIRAS Memory Module]
  ↓
[LayerNorm + Gating]
  ↓
Output (N, d)
```

**Variables:**
| Variable | Shape | Description |
|----------|-------|-------------|
| W_q, W_k, W_v | (d, d) | Projection matrices |
| conv1d | kernel=4 | Depthwise-separable convolution |
| W_gate | (d, d) | Gating linear layer |

**Implementation Notes:**
- Follow modern linear RNN design (GLA, Gated DeltaNet)
- Convolution adds local context
- ℓ2 norm improves training stability
- Gating from Mehta et al. 2023

---

### 6.2 Full Model Architecture

**Macro Architecture (Section 5.4):**
- Based on Llama architecture
- Replace attention with MIRAS layer

**Components:**
```
Embedding
  ↓
[MIRAS Layer
  ↓
 MLP (SwiGLU)
  ↓
 RMSNorm
] × L layers
  ↓
RMSNorm
  ↓
LM Head
```

**MLP:**
- SwiGLU activation (Gated Linear Unit with Swish)
- Formula: MLP(x) = (W_1 x ⊙ swish(W_2 x)) @ W_3

**Positional Encoding:**
- RoPE (Rotary Position Embedding)

**Normalization:**
- RMSNorm (Root Mean Square Layer Normalization)

**Model Sizes (Table 5):**
| Size | Layers | Dim | Heads | Tokens | Peak LR |
|------|--------|-----|-------|--------|---------|
| 170M | 12 | 768 | 16 | 15B | 3e-3 |
| 340M | 24 | 1024 | 16 | 15B | 1.5e-3 |
| 780M | 24 | 1536 | 16 | 30B | 1.25e-3 |
| 1.3B | - | - | - | 100B | - |

**ASCII Diagram:**
```
Input Tokens
     ↓
[Token Embedding]
     ↓
┌────────────────────┐
│  MIRAS Block 1     │
│  ┌──────────────┐  │
│  │ Q/K/V Proj   │  │
│  │ Conv1D       │  │
│  │ Memory(q,k,v)│  │
│  │ Gate         │  │
│  └──────────────┘  │
│  RMSNorm           │
│  MLP (SwiGLU)      │
│  RMSNorm           │
└────────────────────┘
     ↓
   [... L blocks ...]
     ↓
[Final RMSNorm]
     ↓
[LM Head]
     ↓
Output Logits
```

---

### 6.3 Hybrid Models (with Attention)

**Architecture (Figure 2, Samba-style):**
```
[MIRAS Layer
  ↓
 Sliding Window Attention (SWA)
  ↓
 MLP
  ↓
 RMSNorm
] × L layers
```

**Sliding Window Attention:**
- Window size: typically 128 or 256
- Local attention pattern
- Complements recurrent MIRAS memory

**Implementation Notes:**
- Alternate MIRAS and SWA
- Or stack: MIRAS → SWA in each block
- SWA provides local inductive bias
- MIRAS provides long-range memory

---

## 7. HYPERPARAMETERS

### 7.1 Model Configuration

| Parameter | 340M | 760M | 1.3B | Description | Paper Reference |
|-----------|------|------|------|-------------|-----------------|
| Layers | 24 | 24 | - | Number of blocks | Table 5 |
| Hidden dim | 1024 | 1536 | - | Model dimension | Table 5 |
| Heads | 16 | 16 | - | Number of attention heads (for hybrid) | Table 5 |
| MLP expansion | 4 | 4 | 4 | Memory MLP expansion factor | Section 5.3 |
| Conv kernel | 4 | 4 | 4 | 1D conv kernel size | Section 5.4 |
| Vocab size | - | - | - | Tokenizer dependent (not specified) | - |

### 7.2 Training Configuration

| Parameter | Value | Description | Paper Reference |
|-----------|-------|-------------|-----------------|
| Training context | 4096 | Context window during training | Section 6 |
| Batch size | - | Not specified | - |
| Tokens (340M) | 15B | Total training tokens | Table 5, Section 6 |
| Tokens (760M) | 30B | Total training tokens | Table 5 |
| Tokens (1.3B) | 100B | Total training tokens | Section 6 |
| Peak LR (340M) | 1.5e-3 | Maximum learning rate | Table 5 |
| Peak LR (760M) | 1.25e-3 | Maximum learning rate | Table 5 |
| LR schedule | - | Not specified (likely cosine) | - |
| Warmup | - | Not specified | - |
| Weight decay | - | Not specified | - |
| Optimizer | AdamW | Assumed (standard) | - |

### 7.3 Memory-Specific Parameters

**Moneta:**
| Parameter | Value | Description | Paper Reference |
|-----------|-------|-------------|-----------------|
| p (attentional bias) | 3 | ℓ_p norm for loss | Section 5.3, Figure 4 |
| q (retention) | 4 | ℓ_q norm for retention | Section 5.3, Figure 4 |
| α smooth param | ~10 | For tanh approximation of Sign | Remark 5 |
| ε (abs approx) | 1e-6 | For smooth \|x\| approximation | Section 5.4 |

**Yaad:**
| Parameter | Value | Description | Paper Reference |
|-----------|-------|-------------|-----------------|
| δ_t | data-dependent | Huber threshold (learnable) | Section 5.3 |
| α_t | [0,1]^d | Retention gate (channel-wise) | Section 5.3 |
| β_t | learnable | Global retention (decoupled from η) | Section 5.3 |

**Memora:**
| Parameter | Value | Description | Paper Reference |
|-----------|-------|-------------|-----------------|
| α_t | data-dependent | Retention coefficient | Section 5.3 |
| λ_t | (0,1) | Derived from α_t and η_t | Equation 21 |
| c | 1 (assumed) | Softmax scaling constant | Section 5.2 |

**Channel-wise Parametrization:**
| Parameter | Value | Description | Paper Reference |
|-----------|-------|-------------|-----------------|
| Low-rank k | 32 or 64 | Rank for channel-wise parameter projection | Section 5.4 |

### 7.4 Parallelization Parameters

| Parameter | Value | Description | Paper Reference |
|-----------|-------|-------------|-----------------|
| Chunk size (b) | 16 or 64 | Tokens per parallel chunk | Section 5.4 |
| Lag tokens (Memora) | 1 | Tokens with full nonlinearity at boundary | Section 5.4 |

### 7.5 Architectural Choices Summary Table

| Variant | Memory Arch | Attentional Bias | Retention Gate | Memory Algorithm |
|---------|-------------|------------------|----------------|------------------|
| **Moneta** | 2-layer MLP | ℓ_p (p=3) | ℓ_q (q=4) + ℓ_2 | GD |
| **Yaad** | 2-layer MLP | Huber | ℓ_2 local + ℓ_2 global | GD |
| **Memora** | 2-layer MLP | ℓ_2 | KL divergence | GD |

From Table 1 in paper.

---

## 8. BENCHMARKS

### 8.1 Datasets

**Language Modeling:**
- **FineWeb-Edu** (Penedo et al. 2024)
  - Used for LM and common-sense reasoning
  - Training: 15B tokens (small), 30B tokens (medium), 100B tokens (large)
  - Evaluation metrics: Perplexity on validation set

- **C4** (Raffel et al. 2020)
  - Used for scaling pattern experiments
  - Training context: 4096 tokens

**Specific LM Tasks:**
- **WikiText** (Merity et al. 2017): Perplexity
- **LAMBADA** (Paperno et al. 2016): Perplexity and Accuracy

**Common-sense Reasoning (zero-shot):**
- **PIQA** (Bisk et al. 2020): Physical commonsense, Accuracy
- **HellaSwag** (Zellers et al. 2019): Sentence completion, Accuracy (acc_n)
- **WinoGrande** (Sakaguchi et al. 2021): Pronoun resolution, Accuracy
- **ARC-easy, ARC-challenge** (Clark et al. 2018): Science questions, Accuracy (acc, acc_n)
- **SIQA** (Social IQA): Social reasoning, Accuracy
- **BoolQ** (Clark et al. 2019): Yes/No questions, Accuracy

**Long Context:**
- **RULER S-NIAH** (Hsieh et al. 2024): Needle-in-haystack variants
  - S-NIAH-PK: Passkey retrieval (synthetic noise)
  - S-NIAH-N: Number retrieval
  - S-NIAH-W: Word retrieval
  - Context lengths: 1K, 2K, 4K, 8K tokens
  - Metric: Accuracy (%)

### 8.2 Reported Results

**340M Models (Table 2):**

Best pure recurrent (Moneta):
- WikiText perplexity: 26.19
- LAMBADA perplexity: 29.31
- PIQA: 63.99%
- HellaSwag: 39.23%
- ARC-c: 27.15%

**760M Models (Table 2):**

Best pure recurrent (Yaad):
- WikiText perplexity: 20.99
- LAMBADA perplexity: 21.57
- PIQA: 69.14%
- HellaSwag: 50.02%
- ARC-c: 36.27%

Best hybrid (Moneta-H):
- WikiText perplexity: 18.72
- LAMBADA perplexity: 20.13
- PIQA: 70.84%

**1.3B Models (Table 2):**

Best pure recurrent (Yaad):
- WikiText perplexity: 15.18
- LAMBADA perplexity: 11.89
- PIQA: 72.81%
- HellaSwag: 56.46%
- ARC-c: 40.05%

**Needle-in-Haystack (Table 3, 760M models):**

Moneta (best on synthetic):
- S-NIAH-PK: 99.4% (2K), 98.8% (4K), 98.8% (8K)
- Average: 93.5%

Yaad (best overall):
- Average: 92.9%

**Scaling (Figure 3):**
- FLOPs vs perplexity: All variants outperform baselines
- Context scaling: Superior to Transformers, Mamba2, GSA when training context 2K→32K

### 8.3 Computational Requirements

**Training:**
- 340M model: ~15B tokens, context 4K
- 760M model: ~30B tokens, context 4K
- 1.3B model: ~100B tokens, context 4K
- Hardware: Not specified (likely TPU/GPU cluster)

**Inference:**
- Linear time complexity: O(Nd) vs O(N²d) for Transformers
- Fixed memory footprint (no KV cache growth)
- Parallel training: O(N/b) sequential steps with chunk size b

**Comparison to Baselines (from context):**
- Competitive or better than Transformer++, RetNet, Mamba2, Gated DeltaNet
- Outperforms Titans (predecessor with momentum-based updates)
- Pure recurrent models competitive with hybrid (attention+recurrent)

---

## 9. DEPENDENCIES

### 9.1 Cross-Paper Dependencies

**TITANS (Behrouz et al. 2024c) - CRITICAL DEPENDENCY:**

MIRAS generalizes and extends TITANS:

1. **Memory Architecture:**
   - TITANS: k-layer MLP (deep memory)
   - MIRAS: General (vector, matrix, MLP, etc.)
   - Moneta/Yaad/Memora: 2-layer MLP (simpler than TITANS)

2. **Attentional Bias:**
   - TITANS: ℓ_2 MSE loss only
   - MIRAS: General L (ℓ_p, Huber, robust, etc.)
   - Extends beyond MSE to handle different data distributions

3. **Retention Gate (Forget Gate):**
   - TITANS: ℓ_2 local + ℓ_2 global (specific form)
   - MIRAS: General Ret (ℓ_q, KL, Elastic Net, Bregman, etc.)
   - Reinterprets TITANS forget gate as ℓ_2 retention regularization

4. **Memory Algorithm:**
   - TITANS: GD with momentum
   - MIRAS: General optimizer (GD, momentum, Newton, non-parametric)
   - Moneta/Yaad/Memora: GD (simpler, no momentum)

5. **Specific Relation (from Table 1):**
   ```
   Titans-LMM:
     - Memory: k-layer MLP
     - Attentional Bias: ℓ_2
     - Retention: ℓ_2 local + ℓ_2 global
     - Algorithm: GD + Momentum
     - Update: M_t = α_t M_{t-1} - S_t
              where S_t = η_t S_{t-1} - θ_t ∇L(M_{t-1}; k_t, v_t)
   ```

6. **Cold Start Strategy (Footnote 2):**
   - TITANS uses "cold start" for full memory erase
   - Different from Mamba2/Gated DeltaNet which treat next token as first-ever
   - TITANS uses previous state to measure surprise before erasing

**Implementation Order:**
1. Implement basic memory module (matrix-valued, linear)
2. Implement ℓ_2 attentional bias and retention (baseline)
3. Test against TITANS-LMM (should be special case)
4. Add novel attentional biases (ℓ_p, Huber)
5. Add novel retention gates (KL, Elastic Net)
6. Implement parallelization

---

### 9.2 Other Dependencies

**TTT (Test-Time Training) (Sun et al. 2024):**
- TTT-Linear, TTT-MLP: Use ℓ_2 loss, no retention
- MIRAS unifies TTT as non-parametric/parametric solution to MSE
- TTT-Linear: M_t = M_{t-1} - η∇L(M_{t-1}, x_t)
- TTT-MLP: Same but with MLP memory

**Linear Attention (Katharopoulos et al. 2020):**
- Hebbian rule: M_t = M_{t-1} + v_t k_t^⊤
- MIRAS special case: α=1, dot-product bias, no retention

**RetNet (Sun et al. 2023):**
- M_t = αM_{t-1} + v_t k_t^⊤
- MIRAS special case: scalar α, dot-product bias, ℓ_2 retention

**Mamba2 (Dao et al. 2024):**
- Data-dependent α_t: M_t = α_t M_{t-1} + v_t k_t^⊤
- MIRAS special case: dot-product bias, ℓ_2 retention

**GLA (Yang et al. 2024b):**
- Diagonal α_t: M_t = Diag(α_t)M_{t-1} + v_t k_t^⊤
- MIRAS special case: dot-product bias, ℓ_2 retention

**DeltaNet (Schlag et al. 2021):**
- M_t = (I - βk_t k_t^⊤)M_{t-1} + βv_t k_t^⊤
- MIRAS special case: ℓ_2 bias, no retention (α=1)

**Gated DeltaNet (Yang et al. 2024a):**
- M_t = α_t(I - βk_t k_t^⊤)M_{t-1} + βv_t k_t^⊤
- MIRAS special case: ℓ_2 bias, ℓ_2 retention

**RWKV-7 (Peng et al. 2025b):**
- Channel-wise gated delta rule
- MIRAS special case: ℓ_2 bias, ℓ_2 retention, channel-wise α

**HGRN2 (Qin et al. 2024):**
- Uses ℓ_1 loss (outside standard MIRAS instantiations in Table 1)
- Shows MIRAS can express non-ℓ_2 objectives

**Transformer (Vaswani et al. 2017):**
- Non-parametric solution to ℓ_2 MSE with Nadaraya-Watson estimator
- No retention (keeps all key-value pairs)

---

## 10. IMPLEMENTATION CHECKLIST

### 10.1 Core Equations
- [ ] Equation 1-2: Transformer attention (baseline reference)
- [ ] Equation 3: Hebbian memory update (baseline)
- [ ] Equation 4: Associative memory definition (core framework)
- [ ] Equation 5: Gradient descent update (general)
- [ ] Equation 7: FTRL viewpoint (alternative formulation)
- [ ] Learning-Retaining viewpoint (core implementation approach)
- [ ] Equation 9: Delta rule memory update (baseline)

### 10.2 Moneta
- [ ] Equation 10: ℓ_p attentional bias definition
- [ ] Equation 11: ℓ_p gradient (general p)
- [ ] Equation 12: ℓ_1 special case (value-less memory)
- [ ] Remark 5: Smooth approximations (Sign, |·|)
- [ ] Equation 24: Moneta update rule (main)
- [ ] Equation 25: Moneta gradient (explicit)
- [ ] Equation 28-29: Moneta parallelization
- [ ] ℓ_q retention normalization (W_t = A_t / ∥A_t∥_q^{q-2})
- [ ] Test p=3, q=4 configuration

### 10.3 Yaad
- [ ] Equation 13: Huber function H(a)
- [ ] Equation 14: Coordinate-wise Huber (variant 1)
- [ ] Equation 15: ℓ_2 norm-based Huber (variant 2)
- [ ] Equation 16: Smooth mixture Huber (variant 3, main)
- [ ] Equation 26: Yaad update rule (main)
- [ ] ℓ_2 retention (local + global)
- [ ] Data-dependent δ_t threshold
- [ ] Decoupled η_t and α_t (via β_t)
- [ ] Parallelization with dual gradients (ℓ_1 and ℓ_2)

### 10.4 Memora
- [ ] Equation 18: f-divergence update (general)
- [ ] Equation 19-20: KL divergence retention formulation
- [ ] Equation 21: KL update with Softmax (main)
- [ ] Equation 27: Memora update rule (simplified form)
- [ ] Equation 30: Lag token (chunk boundary)
- [ ] Equation 31-32: Memora parallelization (linear approx)
- [ ] Softmax per-slice for neural networks
- [ ] Log-domain stability

### 10.5 Alternative Retention Gates (Optional Extensions)
- [ ] Equation 22: Elastic Net retention (Learning-Retaining)
- [ ] Equation 23: Elastic Net retention (FTRL)
- [ ] Soft-thresholding operator S_γ
- [ ] ℓ_q memory stability (Section 5.2, Variant 4)
- [ ] Bregman divergence retention (Section 5.2, Variant 5)
- [ ] Robust to value shifts (Equation 17)

### 10.6 Architecture Components
- [ ] 2-layer MLP memory (M(x) = x + LN(W₁σ(W₂x)))
- [ ] GELU activation
- [ ] Layer normalization
- [ ] Residual connections
- [ ] Q/K/V projections
- [ ] 1D depthwise-separable convolution (kernel=4)
- [ ] ℓ_2 normalization for q, k
- [ ] Output gating (sigmoid)
- [ ] RMSNorm (full model)
- [ ] RoPE positional encoding
- [ ] SwiGLU MLP (full model)

### 10.7 Channel-wise Parametrization
- [ ] Low-rank projection for η_t (d → k → d, k=32 or 64)
- [ ] Low-rank projection for α_t
- [ ] Low-rank projection for δ_t (Yaad)
- [ ] Data-dependent parameter computation

### 10.8 Parallelization
- [ ] Chunk-based processing (chunk size b=16 or 64)
- [ ] Batch gradient computation (Equation 29)
- [ ] E_b and B_b construction (learning rate and decay masks)
- [ ] Matrix operations (K, V stacking)
- [ ] Smooth approximations (tanh, sqrt)
- [ ] Chunk boundary handling (normalization for Moneta)
- [ ] Lag token processing (Memora)
- [ ] Dual gradient computation (Yaad)

### 10.9 Training Infrastructure
- [ ] AdamW optimizer
- [ ] Learning rate schedule (cosine decay)
- [ ] Gradient clipping (if needed)
- [ ] Mixed precision training (FP16/BF16)
- [ ] Data loading (FineWeb-Edu, C4)
- [ ] Tokenization (Llama tokenizer or similar)
- [ ] Evaluation harness (lm-eval or similar)

### 10.10 Testing & Validation
- [ ] Unit tests: each equation
- [ ] Integration tests: full forward/backward pass
- [ ] Numerical gradient checks
- [ ] Causality verification (M_t only depends on inputs ≤ t)
- [ ] Memory stability tests (no overflow/underflow)
- [ ] Chunk parallelization equivalence (vs. sequential)
- [ ] Reproduction of baseline results (RetNet, DeltaNet, etc.)
- [ ] Ablation studies (p, q, δ values)

---

## 11. NOTES FOR IMPLEMENTER

### 11.1 Critical Gotchas

1. **Smooth Approximations:**
   - Sign(x) and |x| are non-differentiable
   - MUST use smooth approximations for backprop
   - Sign(x) ≈ tanh(αx), α large (10)
   - |x| = sqrt(x² + ε), ε = 1e-6
   - Test gradients numerically

2. **Normalization Timing (Moneta):**
   - ℓ_q normalization only at chunk boundaries during training
   - Inside chunk: linear accumulation
   - Prevents compounding non-linearity
   - At inference: can normalize every step

3. **Log/Softmax Stability (Memora):**
   - Never compute exp(large number)
   - Use log-sum-exp trick in Softmax
   - Softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
   - Log-domain updates: log(W_t) = ...
   - Clamp log(W) to avoid -∞

4. **Chunk Boundary Handling:**
   - Memory state from previous chunk is frozen (M_0)
   - Gradients computed w.r.t. M_0, not M_{t-1}
   - Approximation error controlled by chunk size
   - Smaller chunk = better approximation, slower training

5. **Channel-wise Parameters:**
   - η_t, α_t, δ_t all shape (d,) not scalar
   - Use broadcasting carefully
   - Low-rank projection reduces parameters
   - Make sure gradients flow through projection

6. **Retention Gate vs. Learning Rate:**
   - In Yaad/Memora: η_t and α_t are DECOUPLED
   - Don't assume α_t = f(η_t)
   - β_t is independent parameter
   - More expressive but more parameters

7. **Huber Threshold (Yaad):**
   - δ_t can be data-dependent (learnable)
   - Shape: scalar or (d,)
   - Needs careful initialization (e.g., 1.0)
   - Too small: behaves like ℓ_2
   - Too large: behaves like ℓ_1

8. **MLP Memory Architecture:**
   - Expansion factor 4: (d → 4d → d)
   - GELU activation between layers
   - Layer norm AFTER MLP
   - Residual connection: x + LN(MLP(x))
   - Gradient flow through residual critical

9. **KV Cache Equivalent:**
   - Unlike Transformers, no growing KV cache
   - Memory size fixed: O(d²) for matrix, O(d × 4d) for 2-layer MLP
   - Enables true O(1) memory for inference
   - BUT: can't perfectly recall like Transformer

10. **Parallelization Trade-off:**
    - Chunk size b: larger = faster, less accurate
    - b=64 good for training, b=1 for inference
    - Parallelization only for training
    - Inference is sequential (but fast)

### 11.2 Implementation Clarifications

1. **What is "Memory"?**
   - For linear: M(W, k) = Wk, W is (d, d) or (d, n)
   - For MLP: M(W, k) = W₁σ(W₂k), W₁ is (d, 4d), W₂ is (4d, d)
   - For full: M(W, k) = k + LN(W₁σ(W₂k)) with residual
   - Memory "state" = parameters W (not hidden state like RNN)

2. **Attentional Bias vs. Outer Loss:**
   - Attentional bias L: memory's internal objective
   - Outer loss: language modeling cross-entropy
   - Meta-learning: inner loop (L), outer loop (LM loss)
   - Gradients from outer loss flow through memory update

3. **What is "Test-Time" Learning?**
   - During forward pass (inference or training), memory updates
   - Memory learns to map (k_t, v_t) as tokens arrive
   - Parameters W updated based on current input
   - Different from weight updates via backprop (those happen in outer loop)

4. **Retention ≠ Forget:**
   - Paper: "Retention gate" not "forget gate"
   - No actual erasing, just weighted retention
   - α_t close to 1: retain strongly
   - α_t close to 0: don't retain (effective forgetting)
   - Matches human memory: retrieval failure, not erasure

5. **FTRL vs. Learning-Retaining:**
   - Mathematically equivalent under conditions (Prop 3.2)
   - Learning-Retaining more general (doesn't need strict convexity)
   - FTRL useful for analysis
   - Implement Learning-Retaining (cleaner)

6. **Gradient Computation:**
   - For linear memory: gradient is rank-1 matrix (v_t k_t^⊤ for Hebbian)
   - For MLP: use autodiff, but can derive analytically
   - Parallelization: batch all gradients in chunk
   - Use einsum for efficient computation

7. **Data-Dependent Parameters:**
   - η_t = MLP_η(x_t): function of input
   - α_t = sigmoid(MLP_α(x_t)): ensures [0,1]
   - δ_t = softplus(MLP_δ(x_t)): ensures > 0
   - Low-rank: x_t → Linear₁ → ReLU → Linear₂ → param

8. **Hybrid Models:**
   - Sequential: [MIRAS → SWA → MLP] per block
   - SWA: sliding window attention (window=128 or 256)
   - Use flash-attention or similar for SWA
   - MIRAS provides global, SWA provides local

### 11.3 Optimization Opportunities

1. **Kernel Fusion:**
   - Fuse Sign/abs/power operations
   - Fuse Softmax + log for Memora
   - Fuse element-wise ⊙ operations
   - Custom CUDA kernels for Equation 29

2. **Memory Layout:**
   - Contiguous tensors for K, V batches
   - Transpose once, use multiple times
   - Avoid strided operations in inner loops

3. **Numerical Precision:**
   - BF16 for most operations
   - FP32 for normalization (ℓ_q, Softmax)
   - Mixed precision: store W in FP32, compute in BF16
   - Gradient accumulation in FP32

4. **Checkpoint/Recomputation:**
   - Save M_0 at chunk boundaries
   - Recompute chunk internals during backward
   - Trade memory for computation
   - Critical for long sequences

5. **Distributed Training:**
   - Tensor parallelism: split d dimension
   - Sequence parallelism: split N dimension
   - Pipeline parallelism: split layers
   - Chunk parallelism: process chunks on different devices

6. **Initialization:**
   - Memory W: Xavier/He initialization
   - α_t: initialize close to 1 (high retention initially)
   - η_t: initialize small (1e-2 to 1e-3)
   - δ_t: initialize to 1.0

7. **Gradient Clipping:**
   - Memory gradients can be large (especially ℓ_p with p>2)
   - Clip by norm: max_norm = 1.0
   - Or use adaptive clipping

8. **Learning Rate Schedule:**
   - Cosine decay from peak LR
   - Warmup: 2000-5000 steps
   - Peak LR: see Table 5 (1.5e-3 for 340M)
   - Min LR: 10% of peak

### 11.4 Debugging Tips

1. **Gradient Checks:**
   - Numerical gradient vs. autodiff
   - Check each component separately (ℓ_p, ℓ_q, Huber, KL)
   - Use small models (d=32) for debugging

2. **Monitor These:**
   - Memory norm: should be bounded
   - α_t distribution: should be in [0,1]
   - η_t distribution: should be positive, reasonable scale
   - Gradient norms: should not explode
   - Loss: should decrease (eventually)

3. **Common Errors:**
   - NaN: check Softmax (log of negative), sqrt of negative
   - Inf: check exp(large), division by zero
   - Exploding: check α_t > 1, η_t too large
   - No learning: check α_t ≈ 0, η_t ≈ 0

4. **Unit Tests:**
   - RetNet: set p=2, q=2, α=const, no MLP → should match RetNet exactly
   - DeltaNet: set p=2, α=1, no retention → should match DeltaNet
   - Chunk=1: should match sequential (up to numerical precision)

5. **Ablations to Run:**
   - Vary p: {1, 1.5, 2, 2.8, 3, 3.2, 4}
   - Vary q: {2, 3, 4, 5}
   - Vary chunk size: {1, 4, 16, 64}
   - Vary δ: {0.1, 0.5, 1.0, 2.0}

### 11.5 Paper-Specific Notes

1. **Relationship to TITANS:**
   - MIRAS is generalization, not replacement
   - Titans-LMM: ℓ_2 bias, ℓ_2 retention, GD+momentum, k-layer MLP
   - Moneta/Yaad/Memora: simpler (2-layer, GD only) but different objectives/gates
   - TITANS likely better on some tasks (momentum helps)
   - MIRAS provides framework to understand why

2. **Table 1 Unification:**
   - All existing models fit in MIRAS framework
   - Most use ℓ_2 or dot-product bias
   - Most use ℓ_2 retention
   - Novel contribution: go beyond these defaults
   - Opportunity: explore vast design space

3. **Performance Claims:**
   - Moneta/Yaad/Memora beat Mamba2, Gated DeltaNet, TTT
   - Competitive with or beat Transformers (small scale)
   - Better scaling with context length
   - Hybrid versions best overall

4. **When to Use Which Variant:**
   - Moneta: noisy data, need robustness (best on S-NIAH-PK)
   - Yaad: outliers, need adaptive robustness (best overall balance)
   - Memora: need hard forgetting, sparsity (KL divergence)
   - Hybrid: when maximum performance needed (adds SWA)

5. **Limitations (not in paper, inferred):**
   - Fixed memory size: can't grow like Transformer KV cache
   - Approximation in chunk parallelization
   - Many hyperparameters (p, q, δ, α, η)
   - Unclear how to choose attentional bias for new task
   - No theoretical guarantees on memorization capacity

6. **Future Work Suggestions (from paper):**
   - Explore other attentional biases (KL, Wasserstein, etc.)
   - Other retention gates (Section 5.2 has many)
   - Other memory architectures (beyond MLP)
   - Other optimizers (Newton, momentum, adaptive)
   - Theoretical analysis (capacity, expressivity)
   - Application to other domains (vision, biology)

---

## 12. METADATA

```yaml
paper_id: "MIRAS"
paper_title: "It's All Connected: A Journey Through Test-Time Memorization, Attentional Bias, Retention, and Online Optimization"
authors: "Ali Behrouz, Meisam Razaviyayn, Peilin Zhong, Vahab Mirrokni"
affiliation: "Google Research"
arxiv: "arXiv:2504.13173v1"
year: 2025
date_published: "2025-04-17"

equations_extracted: 32
algorithms_extracted: 3
architectures_extracted: 3
hyperparameters_extracted: 20+

key_concepts:
  - "Associative Memory"
  - "Attentional Bias"
  - "Retention Gate (Forget Gate)"
  - "Test-Time Memorization"
  - "Meta-Learning"
  - "Online Optimization"
  - "Linear RNNs"
  - "Efficient Transformers"

novel_contributions:
  - "MIRAS framework (4 design choices)"
  - "Moneta (ℓ_p bias + ℓ_q retention)"
  - "Yaad (Huber bias + ℓ_2 retention)"
  - "Memora (ℓ_2 bias + KL retention)"
  - "Unification of existing models (Table 1)"
  - "Novel retention gates (f-divergence, Elastic Net, etc.)"
  - "Parallelization for non-linear RNNs"

dependencies:
  critical:
    - "TITANS (Behrouz et al. 2024c)"
  important:
    - "TTT (Sun et al. 2024)"
    - "RetNet (Sun et al. 2023)"
    - "Mamba2 (Dao et al. 2024)"
    - "GLA (Yang et al. 2024b)"
    - "Gated DeltaNet (Yang et al. 2024a)"

implementation_priority:
  high:
    - "Core framework (Equation 4, Learning-Retaining)"
    - "Moneta (Equations 24-25, 28-29)"
    - "Yaad (Equations 16, 26)"
    - "Memora (Equations 21, 27, 30-32)"
    - "Parallelization (Section 5.4)"
    - "Architecture (Section 5.4, Figure 2)"
  medium:
    - "Alternative retention gates (Section 5.2)"
    - "Hybrid models (with SWA)"
    - "Channel-wise parametrization"
  low:
    - "Other attentional biases (Equation 17)"
    - "Theoretical analysis"

code_modules:
  - "memory.py: Memory architectures (linear, MLP)"
  - "attentional_bias.py: Loss functions (ℓ_p, Huber, robust, KL)"
  - "retention_gate.py: Retention functions (ℓ_q, KL, Elastic Net, Bregman)"
  - "optimizer.py: Memory learning algorithms (GD, momentum)"
  - "moneta.py: Moneta model"
  - "yaad.py: Yaad model"
  - "memora.py: Memora model"
  - "miras_layer.py: MIRAS layer block"
  - "miras_model.py: Full model (with embedding, LM head)"
  - "parallelization.py: Chunk-based training"
  - "utils.py: Smooth approximations, helper functions"

testing_requirements:
  - "Unit tests for each equation"
  - "Gradient checks (numerical vs. autodiff)"
  - "Baseline reproductions (RetNet, DeltaNet)"
  - "Chunk parallelization equivalence"
  - "Causality verification"
  - "Memory stability tests"
  - "Ablation studies (p, q, δ, chunk size)"

estimated_complexity:
  implementation: "High (many components, careful numerical handling)"
  debugging: "High (meta-learning, multiple loss terms, parallelization)"
  tuning: "High (many hyperparameters, task-dependent optimal choices)"

recommended_approach:
  1. "Implement basic framework (linear memory, ℓ_2 bias, ℓ_2 retention, GD)"
  2. "Test against RetNet/DeltaNet (should match exactly)"
  3. "Add MLP memory architecture"
  4. "Add Moneta components (ℓ_p, ℓ_q)"
  5. "Add Yaad components (Huber)"
  6. "Add Memora components (KL, Softmax)"
  7. "Implement parallelization (critical for scale)"
  8. "Full architecture (MIRAS layer, full model)"
  9. "Training infrastructure"
  10. "Evaluation harness"
  11. "Ablation studies"
  12. "Reproduce paper results"

notes:
  - "MIRAS is framework, not single model"
  - "3 novel models: Moneta, Yaad, Memora"
  - "Unifies 15+ existing models (Table 1)"
  - "Generalizes TITANS (critical dependency)"
  - "Extensive design space (4 choices × many options each)"
  - "Parallelization essential for practical training"
  - "Paper lacks some implementation details (will need experimentation)"
  - "Many hyperparameters (p, q, δ, α, η, chunk size, etc.)"
  - "Task-dependent optimal configuration"
```

---

## END OF CONTEXT FILE

**Total Equations Extracted:** 32+
**Total Algorithms Extracted:** 3 (Moneta, Yaad, Memora) + parallelization variants
**Total Architectures:** 3 variants + hybrid + full model architecture
**Total Hyperparameters:** 20+

This context file provides complete mathematical formulations, implementation guidance, and architectural details for reproducing the MIRAS paper. All equations include LaTeX, plain English descriptions, variable tables with shapes, implementation notes, dependencies, and test criteria as requested.
