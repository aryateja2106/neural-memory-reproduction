# Nested Learning: The Illusion of Deep Learning Architecture - Implementation Context

**Paper ID:** NL
**Full Title:** Nested Learning: The Illusion of Deep Learning Architecture
**Authors:** Ali Behrouz, Meisam Razaviyayn, Peilin Zhong, and Vahab Mirrokni
**arXiv:** Published at NeurIPS 2025
**Year:** 2025

**Key Contribution (1 sentence):**
NL presents a paradigm where machine learning models are represented as nested, multi-level optimization problems with separate context flows, showing that architectures and optimizers are associative memories, and introducing Hope—a continual learning architecture with Continuum Memory Systems.

---

## Quick Reference
| Aspect | Count | Key Items |
|--------|-------|-----------|
| Equations | 121 | Gradient Descent (1-3), Meta Learning (4), FWP Update (5), Adam Decomposition (100-105), DGD (113-121), Self-Referential Titans (83-97), CMS (70-71) |
| Algorithms | 1 | Multi-scale Momentum Muon (M3) |
| Core Architectures | 3 | Self-Referential Titans, Continuum Memory System (CMS), Hope |
| Optimizers | 5 | Delta Gradient Descent (DGD), Delta Momentum, Deep Momentum, M3, standard optimizers as associative memories |

---

## 1. EQUATIONS

### 1.1 Gradient Descent (GD)

**Equation 1: Standard Stochastic Gradient Descent**
```
W_{t+1} = W_t - η_t ∇_{W_t} L(W_t; x_t)
```

**Plain English:** Update weights by moving in the negative gradient direction scaled by learning rate.

**Variables:**
| Variable | Shape | Description |
|----------|-------|-------------|
| W_t | [d_out, d_in] | Weight matrix at step t |
| η_t | scalar | Learning rate at step t |
| ∇_{W_t} L | [d_out, d_in] | Gradient of loss w.r.t. weights |
| x_t | [d_in] | Input data sample |

**Implementation Notes:**
- Standard SGD update
- Can be reformulated as steepest descent in Euclidean metric

**Dependencies:** None
**Test Criteria:** Loss should decrease monotonically with proper learning rate

---

**Equation 2: Steepest Descent Formulation**
```
W_{t+1} = arg min_W { ⟨∇_W L(W_t; x_t), W⟩ + (1/(2η_t)) ||W - W_t||²_2 }
```

**Plain English:** GD minimizes a first-order Taylor approximation regularized by quadratic proximal term.

**Variables:**
| Variable | Shape | Description |
|----------|-------|-------------|
| W | [d_out, d_in] | Optimization variable |
| W_t | [d_out, d_in] | Current weights |

**Implementation Notes:**
- Equivalent to Equation 1
- Reveals implicit bias toward small moves in L2-distance
- Solution obtained by setting gradient to zero

**Dependencies:** Equation 1
**Test Criteria:** Should give identical results to standard GD

---

**Equation 3: FTRL (Follow-The-Regularized-Leader) Form**
```
W_{t+1} = arg min_W { ∑_{s=1}^t ⟨∇L(W_s; x_s), W⟩ + (1/(2η)) ||W - W_1||²_2 }
```

**Plain English:** Accumulates all past gradients with constant learning rate η.

**Variables:**
| Variable | Shape | Description |
|----------|-------|-------------|
| W_s | [d_out, d_in] | Weights at step s |
| η | scalar | Constant learning rate |

**Implementation Notes:**
- Requires constant learning rate
- Solution: W_{t+1} = W_1 - η ∑_{s=1}^t ∇L(W_s; x_s)
- Used interchangeably with steepest descent form

**Dependencies:** Equations 1-2
**Test Criteria:** Equivalent to iterative GD with constant η

---

### 1.2 Meta Learning

**Equation 4: Meta Learning Outer Loop**
```
Φ* = arg min_Φ E_{T_i ~ p(T)} [ℓ(θ, T_i; Φ)]
```

**Plain English:** Meta-learn parameter Φ that optimizes performance across distribution of tasks.

**Variables:**
| Variable | Shape | Description |
|----------|-------|-------------|
| Φ | varies | Outer loop parameters |
| θ | varies | Inner loop parameters |
| T_i | - | Task sampled from distribution |
| p(T) | - | Task distribution |

**Implementation Notes:**
- Two-level optimization
- Outer loop meta-learns, inner loop task-learns
- Can be supervised or unsupervised

**Dependencies:** None
**Test Criteria:** Should improve few-shot performance on new tasks

---

### 1.3 Fast Weight Programmers (FWP)

**Equation 5: Vanilla FWP Update**
```
M_t = α_t M_{t-1} + v_t φ(k_t)^T
```

**Plain English:** Update matrix-valued memory with outer product of value and key, with decay α.

**Variables:**
| Variable | Shape | Description |
|----------|-------|-------------|
| M_t | [d_out, d_key] | Memory matrix at step t |
| α_t | scalar | Decay factor |
| v_t | [d_out] | Value vector |
| k_t | [d_key] | Key vector |
| φ(·) | [d_key] → [d_key] | Element-wise feature map |

**Implementation Notes:**
- Hebbian/outer-product update
- Retrieval: y_t = M_t φ(q_t)
- Matrix state enables key-value memory

**Dependencies:** None
**Test Criteria:** Should learn associative mappings

---

### 1.4 Associative Memory

**Equation 6: Associative Memory Optimization**
```
M* = arg min_M L̃(M(K); V)
```

**Plain English:** Find memory operator M that best maps keys K to values V under objective L̃.

**Variables:**
| Variable | Shape | Description |
|----------|-------|-------------|
| M | function | Memory operator |
| K | [N, d_k] | Set of keys |
| V | [N, d_v] | Set of values |
| L̃ | - | Quality measure (e.g., MSE) |

**Implementation Notes:**
- General framework for sequence models
- Choice of L̃ and optimization determines architecture
- Can be parametric or non-parametric

**Dependencies:** None
**Test Criteria:** Reconstruction error on training pairs

---

### 1.5 Backpropagation as Associative Memory

**Equation 8: Training Linear Layer with Gradient Descent**
```
W_{t+1} = W_t - η_{t+1} ∇_y_{t+1} L(W_t; x_{t+1}) ⊗ x_{t+1}
```

**Plain English:** Update weights proportional to outer product of local error signal and input.

**Variables:**
| Variable | Shape | Description |
|----------|-------|-------------|
| ∇_y L | [d_out] | Local surprise signal (LSS) |
| x_t | [d_in] | Input data |

**Implementation Notes:**
- LSS measures surprise of layer output
- Backprop = learning to map inputs to their errors
- Self-referential: values depend on current state

**Dependencies:** Equation 1
**Test Criteria:** Should match standard backprop exactly

---

**Equation 9: Associative Memory Formulation of Backprop**
```
W_{t+1} = arg min_W { ⟨W x_t, ∇_y_{t+1} L(W_t; x_{t+1})⟩ + (1/(2η_{t+1})) ||W - W_t||²_2 }
```

**Plain English:** Training = finding mapping from inputs to their local error signals.

**Variables:** Same as Equation 8

**Implementation Notes:**
- Equivalent to Equation 8
- Makes compression interpretation explicit
- Key insight: backprop compresses input→error mappings

**Dependencies:** Equation 8
**Test Criteria:** Identical to standard GD on loss

---

### 1.6 Momentum-Based Optimizers

**Equation 10-11: Gradient Descent with Momentum**
```
W_{t+1} = W_t - m_{t+1}
m_{t+1} = m_t + η_{t+1} ∇_W L(W_t; x_{t+1})
```

**Plain English:** Accumulate gradient in momentum term, use it to update weights.

**Variables:**
| Variable | Shape | Description |
|----------|-------|-------------|
| m_t | [d_out, d_in] | Momentum term (gradient accumulator) |

**Implementation Notes:**
- Two-level optimization: m updated by GD, W updated by m
- Momentum = memory of past gradients
- Can set m_t = m_t + η ∇_y L ⊗ x to avoid chain rule

**Dependencies:** Equation 1
**Test Criteria:** Should converge faster than vanilla GD

---

**Equation 12-13: Momentum as Associative Memory**
```
W_{t+1} = W_t - m_{t+1}
m_{t+1} = arg min_m { -⟨m, ∇_{W_t} L(W_t; x_{t+1})⟩ + (1/(2η_{t+1})) ||m - m_t||²_2 }
```

**Plain English:** Momentum solves optimization to compress gradients into its parameters.

**Variables:** Same as Equations 10-11

**Implementation Notes:**
- Momentum = value-less associative memory
- Maps gradients to scalar 1 (dot-product objective)
- Two-level nested optimization

**Dependencies:** Equations 10-11
**Test Criteria:** Equivalent to standard momentum

---

### 1.7 Linear Attention as Associative Memory

**Equation 14-16: Linear Attention**
```
k_t = W_k x_t,  v_t = W_v x_t,  q_t = W_q x_t
M_t = M_{t-1} + v_t k_t^T
y_t = M_t q_t
```

**Plain English:** Project input to keys/values/queries, update memory with outer product, retrieve with query.

**Variables:**
| Variable | Shape | Description |
|----------|-------|-------------|
| W_k, W_v, W_q | [d, d_in] | Projection matrices |
| M_t | [d, d] | Memory matrix |

**Implementation Notes:**
- Unnormalized linear attention
- Memory updated with Hebbian rule
- Projections in outer level, memory in inner level

**Dependencies:** Equation 5
**Test Criteria:** Should learn in-context patterns

---

**Equation 17-18: Linear Attention as Optimization**
```
M_{t+1} = arg min_M { -⟨M k_{t+1}, v_{t+1}⟩ + (1/2) ||M - M_t||²_F }
⟹ M_{t+1} = M_t + v_{t+1} k_{t+1}^T
```

**Plain English:** Linear attention = gradient descent on dot-product objective with learning rate 1.

**Variables:** Same as Equations 14-16

**Implementation Notes:**
- Objective: L̃(M; k, v) = -2⟨Mk, v⟩
- Gradient: ∇L̃ = -v k^T
- Recovers Hebbian update with η=1

**Dependencies:** Equations 14-16
**Test Criteria:** Matches linear attention exactly

---

### 1.8 Nested System Definitions

**Definition 3: Nested System (Equation 19)**
```
θ_i^(k)_{t+1} = arg min_{Φ_i^(k)} { ⟨Φ_i^(k) x_{t+1}, -∇L_i^(k)(θ_it^(k); x_{t+1})⟩
                                     + (1/(2η_i^(k)_{t+1})) ||Φ_i^(k) - θ_it^(k)||²_2 }
```

**Plain English:** Each level k has problems i, each optimized by GD on its context.

**Variables:**
| Variable | Shape | Description |
|----------|-------|-------------|
| k | - | Level index (1 to K) |
| i | - | Problem index in level k |
| θ_i^(k) | varies | Parameters of i-th problem in level k |
| L_i^(k) | - | Objective of i-th problem in level k |
| C_i^(k) | - | Context of i-th problem |

**Implementation Notes:**
- Ordered by update frequency
- Each box has own gradient flow
- No backprop between levels initially

**Dependencies:** Equations 1-3
**Test Criteria:** Should decompose existing architectures

---

**Definition 4: NSAM (Equation 20)**
```
θ_i^(k)_{t+1} = arg min_{Φ_i^(k)} { ⟨Φ_i^(k) k_t^(i), -∇L_i^(k)(θ_it^(k); k_t^(i), v_t^(i))⟩
                                     + (1/(2η_i^(k)_{t+1})) ||Φ_i^(k) - θ_it^(k)||²_2 }
```

**Plain English:** Nested System where each problem is associative memory mapping keys to values.

**Variables:**
| Variable | Shape | Description |
|----------|-------|-------------|
| k_t^(i), v_t^(i) | varies | Key-value pairs for problem i |
| C_i^(k) | {(k_j, v_j)} | Context = set of key-value pairs |

**Implementation Notes:**
- Specialization of Definition 3
- All problems are associative memories
- Unified framework for architectures + optimizers

**Dependencies:** Definition 3, Equation 6
**Test Criteria:** Should recover existing sequence models

---

### 1.9 Delta Gradient Descent (DGD)

**Equation 56: DGD Objective**
```
W_{t+1} = arg min_W { (1/2) ||W x_t - u_t||²_2 + (1/(2η_t)) ||W - W_t||²_2 }
```

**Plain English:** Use L2 regression instead of dot-product to learn input→error mapping.

**Variables:**
| Variable | Shape | Description |
|----------|-------|-------------|
| u_t | [d_out] | -∇_y L (target for regression) |

**Implementation Notes:**
- More expressive than vanilla GD (Hebbian)
- Captures dependencies between samples
- Requires normalized inputs for closed form

**Dependencies:** Equation 9
**Test Criteria:** Should outperform GD on non-i.i.d. data

---

**Equation 57: DGD Closed Form (Normalized Inputs)**
```
W_{t+1} = W_t (I - η'_t x_t x_t^T) - η'_t ∇_y_t L(W_t; x_t) ⊗ x_t
where η'_t = η_t / (1 + η_t)
```

**Plain English:** Update includes adaptive weight decay based on current input.

**Variables:**
| Variable | Shape | Description |
|----------|-------|-------------|
| ||x_t||_2 | scalar | Assumed to be constant λ |
| η'_t | scalar | Adjusted learning rate |

**Implementation Notes:**
- Derived using Sherman-Morrison lemma
- First term = adaptive decay
- Second term = gradient step
- Requires input normalization

**Dependencies:** Equation 56
**Test Criteria:** Should match Equation 56 numerically

---

### 1.10 Continuum Memory System (CMS)

**Equation 70: CMS Forward Pass**
```
y_t = MLP^(f_k)(MLP^(f_{k-1})(... MLP^(f_1)(x_t)))
```

**Plain English:** Chain of MLP blocks, each with different update frequency.

**Variables:**
| Variable | Shape | Description |
|----------|-------|-------------|
| f_ℓ | scalar | Frequency of ℓ-th MLP block |
| k | - | Number of memory levels |

**Implementation Notes:**
- Generalizes long/short-term memory
- Higher frequency = less persistent memory
- Each level compresses its own context

**Dependencies:** None
**Test Criteria:** Should enable memory loop (recover forgotten knowledge)

---

**Equation 71: CMS Update Rule**
```
θ^(f_ℓ)_{i+1} = θ^(f_ℓ)_i - { ∑_{t=i-C^(ℓ)}^i η_t^(ℓ) f(θ_t^(f_ℓ); x_t)  if i ≡ 0 (mod C^(ℓ))
                               { 0                                            otherwise
```

**Plain English:** Update ℓ-th block every C^(ℓ) steps by accumulating errors over chunk.

**Variables:**
| Variable | Shape | Description |
|----------|-------|-------------|
| C^(ℓ) | int | Chunk size for level ℓ |
| f(·) | - | Error function (e.g., ∇L for GD) |

**Implementation Notes:**
- Allows parallelization within chunks
- Different blocks update at different rates
- Frequency f_ℓ = max_i C^(i) / C^(ℓ)

**Dependencies:** Equation 70
**Test Criteria:** Should maintain performance with fewer updates

---

### 1.11 Adam as Associative Memory

**Equation 101: Momentum Objective**
```
L̃_t = ∑_{i=1}^t ||m_ℓ_t ⊙ g_ℓ_{i+1} - P_ℓ_t||²_2 + λ_ℓ ||m_ℓ_t||²_F
```

**Plain English:** Find momentum that maps gradients to global property P.

**Variables:**
| Variable | Shape | Description |
|----------|-------|-------------|
| m_ℓ_t | [d_out, d_in] | Momentum (to optimize) |
| g_ℓ_i | [d_out, d_in] | Gradient at step i |
| P_ℓ_t | [d_out, d_in] | Target property (e.g., variance) |
| ⊙ | - | Element-wise product |

**Implementation Notes:**
- Momentum = associative memory for gradients
- P determines what gradient info is compressed
- Optimal solution depends on P choice

**Dependencies:** Equations 10-13
**Test Criteria:** Should recover known optimizers

---

**Equation 102: Optimal Momentum (Element-wise)**
```
m_ℓ,i^(t)* = [H_ℓ,i^(t) + λ_ℓ I]^(-1) ⊙ M̃_ℓ,i+1^(t) ⊙ P_ℓ_t
where:
M̃_ℓ,i+1^(t) = M̃_ℓ,i^(t) + β_1 g_ℓ_{i+1}
H_ℓ,i+1^(t) = H_ℓ,i^(t) + β_2 g_ℓ_{i+1}²
```

**Plain English:** Optimal momentum divides accumulated gradients by accumulated squared gradients.

**Variables:**
| Variable | Shape | Description |
|----------|-------|-------------|
| M̃ | [d_out, d_in] | First moment (gradient accumulator) |
| H | [d_out, d_in] | Second moment (variance accumulator) |
| β_1, β_2 | scalar | Decay rates |

**Implementation Notes:**
- Setting P = √(∑ g²) gives Adam
- M̃ = momentum, H = variance estimator
- Element-wise operations throughout

**Dependencies:** Equation 101
**Test Criteria:** Should derive Adam exactly

---

**Equation 105: Adam Update (Derived)**
```
W_ℓ_{i+1} = W_ℓ_i - (η_t / √β_2) M̃_ℓ,i^(t) / (H_ℓ,i^(t))^(1/2) + ε
```

**Plain English:** Adam = optimal associative memory for L2 regression to gradient variance.

**Variables:**
| Variable | Shape | Description |
|----------|-------|-------------|
| ε | scalar | Numerical stability constant |

**Implementation Notes:**
- Recovered from first principles
- Shows Adam optimizes specific objective
- Two memories: M̃ and H, same frequency

**Dependencies:** Equation 102
**Test Criteria:** Should match standard Adam implementation

---

### 1.12 Self-Referential Titans

**Equation 83: Self-Referential Titans (General Form)**
```
y_t = M_{memory,t-1}(q_t)
k_t = M_{k,t-1}(x_t),  v_t = M_{v,t-1}(x_t),  η_t = M_{η,t-1}(x_t),  α_t = M_{α,t-1}(x_t)
v̂_{□,t} = M_{□,t-1}(v_t)  (generating own values)
M_{□,t} = M_{□,t-1}(α_t I - η_t k_t k_t^T) - η_t ∇L_{M_{□,t-1}}(M_{□,t-1}; k_t, v̂_{□,t})
for □ ∈ {k, v, q, η, α, memory}
```

**Plain English:** All components (keys, values, learning rates, weight decays) are themselves memories that adapt in-context and generate their own target values.

**Variables:**
| Variable | Shape | Description |
|----------|-------|-------------|
| M_{□} | varies | Memory for component □ |
| α_t | scalar/matrix | Weight decay (input-dependent) |
| η_t | scalar/matrix | Learning rate (input-dependent) |
| v̂_{□,t} | varies | Self-generated value for component □ |

**Implementation Notes:**
- All projections are adaptive memories
- Self-modifying: generates own values v̂
- Uses DGD (delta rule) for update
- q_t = x_t W_q is the only non-adaptive projection

**Dependencies:** Equations 56-57 (DGD), Definition 4 (NSAM)
**Test Criteria:** Should outperform static projections on continual learning

---

**Equation 88: Self-Referential Titans Update (With DGD)**
```
M_{□,t} = M_{□,t-1}(α_t I - η_t k_t k_t^T) - η_t ∇L_{M_{□,t-1}}(M_{□,t-1}; k_t, v̂_{□,t})
```

**Plain English:** Each memory updated using Delta rule with adaptive decay and learning rate.

**Variables:** Same as Equation 83

**Implementation Notes:**
- Uses L2 regression objective
- Adaptive decay term: α_t I - η_t k_t k_t^T
- Gradient computed w.r.t. last chunk state
- All memories initialized via meta-learning

**Dependencies:** Equation 83, 57
**Test Criteria:** Should learn to modify itself based on context

---

**Equation 90: Chunk-wise Self-Referential Titans (Efficient)**
```
y_t = M_{memory,C×⌈t/C⌉}(q_t)
k_t = M_{k,C×⌈t/C⌉}(x_t), ...
v̂_{□,t} = M_{□,C×⌈t/C⌉}(v_t)
M_{□,t} = M_{□,t-1}(α_t I - η_t k_t k_t^T) - η_t ∇L_{M_{□,C×⌈t/C⌉}}(M_{□,C×⌈t/C⌉}; k_t, v̂_{□,t})
```

**Plain English:** Generate keys/values/etc. once per chunk for parallelization.

**Variables:**
| Variable | Shape | Description |
|----------|-------|-------------|
| C | int | Chunk size |
| ⌈t/C⌉ | int | Chunk index |

**Implementation Notes:**
- Compute all chunk parameters before processing
- Enables sequence parallelization (dual form)
- Gradients taken w.r.t. chunk start state
- Trade-off: accuracy vs. efficiency

**Dependencies:** Equation 88
**Test Criteria:** Should approximate Equation 88 with computational savings

---

**Equation 92-93: Matrix-Valued Titans (Dot-Product vs L2)**

**Dot-Product Objective:**
```
M_{□,t} = M_{□,t-1}(α_t I - η_t k_t k_t^T) - η_t v̂_{□,t} k_t^T
```

**L2 Regression Objective:**
```
M_{□,t} = M_{□,t-1}(α_t I - η_t k_t k_t^T) - η_t (M_{□,C×⌈t/C⌉} k_t - v̂_{□,t}) k_t^T
```

**Plain English:** Two choices for recurrence: Hebbian (dot-product) or Delta rule (L2).

**Variables:**
| Variable | Shape | Description |
|----------|-------|-------------|
| M_{□} | [d, d] | Matrix-valued memory |

**Implementation Notes:**
- Dot-product = simpler, Hebbian update
- L2 = better memory management (Delta rule)
- Choice affects capacity and forgetting
- L2 recommended for continual learning

**Dependencies:** Equation 90
**Test Criteria:** L2 should outperform dot-product on long sequences

---

### 1.13 Hope Architecture

**Equation 94-97: Hope (Self-Referential Titans + CMS)**
```
o_t = M_{memory,t-1}(q_t)
k_t = M_{k,t-1}(x_t), v_t = M_{v,t-1}(x_t), η_t = M_{η,t-1}(x_t), α_t = M_{α,t-1}(x_t)
v̂_{□,t} = M_{□,t-1}(v_t)
M_{□,t} = M_{□,t-1}(α_t I - η_t k_t k_t^T) - η_t ∇L_{M_{□,t-1}}(M_{□,t-1}; k_t, v̂_{□,t})
y_t = MLP^(f_k)(MLP^(f_{k-1})(... MLP^(f_1)(o_t)))
```

**Plain English:** Hope = self-modifying Titans (high-frequency in-context learning) followed by CMS (multi-frequency persistent memory).

**Variables:**
| Variable | Shape | Description |
|----------|-------|-------------|
| o_t | [d] | Output of Titans block |
| y_t | [d] | Final output after CMS |

**Implementation Notes:**
- Two complementary systems:
  - Titans: expressive learning rule, small capacity
  - CMS: simple rule, large capacity
- Also includes local convolutions (window=4)
- L2 normalization on q and k

**Dependencies:** Equations 83-97 (Titans), 70-71 (CMS)
**Test Criteria:** Should excel at continual learning and long-context tasks

---

### 1.14 Multi-scale Momentum Muon (M3)

**Algorithm 1: M3 Optimizer Pseudocode**
```python
# Inputs: θ_0, L, η, T, β_1, β_2, β_3, α, ε, f
# Initialize: M_0^(1), M_0^(2) ← 0, V_0 ← 0

for k = 0, 1, 2, ... do:
    # Slow Memory (updated every f steps)
    M_t^(2) = M_{t-1}^(2) + β_3 ∑_{i=(k-1)f}^{kf} g_i
    O_t^(2) = NewtonSchulz_T(M_t^(2))

    for t = kf+1, kf+2, ..., (k+1)f do:
        # Gradient
        g_t = ∇_{θ_t} L(θ_t)

        # First Momentum (fast)
        M_t^(1) = M_{t-1}^(1) + β_1 g_t

        # Second Momentum (variance)
        V_t = V_{t-1} + β_2 g_t²

        # Orthogonalize fast momentum
        O_t^(1) = NewtonSchulz_T(M_t^(1))

        # Update
        θ_t = θ_{t-1} - η (O_t^(1) + α O_t^(2)) / (√V_t + ε)
```

**Plain English:** Two-frequency momentum (fast + slow) with orthogonalization and variance normalization.

**Variables:**
| Variable | Shape | Description |
|----------|-------|-------------|
| M^(1) | [d_out, d_in] | Fast momentum (updated every step) |
| M^(2) | [d_out, d_in] | Slow momentum (updated every f steps) |
| V | [d_out, d_in] | Variance estimator |
| O^(1), O^(2) | [d_out, d_in] | Orthogonalized momentums |
| f | int | Frequency ratio (slow update interval) |
| α | scalar | Slow momentum weight |

**Implementation Notes:**
- CMS applied to optimizer context (gradients)
- NewtonSchulz_T = iterative orthogonalization (T steps)
- Combines Adam (V term) + Muon (orthog) + CMS (two frequencies)
- Slow momentum provides long-term gradient info

**Dependencies:** Equation 101-105 (Adam as memory), 70-71 (CMS), 42-44 (Muon)
**Test Criteria:** Should find better solutions than single-scale optimizers

---

## 2. ALGORITHMS

### Algorithm 1: Multi-scale Momentum Muon (M3)

**Purpose:** Optimizer with continuum memory for gradient compression

**Pseudocode:**
```
Input: θ_0, L(·), η > 0, T, β_1, β_2, β_3 ∈ [0,1), α ≥ 0, ε > 0, f
Initialize: M_0^(1), M_0^(2) ← 0, V_0 ← 0

for lower-frequency iteration k = 0, 1, 2, ... do:
    # Update slow memory every f steps
    M_t^(2) = M_{t-1}^(2) + β_3 ∑_{i=(k-1)f}^{kf} g_i
    O_t^(2) ← NewtonSchulz_T(M_t^(2))

    for t = kf+1, ..., (k+1)f do:
        g_t = ∇_{θ_t} L(θ_t)
        M_t^(1) = M_{t-1}^(1) + β_1 g_t
        V_t = V_{t-1} + β_2 g_t²
        O_t^(1) ← NewtonSchulz_T(M_t^(1))
        θ_t ← θ_{t-1} - η (O_t^(1) + α O_t^(2)) / (√V_t + ε)
```

**Line-by-Line Mapping:**
1. **Input parameters**: Learning rate η, Newton-Schulz iterations T, momentum rates β_1/β_2/β_3, slow weight α, stability ε, frequency f
2. **Initialize moments**: Two momentum matrices (fast/slow), one variance matrix
3. **Outer loop**: Iterate over chunks of size f
4. **Update slow momentum**: Accumulate gradients from last f steps
5. **Orthogonalize slow**: Newton-Schulz on M^(2) for better geometry
6. **Inner loop**: Iterate within chunk
7. **Compute gradient**: Standard backprop
8. **Update fast momentum**: EMA of gradients with β_1
9. **Update variance**: EMA of squared gradients with β_2
10. **Orthogonalize fast**: Newton-Schulz on M^(1)
11. **Weight update**: Combine both orthogonalized momentums, normalize by variance

**Implementation Considerations:**
- NewtonSchulz_T: Iterative method to find Q s.t. Q^T Q ≈ I
  - Initialize Q_0 = M / ||M||
  - Iterate: Q_{i+1} = Q_i (3I - Q_i^T Q_i) / 2 for T steps
- Computational cost: O(Td² per update where d = param size
- Memory: 3 matrices (M^(1), M^(2), V) + temporaries
- Frequency f controls update interval of slow memory
- α controls contribution of slow vs fast momentum

**Dependencies:**
- Equations 75 (CMS in optimizer), 101-105 (Adam decomposition)
- NewtonSchulz orthogonalization (Equation 44)

**Test Criteria:**
- Should converge faster than AdamW or Muon on vision tasks
- Slow momentum should capture long-term gradient structure
- May have overhead for small models

---

## 3. ARCHITECTURE

### 3.1 Nested Learning Module Structure

```
Neural Learning Module (NLM)
├── Level 1 (Frequency: ∞) [In-Context Learning]
│   ├── Non-Parametric Blocks (Attention, etc.)
│   └── Context: Token sequence
│
├── Level 2 (Frequency: 1/C) [Test-Time Learning]
│   ├── Parametric Memories (Linear Attention, Titans)
│   └── Context: Token sequence (updated every C steps)
│
├── Level 3+ (Frequency: 1/C^(ℓ)) [Persistent Memory]
│   ├── MLP Blocks in CMS
│   └── Context: Accumulated representations
│
└── Level ∞ (Frequency: 0) [Pre-Training]
    ├── All slow weights (W_k, W_v, W_q, MLP_init)
    └── Context: Entire pre-training dataset
```

**Key Insight:** Transformers are 2-level systems (Level 1=Attention at freq ∞, Level ∞=MLP at freq 0)

---

### 3.2 Hope Architecture Diagram

```
Input Sequence {x_1, ..., x_T}
    │
    ▼
┌─────────────────────────────────────────┐
│ Self-Referential Titans Block           │
│                                          │
│  q_t = x_t W_q  (static)                │
│                                          │
│  ┌──────────────────────────────────┐   │
│  │ Adaptive Projection Memories     │   │
│  │  M_k(x_t) → k_t                  │   │
│  │  M_v(x_t) → v_t                  │   │
│  │  M_η(x_t) → η_t  (learning rate) │   │
│  │  M_α(x_t) → α_t  (weight decay)  │   │
│  └──────────────────────────────────┘   │
│                                          │
│  ┌──────────────────────────────────┐   │
│  │ Self-Value Generation            │   │
│  │  M_□(v_t) → v̂_□,t  for each □    │   │
│  └──────────────────────────────────┘   │
│                                          │
│  ┌──────────────────────────────────┐   │
│  │ Memory Update (DGD)              │   │
│  │  M_□,t = M_□,t-1 (α_t I - η_t k k^T) │
│  │          - η_t ∇L(M; k_t, v̂_□,t)     │
│  └──────────────────────────────────┘   │
│                                          │
│  Retrieval: o_t = M_memory(q_t)         │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Continuum Memory System (CMS)           │
│                                          │
│  ┌──────────────────────────────────┐   │
│  │ Level 1: High Freq (f_1 = L/C_1) │   │
│  │  MLP^(f_1)(·)                     │   │
│  │  Updates every C_1 tokens         │   │
│  └──────────────────────────────────┘   │
│         │                                │
│  ┌──────────────────────────────────┐   │
│  │ Level 2: Mid Freq (f_2 = L/C_2)  │   │
│  │  MLP^(f_2)(·)                     │   │
│  │  Updates every C_2 tokens         │   │
│  └──────────────────────────────────┘   │
│         │                                │
│       ...                                │
│         │                                │
│  ┌──────────────────────────────────┐   │
│  │ Level k: Low Freq (f_k = L/C_k)  │   │
│  │  MLP^(f_k)(·)                     │   │
│  │  Updates every C_k tokens         │   │
│  └──────────────────────────────────┘   │
│                                          │
│  Output: y_t (token representation)     │
└─────────────────────────────────────────┘
```

**Module Breakdown:**

1. **Self-Referential Titans** (High-Frequency Level)
   - Input: x_t ∈ R^d
   - Memories: M_k, M_v, M_η, M_α, M_memory (each is 2-layer MLP)
   - Update: Every token (frequency = 1)
   - Purpose: Expressive in-context learning with self-modification
   - Output: o_t ∈ R^d

2. **Continuum Memory System** (Multi-Frequency Levels)
   - Input: o_t ∈ R^d
   - Blocks: k MLP layers with frequencies f_1 > f_2 > ... > f_k
   - Update: Each MLP^(f_ℓ) updates every C^(ℓ) tokens
   - Purpose: Persistent memory with different timescales
   - Output: y_t ∈ R^d

**Layer Details:**

**Titans Memory Architecture:**
```python
class TitansMemory(nn.Module):
    def __init__(self, d_model, d_hidden):
        self.W1 = nn.Linear(d_model, d_hidden)
        self.W2 = nn.Linear(d_hidden, d_model)
        self.activation = nn.GELU()

    def forward(self, x):
        return x + self.W2(self.activation(self.W1(x)))
```

**CMS MLP Architecture:**
```python
class CMSMLP(nn.Module):
    def __init__(self, d_model, expansion_factor=4):
        self.up = nn.Linear(d_model, expansion_factor * d_model)
        self.down = nn.Linear(expansion_factor * d_model, d_model)
        self.activation = nn.GELU()

    def forward(self, x):
        return x + self.down(self.activation(self.up(x)))
```

---

### 3.3 Hope-Attention Variant

```
Input Sequence {x_1, ..., x_T}
    │
    ▼
┌─────────────────────────────────────────┐
│ Standard Softmax Attention              │
│  (non-parametric, frequency = ∞)        │
│                                          │
│  Q, K, V = x W_q, x W_k, x W_v          │
│  Attn = softmax(QK^T / √d) V            │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Continuum Memory System (CMS)           │
│  (replaces MLP block)                   │
│                                          │
│  Multi-frequency MLP chain              │
│  (same as above)                        │
└─────────────────────────────────────────┘
```

**Purpose:** Simpler variant replacing Titans with standard attention, only adding CMS

---

## 4. HYPERPARAMETERS

### 4.1 Hope Model Configuration

| Parameter | Value | Section/Table | Description |
|-----------|-------|---------------|-------------|
| **Architecture** | | | |
| d_model | 760M: 1024, 1.3B: 1536 | Section 9.3 | Model dimension |
| n_layers | varies | Section 9.3 | Number of Hope blocks |
| vocab_size | 32K | Section 9.2 | Tokenizer vocabulary |
| **Titans Block** | | | |
| memory_architecture | 2-layer MLP | Eq 89, 91 | M_□(·) = · + W_1 σ(W_2(·)) |
| chunk_size_titans | varies | Section 8.2 | Chunk size for Titans update |
| titans_objective | L2 regression | Eq 93 | Internal loss for memories |
| normalization | L2 on q, k | Section 8.3 | Normalize query and key |
| local_conv_window | 4 | Section 8.3 | Local convolution window |
| **CMS Block** | | | |
| num_levels | 4 | Section 7 | Number of frequency levels |
| chunk_sizes | [512, 2K, 8K, 32K] | Figure 7 | C^(ℓ) for each level |
| frequencies | [L/512, L/2K, L/8K, L/32K] | Figure 7 | Derived from chunks |
| mlp_expansion | 4 | Standard | Hidden dim = 4 × d_model |
| cms_connection | Sequential | Eq 73 | MLP^(k)(...MLP^(1)(·)) |

### 4.2 Training Configuration

| Parameter | Value | Section/Table | Description |
|-----------|-------|---------------|-------------|
| **Data** | | | |
| dataset | FineWeb-Edu + long-context | Sec 9.2, 9.3 | Training corpus |
| tokens_760M | 30B | Section 9.3 | Tokens for 760M model |
| tokens_1.3B | 100B | Section 9.3 | Tokens for 1.3B model |
| context_length | 4K-32K | Tables 1-2 | Training context window |
| **Optimizer** | | | |
| optimizer | AdamW | Section 9.3 | Default optimizer |
| learning_rate | tuned per model | Section 9.3 | Model-specific tuning |
| adam_beta1 | 0.9 (typical) | Standard | First moment decay |
| adam_beta2 | 0.999 (typical) | Standard | Second moment decay |
| weight_decay | tuned | Section 9.3 | L2 regularization |
| **Meta-Learning** | | | |
| memory_init | meta-learned | Eq 72-73, Sec 8.1 | Initial state of all M_□ |
| meta_objective | E_T[ℓ(Θ,T;Φ)] | Eq 72 | Outer loop loss |

### 4.3 M3 Optimizer Configuration

| Parameter | Value | Section/Table | Description |
|-----------|-------|---------------|-------------|
| η (learning_rate) | tuned | Algorithm 1 | Base learning rate |
| T (newton_schulz_iters) | varies | Algorithm 1 | Orthogonalization iterations |
| β_1 | 0.9 (typical) | Algorithm 1 | Fast momentum decay |
| β_2 | 0.999 (typical) | Algorithm 1 | Variance decay |
| β_3 | tuned | Algorithm 1 | Slow momentum decay |
| α (slow_weight) | tuned | Algorithm 1 | Slow momentum contribution |
| ε (epsilon) | 1e-8 (typical) | Algorithm 1 | Numerical stability |
| f (frequency) | tuned | Algorithm 1 | Slow update interval |

### 4.4 Continual Learning Tasks

| Parameter | Value | Section/Table | Description |
|-----------|-------|---------------|-------------|
| **Class-Incremental** | | | |
| base_model | Llama-3B, Llama3-8B | Section 9.1 | Backbone for adaptation |
| continual_pretrain_tokens | 15B | Section 9.1 | After adding CMS |
| **CTNL (Translation)** | | | |
| languages | Kalamang, Manchu | Section 9.1 | Novel languages to learn |
| setup | sequential | Figure 8 | Learn L1 then L2 |
| **Memory Levels (Ablation)** | | | |
| num_levels_tested | 1, 2, 3, 4 | Figure 7 | Effect of CMS depth |
| lowest_freq_tested | 512, 2K, 8K | Figure 7 | Effect of frequency |

---

## 5. BENCHMARKS

### 5.1 Datasets

| Dataset | Type | Details | Results Table |
|---------|------|---------|---------------|
| **Continual Learning** | | | |
| CLINC | Intent classification | 150 classes, 23.7K queries, 10 domains | Figure 6 |
| Banking | Intent classification | 77 classes, 3083 examples, single-domain | Figure 6 |
| DBpedia | Topic classification | 70 classes, 340K examples (10K used) | Figure 6 |
| CTNL | Translation | Kalamang + Manchu → English | Figure 8 |
| **Long Context** | | | |
| MK-NIAH | Multi-key retrieval | RULER benchmark, 4K-16K context | Table 1, Fig 7 |
| LongHealth | Medical QA | 5.1K-6.8K word patient records, 200 Qs | Figure 7 |
| QASPER | Paper QA | 1.6K NLP papers, 5K QA pairs | Figure 7 |
| S-NIAH-1/2/3 | Single needle | Passkey, number, UUID retrieval | Table 1 |
| MQ-NIAH | Multi-query | Multiple questions per context | Table 1 |
| MV-NIAH | Multi-value | Multiple values to retrieve | Table 1 |
| BABILong | Reasoning | Sequences up to 10M tokens | Figure 9 |
| **Language Modeling** | | | |
| Wikitext | Perplexity | Standard LM benchmark | Table 2 |
| LambadaStory (LMB) | Perplexity + Accuracy | Story completion | Table 2 |
| **Reasoning** | | | |
| PIQA | Physical QA | Physical commonsense | Table 2 |
| HellaSwag | Sentence completion | Commonsense reasoning | Table 2 |
| WinoGrande | Pronoun resolution | Winograd schema | Table 2 |
| ARC-easy/challenge | Science QA | Grade-school science | Table 2 |
| SIQA | Social QA | Social interactions | Table 2 |
| BoolQ | Yes/No QA | Boolean questions | Table 2 |
| **In-Context Recall** | | | |
| SWDE | Information extraction | Short context recall | Table 3 |
| NaturalQuestions (NQ) | QA | Wikipedia-based | Table 3 |
| DROP | Reading comprehension | Discrete reasoning | Table 3 |
| FDA | Domain-specific QA | Financial documents | Table 3 |
| SQUAD | QA | Reading comprehension | Table 3 |
| TQA | Textbook QA | Science textbooks | Table 3 |
| **Synthetic** | | | |
| MAD | Synthetic memory | Compression, ICR, copying, selective memory | Table 4 |
| Formal Languages | Automata recognition | Parity, regex, counter, shuffle | Table 5 |

### 5.2 Reported Results (Key Highlights)

**Continual Learning (Figure 6):**
- CLINC (Llama3-3B): Hope 93.4%, InCA 69.2%, ICL 63.7%, EWC 60.1%
- Banking (Llama3-3B): Hope 85.2%, InCA 63.9%, ICL 58.1%, EWC 55.7%
- DBpedia (Llama3-3B): Hope 82.1%, InCA 42.6%, ICL 38.2%, EWC 36.8%

**Long Context (Table 1, 16K context):**
- S-NIAH-1 (passkey): Hope 100%, Transformer 79.8%, RWKV-7 99.6%
- S-NIAH-3 (uuid): Hope 24.8%, Transformer 40.8%, Titans 21.2%
- MK-NIAH-1: Hope 14.8%, Transformer 61.4%, Titans 8.2%

**Language Modeling (1.3B, Table 2):**
- Wiki ppl: Hope 14.39, Titans 15.60, Transformer++ 17.92
- Avg reasoning acc: Hope 58.04%, Titans 56.82%, Transformer++ 53.38%

**BABILong (Figure 9):**
- Hope maintains >80% accuracy up to 10M context
- Titans/ARMT drop to ~70% beyond 1M
- Large models (GPT4) fail around 128K-256K

### 5.3 Computational Requirements

| Model Size | Tokens | Wall Time (est.) | Hardware | Source |
|------------|--------|------------------|----------|--------|
| 760M params | 30B | ~days | Multi-GPU | Section 9.3 |
| 1.3B params | 100B | ~weeks | Multi-GPU | Section 9.3 |
| Hope overhead | ~1.2-1.5× | vs Transformer | GPU memory | Section 7.1 |
| M3 overhead | ~1.3× | vs Muon | Training time | Figure 12 |

**Memory Requirements:**
- Self-Referential Titans: 6× memory matrices (M_k, M_v, M_q, M_η, M_α, M_memory)
- CMS: k MLP blocks (typically k=4)
- Total: ~1.5× parameter count vs. standard Transformer

**Inference Speed:**
- Hope: O(L·d²) for Titans + O(L·d²) for CMS per layer
- Titans chunk-wise: Parallelizable within chunks
- CMS: Only update subset of blocks per token (frequency-dependent)

---

## 6. DEPENDENCIES

### 6.1 Cross-Paper Dependencies

**Relationship to TITANS:**
- **NL generalizes TITANS**: TITANS (Behrouz et al. 2025c) is a specific instance of NSAM
- **TITANS = 2-level nested system**:
  - Level 1 (outer): Projection matrices W_k, W_v, W_q optimized via pre-training
  - Level 2 (inner): Memory M_t optimized in-context with Delta rule
- **NL extends TITANS with**:
  - Self-referential design (Titans generate own values)
  - Adaptive projections (all W become memories M)
  - CMS for persistent memory across levels
- **Equations inherited from TITANS**:
  - Delta rule update (Eq 65 in NL = core TITANS update)
  - Matrix-valued memory formulation
  - Meta-learned initialization

**Relationship to MIRAS:**
- **MIRAS = general associative memory framework**: Behrouz et al. 2025b
- **NL uses MIRAS Definition 1**: Associative memory as optimization (Eq 6)
- **MIRAS provides**:
  - Attentional bias (internal objective L̃)
  - Optimization algorithm choices (GD, Newton, etc.)
  - Non-Euclidean objectives (L_p norms)
- **NL extends MIRAS with**:
  - Nested/multi-level formulation (MIRAS is single-level)
  - Knowledge transfer between levels
  - Optimizer decomposition as associative memory
- **Key MIRAS results used in NL**:
  - Linear attention = dot-product bias + GD (Eq 17-18)
  - DeltaNet = L2 bias + GD (Eq 65)
  - Higher-order feature maps for capacity

**Other Dependencies:**
- **Linear Transformers** (Katharopoulos et al. 2020): Hebbian FWP formulation (Eq 5, 64)
- **Adam** (Kingma & Ba 2014): Decomposed as nested memory (Eq 101-105)
- **MAML** (Finn et al. 2017): Knowledge transfer via initialization (Eq 28, 72-73)
- **Muon** (Jordan et al. 2024): Orthogonalization in momentum (Eq 42-44)
- **FWP** (Schlag et al. 2021): Fast weight programmer formulation (Eq 5)

### 6.2 Implementation Order

**Phase 1: Core Foundations**
1. Implement associative memory base class (Eq 6)
2. Implement gradient descent variants:
   - Standard GD (Eq 1-3)
   - GD with momentum (Eq 10-13)
   - Delta Gradient Descent (Eq 56-57, 113-121)
3. Test on simple linear regression tasks

**Phase 2: Sequence Models**
4. Implement linear attention (Eq 14-18, 64)
5. Implement DeltaNet (Eq 65)
6. Test on sequence memorization tasks
7. Verify equivalence to existing implementations

**Phase 3: Nested Systems**
8. Implement NSAM framework (Definition 4, Eq 19-20)
9. Decompose existing models (Transformers, linear RNNs)
10. Test that decomposition matches original performance

**Phase 4: Optimizers as Memories**
11. Implement Adam decomposition (Eq 101-105)
12. Implement M3 optimizer (Algorithm 1)
13. Test on vision tasks (ImageNet)

**Phase 5: Continuum Memory System**
14. Implement CMS (Eq 70-71)
15. Test frequency ablations (Figure 7)
16. Implement ad-hoc stacking from pre-trained models (Section 7.3)

**Phase 6: Self-Referential Titans**
17. Implement basic Titans (from TITANS paper, Eq 93)
18. Add self-value generation (Eq 84, 87)
19. Make all projections adaptive (Eq 83, 88)
20. Implement chunk-wise training (Eq 90)
21. Test on continual learning tasks

**Phase 7: Hope Architecture**
22. Combine Titans + CMS (Eq 94-97)
23. Add local convolutions and normalization
24. Implement Hope-Attention variant
25. Full evaluation suite (Tables 1-5)

**Phase 8: Scaling and Optimization**
26. Implement sequence parallelization (dual form)
27. Memory optimization for large models
28. Distributed training setup
29. Hyperparameter tuning

---

## 7. IMPLEMENTATION CHECKLIST

### 7.1 Equations to Implement

**Core Optimization:**
- [ ] Eq 1: Standard SGD
- [ ] Eq 2: Steepest descent formulation
- [ ] Eq 3: FTRL form
- [ ] Eq 56-57: Delta Gradient Descent (DGD)
- [ ] Eq 113-121: DGD with normalization (Appendix C)

**Momentum Variants:**
- [ ] Eq 10-11: GD with momentum
- [ ] Eq 12-13: Momentum as associative memory
- [ ] Eq 33: Momentum update (general form)
- [ ] Eq 34: Momentum objective (dot-product)
- [ ] Eq 37: Generalized momentum
- [ ] Eq 48-49: Delta momentum
- [ ] Eq 50: Deep momentum (DMGD)
- [ ] Eq 51: Higher-order feature map momentum
- [ ] Eq 52: Nonlinear output momentum (Muon)

**Adam Decomposition:**
- [ ] Eq 101: Momentum objective for Adam
- [ ] Eq 102: Optimal momentum (element-wise)
- [ ] Eq 103: Adam update (derived)
- [ ] Eq 105: Adam final form
- [ ] Eq 106-111: AdaGrad as associative memory

**Sequence Models:**
- [ ] Eq 5: Vanilla FWP update
- [ ] Eq 6: Associative memory definition
- [ ] Eq 14-18: Linear attention + optimization view
- [ ] Eq 64: Linear attention with Hebbian rule
- [ ] Eq 65: DeltaNet (Delta rule)
- [ ] Eq 92-93: Matrix-valued updates (dot-product vs L2)

**Nested Systems:**
- [ ] Eq 19: Nested system definition
- [ ] Eq 20: NSAM definition
- [ ] Eq 24-27: Knowledge transfer methods

**Continuum Memory System:**
- [ ] Eq 70: CMS forward pass
- [ ] Eq 71: CMS update rule
- [ ] Eq 72-73: CMS knowledge transfer variants

**Self-Referential Titans:**
- [ ] Eq 83: Titans general form
- [ ] Eq 84: Self-value generation
- [ ] Eq 85: Titans optimization
- [ ] Eq 86-88: Titans with DGD
- [ ] Eq 89: Memory architecture (2-layer MLP)
- [ ] Eq 90: Chunk-wise Titans (efficient)
- [ ] Eq 92-93: Matrix-valued Titans

**Hope Architecture:**
- [ ] Eq 94-97: Hope (Titans + CMS)
- [ ] Local convolutions (window=4)
- [ ] L2 normalization on q, k

### 7.2 Algorithms to Implement

- [ ] Algorithm 1: Multi-scale Momentum Muon (M3)
  - [ ] Fast momentum update
  - [ ] Slow momentum update
  - [ ] Variance estimator
  - [ ] NewtonSchulz orthogonalization
  - [ ] Weight update with aggregation

### 7.3 Architectures to Implement

**Core Components:**
- [ ] Associative Memory base class
- [ ] MLP module (for memory architecture)
- [ ] Linear attention block
- [ ] DeltaNet block

**CMS Components:**
- [ ] CMS block (multi-frequency MLPs)
- [ ] Chunk-based update scheduler
- [ ] Frequency controller

**Titans Components:**
- [ ] Adaptive projection memories (M_k, M_v, M_η, M_α)
- [ ] Self-value generator (M_□)
- [ ] Main memory (M_memory)
- [ ] DGD updater for memories
- [ ] Chunk-wise parallelizer

**Hope:**
- [ ] Hope block (Titans + CMS)
- [ ] Hope-Attention variant
- [ ] Full Hope model with stacking

### 7.4 Test Cases

**Unit Tests:**
- [ ] GD converges on convex quadratic
- [ ] DGD matches GD on i.i.d. data
- [ ] DGD outperforms GD on sequential data
- [ ] Momentum accelerates convergence
- [ ] Adam derivation matches standard implementation
- [ ] Linear attention matches FWP formulation
- [ ] DeltaNet matches Delta rule math
- [ ] CMS blocks update at correct frequencies

**Integration Tests:**
- [ ] Decomposed Transformer matches original
- [ ] Titans learns in-context on synthetic tasks
- [ ] CMS enables memory recovery (loop test)
- [ ] Hope trains without NaNs or instability
- [ ] M3 converges on vision tasks

**Benchmark Tests:**
- [ ] Hope > ICL on continual learning (CLINC, Banking)
- [ ] Hope > Transformers on long context (NIAH)
- [ ] Hope ≥ Titans on language modeling
- [ ] Hope maintains performance on BABILong to 10M tokens
- [ ] M3 finds better solutions than Adam/Muon

---

## 8. NOTES FOR IMPLEMENTER

### 8.1 Critical Gotchas

**1. Frequency and Update Timing**
- **Issue**: Easy to confuse frequency f with chunk size C
- **Fix**: f = L/C where L = total sequence length
- Higher frequency = more updates = LESS persistent memory
- Update at step t if `t % C == 0`

**2. Self-Referential Titans Value Generation**
- **Issue**: v̂_{□,t} depends on M_{□,t-1}, creating circular dependency
- **Fix**: Use chunk-wise update (Eq 90)
  - Compute all v̂ at chunk boundaries
  - Use same v̂ for entire chunk
  - Enables parallelization

**3. DGD Requires Normalization**
- **Issue**: Closed form (Eq 57) only valid if ||x_t||_2 = constant
- **Fix**: Apply L2 normalization to inputs before DGD
- Alternative: Use iterative solver without normalization

**4. Meta-Learning Initialization**
- **Issue**: Memories need good initialization to adapt fast
- **Fix**: Meta-learn M_{□,0} across tasks (Eq 72-73)
- Cold start: Initialize with small random values
- Can also initialize from pre-trained MLP weights (Section 7.3)

**5. Memory Requirements**
- **Issue**: Hope requires ~1.5× parameters vs Transformer
- **Fix**:
  - Use smaller d_hidden for Titans memories
  - Reduce number of CMS levels for small models
  - Gradient checkpointing for long sequences

**6. Numerical Stability**
- **Issue**: NewtonSchulz can explode if M is poorly conditioned
- **Fix**:
  - Clip gradient norms
  - Initialize M near identity
  - Use fewer iterations T (3-5 sufficient)

**7. Chunk Size Selection**
- **Issue**: Small C = more updates = slower; large C = less adaptive
- **Fix**:
  - Titans: C ∈ [16, 128] (balance speed/adaptation)
  - CMS: Exponential spacing [512, 2K, 8K, 32K]
  - Task-dependent: shorter for continual learning

**8. Knowledge Transfer**
- **Issue**: No gradient flow between levels by default
- **Fix**: Choose transfer mechanism:
  - Initialization (MAML-style): Eq 28, 72-73
  - Direct connection: Eq 24-27
  - Backpropagation: Equation in Section 3.3
- Hope uses initialization for CMS, self-generation for Titans

### 8.2 Clarifications

**What is "context flow"?**
- The data that an optimization problem operates on
- Examples:
  - Tokens: Context flow for architectures
  - Gradients: Context flow for optimizers
  - Activations: Context flow for normalization layers
- Each level has its own context flow

**What makes Titans "self-referential"?**
- Standard memory: v_t given externally
- Self-referential: v̂_t = M(v_t) generated by memory itself
- Enables self-modification: memory controls its own learning

**Why is Adam "optimal" for L2 regression on gradient variance?**
- Eq 101 defines objective: map gradients → variance
- Eq 102 derives optimal solution analytically
- Result: m* = M̃ / √H (Adam's form)
- Optimality = closed-form solution to that specific objective

**What's the difference between Hope and Hope-Attention?**
- Hope: Self-Referential Titans + CMS
- Hope-Attention: Standard Softmax Attention + CMS
- Hope-Attention simpler but less adaptive

**How does CMS enable "memory loop"?**
- Without CMS: Forgotten knowledge is gone
- With CMS: Knowledge stored in multiple frequencies
- If high-freq forgets, low-freq retains
- Knowledge transfers back via backprop or initialization
- See Figure 10 for empirical evidence

**Why are Transformers "2-level"?**
- Level 1: Attention (frequency = ∞, updated every token non-parametrically)
- Level 2: MLP + projections (frequency = 0, never updated post-training)
- No intermediate frequencies → static after pre-training

### 8.3 Optimization Opportunities

**1. Sequence Parallelization**
- Titans chunk-wise: All tokens in chunk computed in parallel
- CMS: High-freq blocks parallelizable within chunks
- Use dual form (Sun et al. 2024, Behrouz et al. 2025c)

**2. Memory Sharing**
- Share v_t across all M_{□} (reduces 6× to 1× value storage)
- Share MLP weights with different inputs (weight tying)

**3. Sparse Updates**
- Only update memories that hit chunk boundary
- Lazy evaluation: compute M_{□,t} only when needed
- Cache k_t, v_t, η_t, α_t across chunk

**4. Mixed Precision**
- FP16 for forward pass
- FP32 for memory updates (stability)
- BF16 for gradients

**5. Gradient Checkpointing**
- Recompute Titans activations in backward
- Checkpoint CMS at level boundaries
- Trade compute for memory (essential for long sequences)

**6. Kernel Fusion**
- Fuse M_{□,t-1}(x_t) calls (single kernel for all projections)
- Fuse CMS forward pass (minimize memory movement)
- Custom CUDA kernels for DGD update

**7. Adaptive Chunk Sizing**
- Smaller chunks early in training (more updates)
- Larger chunks later (less forgetting)
- Task-dependent: short for continual learning, long for language modeling

**8. Initialization Strategies**
- Cold start: Xavier/He for M_{□,0}
- Warm start: Copy pre-trained MLP weights (Section 7.3)
- Progressive stacking: Add levels incrementally during training

### 8.4 Connection to Other Works

**TTT (Test-Time Training):**
- TTT = parametric in-context learning (Section 6)
- Hope's Titans = generalization of TTT with self-modification
- TTT uses single objective; Hope uses multiple nested objectives

**Cartridges (Eyuboglu et al. 2025):**
- Different approach to long-context: retrieval + streaming
- Hope: Continual compression at multiple frequencies
- Complementary: Could combine retrieval with Hope's CMS

**Loop Transformers:**
- Depth of computation via looping layers
- NL: Depth via stacking optimization levels
- Hope achieves looping implicitly (memory recovery in CMS)

**Learned Optimizers:**
- Explicitly learn optimizer update rule
- NL: Show standard optimizers are already learned (associative memories)
- Different levels: Learned optimizers in outer loop, NL optimizers in all levels

**Hypernetworks:**
- Generate weights of one network by another
- NL: Special case of knowledge transfer (weight generation, Eq 29-30)
- Hope's Titans: Hypernetwork where memory generates own values

---

## METADATA

```yaml
paper_id: "NL"
equations_extracted: 121
algorithms_extracted: 1
core_architectures: 3
optimizer_variants: 5

implementation_complexity:
  foundations: "Medium (GD variants, associative memory)"
  sequence_models: "Medium (linear attention, DeltaNet)"
  nested_systems: "High (NSAM framework, decomposition)"
  cms: "Medium (multi-frequency updates, chunking)"
  titans: "High (self-referential, chunk-wise training)"
  hope: "Very High (Titans + CMS + all optimizations)"
  m3_optimizer: "Medium (momentum + orthogonalization)"

key_innovations:
  - "Nested Learning paradigm (multi-level optimization)"
  - "Optimizers as associative memories (Adam = optimal L2 regressor)"
  - "Self-referential Titans (memories generate own values)"
  - "Continuum Memory System (multi-frequency persistent memory)"
  - "Delta Gradient Descent (state-dependent weight decay)"
  - "Hope architecture (continual learning without catastrophic forgetting)"
  - "Multi-scale Momentum Muon (CMS applied to optimizer gradients)"

dependencies:
  critical_papers:
    - "TITANS (Behrouz et al. 2025c)"
    - "MIRAS (Behrouz et al. 2025b)"
    - "Linear Transformers (Katharopoulos et al. 2020)"
    - "DeltaNet (Schlag et al. 2021)"
    - "Adam (Kingma & Ba 2014)"
    - "MAML (Finn et al. 2017)"
    - "Muon (Jordan et al. 2024)"

implementation_order:
  1: "Core optimization (GD, DGD, momentum)"
  2: "Sequence models (linear attention, DeltaNet)"
  3: "NSAM framework (nested decomposition)"
  4: "Optimizer decomposition (Adam, M3)"
  5: "CMS (multi-frequency MLPs)"
  6: "Titans (self-referential memories)"
  7: "Hope (full integration)"
  8: "Scaling and optimization"

recommended_starting_point: "Implement DGD (Eq 56-57) and verify it outperforms GD on sequential data, then build CMS (Eq 70-71) as it's simpler than Titans"

compute_requirements:
  training_760M: "Multi-GPU, ~days for 30B tokens"
  training_1.3B: "Multi-GPU, ~weeks for 100B tokens"
  memory_overhead: "1.5× vs Transformer"
  inference_overhead: "1.2× vs Transformer"

testing_priorities:
  1: "Continual learning (CLINC, Banking, CTNL)"
  2: "Long context (NIAH, BABILong)"
  3: "Language modeling (perplexity)"
  4: "Reasoning (ARC, HellaSwag)"
  5: "Optimizer performance (M3 on ImageNet)"
```
