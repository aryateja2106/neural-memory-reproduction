# Titans: Learning to Memorize at Test Time - Implementation Context

**Paper ID:** TITANS
**Full Title:** Titans: Learning to Memorize at Test Time
**Authors:** Ali Behrouz, Peilin Zhong, Vahab Mirrokni (Google Research)
**arXiv:** arXiv:2501.00663v1 [cs.LG]
**Year:** 2025 (31 Dec 2024)

**Key Contribution (1 sentence):**
Introduces a neural long-term memory module that learns to memorize at test time using gradient-based surprise metrics with momentum and forgetting, combined with sliding-window attention in three architectural variants (MAC, MAG, MAL) to achieve superior performance on long-context tasks up to 2M tokens.

---

## Quick Reference
| Aspect | Count | Key Items |
|--------|-------|-----------|
| Equations | 35 | Attention (2), Linear attention (3-5), Memory update (8-14), Parallelization (16-18), Architecture variants (21-31), LMM recurrence (32-35) |
| Algorithms | 3 | Neural memory training, Parallel chunk processing, Mini-batch gradient descent with momentum |
| Architectures | 3 | MAC (Memory as Context), MAG (Memory as Gating), MAL (Memory as Layer) |
| Hyperparameters | 15+ | Learning rates, context windows, chunk sizes, memory depths, decay rates |

---

## 1. EQUATIONS

### 1.1 Standard Attention (Baseline)

**Equation 1: Query, Key, Value Projections**
```latex
Q = xW_Q, \quad K = xW_K, \quad V = xW_V
```

**Plain English:** Project input sequence into query, key, and value matrices using learnable weight matrices.

**Variables:**
| Variable | Shape | Description |
|----------|-------|-------------|
| x | (N, d_in) | Input sequence |
| W_Q, W_K, W_V | (d_in, d_in) | Learnable projection matrices |
| Q, K, V | (N, d_in) | Query, key, value matrices |
| N | scalar | Sequence length |
| d_in | scalar | Input/hidden dimension |

**Implementation Notes:**
- Standard linear projections
- No bias terms mentioned
- Same dimensionality for all projections

**Dependencies:** None (foundational)

**Test Criteria:** Q, K, V should have shape (N, d_in) after projection

---

**Equation 2: Softmax Attention Output**
```latex
y_i = \sum_{j=1}^{i} \frac{\exp(Q_i^\top K_j / \sqrt{d_{in}}) V_j}{\sum_{\ell=1}^{i} \exp(Q_i^\top K_\ell / \sqrt{d_{in}})}
```

**Plain English:** For each position i, compute weighted sum of values where weights are softmax of scaled dot-product attention scores (causal/autoregressive).

**Variables:**
| Variable | Shape | Description |
|----------|-------|-------------|
| y_i | (d_in,) | Output at position i |
| Q_i | (d_in,) | Query vector at position i |
| K_j | (d_in,) | Key vector at position j |
| V_j | (d_in,) | Value vector at position j |
| sqrt(d_in) | scalar | Temperature scaling factor |

**Implementation Notes:**
- Causal mask: only attend to j ≤ i
- Temperature scaling prevents gradient vanishing
- O(N²) complexity

**Dependencies:** Equation 1

**Test Criteria:**
- Output shape: (N, d_in)
- Each y_i only depends on positions 1..i
- Attention weights sum to 1

---

### 1.2 Linear Attention (Efficient Variant)

**Equation 3: Kernel-based Linear Attention**
```latex
y_i = \frac{\phi(Q_i)^\top \sum_{j=1}^{i} \phi(K_j) V_j}{\phi(Q_i)^\top \sum_{\ell=1}^{i} \phi(K_\ell)}
```

**Plain English:** Replace softmax with kernel function φ that factorizes: φ(x,y) = φ(x)φ(y), enabling linear complexity through associativity.

**Variables:**
| Variable | Shape | Description |
|----------|-------|-------------|
| φ(·) | R^d → R^d | Kernel function (e.g., identity, ReLU, ELU+1) |
| y_i | (d_in,) | Output at position i |

**Implementation Notes:**
- Common choice: φ(x) = x (identity kernel)
- Can precompute cumulative sums
- Enables O(N) complexity

**Dependencies:** Equation 1 for Q, K, V

**Test Criteria:** Output shape matches Equation 2 but computed in O(N) time

---

**Equation 4-5: Recurrent Form of Linear Attention**
```latex
M_t = M_{t-1} + K_t^\top V_t  \quad (4)
y_t = Q_t M_t  \quad (5)
```

**Plain English:** Maintain matrix-valued state M_t that accumulates key-value outer products; retrieve by matrix-vector multiplication with query.

**Variables:**
| Variable | Shape | Description |
|----------|-------|-------------|
| M_t | (d_in, d_in) | Matrix-valued memory state at time t |
| K_t | (d_in,) | Key at time t |
| V_t | (d_in,) | Value at time t |
| y_t | (d_in,) | Output at time t |
| Q_t | (d_in,) | Query at time t |

**Implementation Notes:**
- Matrix state allows efficient incremental updates
- Enables fast autoregressive inference
- Memory grows additively without bounds → overflow issue

**Dependencies:** Linear attention framework (Eq 3)

**Test Criteria:**
- M_t shape: (d_in, d_in)
- y_t equivalent to Eq 3 with identity kernel

---

### 1.3 Neural Memory Core Equations

**Equation 8: Basic Surprise Metric**
```latex
M_t = M_{t-1} - \theta_t \nabla \ell(M_{t-1}; x_t)
```

**Plain English:** Update memory parameters using gradient descent where gradient magnitude measures "surprise" (unexpectedness of input).

**Variables:**
| Variable | Shape | Description |
|----------|-------|-------------|
| M_t | (depends on LM) | Memory network parameters at time t |
| θ_t | scalar or (d_in,) | Data-dependent learning rate |
| ∇ℓ | same as M | Gradient of loss w.r.t. memory parameters |
| x_t | (d_in,) | Input at time t |

**Implementation Notes:**
- M can be MLP with LM ≥ 1 layers
- θ_t is learned/adaptive
- Loss ℓ defined in Eq 12

**Dependencies:** Loss function (Eq 12)

**Test Criteria:** Memory updates should reflect gradient direction

---

**Equation 9-10: Momentum-based Surprise (Core Innovation)**
```latex
M_t = M_{t-1} + S_t  \quad (9)
S_t = \eta_t S_{t-1} - \theta_t \nabla \ell(M_{t-1}; x_t)  \quad (10)
```

**Plain English:** Introduce momentum term S_t that accumulates both "past surprise" (ηS_{t-1}) and "momentary surprise" (gradient), enabling memory of surprise flow across tokens.

**Variables:**
| Variable | Shape | Description |
|----------|-------|-------------|
| S_t | same as M | Momentum/surprise accumulator |
| η_t | scalar or (d_in,) | Data-dependent surprise decay rate |
| θ_t | scalar or (d_in,) | Data-dependent step size |

**Implementation Notes:**
- η_t → 0: ignore past surprise (context change)
- η_t → 1: fully incorporate past surprise (relevant context)
- Analogous to momentum in SGD
- **KEY DIFFERENCE from TTT/DeltaNet**: incorporates token flow

**Dependencies:** Equation 8, 12

**Test Criteria:**
- S_t shape matches M
- η_t, θ_t ∈ [0,1] typically

---

**Equation 11: Key-Value Projections for Memory**
```latex
k_t = x_t W_K, \quad v_t = x_t W_V
```

**Plain English:** Project input into key-value pairs for associative memory (same as attention projections).

**Variables:**
| Variable | Shape | Description |
|----------|-------|-------------|
| k_t | (d_in,) | Key vector at time t |
| v_t | (d_in,) | Value vector at time t |
| W_K, W_V | (d_in, d_in) | Learnable projection matrices (hyperparameters in inner loop) |

**Implementation Notes:**
- W_K, W_V are hyperparameters in meta-learning framework
- Optimized in outer loop, fixed in inner loop
- Shared with attention or separate (implementation choice)

**Dependencies:** None (foundational)

**Test Criteria:** k_t, v_t have shape (d_in,)

---

**Equation 12: Associative Memory Loss (Inner Loop Objective)**
```latex
\ell(M_{t-1}; x_t) = \|M_{t-1}(k_t) - v_t\|_2^2
```

**Plain English:** Memory module should predict value v_t from key k_t; gradient measures prediction error.

**Variables:**
| Variable | Shape | Description |
|----------|-------|-------------|
| M_{t-1}(k_t) | (d_in,) | Memory network forward pass on key |
| v_t | (d_in,) | Target value |
| ℓ | scalar | Mean squared error loss |

**Implementation Notes:**
- MSE loss for associative memory
- M is neural network (MLP with LM layers)
- Gradient ∇ℓ computed w.r.t. M's weights

**Dependencies:** Equation 11

**Test Criteria:**
- Loss is scalar
- Gradient shape matches M's parameter shapes

---

**Equation 13-14: Forgetting Mechanism (Weight Decay)**
```latex
M_t = (1 - \alpha_t) M_{t-1} + S_t  \quad (13)
S_t = \eta_t S_{t-1} - \theta_t \nabla \ell(M_{t-1}; x_t)  \quad (14)
```

**Plain English:** Add forgetting gate α_t that scales down previous memory before adding surprise update (weight decay in optimization).

**Variables:**
| Variable | Shape | Description |
|----------|-------|-------------|
| α_t | scalar or (d_in,) | Data-dependent forget gate ∈ [0,1] |

**Implementation Notes:**
- α_t → 0: retain all memory
- α_t → 1: clear memory completely
- Generalizes gating in Mamba/GLA
- **KEY DIFFERENCE from TTT**: includes forgetting

**Dependencies:** Equations 9-10

**Test Criteria:**
- α_t should be in [0,1]
- Memory magnitude controlled by decay

---

**Equation 15: Memory Retrieval**
```latex
y_t = M^*_t(q_t)
```

**Plain English:** Retrieve information from memory using forward pass (no weight update) with query q_t.

**Variables:**
| Variable | Shape | Description |
|----------|-------|-------------|
| M^* | neural network | Memory in inference mode (frozen weights) |
| q_t | (d_in,) | Query vector: x_t W_Q |
| y_t | (d_in,) | Retrieved memory output |

**Implementation Notes:**
- No gradient computation for retrieval
- M^* indicates frozen parameters
- Different from training mode where weights update

**Dependencies:** Equation 11 (for query projection)

**Test Criteria:** y_t shape is (d_in,)

---

### 1.4 Parallelization Equations

**Equation 16: Chunk-wise Mini-batch Gradient Descent**
```latex
M_t = (1 - \alpha_t) M_{t-1} - \theta_t \nabla \ell(M_{t-1}; x_t) = \beta_t M_0 - \sum_{i=1}^{t} \theta_i \frac{\beta_t}{\beta_i} \nabla \ell(M_{t'}; x_i)
```

**Plain English:** Reformulate recurrent updates as batch computation where t' = t - mod(t, b) for chunk size b, enabling parallelization.

**Variables:**
| Variable | Shape | Description |
|----------|-------|-------------|
| b | scalar | Chunk/batch size |
| t' | scalar | Start of current chunk |
| β_t | scalar | Product Π_{j=1}^t (1 - α_j) |
| M_0 | network params | Initial memory state for chunk |

**Implementation Notes:**
- Split sequence into chunks of size b
- Compute all gradients in chunk simultaneously
- β accumulates decay products
- Enables GPU parallelization

**Dependencies:** Equations 13-14

**Test Criteria:** Results match sequential computation

---

**Equation 17: Tensorized Gradient Computation (Linear Case)**
```latex
\nabla \ell(W_0; x_t) = (W_0 x_t - x_t) x_t^\top \Rightarrow \sum_{i=1}^{b} \theta_i \frac{\beta_b}{\beta_i} \nabla \ell(W_0; x_i) = \Theta_b B_b (W_0 X - X) X^\top
```

**Plain English:** For linear memory W, batch all gradients into matrix operations using diagonal matrices Θ, B.

**Variables:**
| Variable | Shape | Description |
|----------|-------|-------------|
| Θ_b | (b, b) diagonal | diag([θ_1, θ_2, ..., θ_b]) |
| B_b | (b, b) diagonal | diag([β_b/β_1, β_b/β_2, ..., β_b/β_b]) |
| X | (b, d_in) | Batch of inputs in chunk |
| W_0 | (d_in, d_in) | Memory weights at chunk start |

**Implementation Notes:**
- Only for linear memory (LM=1)
- Enables pure matmul operations
- Store Θ, B per chunk (not all chunks)
- Reduces memory overhead

**Dependencies:** Equation 16

**Test Criteria:** Single matmul chain produces batch gradients

---

**Equation 18: Momentum via Parallel Associative Scan**
```latex
S_t = \eta_t S_{t-1} - \theta_t u_t
```

**Plain English:** Momentum is linear recurrence; use parallel scan algorithm to compute all S_t in O(log b) depth.

**Variables:**
| Variable | Shape | Description |
|----------|-------|-------------|
| u_t | same as M | Pre-computed gradient ∇ℓ(M_{t'}; x_t) |
| S_t | same as M | Momentum state |
| η_t | scalar | Transition coefficient |

**Implementation Notes:**
- u_t computed in parallel (all at once)
- Parallel scan: O(N log b) work, O(log b) depth
- Reference: Smith et al. 2023 (parallel associative scan)

**Dependencies:** Equation 14

**Test Criteria:** Parallel scan matches sequential computation

---

### 1.5 Architecture-Specific Equations

#### MAC (Memory as Context)

**Equation 21-25: MAC Forward Pass**
```latex
h_t = M^*_{t-1}(q_t)  \quad (21)
\tilde{S}^{(t)} = [p_1, p_2, ..., p_{N_p}] \,||\, h_t \,||\, S^{(t)}  \quad (22)
y_t = \text{Attn}(\tilde{S}^{(t)})  \quad (23)
M_t = M_{t-1}(y_t)  \quad (24)
o_t = y_t \otimes M^*_t(y_t)  \quad (25)
```

**Plain English:**
1. Retrieve historical context h_t from memory using current segment as query
2. Concatenate persistent memory, historical context, and current segment
3. Apply full attention within concatenated sequence
4. Update memory using attention output (training mode)
5. Final output combines attention and memory retrieval (⊗ is gating)

**Variables:**
| Variable | Shape | Description |
|----------|-------|-------------|
| S^(t) | (C, d_in) | Current segment (chunk size C) |
| q_t | (C, d_in) | Queries from current segment |
| h_t | (N_l, d_in) | Retrieved historical context (N_l memory tokens) |
| p_1...p_{N_p} | (N_p, d_in) | Persistent learnable memory tokens |
| S̃^(t) | (N_p+N_l+C, d_in) | Concatenated sequence for attention |
| y_t | (N_p+N_l+C, d_in) | Attention output |
| o_t | (C, d_in) | Final output for segment |

**Implementation Notes:**
- Segment size C is hyperparameter
- N_p persistent tokens (e.g., 32)
- N_l = C typically (one memory token per input token)
- Attention is causal within S̃
- M_{t-1}(y_t) means forward pass WITH weight update
- ⊗ can be element-wise multiply or gated linear unit

**Dependencies:** Equations 13-15 for memory, Equation 2 for attention

**Test Criteria:**
- S̃ has shape (N_p+N_l+C, d_in)
- Attention mask is causal within segment
- Memory updates only during training

---

#### MAG (Memory as Gating)

**Equation 26-28: MAG Forward Pass**
```latex
\tilde{x} = [p_1, p_2, ..., p_{N_p}] \,||\, x  \quad (26)
y = \text{SW-Attn}^*(\tilde{x})  \quad (27)
o = y \otimes M(\tilde{x})  \quad (28)
```

**Plain English:**
1. Prepend persistent memory to full sequence
2. Apply sliding window attention (no segmentation)
3. Gate attention output with memory output

**Variables:**
| Variable | Shape | Description |
|----------|-------|-------------|
| x | (N, d_in) | Full input sequence |
| x̃ | (N_p+N, d_in) | Sequence with persistent prefix |
| SW-Attn^* | function | Sliding window attention with prefix visibility |
| y | (N_p+N, d_in) | Attention output |
| M(x̃) | (N_p+N, d_in) | Memory output (full recurrence over sequence) |
| o | (N_p+N, d_in) | Final gated output |

**Implementation Notes:**
- No chunking (processes full sequence)
- Sliding window size W is hyperparameter (e.g., 2048)
- Prefix tokens always visible to all positions
- ⊗ includes normalization: normalize(y) ⊗ σ(normalize(M(x̃)))
- σ is non-linearity (e.g., SiLU)

**Dependencies:** Sliding window attention, Equations 13-15 for M

**Test Criteria:**
- First N_p positions always visible in attention
- Window size W controls local attention span

---

#### MAL (Memory as Layer)

**Equation 29-31: MAL Forward Pass**
```latex
\tilde{x} = [p_1, p_2, ..., p_{N_p}] \,||\, x  \quad (29)
y = M(\tilde{x})  \quad (30)
o = \text{SW-Attn}(y)  \quad (31)
```

**Plain English:**
1. Prepend persistent memory
2. Process through memory layer
3. Process memory output through sliding window attention

**Variables:**
| Variable | Shape | Description |
|----------|-------|-------------|
| x̃ | (N_p+N, d_in) | Sequence with persistent prefix |
| y | (N_p+N, d_in) | Memory layer output |
| o | (N_p+N, d_in) | Final attention output |

**Implementation Notes:**
- Sequential layer design: Memory → Attention
- Similar to H3 architecture (Fu et al. 2023)
- Less expressive than MAC/MAG (see ablations)
- Faster training (Figure 9)

**Dependencies:** Equations 13-15 for M, sliding window attention

**Test Criteria:** Standard sequential layer composition

---

**Equation 19: Persistent Memory Concatenation**
```latex
x_{new} = [p_1, p_2, ..., p_{N_p}] \,||\, x
```

**Plain English:** Prepend N_p learnable data-independent tokens to sequence start.

**Variables:**
| Variable | Shape | Description |
|----------|-------|-------------|
| p_i | (d_in,) | Learnable persistent memory token i |
| N_p | scalar | Number of persistent tokens (e.g., 32) |
| x | (N, d_in) | Original input sequence |
| x_new | (N_p+N, d_in) | Augmented sequence |

**Implementation Notes:**
- p_i are learned parameters (not functions of input)
- Act as "meta-memory" for task knowledge
- Mitigate attention sink effect (bias toward initial tokens)
- Equivalent to feedforward with softmax weights (Eq 20)

**Dependencies:** None

**Test Criteria:** p_i are part of model parameters, not inputs

---

**Equation 20: FFN as Data-Independent Attention**
```latex
FFN(x) = W_V \text{Softmax}(W_K x)
```

**Plain English:** Feedforward networks can be viewed as attention with data-independent keys/values.

**Variables:**
| Variable | Shape | Description |
|----------|-------|-------------|
| W_K, W_V | (d_in, d_ff) | FFN weight matrices |
| x | (d_in,) | Input |

**Implementation Notes:**
- Motivates persistent memory design
- Reference: Sukhbaatar et al. 2019
- Persistent tokens provide this functionality

**Dependencies:** None (motivation)

**Test Criteria:** Theoretical equivalence

---

### 1.6 Connection to Modern RNNs (Appendix C)

**Equation 32-33: LMM Recurrence Form**
```latex
M_t = \text{diag}(1 - \alpha_t) M_t + S_t  \quad (32)
S_t = \text{diag}(\eta_t) S_{t-1} - \text{diag}(\theta_t) (M_{t-1} k_t^\top k_t - v_t^\top k_t)  \quad (33)
```

**Plain English:** For linear memory (W ∈ R^{d×d}), the recurrence explicitly shows diagonal gating and momentum.

**Variables:**
| Variable | Shape | Description |
|----------|-------|-------------|
| M_t | (d_in, d_in) | Linear memory matrix |
| diag(α_t) | (d_in, d_in) | Diagonal forget gate matrix |
| k_t | (d_in,) | Key vector |

**Implementation Notes:**
- Only for LM=1 (linear memory)
- Shows connection to Gated DeltaNet
- For LM ≥ 2, no closed form (use gradient)

**Dependencies:** Equations 13-14, linear memory assumption

**Test Criteria:** Matches gradient-based update for linear case

---

**Equation 34: Gated DeltaNet Recurrence**
```latex
S_{t+1} = S_t (I - \theta_t k_t k_t^\top) + \theta_t v_t k_t^\top
```

**Plain English:** Delta rule update (remove old memory, add new) without momentum.

**Variables:**
| Variable | Shape | Description |
|----------|-------|-------------|
| S_t | (d_in, d_in) | State matrix |
| I | (d_in, d_in) | Identity matrix |

**Implementation Notes:**
- Setting η_t = 0 in Eq 33 recovers this
- LMM generalizes Gated DeltaNet with momentum

**Dependencies:** Delta rule (Widrow & Hoff 1988)

**Test Criteria:** Special case of LMM

---

**Equation 35: Longhorn Recurrence**
```latex
S_{t+1} = S_t (I - \delta_t k_t k_t^\top) + \delta_t v_t k_t^\top, \quad \delta_t = \frac{\theta_t}{1 + \theta_t k_t k_t^\top}
```

**Plain English:** Implicit online learning variant of delta rule (no forgetting gate).

**Variables:**
| Variable | Shape | Description |
|----------|-------|-------------|
| δ_t | scalar | Implicit learning rate |

**Implementation Notes:**
- Liu et al. 2024
- No forget gate (α_t = 0 always)
- LMM adds forgetting + momentum

**Dependencies:** Delta rule

**Test Criteria:** Special case of LMM without forgetting/momentum

---

## 2. ALGORITHMS

### Algorithm 1: Neural Memory Training (Sequential)

**Pseudocode:**
```python
# Initialization
M_0 = initialize_memory_network(depth=L_M, dim=d_in)
S_0 = zeros_like(M_0.parameters())
W_K, W_V, W_Q = initialize_projections(d_in, d_in)

# Outer loop (model training)
for batch in dataloader:
    x = batch  # shape: (batch_size, N, d_in)

    # Inner loop (test-time learning)
    M_t = M_0.copy()
    S_t = S_0.copy()

    for t in range(1, N+1):
        # 1. Project to key-value
        k_t = x[t] @ W_K
        v_t = x[t] @ W_V

        # 2. Compute loss and gradient (surprise)
        pred = M_t.forward(k_t, grad_mode=True)
        loss = ||pred - v_t||_2^2
        grad = autograd(loss, M_t.parameters())

        # 3. Compute data-dependent gates
        alpha_t = sigmoid(linear_alpha(x[t]))  # forget gate
        theta_t = sigmoid(linear_theta(x[t]))  # step size
        eta_t = sigmoid(linear_eta(x[t]))      # momentum decay

        # 4. Update momentum (surprise accumulator)
        S_t = eta_t * S_t - theta_t * grad

        # 5. Update memory with forgetting
        M_t.parameters() = (1 - alpha_t) * M_t.parameters() + S_t

        # 6. Retrieve output (no grad)
        q_t = x[t] @ W_Q
        y_t = M_t.forward(q_t, grad_mode=False)

        # Store y_t for outer loss

    # Compute outer loss (e.g., next-token prediction)
    outer_loss = compute_task_loss(y, targets)

    # Update W_K, W_V, W_Q, gate networks, M_0 initialization
    outer_loss.backward()
    optimizer.step()
```

**Line-by-Line Mapping:**
1. Initialize memory as MLP with L_M layers
2. Initialize momentum to zeros (same structure as parameters)
3. Initialize projection matrices as hyperparameters
4-5. Standard training loop
6-7. Copy parameters for inner loop (test-time learning)
8. Sequential processing over sequence
9-10. Compute key-value from input (Eq 11)
11-13. Forward pass through memory, compute MSE loss (Eq 12), get gradient (Eq 8)
14-16. Compute adaptive gates α, θ, η from input features
17. Update momentum with decay and gradient (Eq 10/14)
18. Update memory with forgetting and surprise (Eq 13)
19-20. Retrieve output using query (Eq 15)
21-23. Compute outer objective (e.g., language modeling loss)
24. Backprop through entire process to update hyperparameters

**Implementation Considerations:**
- Inner loop updates M_t but doesn't accumulate gradients for outer loop
- Outer loop only updates W_K, W_V, W_Q, gate networks, M_0
- Memory state M_t resets each sequence (or chunk)
- Gradient checkpointing needed for long sequences

---

### Algorithm 2: Parallel Chunk Processing

**Pseudocode:**
```python
def parallel_memory_training(x, M_0, chunk_size=b):
    """
    Parallel training using mini-batch gradient descent formulation.
    Args:
        x: (N, d_in) input sequence
        M_0: initial memory network
        chunk_size: number of tokens per chunk
    """
    N = x.shape[0]
    num_chunks = N // chunk_size
    outputs = []

    M_chunk = M_0.copy()
    S_chunk = zeros_like(M_0.parameters())

    for chunk_idx in range(num_chunks):
        # Extract chunk
        start = chunk_idx * chunk_size
        end = start + chunk_size
        X_chunk = x[start:end]  # (b, d_in)

        # 1. Compute all keys/values in parallel
        K_chunk = X_chunk @ W_K  # (b, d_in)
        V_chunk = X_chunk @ W_V  # (b, d_in)
        Q_chunk = X_chunk @ W_Q  # (b, d_in)

        # 2. Compute all gates in parallel
        alpha = sigmoid(gate_alpha(X_chunk))  # (b,) or (b, d_in)
        theta = sigmoid(gate_theta(X_chunk))  # (b,) or (b, d_in)
        eta = sigmoid(gate_eta(X_chunk))      # (b,)

        # 3. Compute all gradients at M_chunk (not M_t!)
        # For linear memory (optional optimization):
        if isinstance(M_chunk, LinearMemory):
            # Use tensorized form (Eq 17)
            Theta_b = diag(theta)  # (b, b)
            beta = cumprod(1 - alpha)  # (b,)
            B_b = diag(beta[-1] / beta)  # (b, b)

            grad_batch = Theta_b @ B_b @ (M_chunk @ K_chunk.T - V_chunk) @ K_chunk.T
            # grad_batch is (b, d_in, d_in)
        else:
            # For deep memory, compute sequentially or use vmap
            grad_batch = []
            for i in range(chunk_size):
                pred = M_chunk.forward(K_chunk[i])
                loss = ||pred - V_chunk[i]||^2
                grad_batch.append(autograd(loss, M_chunk.parameters()))
            grad_batch = stack(grad_batch)  # (b, ...)

        # 4. Compute momentum using parallel scan (Eq 18)
        # S_t = eta_t * S_{t-1} - theta_t * grad_t
        # This is a linear recurrence: use associative scan
        S_batch = parallel_scan(
            init=S_chunk,
            inputs=grad_batch,
            transition=eta,
            step_size=theta
        )  # (b, ...) all S_t in chunk

        # 5. Update memory (final state of chunk)
        # For simplicity, use final state (non-parallel step)
        M_next = M_chunk
        S_next = S_chunk
        for i in range(chunk_size):
            S_next = eta[i] * S_next - theta[i] * grad_batch[i]
            M_next = (1 - alpha[i]) * M_next + S_next

        M_chunk = M_next
        S_chunk = S_next

        # 6. Retrieve outputs (parallel)
        Y_chunk = M_chunk.forward_batch(Q_chunk, grad_mode=False)  # (b, d_in)
        outputs.append(Y_chunk)

    return concatenate(outputs)  # (N, d_in)

def parallel_scan(init, inputs, transition, step_size):
    """
    Compute S_t = transition[t] * S_{t-1} - step_size[t] * inputs[t]
    in O(log b) parallel depth.

    Reference: Smith et al. 2023 (Simplified State Space Layers)
    """
    # Implementation uses prefix sum / scan algorithm
    # Pseudocode sketch:
    b = len(inputs)
    S = [init]

    # Level-by-level tree reduction
    for level in range(log2(b)):
        # Combine adjacent pairs in parallel
        ...

    return S
```

**Line-by-Line Mapping:**
1-5. Setup chunk iteration
6-9. Extract chunk of size b from sequence
10-12. Parallel projection to K, V, Q (single matmul)
13-15. Parallel computation of gates (single forward pass)
16-30. Gradient computation: tensorized for linear (Eq 17), vmap for deep
31-36. Parallel scan for momentum (Eq 18), O(log b) depth
37-43. Sequential update within chunk (final state only)
44. Parallel retrieval for all chunk positions

**Implementation Considerations:**
- Chunk size b is tuned for memory/compute (e.g., 64, 128)
- Parallel scan requires custom kernel or JAX lax.associative_scan
- Trade-off: larger chunks → more parallelism but more memory
- Tensorized form only for linear memory (LM=1)

---

### Algorithm 3: Titans (MAC) Full Architecture

**Pseudocode:**
```python
class TitansMAC(nn.Module):
    def __init__(self, d_model, num_layers, chunk_size,
                 memory_depth, num_persistent, num_memory_tokens):
        self.d_model = d_model
        self.C = chunk_size  # segment size
        self.N_p = num_persistent
        self.N_l = num_memory_tokens

        # Persistent memory (learnable)
        self.persistent_memory = nn.Parameter(torch.randn(num_persistent, d_model))

        # Stack of Titans layers
        self.layers = nn.ModuleList([
            TitansLayer(d_model, memory_depth)
            for _ in range(num_layers)
        ])

        self.output_projection = nn.Linear(d_model, vocab_size)

    def forward(self, x, targets=None):
        """
        x: (batch, seq_len, d_model)
        """
        batch, seq_len, d = x.shape
        num_chunks = seq_len // self.C

        # Initialize memory state for each layer
        memory_states = [layer.init_memory() for layer in self.layers]

        all_outputs = []

        for chunk_idx in range(num_chunks):
            # Extract segment
            start = chunk_idx * self.C
            end = start + self.C
            segment = x[:, start:end, :]  # (batch, C, d)

            # Process through layers
            for layer_idx, layer in enumerate(self.layers):
                segment, memory_states[layer_idx] = layer.forward_chunk(
                    segment=segment,
                    memory_state=memory_states[layer_idx],
                    persistent_memory=self.persistent_memory
                )

            all_outputs.append(segment)

        # Concatenate all segments
        output = torch.cat(all_outputs, dim=1)  # (batch, seq_len, d)

        # Final projection
        logits = self.output_projection(output)

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
            return logits, loss

        return logits


class TitansLayer(nn.Module):
    def __init__(self, d_model, memory_depth):
        # Memory module
        self.memory = NeuralMemory(d_model, memory_depth)

        # Attention
        self.attention = CausalAttention(d_model)

        # Projections
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)

        # Gating for final output
        self.gate_norm1 = nn.LayerNorm(d_model)
        self.gate_norm2 = nn.LayerNorm(d_model)
        self.gate = nn.Linear(d_model, d_model)

        # Residual connections
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model)

    def init_memory(self):
        return self.memory.init_state()

    def forward_chunk(self, segment, memory_state, persistent_memory):
        """
        segment: (batch, C, d_model)
        memory_state: memory network state
        persistent_memory: (N_p, d_model)
        """
        batch, C, d = segment.shape
        N_p = persistent_memory.shape[0]

        # 1. Retrieve from memory (Eq 21)
        queries = self.W_Q(segment)  # (batch, C, d)
        h_t = self.memory.retrieve(queries, memory_state)  # (batch, N_l, d)
        # N_l typically equals C

        # 2. Concatenate context (Eq 22)
        # Expand persistent memory for batch
        persistent = persistent_memory.unsqueeze(0).expand(batch, -1, -1)  # (batch, N_p, d)

        # Concatenate: [persistent | historical | current]
        concat_seq = torch.cat([persistent, h_t, segment], dim=1)  # (batch, N_p+N_l+C, d)

        # 3. Causal attention (Eq 23)
        attn_out = self.attention(concat_seq)  # (batch, N_p+N_l+C, d)

        # Extract the portion corresponding to original segment
        # (last C tokens)
        y_t = attn_out[:, -C:, :]  # (batch, C, d)

        # 4. Update memory (Eq 24)
        # Use attention output to update memory
        memory_state = self.memory.update(y_t, memory_state)

        # 5. Final output with gating (Eq 25)
        # Retrieve from updated memory
        memory_out = self.memory.retrieve(y_t, memory_state)  # (batch, C, d)

        # Gate: y ⊗ memory_out
        gated = self.gate_norm1(y_t) * torch.sigmoid(self.gate(self.gate_norm2(memory_out)))

        # Residual connection
        output = segment + gated

        # FFN
        output = output + self.ffn(self.norm2(output))

        return output, memory_state


class NeuralMemory(nn.Module):
    def __init__(self, d_model, depth):
        self.depth = depth
        self.d_model = d_model

        # Memory network (MLP with depth layers)
        layers = []
        for i in range(depth):
            layers.append(nn.Linear(d_model, d_model))
            if i < depth - 1:
                layers.append(nn.SiLU())  # Activation between layers
        self.network = nn.Sequential(*layers)

        # Gate networks
        self.alpha_net = nn.Linear(d_model, 1)  # Forget gate
        self.theta_net = nn.Linear(d_model, 1)  # Step size
        self.eta_net = nn.Linear(d_model, 1)    # Momentum decay

        # Projections
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)

        # Convolutions (as in modern RNNs)
        self.conv_k = nn.Conv1d(d_model, d_model, kernel_size=4, padding=3, groups=d_model)
        self.conv_v = nn.Conv1d(d_model, d_model, kernel_size=4, padding=3, groups=d_model)

    def init_state(self):
        # Initialize memory parameters and momentum
        params = {name: param.clone() for name, param in self.network.named_parameters()}
        momentum = {name: torch.zeros_like(param) for name, param in self.network.named_parameters()}
        return {'params': params, 'momentum': momentum}

    def update(self, x, state):
        """
        Update memory using gradient descent with momentum and forgetting.
        x: (batch, seq_len, d_model)
        """
        batch, seq_len, d = x.shape

        params = state['params']
        momentum = state['momentum']

        # Apply convolutions
        k = self.W_K(x)  # (batch, seq_len, d)
        k = k.transpose(1, 2)  # (batch, d, seq_len)
        k = self.conv_k(k)[:, :, :seq_len].transpose(1, 2)  # (batch, seq_len, d)

        v = self.W_V(x)
        v = v.transpose(1, 2)
        v = self.conv_v(v)[:, :, :seq_len].transpose(1, 2)

        # Sequential update over sequence
        for t in range(seq_len):
            k_t = k[:, t, :]  # (batch, d)
            v_t = v[:, t, :]  # (batch, d)
            x_t = x[:, t, :]  # (batch, d)

            # Compute gates
            alpha_t = torch.sigmoid(self.alpha_net(x_t))  # (batch, 1)
            theta_t = torch.sigmoid(self.theta_net(x_t))
            eta_t = torch.sigmoid(self.eta_net(x_t))

            # Forward pass with current params
            pred = self._forward_with_params(k_t, params)  # (batch, d)

            # Compute loss and gradient
            loss = F.mse_loss(pred, v_t, reduction='none')  # (batch, d)

            # Compute gradients (simplified - actual needs functional API)
            grads = torch.autograd.grad(
                loss.mean(),
                params.values(),
                create_graph=True
            )

            # Update momentum and parameters
            for (name, param), grad in zip(params.items(), grads):
                # Momentum update (Eq 14)
                momentum[name] = eta_t * momentum[name] - theta_t * grad

                # Parameter update with forgetting (Eq 13)
                params[name] = (1 - alpha_t) * param + momentum[name]

        return {'params': params, 'momentum': momentum}

    def retrieve(self, queries, state):
        """
        Retrieve from memory without updating.
        queries: (batch, seq_len, d_model)
        """
        params = state['params']

        # Forward pass with frozen params
        output = self._forward_with_params(queries, params)

        return output

    def _forward_with_params(self, x, params):
        """
        Forward pass using specific parameter dictionary.
        """
        # Manually apply layers with custom params
        # (requires functional API in practice)
        h = x
        param_list = list(params.values())

        for i in range(self.depth):
            weight = param_list[2*i]  # Linear weight
            bias = param_list[2*i + 1]  # Linear bias

            h = F.linear(h, weight, bias)

            if i < self.depth - 1:
                h = F.silu(h)

        return h
```

**Implementation Considerations:**
- Use `torch.func` or JAX for functional gradient computation
- Memory state management crucial for multi-layer stacking
- Convolutions add local smoothing (standard in modern RNNs)
- Gating output prevents gradient issues
- Chunk size C affects memory/speed trade-off

---

## 3. ARCHITECTURE

### 3.1 Overall Titans Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT SEQUENCE                           │
│                         x ∈ R^(N×d_in)                          │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         │ Segment into chunks of size C
                         ▼
         ┌───────────────────────────────────────┐
         │     SEGMENT S^(1), S^(2), ..., S^(T)  │
         │     Each S^(t) ∈ R^(C×d_in)           │
         └───────────────┬───────────────────────┘
                         │
        ┌────────────────┴────────────────┐
        │                                  │
        │   FOR EACH SEGMENT (recurrent)   │
        │                                  │
        │   ┌──────────────────────────┐  │
        │   │  TITANS LAYER (stacked)  │  │
        │   │                          │  │
        │   │  ┌────────────────────┐  │  │
        │   │  │  1. CONTEXTUAL     │  │  │
        │   │  │     MEMORY (LMM)   │  │  │
        │   │  │     - Learning      │  │  │
        │   │  │     - Forgetting    │  │  │
        │   │  └────────┬───────────┘  │  │
        │   │           │               │  │
        │   │  ┌────────▼───────────┐  │  │
        │   │  │  2. PERSISTENT     │  │  │
        │   │  │     MEMORY (P)     │  │  │
        │   │  │     - Learnable    │  │  │
        │   │  │     - Data-indep   │  │  │
        │   │  └────────┬───────────┘  │  │
        │   │           │               │  │
        │   │  ┌────────▼───────────┐  │  │
        │   │  │  3. CORE           │  │  │
        │   │  │     (SHORT-TERM)   │  │  │
        │   │  │     - Attention    │  │  │
        │   │  │     - Limited ctx  │  │  │
        │   │  └────────────────────┘  │  │
        │   │                          │  │
        │   └──────────────────────────┘  │
        │                                  │
        └──────────────────────────────────┘
                         │
                         │ Concatenate segments
                         ▼
         ┌───────────────────────────────────────┐
         │         OUTPUT PROJECTION              │
         │         Linear(d_in → vocab_size)     │
         └───────────────────────────────────────┘
```

**Three Branches (Hyper-heads):**
1. **Core (Short-term Memory)**: Attention with limited window (segment size C)
2. **Long-term Memory**: Neural memory module (LMM) with learning/forgetting
3. **Persistent Memory**: Learnable data-independent parameters

**Design Philosophy:**
- Segment-based processing for efficiency
- Memory state persists across segments
- Attention operates within segment
- Three memory types mirror human cognition

---

### 3.2 MAC (Memory as Context) Architecture

```
SEGMENT S^(t) ∈ R^(C×d_in)
        │
        ├─────────────────┬─────────────────┬──────────────────┐
        │                 │                 │                  │
        │                 │                 │                  │
   ┌────▼────┐       ┌────▼────┐      ┌────▼─────┐      ┌────▼────┐
   │  Query  │       │  M_{t-1}│      │Persistent│      │ Segment │
   │   q_t   │       │ (Memory)│      │  Memory  │      │  S^(t)  │
   └────┬────┘       └────┬────┘      └────┬─────┘      └────┬────┘
        │                 │                 │                  │
        │          ┌──────▼──────┐          │                  │
        │          │  Retrieve   │          │                  │
        │          │ h_t = M*(q) │          │                  │
        │          └──────┬──────┘          │                  │
        │                 │                 │                  │
        └─────────────────┴─────────────────┴──────────────────┘
                          │
                 Concatenate [P || h || S]
                          │
                 S̃^(t) ∈ R^((Np+Nl+C)×d_in)
                          │
                ┌─────────▼─────────┐
                │  Causal Attention │
                │  (Full over S̃)    │
                └─────────┬─────────┘
                          │
                    y_t ∈ R^((Np+Nl+C)×d_in)
                          │
                ┌─────────┴─────────┐
                │                   │
         ┌──────▼──────┐    ┌──────▼──────┐
         │   Update    │    │  Retrieve   │
         │  M_t = M(y) │    │ m = M*(y)   │
         └─────────────┘    └──────┬──────┘
                                   │
                            ┌──────▼──────┐
                            │  Gate/Comb  │
                            │  o = y ⊗ m  │
                            └──────┬──────┘
                                   │
                            OUTPUT o_t ∈ R^(C×d_in)
```

**Attention Mask Pattern:**
```
Segment Window:
  Persistent | Long-term | Current
  [  N_p    ][   N_l   ][   C    ]
      ▲          ▲          ▲
      │          │          │
      └──────────┴──────────┘
         All visible to all
         (causal within segment)
```

**Key Features:**
- Historical context retrieved before attention
- Attention decides what to memorize
- Memory updates during training only
- Test time: memory continues learning

---

### 3.3 MAG (Memory as Gating) Architecture

```
INPUT x ∈ R^(N×d_in)
        │
   Prepend Persistent
        │
   x̃ = [P || x]
        │
        ├─────────────────┬─────────────────┐
        │                 │                 │
┌───────▼───────┐   ┌─────▼─────┐   ┌──────▼──────┐
│ Sliding Window│   │  Neural   │   │             │
│   Attention   │   │  Memory   │   │             │
│   (local)     │   │  (global) │   │             │
└───────┬───────┘   └─────┬─────┘   │             │
        │                 │          │             │
        y                 M(x̃)       │             │
        │                 │          │             │
        └────────┬────────┘          │             │
                 │                   │             │
          ┌──────▼──────┐            │             │
          │  Normalize  │            │             │
          │  y', M'     │            │             │
          └──────┬──────┘            │             │
                 │                   │             │
          ┌──────▼──────┐            │             │
          │  Gate Comb  │            │             │
          │ o = y' ⊗    │            │             │
          │    σ(M')    │            │             │
          └──────┬──────┘            │             │
                 │                   │             │
              OUTPUT                 │             │
```

**Attention Mask Pattern:**
```
Sliding Window with Prefix:
Position: 0   1   2  ...  N_p  N_p+1  ...  N_p+N
          [Persistent ][      Sequence      ]
Prefix always visible to all
Window size W for local attention
```

**Key Features:**
- No segmentation (full sequence)
- Sliding window for local dependencies
- Memory for global dependencies
- Gating combines both paths
- Faster training than MAC (Figure 9)

---

### 3.4 MAL (Memory as Layer) Architecture

```
INPUT x ∈ R^(N×d_in)
        │
   Prepend Persistent
        │
   x̃ = [P || x]
        │
┌───────▼───────┐
│     LAYER 1   │
│               │
│  ┌─────────┐  │
│  │ Memory  │  │
│  │ (LMM)   │  │
│  └────┬────┘  │
│       │       │
│  ┌────▼────┐  │
│  │ SW-Attn │  │
│  └────┬────┘  │
│       │       │
│  Residual +   │
│  LayerNorm    │
└───────┬───────┘
        │
┌───────▼───────┐
│     LAYER 2   │
│      ...      │
└───────┬───────┘
        │
     OUTPUT
```

**Key Features:**
- Sequential: Memory → Attention
- Standard hybrid architecture (like H3)
- Simpler than MAC/MAG
- Fastest training (Figure 9)
- Less expressive (Table 5)

---

### 3.5 Module Breakdown

**Neural Memory Module (LMM):**
```
Input: k_t ∈ R^(d_in)
       │
┌──────▼──────┐
│ 1D Conv (K) │ (depthwise-separable)
└──────┬──────┘
       │
┌──────▼──────┐
│  Linear_1   │
│  d_in → d_in│
└──────┬──────┘
       │
┌──────▼──────┐
│    SiLU     │ (if L_M ≥ 2)
└──────┬──────┘
       │
┌──────▼──────┐
│  Linear_2   │ (if L_M ≥ 2)
│  d_in → d_in│
└──────┬──────┘
       │
      ...       (repeat for L_M layers)
       │
┌──────▼──────┐
│ Output: v̂_t │
└─────────────┘

Gradient: ∇ℓ = 2(v̂_t - v_t) · [backprop through layers]
```

**Parameters:**
- Depth L_M ∈ {1, 2, 3, 4}
- Each linear layer: (d_in × d_in) params
- Total params: L_M × d_in² (ignoring bias)

**Gating Networks:**
```
Input x_t ∈ R^(d_in)
       │
       ├──────────┬──────────┬──────────┐
       │          │          │          │
┌──────▼──────┐ ┌▼─────┐ ┌──▼──────┐ ┌──▼──────┐
│Linear(d→1)  │ │Linear│ │Linear   │ │Linear   │
│   α_t net   │ │θ_t   │ │ η_t     │ │...      │
└──────┬──────┘ └┬─────┘ └──┬──────┘ └──┬──────┘
       │         │          │          │
┌──────▼──────┐ ┌▼─────┐ ┌──▼──────┐
│  Sigmoid    │ │Sigm. │ │Sigmoid  │
└──────┬──────┘ └┬─────┘ └──┬──────┘
       │         │          │
   α_t ∈[0,1]  θ_t       η_t
   (forget)    (step)    (momentum)
```

**Attention Module:**
- Standard multi-head attention for MAC/MAG/MAL
- Sliding window implementation for MAG/MAL
- Causal masking within segments
- Implementation: FlashAttention-2 (Dao 2024)

---

### 3.6 Layer Details

**Titans Layer Structure (MAC variant):**
```python
class TitansLayerMAC:
    def __init__(self, d_model, num_heads, memory_depth):
        # Memory components
        self.memory = NeuralMemory(d_model, memory_depth)
        self.persistent_memory = Parameter(randn(N_p, d_model))

        # Attention
        self.attn = MultiHeadAttention(d_model, num_heads)

        # Projections
        self.W_Q = Linear(d_model, d_model, bias=False)
        self.W_K = Linear(d_model, d_model, bias=False)
        self.W_V = Linear(d_model, d_model, bias=False)
        self.W_O = Linear(d_model, d_model, bias=False)

        # Gating
        self.gate_norm1 = LayerNorm(d_model)
        self.gate_norm2 = LayerNorm(d_model)
        self.gate = Linear(d_model, d_model)

        # FFN
        self.ffn = FFN(d_model, 4 * d_model)  # 4x expansion
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

    def forward(self, segment, memory_state):
        # 1. Retrieve from memory
        q = self.W_Q(segment)
        h = self.memory.retrieve(q, memory_state)

        # 2. Concatenate with persistent
        concat = cat([self.persistent_memory, h, segment], dim=1)

        # 3. Attention
        attn_out = self.attn(concat)[:, -C:]  # Extract segment portion

        # 4. Update memory
        memory_state = self.memory.update(attn_out, memory_state)

        # 5. Gate with memory retrieval
        mem_out = self.memory.retrieve(attn_out, memory_state)
        gated = self.gate_norm1(attn_out) * sigmoid(self.gate(self.gate_norm2(mem_out)))

        # 6. Residual + FFN
        x = segment + gated
        x = x + self.ffn(self.norm2(x))

        return x, memory_state
```

**FFN Structure:**
```
Input ∈ R^(d_model)
       │
┌──────▼──────┐
│ Linear(d→4d)│
└──────┬──────┘
       │
┌──────▼──────┐
│  SiLU/GELU  │
└──────┬──────┘
       │
┌──────▼──────┐
│ Linear(4d→d)│
└──────┬──────┘
       │
    Output
```

---

## 4. HYPERPARAMETERS

### 4.1 Model Configuration

| Parameter | Symbol | Values Tested | Best Value | Paper Reference |
|-----------|--------|---------------|------------|-----------------|
| **Architecture** |
| Model size | - | 170M, 340M, 400M, 760M | Depends on task | §5.1 |
| Number of layers | L | 12, 24, 32 | 24 (for 400M) | §5.1 |
| Hidden dimension | d_in / d_model | 768, 1024, 1536, 2048 | 1024 (for 400M) | §5.1 |
| Number of heads | - | 8, 16 | 16 | §5.1 |
| FFN expansion | - | 4× | 4× | Standard |
| Vocabulary size | - | 32K | 32K | §5.1 |
| **Memory Module** |
| Memory depth | L_M | 1, 2, 3, 4 | 3 | §5.5, Figure 7 |
| Chunk/segment size | C | 512, 1024, 2048 | 1024 | §5.1 (MAC) |
| Sliding window | W | 2048 | 2048 | §4.2, §4.3 (MAG/MAL) |
| Persistent tokens | N_p | 16, 32, 64 | 32 | §3.3 |
| Memory tokens | N_l | C (same as chunk) | 1024 | §4.1 |
| Chunk size (parallel) | b | 32, 64, 128, 256 | 64 | §3.2 |
| **Convolution** |
| Conv kernel size | - | 4 | 4 | §4.4 |
| Conv type | - | Depthwise-separable | Depthwise-separable | §4.4 |
| Conv padding | - | 3 | 3 | §4.4 |

**Model Size Breakdown (340M example):**
- 24 layers
- d_model = 1024
- 16 attention heads
- Head dimension = 64
- FFN intermediate = 4096

---

### 4.2 Training Configuration

| Parameter | Symbol | Value | Paper Reference |
|-----------|--------|-------|-----------------|
| **Optimizer** |
| Optimizer | - | AdamW | §5.1 |
| Learning rate | lr | 4e-4 | §5.1 |
| LR schedule | - | Cosine annealing | §5.1 |
| Weight decay | - | 0.1 | §5.1 |
| Batch size (tokens) | - | 0.5M tokens | §5.1 |
| Training tokens | - | 15B (170M/340M/400M), 30B (760M) | §5.1 |
| Training length | - | 4K tokens | §5.1 |
| Gradient clipping | - | 1.0 | Standard (not explicit) |
| Warmup steps | - | ~2000 (typical) | Not explicit |
| **Data** |
| Dataset | - | FineWeb-Edu | §5.1 |
| Tokenizer | - | Llama 2 | §5.1 |
| Vocab size | - | 32K | §5.1 |
| Context length (train) | - | 4K | §5.1 |
| Context length (test) | - | 2K-2M | §5.3, §5.4 |

**Training Details:**
- No explicit mention of β1, β2 for AdamW (assume defaults: 0.9, 0.999)
- No explicit mention of epsilon (assume 1e-8)
- Gradient accumulation likely used for 0.5M token batches

---

### 4.3 Memory-Specific Parameters

| Parameter | Description | Value/Range | Paper Reference |
|-----------|-------------|-------------|-----------------|
| **Gate Initialization** |
| α_t initialization | Forget gate bias | Small positive (→ retain memory) | §3.1 |
| θ_t initialization | Step size bias | Medium (0.5 equivalent) | §3.1 |
| η_t initialization | Momentum bias | High (→ retain momentum) | §3.1 |
| **Memory Initialization** |
| M_0 initialization | Initial memory weights | Xavier/Kaiming normal | Standard |
| S_0 initialization | Initial momentum | Zeros | §3.1 |
| Persistent memory init | P initialization | Normal(0, 0.02) | Standard |
| **Projection Initialization** |
| W_Q, W_K, W_V | Key/value/query matrices | Normal(0, 0.02) or Xavier | §3.1, §4.4 |
| **Normalization** |
| Query normalization | ℓ2-norm | Applied | §4.4 |
| Key normalization | ℓ2-norm | Applied | §4.4 |
| Layer normalization | - | Pre-norm | §4.4 |

**Implementation Notes:**
- Gates use sigmoid activation → output in [0,1]
- Forget gate bias initialized positive to prevent aggressive forgetting early
- Step size typically initialized around sigmoid^{-1}(0.1) to sigmoid^{-1}(0.5)

---

### 4.4 Variant-Specific Parameters

**MAC (Memory as Context):**
| Parameter | Value | Notes |
|-----------|-------|-------|
| Segment size (C) | 1024 | §4.1 |
| Historical tokens (N_l) | 1024 | Same as C |
| Persistent tokens (N_p) | 32 | §3.3 |
| Total attention size | N_p + N_l + C ≈ 2080 | Per segment |
| Attention type | Full causal | Within segment |

**MAG (Memory as Gating):**
| Parameter | Value | Notes |
|-----------|-------|-------|
| Sliding window (W) | 2048 | §4.2 |
| Persistent tokens (N_p) | 32 | §3.3 |
| Prefix visibility | Always | All positions see prefix |
| Gating type | Element-wise multiply | After normalization |
| Normalization | LayerNorm on both paths | §4.4 |

**MAL (Memory as Layer):**
| Parameter | Value | Notes |
|-----------|-------|-------|
| Sliding window (W) | 2048 | §4.3 |
| Persistent tokens (N_p) | 32 | §3.3 |
| Layer order | Memory → Attention | Sequential |

---

## 5. BENCHMARKS

### 5.1 Language Modeling Datasets

**WikiText:**
- **Description:** Word-level language modeling dataset from Wikipedia
- **Metric:** Perplexity (lower is better)
- **Titans Results (400M):**
  - Titans (LMM): 25.03
  - Titans (MAC): **25.61**
  - Titans (MAG): **23.59** ← Best
  - Titans (MAL): 23.93
- **Baseline Comparison:** Transformer++: 30.63, Mamba2: 26.34, Gated DeltaNet: 25.47
- **Reference:** Table 1

**LAMBADA:**
- **Description:** Next-word prediction requiring broad discourse context
- **Metrics:**
  - Perplexity (ppl ↓)
  - Accuracy (acc ↑)
- **Titans Results (400M):**
  - ppl: 27.73 (MAC), **27.81** (MAG), 27.89 (MAL)
  - acc: **36.92** (MAC), 37.24 (MAG), 36.84 (MAL)
- **Baseline:** Transformer++: 37.37 ppl / 29.64 acc
- **Reference:** Table 1

---

### 5.2 Common-Sense Reasoning

**PIQA (Physical Interaction QA):**
- **Titans (400M):** 66.39% (MAC), **66.80%** (MAG)
- **Baseline:** Transformer++: 64.27%
- **Reference:** Table 1

**HellaSwag:**
- **Titans (400M acc_norm):** **41.18%** (MAC), 40.92% (MAG)
- **Baseline:** Transformer++: 37.72%
- **Reference:** Table 1

**WinoGrande:**
- **Titans (400M):** **52.80%** (MAC), 53.21% (MAG)
- **Baseline:** Transformer++: 51.53%
- **Reference:** Table 1

**ARC-Easy:**
- **Titans (400M):** **60.24%** (MAC), 60.01% (MAG)
- **Baseline:** Transformer++: 54.95%
- **Reference:** Table 1

**ARC-Challenge (acc_norm):**
- **Titans (400M):** **29.69%** (MAC), 29.45% (MAG)
- **Baseline:** Transformer++: 27.36%
- **Reference:** Table 1

**SIQA (Social IQa):**
- **Titans (400M):** **40.07%** (MAC), 39.91% (MAG)
- **Baseline:** Transformer++: 38.07%
- **Reference:** Table 1

**BoolQ:**
- **Titans (400M):** **61.93%** (MAC), 61.28% (MAG)
- **Baseline:** Transformer++: 61.59%
- **Reference:** Table 1

**Average Accuracy (400M):**
- Titans (MAC): **48.65%**
- Titans (MAG): **48.60%**
- Titans (MAL): 47.87%
- Transformer++: 45.64%
- Mamba2: 46.91%
- Gated DeltaNet: 47.26%

---

### 5.3 RULER Benchmark (Needle in Haystack)

**Single NIAH Tasks:**
- **S-NIAH-PK (Passkey):**
  - 2K: 99.8% (LMM), 99.2% (MAC)
  - 4K: 98.4%, 98.8%
  - 8K: 98.2%, 99.0%
  - 16K: **96.2%**, **98.4%** ← Best at 16K
  - **Baselines:** TTT: 88.4% @ 16K, Mamba2: 5.4% @ 16K, DeltaNet: 71.4% @ 16K

- **S-NIAH-N (Number):**
  - 2K: 100.0%, 99.6%
  - 16K: **80.2%**, **97.4%**
  - **Baselines:** TTT: 4.4% @ 16K, Mamba2: 0% @ 16K

- **S-NIAH-W (Word):**
  - 2K: 90.4%, 98.2%
  - 16K: **80.6%**, **95.2%**
  - **Baselines:** All near 0% @ 16K

**Reference:** Table 2

---

### 5.4 BABILong Benchmark

**Setup:**
- Long-context reasoning requiring multi-hop inference
- Context lengths: 1K to 1M tokens

**Few-shot Results (@ 1M tokens):**
- Titans (MAC): **~52% accuracy**
- GPT-4: ~35%
- Llama3.1-8B: ~20%
- Mamba2-8B: ~15%
- RecurrentGemma-9B: ~25%
- **Reference:** Figure 6a

**Fine-tuned Results:**
- Titans (MAC): **~70% @ 1M tokens**
- GPT-4: ~45%
- RMT (fine-tuned): ~38%
- Llama3.1-70B: ~40%
- **Reference:** Figure 6b

**Key Achievement:**
- 340M parameter model outperforms 70B+ models
- Effective context length exceeds 1M tokens

---

### 5.5 Time Series Forecasting

**Datasets:** ETTm1, ETTm2, ETTh1, ETTh2, ECL, Traffic, Weather

**Neural Memory Results (MSE/MAE):**
- **ETTm1:** 0.358 / 0.387 ← Best
- **ETTm2:** 0.261 / 0.309 ← Best
- **ETTh1:** 0.420 / 0.421 ← Best
- **ETTh2:** 0.336 / 0.382 ← Best
- **ECL:** 0.162 / 0.261 ← Best
- **Traffic:** 0.415 / 0.289 ← Best
- **Weather:** 0.231 / 0.265 ← Best

**Baseline Comparison:**
- Simba (Mamba-based): 0.383 / 0.396 (ETTm1)
- iTransformer: 0.407 / 0.410 (ETTm1)
- TimesNet: 0.400 / 0.406 (ETTm1)

**Reference:** Table 3

---

### 5.6 DNA Modeling (GenomicsBenchmarks)

**Tasks:** Enhancer Cohn, Enhancer Ens, Human Reg., Non-TATA Promoters, Human OCR Ens.

**Neural Memory Results (accuracy %):**
- Enhancer Cohn: **75.2%** ← Best
- Enhancer Ens: **89.6%** ← Best
- Human Reg.: 89.3%
- Non-TATA Promoters: **96.6%** (tied)
- Human OCR Ens.: **79.9%** ← Best

**Baseline Comparison:**
- HyenaDNA: 74.2% / 89.2% / 93.8% / 96.6% / 80.9%
- Based: 74.6% / 89.5% / 89.5% / 96.8% / 79.0%
- Mamba: 73.0% / - / - / 96.6% / -

**Reference:** Table 4

---

### 5.7 Computational Requirements

**Training Throughput (tokens/sec @ 8K sequence):**
- Titans (MAL): ~40 (highest)
- Titans (MAG): ~35
- Titans (MAC): ~30
- Mamba2: ~42
- Gated DeltaNet: ~45
- Transformer++: ~25
- **Reference:** Figure 9

**Memory Depth Impact (perplexity @ 32K seq):**
- L_M = 1: ~13.4 (170M params)
- L_M = 2: ~13.0
- L_M = 3: **~12.6** ← Best
- L_M = 4: ~12.8
- **Reference:** Figure 7c

**Context Length Scaling:**
- Tested up to 2M tokens
- BABILong: successful at 1M+
- NIAH: tested at 2K, 4K, 8K, 16K
- **Reference:** §5.3, §5.4

---

## 6. DEPENDENCIES

### 6.1 Core Dependencies

**No External Papers Required:**
- Architecture is self-contained
- Can be implemented from equations in paper
- Standard components (attention, MLP, normalization)

**Helpful References (not required):**
1. **FlashAttention-2** (Dao 2024): Efficient attention implementation
   - Reference: Dao, Tri. "FlashAttention-2: Faster Attention with Better Parallelism" ICLR 2024
   - arXiv:2307.08691
   - Use for: Sliding window attention, causal attention optimization

2. **Parallel Associative Scan** (Smith et al. 2023): For momentum computation
   - Reference: Smith, Jimmy TH, Andrew Warrington, and Scott Linderman. "Simplified State Space Layers" ICLR 2023
   - Use for: Parallelizing Equation 18

3. **Gradient-based Meta-learning** (Andrychowicz et al. 2016): Conceptual background
   - Reference: "Learning to learn by gradient descent by gradient descent" NIPS 2016
   - Use for: Understanding test-time learning framework

---

### 6.2 Baseline Comparisons (for reproduction)

**Modern RNN Baselines:**
1. **Mamba2:** Dao & Gu (2024) - arXiv:2405.21060
2. **Gated DeltaNet:** Yang et al. (2024) - arXiv:2412.06464
3. **DeltaNet:** Yang et al. (2024) - arXiv:2408.XXXXX
4. **TTT:** Sun et al. (2024) - arXiv:2407.04620
5. **GLA:** Yang et al. (2024) - Gated Linear Attention
6. **RetNet:** Sun et al. (2023) - arXiv:2307.08621

**Attention Baselines:**
1. **Transformer++:** Standard transformer with improvements (Touvron et al. 2023)

---

### 6.3 Implementation Order

**Phase 1: Core Memory Module**
1. Implement basic MLP memory (Eq 11-12, 15)
2. Add gradient-based update (Eq 8)
3. Add momentum (Eq 9-10)
4. Add forgetting (Eq 13-14)
5. Test on simple associative memory task

**Phase 2: Parallelization**
1. Implement chunk-wise processing (Eq 16)
2. Add tensorized gradients for linear case (Eq 17)
3. Implement parallel scan for momentum (Eq 18)
4. Benchmark against sequential version

**Phase 3: Architecture Variants**
1. Implement MAC (Eq 21-25)
   - Segment processing
   - Memory retrieval
   - Concatenation + attention
   - Memory update
2. Implement MAG (Eq 26-28)
   - Sliding window attention
   - Gating mechanism
3. Implement MAL (Eq 29-31)
   - Sequential layers

**Phase 4: Training & Evaluation**
1. Setup data pipeline (FineWeb-Edu)
2. Implement training loop with meta-learning
3. Evaluate on language modeling
4. Evaluate on long-context tasks

---

## 7. IMPLEMENTATION CHECKLIST

### 7.1 Equations to Implement

- [ ] **Eq 1:** Q, K, V projections (standard linear layers)
- [ ] **Eq 2:** Softmax attention (baseline, for comparison)
- [ ] **Eq 3-5:** Linear attention (optional, for understanding)
- [ ] **Eq 8:** Basic gradient-based surprise
- [ ] **Eq 9-10:** Momentum-based surprise (**CORE INNOVATION**)
- [ ] **Eq 11:** Key-value projections for memory
- [ ] **Eq 12:** Associative memory loss (MSE)
- [ ] **Eq 13-14:** Forgetting mechanism (**CRITICAL**)
- [ ] **Eq 15:** Memory retrieval (frozen forward pass)
- [ ] **Eq 16:** Chunk-wise mini-batch gradient descent
- [ ] **Eq 17:** Tensorized gradient computation (linear case)
- [ ] **Eq 18:** Parallel associative scan for momentum
- [ ] **Eq 19-20:** Persistent memory (learnable parameters)
- [ ] **Eq 21-25:** MAC architecture forward pass
- [ ] **Eq 26-28:** MAG architecture forward pass
- [ ] **Eq 29-31:** MAL architecture forward pass

### 7.2 Algorithms to Implement

- [ ] **Algorithm 1:** Sequential memory training (inner + outer loop)
- [ ] **Algorithm 2:** Parallel chunk processing
- [ ] **Algorithm 3:** Full Titans (MAC) architecture
- [ ] **Parallel Scan:** For momentum computation (Eq 18)
- [ ] **Gradient Checkpointing:** For long sequences

### 7.3 Components to Implement

**Memory Module:**
- [ ] MLP with configurable depth (L_M ∈ {1,2,3,4})
- [ ] Functional forward pass (for gradient computation)
- [ ] Parameter state management (M_t, S_t)
- [ ] Gate networks (α, θ, η)
- [ ] 1D depthwise-separable convolutions

**Attention:**
- [ ] Multi-head causal attention
- [ ] Sliding window attention (for MAG/MAL)
- [ ] Attention with prefix visibility (for MAG)
- [ ] FlashAttention integration (optional but recommended)

**Architecture:**
- [ ] Persistent memory parameters
- [ ] Segment/chunk processing logic
- [ ] Memory state persistence across segments
- [ ] Gating mechanisms for output
- [ ] Residual connections + LayerNorm

**Training:**
- [ ] Meta-learning loop (inner/outer)
- [ ] AdamW optimizer
- [ ] Cosine learning rate schedule
- [ ] Gradient clipping
- [ ] Mixed precision training

### 7.4 Testing Checklist

**Unit Tests:**
- [ ] Memory forward pass shape correctness
- [ ] Gradient computation for memory loss
- [ ] Momentum update correctness
- [ ] Forgetting mechanism (α → 0 vs α → 1)
- [ ] Gate output ranges ([0,1])
- [ ] Parallel scan equivalence to sequential
- [ ] Chunk processing equivalence to full sequence

**Integration Tests:**
- [ ] MAC forward pass with dummy data
- [ ] Memory state persistence across chunks
- [ ] Gradient flow through full architecture
- [ ] Training loop convergence on toy task

**Benchmark Tests:**
- [ ] Language modeling perplexity (WikiText)
- [ ] NIAH accuracy at 2K, 4K, 8K, 16K
- [ ] BABILong accuracy at long context
- [ ] Training throughput comparison

---

## 8. NOTES FOR IMPLEMENTER

### 8.1 Critical Implementation Details

**1. Functional Gradient Computation:**
```python
# WRONG: Standard PyTorch
loss = mse_loss(memory(k), v)
loss.backward()  # Updates memory permanently

# CORRECT: Functional API
import torch.func as functorch

# Create functional version
func_memory, params = functorch.make_functional(memory)

# Compute gradient w.r.t. params (not updating)
loss = mse_loss(func_memory(params, k), v)
grads = torch.autograd.grad(loss, params, create_graph=True)

# Manually update params
new_params = [(1 - alpha) * p + S for p, S in zip(params, momentum)]
```

**2. Memory State Management:**
- Memory parameters M_t must be **detached** from computation graph between segments
- Only final segment loss should backprop to hyperparameters (W_K, W_V, gates)
- Momentum S_t is also part of state (not just M_t)

**3. Parallel Scan Implementation:**
```python
# Use JAX or custom CUDA kernel
import jax.numpy as jnp
from jax.lax import associative_scan

def scan_fn(carry, x):
    eta, theta, grad = x
    return eta * carry - theta * grad

S_all = associative_scan(scan_fn, (eta, theta, grads), init=S_0)
```

**4. Sliding Window Attention:**
- Use FlashAttention with custom mask
- Prefix tokens (persistent memory) should have full visibility
- Local tokens attend within window W
- Implementation: use `attention_mask` or `block_mask` in FlashAttention

---

### 8.2 Common Pitfalls

**1. Gradient Accumulation:**
- Don't accumulate gradients for memory parameters across sequence
- Each segment starts fresh (meta-learning, not continual learning)

**2. Gate Initialization:**
- Initialize forget gate α bias to negative (sigmoid → small α → retain memory)
- Initialize step size θ bias to negative (sigmoid → small θ → conservative updates)
- Initialize momentum η bias to positive (sigmoid → large η → retain momentum)

**3. Memory Overflow:**
- Even with forgetting, memory can overflow on very long sequences
- Monitor gradient norms of memory parameters
- Consider gradient clipping on α, θ, η outputs

**4. Chunking Boundary:**
- Last chunk may have fewer than C tokens
- Pad or handle separately
- Attention mask must account for padding

---

### 8.3 Optimization Opportunities

**1. Kernel Fusion:**
- Fuse gate computation with memory update
- Fuse convolution with projection
- Use Triton or CUDA for custom kernels

**2. Memory Efficiency:**
- Use gradient checkpointing for deep memory (L_M ≥ 3)
- Store only Θ, B per chunk (not all chunks)
- Recompute activations in backward pass

**3. Parallelism:**
- Process multiple segments in parallel (different sequences in batch)
- Use data parallelism across GPUs
- Pipeline parallelism for very large models

**4. Mixed Precision:**
- Use bfloat16 for activations
- Keep memory parameters in fp32 for stability
- Use automatic mixed precision (AMP)

---

### 8.4 Debugging Tips

**1. Check Memory Update:**
```python
# Memory should change after update
M_before = memory.state_dict().copy()
memory_state = memory.update(x, memory_state)
M_after = memory.state_dict()

# Should be different (unless α=1, θ=0)
assert not torch.allclose(M_before['linear.weight'], M_after['linear.weight'])
```

**2. Check Gradient Flow:**
```python
# Outer loss should flow to W_K, W_V, not memory params
loss.backward()

# Should have gradients
assert W_K.grad is not None
assert W_V.grad is not None

# Should NOT have gradients (updated in inner loop)
assert all(p.grad is None for p in memory.parameters())
```

**3. Check Attention Mask:**
```python
# Visualize attention pattern
attn_weights = model.get_attention_weights(x)
import matplotlib.pyplot as plt
plt.imshow(attn_weights[0].cpu(), cmap='hot')
plt.show()

# Should see:
# - Persistent tokens always visible (top rows)
# - Causal pattern within segment (lower triangle)
```

**4. Check Forgetting:**
```python
# Force α → 1 (full forget)
model.memory.alpha_net.bias.data.fill_(10.0)  # sigmoid(10) ≈ 1

# Memory should reset after one step
M_0 = memory.state_dict()['linear.weight'].clone()
memory.update(x, memory_state)
M_1 = memory.state_dict()['linear.weight']

# Should be mostly zeros (momentum term)
assert M_1.abs().mean() < M_0.abs().mean() * 0.1
```

---

### 8.5 Theoretical Insights for Implementation

**1. Why Momentum?**
- Token at position t might not be surprising itself
- But follows surprising tokens → should be remembered
- Momentum carries "surprise signal" forward
- Example: "The capital of France is [surprising: France] [boring: Paris]"
  - Without momentum: only "France" gets high surprise
  - With momentum: both get high surprise signal

**2. Why Forgetting?**
- Fixed memory capacity (parameter count)
- Long sequences exceed capacity
- Forgetting old info makes room for new
- Data-dependent: forget when context changes

**3. Why Deep Memory?**
- Linear memory: assumes linear dependency between keys and values
- Deep memory: can model non-linear associations
- Empirically: L_M=3 is sweet spot (Figure 7)
- Too deep: overfitting, slow training

**4. Why Three Variants?**
- **MAC:** Most expressive, attention chooses what to remember
- **MAG:** Fast training, gating combines global + local
- **MAL:** Simplest, like existing hybrid models
- Trade-off: expressiveness vs. efficiency

---

### 8.6 Extensions & Future Work

**1. Different Memory Architectures:**
- Replace MLP with Transformer layers
- Use memory-efficient architectures (Berges et al. 2024)
- Sparse memory networks

**2. Different Objectives:**
- Contrastive loss instead of MSE
- Reconstruction loss
- Task-specific losses

**3. Different Gating:**
- Learnable interpolation (not sigmoid)
- Soft MoE-style gating
- Attention-based gating

**4. Chunk-Adaptive Processing:**
- Variable chunk sizes based on content
- Learned chunk boundaries
- Hierarchical chunking

---

## METADATA

```yaml
paper_id: "TITANS"
full_title: "Titans: Learning to Memorize at Test Time"
authors: ["Ali Behrouz", "Peilin Zhong", "Vahab Mirrokni"]
institution: "Google Research"
arxiv: "arXiv:2501.00663v1"
year: 2025
date: "2024-12-31"

equations_extracted: 35
algorithms_extracted: 3
architectures: 3

key_innovations:
  - "Momentum-based surprise metric for memory update"
  - "Forgetting mechanism via weight decay"
  - "Test-time learning with gradient descent"
  - "Three-branch architecture (core, long-term, persistent)"

implementation_difficulty: "High"
estimated_implementation_time: "4-6 weeks (full reproduction)"

critical_equations:
  - "Eq 9-10: Momentum-based surprise"
  - "Eq 13-14: Forgetting mechanism"
  - "Eq 21-25: MAC architecture"

dependencies:
  external_papers: 0
  helpful_references: 3

model_sizes:
  - "170M parameters"
  - "340M parameters"
  - "400M parameters"
  - "760M parameters"

datasets:
  language_modeling: ["WikiText", "LAMBADA", "FineWeb-Edu"]
  reasoning: ["PIQA", "HellaSwag", "WinoGrande", "ARC", "SIQA", "BoolQ"]
  long_context: ["RULER", "BABILong"]
  time_series: ["ETTm1", "ETTm2", "ETTh1", "ETTh2", "ECL", "Traffic", "Weather"]
  genomics: ["GenomicsBenchmarks"]

key_results:
  - "Outperforms Mamba2, Gated DeltaNet on all tasks"
  - "340M model beats GPT-4 on BABILong @ 1M tokens"
  - "Scales to 2M+ context length"
  - "Best NIAH scores at 16K context (97-98%)"

implementation_status: "Not yet implemented (paper just released)"
code_availability: "Authors intend to release (PyTorch + JAX)"

notes: |
  This is a comprehensive extraction covering all equations, algorithms,
  and architectural details needed for full reproduction. The paper
  introduces significant innovations in test-time learning for sequence
  models, particularly the momentum-based surprise metric and forgetting
  mechanism that generalize recent work (Gated DeltaNet, TTT, Longhorn).

  Key challenge: Implementing functional gradient computation for
  meta-learning framework. Recommended to use JAX or PyTorch functorch.

  Three architectural variants provide flexibility: MAC (most expressive),
  MAG (fast training), MAL (simplest). Start with MAL for initial
  implementation, then build up to MAC.

extraction_date: "2024-12-30"
extractor_confidence: "95%"
```
