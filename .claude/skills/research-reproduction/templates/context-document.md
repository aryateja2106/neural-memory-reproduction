# Context Document Template

This template defines the output format for extraction agents. Each paper gets one `.context.md` file that the orchestrator uses for implementation.

## File Naming

```
{PAPER_ID}.context.md

Examples:
- TITANS.context.md
- MIRAS.context.md  
- HOPE.context.md
```

## Template

```markdown
# {Paper Title} - Implementation Context

**Paper ID:** {PAPER_ID}
**Full Title:** {Full paper title}
**Authors:** {Author list}
**arXiv:** {arXiv URL or DOI}
**Year:** {Publication year}

**Key Contribution (1 sentence):**
{What this paper adds that we need to implement}

**Extraction Date:** {ISO timestamp}
**Extraction Agent:** {Agent identifier}

---

## Quick Reference

| Aspect | Count | Key Items |
|--------|-------|-----------|
| Equations | {N} | Eq 3 (memory), Eq 5 (surprise), ... |
| Algorithms | {N} | Alg 1 (forward), Alg 2 (training), ... |
| Layers/Modules | {N} | MemoryLayer, SurpriseGate, ... |
| Hyperparameters | {N} | See table below |
| Benchmarks | {N} | WikiText-103, BABILong, ... |

---

## 1. EQUATIONS

### Equation {N}: {Descriptive Name}

**Location:** Section {X.Y}, Page {P}

**LaTeX:**
```latex
{exact LaTeX}
```

**Rendered:** 
$$M_{t+1} = M_t + \eta \cdot \nabla_M \ell(M_t; x_t)$$

**Plain English:**
{Human-readable description of what this equation does}

**Variables:**
| Symbol | Name | Type | Shape | Description |
|--------|------|------|-------|-------------|
| $M_t$ | Memory state | Tensor | [B, S, D] | Current memory at time t |
| $x_t$ | Input | Tensor | [B, D] | Current input token embedding |
| $\eta$ | Learning rate | Scalar | [] | Memory update rate |
| $\ell$ | Loss function | Function | - | Associative memory loss |

**Implementation Notes:**
- {Note 1: e.g., "Must be differentiable for end-to-end training"}
- {Note 2: e.g., "Gradient computed via autodiff, not analytical"}
- {Note 3: e.g., "η typically 0.01-0.1 per paper experiments"}

**Dependencies:**
- Requires: {Equation X for gradient computation}
- Used by: {Algorithm 1 line 7, Equation Y}

**Test Criteria:**
- [ ] Output shape matches input M_t shape
- [ ] Gradients flow to M_t
- [ ] Numerically stable for values in [-1000, 1000]
- [ ] Deterministic given same inputs

---

### Equation {N+1}: {Name}
...

---

## 2. ALGORITHMS

### Algorithm {N}: {Name}

**Location:** Section {X.Y}, Page {P}

**Purpose:** {What this algorithm accomplishes}

**Pseudocode:**
```
Algorithm {N}: {Name}
────────────────────────────────────────
Input: 
  - x ∈ ℝ^{B×T×D}: Input sequence
  - M₀ ∈ ℝ^{B×S×D}: Initial memory state
  
Output:
  - y ∈ ℝ^{B×T×D}: Output sequence
  - M_T ∈ ℝ^{B×S×D}: Final memory state

Parameters:
  - W_q, W_k, W_v ∈ ℝ^{D×D}: Projection matrices
  - η: Memory learning rate

────────────────────────────────────────
1:  M ← M₀
2:  for t = 1 to T do
3:      q_t ← W_q · x_t                    // Query projection
4:      k_t ← W_k · M                      // Key from memory
5:      v_t ← W_v · M                      // Value from memory
6:      α_t ← softmax(q_t · k_t^T / √d)   // Attention weights [Eq 2]
7:      y_t ← α_t · v_t                    // Attention output
8:      M ← MemoryUpdate(M, x_t, η)        // Update memory [Eq 3]
9:  end for
10: return y, M
────────────────────────────────────────
```

**Line-by-Line Mapping:**
| Line | Operation | Equation | Module/Function |
|------|-----------|----------|-----------------|
| 3-5 | QKV projection | - | `nn.Linear` |
| 6 | Attention scores | Eq 2 | `scaled_dot_product_attention` |
| 7 | Attention output | Eq 2 | (continued) |
| 8 | Memory update | Eq 3 | `memory_update()` |

**Implementation Considerations:**
- Loop vs vectorized: {Can this be parallelized? How?}
- Memory efficiency: {Gradient checkpointing needed?}
- Numerical: {Any stability concerns?}

**Test Criteria:**
- [ ] Output shape matches expected
- [ ] Memory state changes after processing
- [ ] Gradients flow through entire algorithm
- [ ] Matches paper's computational complexity

---

## 3. ARCHITECTURE

### Overall Structure

```
┌─────────────────────────────────────────────────────────────┐
│                        {Model Name}                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Input: token_ids [B, T]                                    │
│         ↓                                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Token Embedding + Position Embedding                │   │
│  │ Output: [B, T, D]                                   │   │
│  └─────────────────────────────────────────────────────┘   │
│         ↓                                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ {Layer Name} × N_layers                             │   │
│  │ ┌─────────────────────────────────────────────────┐ │   │
│  │ │ Memory-Augmented Attention [Alg 1]              │ │   │
│  │ │ → LayerNorm → FFN → LayerNorm                   │ │   │
│  │ └─────────────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────┘   │
│         ↓                                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Final LayerNorm → Output Projection                 │   │
│  │ Output: logits [B, T, V]                           │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Module Breakdown

| Module | Paper Section | Input Shape | Output Shape | Parameters |
|--------|--------------|-------------|--------------|------------|
| Embedding | §3.1 | [B, T] | [B, T, D] | V×D |
| MemoryLayer | §3.2, Alg 1 | [B, T, D] | [B, T, D] | ~4D² |
| FFN | §3.2 | [B, T, D] | [B, T, D] | 8D² |
| Output | §3.3 | [B, T, D] | [B, T, V] | D×V (tied) |

### Memory Module Detail

```
Memory Module (MemoryLayer)
──────────────────────────────────────────────────
Input: x [B, T, D], memory [B, S, D]

    x ──┬──→ W_q ──→ q [B, T, D]
        │
        │   memory ──┬──→ W_k ──→ k [B, S, D]
        │            └──→ W_v ──→ v [B, S, D]
        │
        └──→ Attention(q, k, v) ──→ attn_out [B, T, D]
                    │
                    └──→ + x (residual) ──→ output [B, T, D]
        
        x, memory ──→ MemoryUpdate [Eq 3] ──→ new_memory [B, S, D]

Output: output [B, T, D], new_memory [B, S, D]
──────────────────────────────────────────────────
```

---

## 4. HYPERPARAMETERS

### Model Configuration

| Parameter | Symbol | Value | Paper Reference | Notes |
|-----------|--------|-------|-----------------|-------|
| Vocab size | V | 50257 | §4.1 | GPT-2 tokenizer |
| Hidden dim | D | 768 | Table 1 | Base model |
| Num layers | N | 12 | Table 1 | Base model |
| Num heads | H | 12 | Table 1 | D/H = 64 |
| Memory size | S | 64 | §3.2 | Slots per layer |
| Memory dim | D_m | 768 | §3.2 | Same as D |
| FFN dim | D_ff | 3072 | Table 1 | 4×D |
| Max seq len | T_max | 2048 | §4.1 | - |

### Training Configuration

| Parameter | Value | Paper Reference | Notes |
|-----------|-------|-----------------|-------|
| Optimizer | AdamW | §4.2 | - |
| Learning rate | 3e-4 | §4.2 | Peak LR |
| LR schedule | Cosine | §4.2 | With warmup |
| Warmup steps | 10000 | §4.2 | - |
| Total steps | 100000 | §4.2 | - |
| Batch size | 32 | §4.2 | Per GPU |
| Gradient clip | 1.0 | §4.2 | Max norm |
| Weight decay | 0.1 | §4.2 | - |
| Dropout | 0.1 | Table 1 | - |

### Memory-Specific

| Parameter | Symbol | Value | Paper Reference | Notes |
|-----------|--------|-------|-----------------|-------|
| Memory LR | η | 0.01 | §3.2 | Memory update rate |
| Surprise threshold | τ | 0.5 | §3.3 | For gating |
| Memory init std | σ | 0.02 | §3.2 | Random init |

---

## 5. BENCHMARKS

### Datasets

| Dataset | Task | Size | Metric | Paper Section |
|---------|------|------|--------|---------------|
| WikiText-103 | Language Modeling | 100M tokens | Perplexity | §4.3 |
| C4 | Language Modeling | 300B tokens | Perplexity | §4.3 |
| BABILong | Long-range QA | 10K-2M ctx | Accuracy | §4.4 |
| PG-19 | Book LM | 11B tokens | Perplexity | §4.3 |

### Reported Results

| Dataset | Baseline | This Paper | Δ | Notes |
|---------|----------|------------|---|-------|
| WikiText-103 (PPL↓) | 18.5 | 17.2 | -7.0% | Table 2 |
| C4 (PPL↓) | 15.8 | 14.9 | -5.7% | Table 2 |
| BABILong 2M (Acc↑) | 45% | 89% | +44pp | Table 3 |
| BABILong 10M (Acc↑) | 12% | 67% | +55pp | Table 3 |

### Evaluation Protocol

```
WikiText-103 Evaluation:
1. Load test split (245K tokens)
2. Process with stride = 512
3. Compute cross-entropy loss
4. Report exp(loss) as perplexity

BABILong Evaluation:
1. Load QA pairs with context length L
2. Insert fact at position P in context
3. Generate answer
4. Exact match accuracy
```

### Computational Requirements

| Aspect | Requirement | Notes |
|--------|-------------|-------|
| Training GPUs | 8× A100 80GB | Paper setup |
| Training time | ~3 days | 100K steps |
| Inference GPU | 1× A100 40GB | Or 2× 3090 |
| Memory (train) | ~40GB/GPU | With grad ckpt |
| Memory (infer) | ~16GB | Batch=1 |

---

## 6. DEPENDENCIES ON OTHER PAPERS

### Required Prior Work

| Paper | What We Need | Our Usage | Priority |
|-------|--------------|-----------|----------|
| {Paper A} | Memory update equations | Eq 3, 5 | Must implement first |
| Transformer | Attention mechanism | Alg 1 lines 6-7 | Use PyTorch impl |
| LayerNorm | Normalization | Throughout | Use PyTorch impl |

### Concepts to Import

From **{Paper A}**:
- Neural Memory Module (§3.1): The core memory structure
- Surprise Metric (§3.3): Gates memory updates
- Implementation: Must implement before this paper

From **Transformers**:
- Multi-head attention: Standard implementation OK
- Position embeddings: Standard implementation OK

### Implementation Order

```
1. [FIRST] Implement {Paper A} memory module
   └── Equations 3, 5
   └── Test independently
   
2. [SECOND] Implement this paper's extensions
   └── Algorithm 1 using Paper A's memory
   └── New components: {list}
   
3. [THIRD] Full model assembly
   └── Combine all components
   └── End-to-end training test
```

---

## 7. IMPLEMENTATION CHECKLIST

### Equations
- [ ] Eq {N}: {name} - tests written
- [ ] Eq {N}: {name} - implementation passes tests
- [ ] Eq {N+1}: {name} - tests written
- [ ] Eq {N+1}: {name} - implementation passes tests
...

### Algorithms
- [ ] Alg {N}: {name} - pseudocode translated
- [ ] Alg {N}: {name} - tests pass
...

### Modules
- [ ] {ModuleName} - implemented
- [ ] {ModuleName} - tested
- [ ] {ModuleName} - integrated
...

### Full Model
- [ ] Model assembles without error
- [ ] Forward pass works
- [ ] Backward pass works
- [ ] Training loop runs
- [ ] Checkpoint save/load works

### Benchmarks
- [ ] WikiText-103 evaluation script
- [ ] BABILong evaluation script
- [ ] Results within X% of paper

---

## 8. NOTES FOR IMPLEMENTER

### Gotchas / Clarifications Needed

1. **{Issue 1}**: {Description of ambiguity in paper}
   - Paper says: "{quote}"
   - Interpretation: {how we'll implement}

2. **{Issue 2}**: {Description}
   - Resolution: {approach}

### Optimization Opportunities

1. {Optimization 1}: {Description, e.g., "Fuse QKV projection"}
2. {Optimization 2}: {Description}

### Known Limitations

1. {Limitation 1}: {What won't work and why}
2. {Limitation 2}: {Description}

---

## METADATA

```yaml
paper_id: "{PAPER_ID}"
extraction_version: "1.0"
extracted_at: "{ISO timestamp}"
extractor: "research-reproduction-skill/extraction-agent"
source_pdf: "{filename or URL}"
source_pages: {total pages}
equations_extracted: {count}
algorithms_extracted: {count}
confidence: "{high|medium|low}"
needs_clarification:
  - "{item 1}"
  - "{item 2}"
```
```

## Usage Notes

1. **Be exhaustive with equations** - Every equation that affects implementation should be captured
2. **Include ALL hyperparameters** - No "default" values; everything explicit
3. **Note ambiguities** - If paper is unclear, document your interpretation
4. **Cross-reference** - Link equations to algorithms, algorithms to modules
5. **Test criteria** - Every equation should have testable criteria
