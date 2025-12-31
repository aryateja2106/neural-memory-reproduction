# Equation Extractor Tool

## Purpose
Extracts, parses, and catalogs mathematical equations from research papers, preserving LaTeX notation and capturing variable definitions, constraints, and dependencies.

## Extraction Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                   EQUATION EXTRACTION PIPELINE                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  MARKDOWN INPUT (from paper-intake)                              │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  The surprise is computed as:                             │   │
│  │  $$S_t = \|h_t - M_{t-1}(k_t)\|_2^2$$                    │   │
│  │  where $h_t$ is the hidden state and $k_t$ is the key.   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│              ┌────────────────────────┐                         │
│              │   LaTeX Detector       │                         │
│              │  • Display math $$..$$  │                         │
│              │  • Inline math $...$   │                         │
│              │  • \begin{equation}    │                         │
│              │  • \begin{align}       │                         │
│              └───────────┬────────────┘                         │
│                          │                                       │
│                          ▼                                       │
│              ┌────────────────────────┐                         │
│              │   Equation Parser      │                         │
│              │  • Extract LHS/RHS     │                         │
│              │  • Identify variables  │                         │
│              │  • Detect operators    │                         │
│              │  • Find subscripts     │                         │
│              └───────────┬────────────┘                         │
│                          │                                       │
│                          ▼                                       │
│              ┌────────────────────────┐                         │
│              │   Context Analyzer     │                         │
│              │  • Find "where" blocks │                         │
│              │  • Extract definitions │                         │
│              │  • Link dependencies   │                         │
│              └───────────┬────────────┘                         │
│                          │                                       │
│                          ▼                                       │
│              ┌────────────────────────┐                         │
│              │   Equation Catalog     │                         │
│              │  equations.yaml        │                         │
│              └────────────────────────┘                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Equation Detection Patterns

### Display Math (Block Equations)
```python
DISPLAY_PATTERNS = [
    r'\$\$(.+?)\$\$',                    # $$..$$
    r'\\begin\{equation\}(.+?)\\end\{equation\}',
    r'\\begin\{equation\*\}(.+?)\\end\{equation\*\}',
    r'\\begin\{align\}(.+?)\\end\{align\}',
    r'\\begin\{align\*\}(.+?)\\end\{align\*\}',
    r'\\begin\{gather\}(.+?)\\end\{gather\}',
    r'\\begin\{multline\}(.+?)\\end\{multline\}',
    r'\\\[(.+?)\\\]',                    # \[..\]
]
```

### Inline Math
```python
INLINE_PATTERNS = [
    r'(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)',  # $..$ (not $$)
    r'\\\\(.+?)\\\\',                          # \\..\\ (less common)
]
```

### Numbered Equations
```python
# Detect equation numbers/labels
LABEL_PATTERNS = [
    r'\\label\{([^}]+)\}',
    r'\((\d+(?:\.\d+)?)\)',  # (1), (2.3), etc. at end of line
    r'Equation\s+(\d+)',
    r'Eq\.\s*\(?(\d+)\)?',
]
```

## Equation Data Structure

```python
@dataclass
class ExtractedEquation:
    """Represents a single extracted equation."""
    
    # Identification
    id: str                          # eq_1, eq_2, etc.
    label: Optional[str]             # Original paper label if any
    name: str                        # Human-readable name
    section: str                     # Section where found
    
    # LaTeX
    latex: str                       # Raw LaTeX string
    latex_display: str               # Formatted for display
    
    # Structure
    lhs: str                         # Left-hand side
    rhs: str                         # Right-hand side
    operator: str                    # =, :=, ≈, ∝, etc.
    
    # Variables
    variables: list[Variable]        # All variables used
    inputs: list[str]                # Input variables
    outputs: list[str]               # Output/computed variables
    
    # Context
    description: str                 # Surrounding text description
    constraints: list[str]           # Constraints mentioned
    assumptions: list[str]           # Assumptions for validity
    
    # Dependencies
    depends_on: list[str]            # Other equation IDs
    used_by: list[str]               # Equations that use this
    
    # Implementation hints
    pytorch_hint: Optional[str]      # Suggested PyTorch operation
    numerical_stability: list[str]   # Stability considerations
    
    # Verification
    test_strategy: str               # How to test this equation
    expected_shapes: dict[str, str]  # Variable → shape mapping
```

```python
@dataclass
class Variable:
    """A variable in an equation."""
    
    symbol: str                      # LaTeX symbol: h_t, M_{t-1}, etc.
    name: str                        # Human name: hidden_state, memory
    type: str                        # tensor, scalar, matrix, etc.
    shape: Optional[str]             # Shape expression: (B, D), (N, N)
    dtype: Optional[str]             # float32, int64, etc.
    constraints: list[str]           # > 0, normalized, etc.
    definition: str                  # Where/how defined
```

## Extraction Algorithm

```python
def extract_equations(markdown_path: Path) -> list[ExtractedEquation]:
    """Extract all equations from a paper's markdown."""
    
    content = markdown_path.read_text()
    equations = []
    
    # 1. Find all LaTeX blocks
    latex_blocks = find_latex_blocks(content)
    
    for i, (latex, position, context) in enumerate(latex_blocks):
        # 2. Parse equation structure
        parsed = parse_equation(latex)
        
        # 3. Extract variables
        variables = extract_variables(latex, context)
        
        # 4. Find surrounding description
        description = extract_description(content, position)
        
        # 5. Detect dependencies
        deps = find_equation_dependencies(latex, equations)
        
        # 6. Generate implementation hints
        hints = generate_pytorch_hints(parsed)
        
        # 7. Create equation object
        eq = ExtractedEquation(
            id=f"eq_{i+1}",
            latex=latex,
            variables=variables,
            description=description,
            depends_on=deps,
            pytorch_hint=hints,
            **parsed
        )
        
        equations.append(eq)
    
    # 8. Build dependency graph
    build_dependency_graph(equations)
    
    return equations


def extract_description(content: str, position: int, window: int = 500) -> str:
    """Extract description text around an equation."""
    
    # Look backwards for "where", "defined as", "computed as", etc.
    before = content[max(0, position - window):position]
    after = content[position:position + window]
    
    # Find the sentence containing/preceding the equation
    intro_patterns = [
        r'([^.]*(?:defined as|given by|computed as|is)[^.]*)\.\s*$',
        r'([^.]*(?:where|such that|with)[^.]*)\s*$',
    ]
    
    for pattern in intro_patterns:
        match = re.search(pattern, before, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    # Look for "where" clause after equation
    where_match = re.search(r'^[^.]*where\s+([^.]+)', after, re.IGNORECASE)
    if where_match:
        return f"where {where_match.group(1)}"
    
    return ""


def extract_variables_from_where_clause(text: str) -> list[Variable]:
    """Parse 'where X is the...' patterns."""
    
    variables = []
    
    # Pattern: $symbol$ is the/a description
    pattern = r'\$([^$]+)\$\s+(?:is|denotes|represents)\s+(?:the|a|an)?\s*([^,.$]+)'
    
    for match in re.finditer(pattern, text, re.IGNORECASE):
        symbol = match.group(1)
        description = match.group(2).strip()
        
        # Infer type from description
        var_type = infer_variable_type(description)
        
        variables.append(Variable(
            symbol=symbol,
            name=symbol_to_name(symbol),
            type=var_type,
            definition=description
        ))
    
    return variables
```

## PyTorch Mapping

```python
# Common LaTeX → PyTorch mappings
PYTORCH_HINTS = {
    # Operations
    r'\\softmax': 'F.softmax(x, dim=-1)',
    r'\\exp': 'torch.exp(x)',
    r'\\log': 'torch.log(x)',
    r'\\sigma': 'torch.sigmoid(x)',
    r'\\tanh': 'torch.tanh(x)',
    r'\\relu': 'F.relu(x)',
    r'\\gelu': 'F.gelu(x)',
    
    # Norms
    r'\\|[^|]+\\|_2': 'torch.norm(x, p=2, dim=-1)',
    r'\\|[^|]+\\|_F': 'torch.norm(x, p="fro")',
    r'\\|[^|]+\\|_1': 'torch.norm(x, p=1, dim=-1)',
    
    # Matrix operations
    r'([A-Z])\\T': 'X.transpose(-1, -2)',
    r'([A-Z])^\{-1\}': 'torch.linalg.inv(X)',
    r'\\det': 'torch.linalg.det(X)',
    r'\\tr': 'torch.trace(X)',
    
    # Products
    r'\\odot': 'x * y  # element-wise',
    r'\\otimes': 'torch.outer(x, y)',
    r'\\cdot': 'torch.dot(x, y) or torch.matmul(X, Y)',
    
    # Summations
    r'\\sum_\{([^}]+)\}': 'torch.sum(x, dim=?)',
    r'\\prod_\{([^}]+)\}': 'torch.prod(x, dim=?)',
    r'\\max_\{([^}]+)\}': 'torch.max(x, dim=?)',
    r'\\min_\{([^}]+)\}': 'torch.min(x, dim=?)',
    
    # Attention-specific
    r'\\text\{softmax\}': 'F.softmax(scores / sqrt(d), dim=-1)',
    r'QK\\T/\\sqrt\{d\}': 'torch.matmul(Q, K.transpose(-1,-2)) / math.sqrt(d)',
}


def generate_pytorch_hints(equation: dict) -> str:
    """Generate PyTorch implementation hint for an equation."""
    
    latex = equation['latex']
    hints = []
    
    for pattern, pytorch in PYTORCH_HINTS.items():
        if re.search(pattern, latex):
            hints.append(pytorch)
    
    if hints:
        return '\n'.join(hints)
    
    # Fallback: general structure hint
    if '=' in latex:
        lhs, rhs = latex.split('=', 1)
        return f"# Compute: {symbol_to_name(lhs.strip())} = ..."
    
    return None
```

## Output Format

### equations.yaml

```yaml
paper: "TITANS"
arxiv_id: "2501.00663"
total_equations: 15
extraction_date: "2025-12-30"

equations:
  - id: eq_1
    name: "Surprise Computation"
    label: "Eq. (1)"
    section: "3.2 Memory Module"
    
    latex: "S_t = \\|h_t - M_{t-1}(k_t)\\|_2^2"
    latex_display: |
      S_t = \|h_t - M_{t-1}(k_t)\|_2^2
    
    lhs: "S_t"
    rhs: "\\|h_t - M_{t-1}(k_t)\\|_2^2"
    operator: "="
    
    variables:
      - symbol: "S_t"
        name: "surprise"
        type: "tensor"
        shape: "(B,)"
        description: "Surprise value at timestep t"
      - symbol: "h_t"
        name: "hidden_state"
        type: "tensor"
        shape: "(B, D)"
        description: "Hidden state from current input"
      - symbol: "M_{t-1}"
        name: "memory"
        type: "callable"
        description: "Memory module at previous timestep"
      - symbol: "k_t"
        name: "key"
        type: "tensor"
        shape: "(B, D_k)"
        description: "Query key for memory retrieval"
    
    description: "Measures how surprising the current input is relative to stored memory"
    
    depends_on: []
    used_by: ["eq_2", "eq_3"]
    
    pytorch_hint: |
      # Retrieve memory prediction
      memory_pred = memory(key)  # (B, D)
      # Compute L2 surprise
      surprise = torch.norm(hidden - memory_pred, p=2, dim=-1) ** 2  # (B,)
    
    numerical_stability:
      - "Add epsilon to avoid zero norm: torch.norm(...) + 1e-8"
      - "Consider clamping for very large values"
    
    test_strategy: "Verify output shape (B,), gradient flow, non-negative values"
    
    expected_shapes:
      input_hidden: "(B, D)"
      input_key: "(B, D_k)"
      output_surprise: "(B,)"

  - id: eq_2
    name: "Memory Update Gate"
    # ... more equations
```

## Integration

### With Context Document

The equation extractor outputs integrate into the context document:

```markdown
## Equations

### Equation 1: Surprise Computation
**LaTeX:** $S_t = \|h_t - M_{t-1}(k_t)\|_2^2$

**Variables:**
- $S_t$: surprise (tensor, shape: (B,))
- $h_t$: hidden_state (tensor, shape: (B, D))
- $M_{t-1}$: memory module (callable)
- $k_t$: key (tensor, shape: (B, D_k))

**Description:** Measures how surprising the current input is relative to stored memory

**PyTorch Hint:**
```python
memory_pred = memory(key)  # (B, D)
surprise = torch.norm(hidden - memory_pred, p=2, dim=-1) ** 2  # (B,)
```

**Dependencies:** None
**Used by:** Equation 2 (Memory Update Gate)
```

### With Verification

```python
# Auto-generated test skeleton
def test_eq1_surprise_computation():
    """Test Equation 1: Surprise Computation
    
    LaTeX: S_t = \|h_t - M_{t-1}(k_t)\|_2^2
    """
    B, D, D_k = 4, 64, 32
    
    # Inputs
    hidden = torch.randn(B, D)
    key = torch.randn(B, D_k)
    memory = MockMemory(D_k, D)
    
    # Compute
    surprise = compute_surprise(hidden, key, memory)
    
    # Verify shape
    assert surprise.shape == (B,), f"Expected (B,), got {surprise.shape}"
    
    # Verify non-negative
    assert (surprise >= 0).all(), "Surprise must be non-negative"
    
    # Verify gradient flow
    surprise.sum().backward()
    assert hidden.grad is not None, "Gradient should flow to hidden"
```

## Command Line

```bash
# Extract equations from a single paper
uv run scripts/extract_equations.py papers/titans.md --output equations/titans.yaml

# Extract with verbose output
uv run scripts/extract_equations.py papers/*.md -v

# Generate test skeletons
uv run scripts/extract_equations.py papers/titans.md --generate-tests

# Validate extracted equations
uv run scripts/extract_equations.py --validate equations/titans.yaml
```
