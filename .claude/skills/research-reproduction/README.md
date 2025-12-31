# Research Paper Reproduction Skill

> **A Claude Code skill for systematically reproducing AI/ML research papers with equation-first verification and multi-agent orchestration.**

## Overview

This skill enables Claude to take research papers (PDFs, arXiv links) and produce:
- Clean, functional PyTorch implementations
- Comprehensive test suites with equation verification
- Quality-checked code (ruff formatted, ty type-checked)
- Git-ready repositories
- Benchmark validation against paper results

## Quick Start

### 1. Tell Claude what you want to reproduce

```
Reproduce the TITANS paper (arXiv:2501.00663). Focus on the memory 
module and attention mechanism. I have a Colab Pro subscription 
with A100 access.
```

### 2. Claude will:

1. **Clarify intent**: Validate/implement/extend? Single or multiple papers?
2. **Extract context**: Spawn agents to process each paper separately
3. **Create implementation plan**: Dependency graph, build order
4. **Implement equation-first**: Write tests before code
5. **Quality check**: Format, lint, type check
6. **Initialize git**: Ready for `gh repo create`

### 3. Execute on Colab

```bash
# Connect to Colab with LeCoder-cgpu
lecoder-cgpu connect --variant gpu

# Upload and run
lecoder-cgpu upload ./titans-reproduction.tar.gz /content/
lecoder-cgpu run "bash setup_colab.sh && python train.py"
```

## Skill Structure

```
research-reproduction/
├── SKILL.md                    # Main orchestrator (read this first)
├── README.md                   # This file
│
├── prompts/                    # Agent prompts
│   ├── extraction-agent.md     # PDF → context document
│   ├── implementation-agent.md # Context → PyTorch code
│   ├── verification-agent.md   # Code → tests
│   └── documentation-agent.md  # Code → docs
│
├── templates/                  # Output templates
│   ├── context-document.md     # Extraction output format
│   ├── implementation-plan.md  # Multi-paper synthesis
│   ├── readme-template.md      # Project README
│   └── test-template.md        # Equation test structure
│
├── scripts/                    # UV single-file scripts
│   ├── extract_paper.py        # PDF processing
│   ├── verify_equations.py     # Equation → test mapping
│   ├── quality_check.py        # Ruff + ty + pytest
│   ├── benchmark_runner.py     # Paper benchmark validation
│   └── generate_docs.py        # Documentation generation
│
├── tools/                      # Tool documentation
│   ├── paper-intake.md         # PDF/URL ingestion
│   ├── equation-extractor.md   # LaTeX extraction
│   ├── verification-engine.md  # Test generation
│   ├── benchmark-validator.md  # Results comparison
│   └── colab-execution.md      # LeCoder-cgpu integration
│
└── references/                 # Reference materials
    ├── equation-patterns.md    # Common equation tests
    └── pytorch-patterns.md     # Implementation patterns
```

## Workflow Phases

### Phase 0: Intent Clarification
Claude asks:
- **Scope**: Validate, implement, or extend the paper?
- **Papers**: Single paper or interconnected papers?
- **Environment**: Local CPU/GPU or Colab?
- **Output**: Notebook, scripts, or both?

### Phase 1: Parallel Extraction
For each paper:
- Spawn extraction agent
- Convert PDF → Markdown
- Extract equations, algorithms, architecture
- Create `.context.md` file

### Phase 2: Context Synthesis
- Read all `.context.md` files
- Build dependency graph
- Create `IMPLEMENTATION_PLAN.md`
- Determine build order

### Phase 3: Equation-First Implementation
For each equation:
1. Write test FIRST
2. Run test (should fail - no impl yet)
3. Implement equation
4. Run test (should pass)
5. Verify shapes, gradients, stability

### Phase 4: Code Quality
```bash
uv run scripts/quality_check.py --fix
```
- Ruff format
- Ruff lint
- ty type check
- pytest with coverage

### Phase 5: Documentation
- Generate README.md
- Create ARCHITECTURE.md
- Document modules with paper references

### Phase 6: Git & Execution
```bash
git init
git add .
git commit -m "Initial implementation of [Paper]"
# Ready for: gh repo create
```

## Output Structure

```
[paper-name]-reproduction/
├── .gitignore
├── .python-version
├── pyproject.toml
├── uv.lock
├── README.md
├── ARCHITECTURE.md
├── IMPLEMENTATION_PLAN.md
│
├── papers/
│   └── [paper].context.md      # Extracted context
│
├── src/
│   ├── __init__.py
│   ├── model.py                # Main model
│   ├── layers/                 # Individual layers
│   │   ├── __init__.py
│   │   ├── attention.py
│   │   └── memory.py
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
│
├── tests/
│   ├── test_equations/         # Equation-level tests
│   │   ├── test_eq1_surprise.py
│   │   └── test_eq2_memory_update.py
│   ├── test_layers/            # Layer-level tests
│   └── test_integration.py     # Full model tests
│
├── configs/
│   ├── default.yaml
│   ├── small.yaml              # For quick testing
│   └── paper.yaml              # Match paper exactly
│
├── notebooks/
│   ├── exploration.ipynb
│   └── quickstart.ipynb
│
├── scripts/
│   ├── train.sh
│   └── evaluate.sh
│
└── docs/
    ├── equations.md
    └── benchmarks.md
```

## Key Features

### Multi-Paper Support
Handle interconnected papers:
```
Paper A (Foundation) → Paper B (Extension) → Paper C (Application)
```
Each creates `.context.md`, orchestrator builds dependency graph.

### Equation-First Verification
Every equation gets 6 standard tests:
- `test_output_shape` - Tensor dimensions match paper
- `test_gradient_flow` - Gradients propagate correctly
- `test_numerical_stability` - Handles edge cases
- `test_deterministic` - Same input → same output
- `test_batch_independence` - Batch elements don't affect each other
- `test_edge_cases` - Empty sequences, single tokens, etc.

### Context Compression
- PDFs processed by subagents (not main context)
- Only `.context.md` loaded into orchestrator
- Prevents context window overflow

### Quality Assurance
```bash
# Format code
uv run ruff format .

# Lint
uv run ruff check . --fix

# Type check
uv run ty check src/

# Test
uv run pytest tests/ -v --cov=src
```

## Scripts Reference

### extract_paper.py
```bash
# Extract from local PDF
uv run scripts/extract_paper.py papers/titans.pdf

# Extract from arXiv
uv run scripts/extract_paper.py "https://arxiv.org/abs/2501.00663"

# Batch extraction
uv run scripts/extract_paper.py papers/*.pdf
```

### verify_equations.py
```bash
# List equations and test status
uv run scripts/verify_equations.py list

# Run equation verification
uv run scripts/verify_equations.py verify

# Generate missing tests
uv run scripts/verify_equations.py generate --paper titans
```

### quality_check.py
```bash
# Check all
uv run scripts/quality_check.py

# Auto-fix issues
uv run scripts/quality_check.py --fix

# With coverage
uv run scripts/quality_check.py --coverage
```

### benchmark_runner.py
```bash
# Run benchmarks
uv run scripts/benchmark_runner.py run --model checkpoints/best.pt

# Compare with paper
uv run scripts/benchmark_runner.py compare --results results.json

# Generate report
uv run scripts/benchmark_runner.py report
```

## Colab Integration

### Resource Planning
| GPU | Units/hr | Max hrs | Best for |
|-----|----------|---------|----------|
| T4 | 1.96 | ~51 | Development |
| L4 | 3.00 | ~33 | Training |
| A100 | 12.00 | ~8 | Benchmarks |

### Workflow
```bash
# Prepare package
uv run scripts/prepare_colab.py

# Connect
lecoder-cgpu connect --variant gpu

# Upload
lecoder-cgpu upload colab_package.tar.gz /content/

# Setup
lecoder-cgpu run "bash setup_colab.sh"

# Train
lecoder-cgpu run "python train.py"

# Download results
lecoder-cgpu download /content/checkpoints ./checkpoints/
lecoder-cgpu download /content/results ./results/
```

## Requirements

- Python 3.11+
- UV (Python package manager)
- Node.js (for lecoder-cgpu)
- Optional: Google Colab Pro (for GPU)

## Example Usage

### Reproduce Single Paper
```
I want to reproduce the TITANS paper focusing on the MAC variant.
Extract all equations and implement with full test coverage.
```

### Reproduce Multi-Paper System
```
Reproduce TITANS, MIRAS, and Nested Learning together. These are 
interconnected - MIRAS extends TITANS memory, and Nested Learning 
uses both. Build in dependency order.
```

### Validate Existing Implementation
```
I have this implementation of the TITANS memory module. Verify it 
matches the paper equations and check numerical stability.
```

## Contributing

This skill is part of the LeCoder ecosystem by LeSearch AI.

- GitHub: [aryateja2106/LeCoder-cgpu-CLI](https://github.com/aryateja2106/LeCoder-cgpu-CLI)
- Website: [lesearch.ai](https://lesearch.ai)

## License

MIT License - See LICENSE file for details.

---

**"Less Code, More Reproduction"** - LeCoder
