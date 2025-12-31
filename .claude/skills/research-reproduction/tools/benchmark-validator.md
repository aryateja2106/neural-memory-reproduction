# Benchmark Validator Tool

## Purpose
Validates reproduction results against paper-reported benchmarks, tracking metrics, comparing performance, and documenting any deviations.

## Validation Philosophy

```
┌─────────────────────────────────────────────────────────────────┐
│                    BENCHMARK VALIDATION FLOW                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  GOAL: Confirm reproduction matches paper within tolerance       │
│                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   Paper     │    │   Our       │    │   Compare   │         │
│  │   Results   │ → │   Results   │ → │   & Report  │         │
│  │   (Table 1) │    │   (Run)     │    │             │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│                                                                  │
│  Acceptable Variance:                                            │
│  • ±1% for accuracy metrics (common in ML)                       │
│  • ±5% for timing/throughput (hardware dependent)                │
│  • Must document and explain any larger deviations               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Benchmark Extraction

### From Paper Tables

```python
@dataclass
class BenchmarkEntry:
    """A single benchmark result from the paper."""
    
    # Identification
    table_id: str              # "Table 1", "Table 2", etc.
    model_name: str            # Model configuration name
    dataset: str               # Dataset name
    
    # Metrics
    metrics: dict[str, float]  # metric_name -> value
    metric_units: dict[str, str]  # metric_name -> unit (%, ms, etc.)
    
    # Conditions
    hardware: Optional[str]    # GPU type if mentioned
    batch_size: Optional[int]
    sequence_length: Optional[int]
    num_params: Optional[int]
    
    # Context
    notes: list[str]           # Footnotes, conditions
    is_baseline: bool          # Is this a baseline model?
    
    # Confidence
    has_std: bool              # Standard deviation reported?
    std_values: dict[str, float]  # If std reported


@dataclass
class BenchmarkTable:
    """Complete benchmark table from paper."""
    
    table_id: str
    title: str
    description: str
    
    # Structure
    columns: list[str]         # Column headers
    row_groups: dict[str, list[str]]  # Group name -> row names
    
    # Data
    entries: list[BenchmarkEntry]
    
    # Metadata
    datasets_used: list[str]
    models_compared: list[str]
    primary_metric: str        # The main metric (e.g., "Accuracy")
    
    # Reproduction guidance
    our_model_row: str         # Which row is our model
    baseline_rows: list[str]   # Which rows are baselines
```

### Extraction from Paper

```python
def extract_benchmarks(context_doc: Path) -> list[BenchmarkTable]:
    """Extract benchmark tables from paper context document."""
    
    content = context_doc.read_text()
    tables = []
    
    # Find benchmark sections
    benchmark_pattern = r'## Benchmarks\s*\n(.*?)(?=\n## |\Z)'
    match = re.search(benchmark_pattern, content, re.DOTALL)
    
    if not match:
        return []
    
    benchmark_section = match.group(1)
    
    # Parse tables
    table_pattern = r'### (Table \d+[^:]*):?\s*([^\n]*)\n(.*?)(?=\n### |\Z)'
    
    for table_match in re.finditer(table_pattern, benchmark_section, re.DOTALL):
        table_id = table_match.group(1)
        title = table_match.group(2).strip()
        content = table_match.group(3)
        
        # Parse markdown table
        entries = parse_markdown_table(content)
        
        tables.append(BenchmarkTable(
            table_id=table_id,
            title=title,
            entries=entries,
            # ... parse additional metadata
        ))
    
    return tables
```

## Benchmark Specification File

### benchmarks.yaml

```yaml
paper: "TITANS"
arxiv_id: "2501.00663"

tables:
  - id: "Table 1"
    title: "Language Modeling Performance"
    description: "Perplexity on various datasets"
    
    primary_metric: "perplexity"
    lower_is_better: true
    
    datasets:
      - name: "WikiText-103"
        split: "test"
        preprocessing: "standard"
      - name: "PG-19"
        split: "test"
    
    models:
      - name: "TITANS-MAC"
        is_ours: true
        config: "configs/titans_mac.yaml"
      
      - name: "Transformer++"
        is_baseline: true
      
      - name: "Mamba-2"
        is_baseline: true
    
    results:
      - model: "TITANS-MAC"
        dataset: "WikiText-103"
        metrics:
          perplexity: 18.2
          perplexity_std: 0.1
        conditions:
          context_length: 8192
          batch_size: 32
      
      - model: "Transformer++"
        dataset: "WikiText-103"
        metrics:
          perplexity: 19.4
        conditions:
          context_length: 8192
    
    tolerance:
      perplexity: 0.5  # Acceptable deviation
      perplexity_relative: 0.02  # Or 2%
  
  - id: "Table 2"
    title: "Inference Speed"
    # ...

reproduction_targets:
  # What we need to reproduce
  primary:
    - table: "Table 1"
      rows: ["TITANS-MAC"]
      datasets: ["WikiText-103"]
  
  secondary:
    - table: "Table 2"
      rows: ["TITANS-MAC"]
```

## Running Benchmarks

### benchmark_runner.py Integration

```python
async def run_benchmarks(
    model_path: str,
    benchmark_spec: str,
    output_dir: str,
    device: str = "cuda"
) -> BenchmarkResults:
    """
    Run benchmarks specified in YAML file.
    
    Usage:
        uv run scripts/benchmark_runner.py \
            --model checkpoints/titans_mac.pt \
            --spec benchmarks/titans.yaml \
            --output results/
    """
    
    # Load spec
    spec = yaml.safe_load(Path(benchmark_spec).read_text())
    
    # Load model
    model = load_model(model_path)
    model.to(device)
    model.eval()
    
    results = BenchmarkResults(spec['paper'])
    
    # Run each benchmark
    for table in spec['tables']:
        for target in spec['reproduction_targets']['primary']:
            if target['table'] != table['id']:
                continue
            
            for model_name in target['rows']:
                for dataset_name in target['datasets']:
                    # Load dataset
                    dataset = load_dataset(dataset_name, table['datasets'])
                    
                    # Run evaluation
                    metrics = evaluate_model(
                        model=model,
                        dataset=dataset,
                        metrics=table['primary_metric']
                    )
                    
                    # Record results
                    results.add_result(
                        table=table['id'],
                        model=model_name,
                        dataset=dataset_name,
                        metrics=metrics
                    )
    
    # Compare with paper
    comparison = compare_with_paper(results, spec)
    
    # Generate report
    report = generate_benchmark_report(comparison)
    
    return results, comparison, report
```

## Comparison Logic

```python
@dataclass
class MetricComparison:
    """Comparison of a single metric."""
    
    metric_name: str
    paper_value: float
    our_value: float
    
    # Computed
    absolute_diff: float
    relative_diff: float  # As percentage
    
    # Tolerance check
    tolerance_absolute: float
    tolerance_relative: float
    within_tolerance: bool
    
    # Assessment
    status: str  # "match", "close", "deviation", "significant_deviation"
    explanation: Optional[str]


def compare_metrics(
    paper_results: dict[str, float],
    our_results: dict[str, float],
    tolerances: dict[str, dict]
) -> list[MetricComparison]:
    """Compare our results against paper."""
    
    comparisons = []
    
    for metric, paper_value in paper_results.items():
        our_value = our_results.get(metric)
        
        if our_value is None:
            comparisons.append(MetricComparison(
                metric_name=metric,
                paper_value=paper_value,
                our_value=None,
                status="missing",
                explanation="Metric not computed"
            ))
            continue
        
        # Compute differences
        abs_diff = abs(our_value - paper_value)
        rel_diff = abs_diff / paper_value if paper_value != 0 else float('inf')
        
        # Get tolerances
        tol = tolerances.get(metric, {})
        tol_abs = tol.get('absolute', float('inf'))
        tol_rel = tol.get('relative', 0.05)  # Default 5%
        
        # Check tolerance
        within_tol = abs_diff <= tol_abs or rel_diff <= tol_rel
        
        # Determine status
        if rel_diff < 0.01:  # < 1%
            status = "match"
        elif rel_diff < 0.03:  # < 3%
            status = "close"
        elif within_tol:
            status = "within_tolerance"
        elif rel_diff < 0.10:  # < 10%
            status = "deviation"
        else:
            status = "significant_deviation"
        
        comparisons.append(MetricComparison(
            metric_name=metric,
            paper_value=paper_value,
            our_value=our_value,
            absolute_diff=abs_diff,
            relative_diff=rel_diff * 100,
            tolerance_absolute=tol_abs,
            tolerance_relative=tol_rel * 100,
            within_tolerance=within_tol,
            status=status,
        ))
    
    return comparisons
```

## Report Generation

### Benchmark Report (Markdown)

```markdown
# Benchmark Validation Report

**Paper:** TITANS (arXiv:2501.00663)
**Date:** 2025-12-30
**Hardware:** NVIDIA A100 (Google Colab)

## Summary

| Status | Count |
|--------|-------|
| ✓ Match (<1% diff) | 8 |
| ~ Close (<3% diff) | 3 |
| ○ Within Tolerance | 1 |
| ⚠ Deviation | 0 |
| ✗ Significant | 0 |

**Overall:** ✓ Reproduction Successful

## Table 1: Language Modeling Performance

### WikiText-103

| Model | Metric | Paper | Ours | Diff | Status |
|-------|--------|-------|------|------|--------|
| TITANS-MAC | Perplexity | 18.2 | 18.3 | +0.55% | ✓ Match |
| TITANS-MAC | Params (M) | 125 | 125 | 0% | ✓ Match |

### Conditions
- Context Length: 8192
- Batch Size: 32
- Hardware: A100 40GB (paper used A100 80GB)

## Table 2: Inference Speed

| Model | Tokens/sec (Paper) | Tokens/sec (Ours) | Diff | Status |
|-------|-------------------|-------------------|------|--------|
| TITANS-MAC | 15,234 | 14,891 | -2.25% | ~ Close |

### Notes
- Slight speed difference likely due to hardware variation (A100 40GB vs 80GB)
- Memory bandwidth difference explains ~2% throughput reduction

## Deviations Explained

### None significant

All metrics within acceptable tolerance. Minor variations explained by:
1. Hardware differences (Colab A100 vs paper's cluster)
2. Random seed variation (results averaged over 3 runs)
3. PyTorch version differences (we use 2.5, paper used 2.3)

## Reproduction Confidence

**High Confidence** - Results closely match paper across all reported metrics.

## Files Generated

- `results/titans_wikitext103.json` - Raw metrics
- `results/comparison.yaml` - Detailed comparison
- `results/benchmark_report.md` - This report
```

## CLI Commands

```bash
# Run benchmarks
uv run scripts/benchmark_runner.py run \
    --model checkpoints/titans.pt \
    --spec benchmarks.yaml

# Compare with paper
uv run scripts/benchmark_runner.py compare \
    --results results/titans.json \
    --paper benchmarks.yaml

# Generate report
uv run scripts/benchmark_runner.py report \
    --comparison results/comparison.yaml \
    --output results/report.md

# Quick validation (run + compare + report)
uv run scripts/benchmark_runner.py validate \
    --model checkpoints/titans.pt \
    --spec benchmarks.yaml

# List available benchmarks
uv run scripts/benchmark_runner.py list
```

## Integration with LeCoder-cgpu

For benchmarks requiring GPU:

```bash
# Connect to Colab with benchmark script
lecoder-cgpu connect --startup-command "cd /content && uv run scripts/benchmark_runner.py run --spec benchmarks.yaml"

# Or run directly
lecoder-cgpu run "uv run scripts/benchmark_runner.py run --spec benchmarks.yaml"

# Download results
lecoder-cgpu download results/ ./results/
```

## Benchmark Datasets

### Supported Datasets

```python
BENCHMARK_DATASETS = {
    # Language Modeling
    'wikitext-103': {
        'source': 'huggingface',
        'path': 'wikitext',
        'name': 'wikitext-103-v1',
        'splits': ['train', 'validation', 'test'],
    },
    'pg-19': {
        'source': 'huggingface',
        'path': 'pg19',
        'splits': ['train', 'validation', 'test'],
    },
    'c4': {
        'source': 'huggingface',
        'path': 'allenai/c4',
        'name': 'en',
        'streaming': True,
    },
    
    # Long Context
    'passkey-retrieval': {
        'source': 'synthetic',
        'generator': 'generate_passkey_test',
        'lengths': [4096, 8192, 16384, 32768, 65536, 131072],
    },
    'needle-in-haystack': {
        'source': 'synthetic',
        'generator': 'generate_needle_test',
    },
    
    # Classification
    'imdb': {
        'source': 'huggingface',
        'path': 'imdb',
    },
    'sst2': {
        'source': 'huggingface',
        'path': 'sst2',
    },
}
```

## Tolerance Guidelines

```yaml
# Standard tolerances for different metric types

accuracy_metrics:
  # Classification, retrieval, etc.
  relative_tolerance: 0.01  # 1%
  absolute_tolerance: 0.5   # 0.5 percentage points

perplexity_metrics:
  relative_tolerance: 0.03  # 3%
  absolute_tolerance: 0.5

speed_metrics:
  # Very hardware dependent
  relative_tolerance: 0.10  # 10%
  absolute_tolerance: null

memory_metrics:
  relative_tolerance: 0.05  # 5%
  absolute_tolerance: 100   # 100 MB

parameter_counts:
  relative_tolerance: 0.001  # 0.1%
  absolute_tolerance: 1000   # 1K params (rounding)
```
