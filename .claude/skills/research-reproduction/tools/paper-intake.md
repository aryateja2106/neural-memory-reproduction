# Paper Intake Tool

## Purpose
Handles ingestion of research papers from multiple sources (PDF files, arXiv URLs, local files) and prepares them for extraction.

## Supported Input Formats

### 1. Local PDF Files
```bash
# Single paper
uv run scripts/extract_paper.py papers/titans.pdf

# Multiple papers
uv run scripts/extract_paper.py papers/*.pdf
```

### 2. arXiv URLs
```bash
# Direct arXiv PDF link
uv run scripts/extract_paper.py "https://arxiv.org/pdf/2501.00663"

# arXiv abstract page (auto-converts to PDF)
uv run scripts/extract_paper.py "https://arxiv.org/abs/2501.00663"
```

### 3. URL List File
```bash
# papers.txt contains one URL per line
uv run scripts/extract_paper.py --from-file papers.txt
```

## Intake Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                      PAPER INTAKE PIPELINE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  INPUT SOURCES                                                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ Local    │  │ arXiv    │  │ URL      │  │ DOI      │       │
│  │ PDF      │  │ ID       │  │ Direct   │  │ Link     │       │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘       │
│       │             │             │             │              │
│       └─────────────┴─────────────┴─────────────┘              │
│                           │                                     │
│                           ▼                                     │
│              ┌────────────────────────┐                        │
│              │     URL Resolver       │                        │
│              │  • Detect source type  │                        │
│              │  • Convert to PDF URL  │                        │
│              │  • Download if needed  │                        │
│              └───────────┬────────────┘                        │
│                          │                                      │
│                          ▼                                      │
│              ┌────────────────────────┐                        │
│              │     PDF Processor      │                        │
│              │  • markitdown (primary)│                        │
│              │  • PyMuPDF (fallback)  │                        │
│              │  • Preserve LaTeX      │                        │
│              └───────────┬────────────┘                        │
│                          │                                      │
│                          ▼                                      │
│              ┌────────────────────────┐                        │
│              │    Markdown Output     │                        │
│              │  papers/[name].md      │                        │
│              │  papers/[name].context.md                       │
│              └────────────────────────┘                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## URL Resolution Logic

```python
def resolve_paper_url(input_str: str) -> tuple[str, str]:
    """
    Resolve various input formats to downloadable PDF URL.
    Returns: (pdf_url, paper_id)
    """
    
    # arXiv abstract page → PDF
    if "arxiv.org/abs/" in input_str:
        arxiv_id = input_str.split("/abs/")[-1].split("v")[0]
        return f"https://arxiv.org/pdf/{arxiv_id}.pdf", arxiv_id
    
    # arXiv PDF direct
    if "arxiv.org/pdf/" in input_str:
        arxiv_id = input_str.split("/pdf/")[-1].replace(".pdf", "")
        return input_str, arxiv_id
    
    # OpenReview
    if "openreview.net" in input_str:
        # Extract forum ID and convert to PDF
        forum_id = extract_openreview_id(input_str)
        return f"https://openreview.net/pdf?id={forum_id}", forum_id
    
    # Semantic Scholar
    if "semanticscholar.org" in input_str:
        paper_id = extract_s2_id(input_str)
        return get_s2_pdf_url(paper_id), paper_id
    
    # Direct PDF URL
    if input_str.endswith(".pdf"):
        filename = Path(input_str).stem
        return input_str, filename
    
    # Local file
    if Path(input_str).exists():
        return input_str, Path(input_str).stem
    
    raise ValueError(f"Cannot resolve paper source: {input_str}")
```

## Output Structure

After intake, each paper produces:

```
papers/
├── titans/
│   ├── original.pdf          # Downloaded PDF (if from URL)
│   ├── full_text.md          # Complete markdown conversion
│   └── titans.context.md     # Extracted context (equations, algorithms, etc.)
├── miras/
│   ├── original.pdf
│   ├── full_text.md
│   └── miras.context.md
└── nested-learning/
    ├── original.pdf
    ├── full_text.md
    └── nested-learning.context.md
```

## Metadata Extraction

Each paper's metadata is captured:

```yaml
# papers/titans/metadata.yaml
title: "TITANS: Learning to Memorize at Test Time"
authors:
  - Ali Behrouz
  - Peilin Zhong
  - Vahab Mirrokni
arxiv_id: "2501.00663"
publication: "arXiv preprint"
year: 2025
abstract: |
  Sequence models are central to modern AI...
keywords:
  - transformers
  - memory
  - attention
  - long-context
citations: 47
pdf_url: "https://arxiv.org/pdf/2501.00663.pdf"
intake_date: "2025-12-30"
```

## Batch Processing

For multiple papers:

```bash
# Create intake manifest
cat > papers/intake.yaml << EOF
papers:
  - name: titans
    source: "https://arxiv.org/abs/2501.00663"
    priority: 1  # Process first (foundation)
    
  - name: miras
    source: "https://arxiv.org/abs/2504.13173"
    priority: 2  # Depends on TITANS
    depends_on: [titans]
    
  - name: nested-learning
    source: "papers/nested_learning.pdf"
    priority: 3  # Depends on both
    depends_on: [titans, miras]
EOF

# Process all papers
uv run scripts/extract_paper.py --manifest papers/intake.yaml
```

## Error Handling

```python
class IntakeError(Exception):
    """Base class for intake errors."""
    pass

class DownloadError(IntakeError):
    """Failed to download paper."""
    pass

class ConversionError(IntakeError):
    """Failed to convert PDF to markdown."""
    pass

class ExtractionError(IntakeError):
    """Failed to extract context."""
    pass

# Retry logic for downloads
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(DownloadError)
)
def download_paper(url: str, output_path: Path) -> None:
    """Download with retry."""
    ...
```

## Integration with Extraction Agent

After intake, the extraction agent is spawned:

```python
# In orchestrator
async def process_paper(paper_source: str):
    # 1. Intake
    pdf_path, paper_id = resolve_and_download(paper_source)
    
    # 2. Convert to markdown
    markdown_path = convert_to_markdown(pdf_path)
    
    # 3. Spawn extraction agent
    await spawn_extraction_agent(
        paper_id=paper_id,
        markdown_path=markdown_path,
        output_path=f"papers/{paper_id}/{paper_id}.context.md"
    )
```

## Command Line Interface

```bash
# Basic usage
uv run scripts/extract_paper.py <source>

# Options
  --output, -o DIR       Output directory (default: papers/)
  --format, -f FORMAT    Output format: markdown, context, both (default: both)
  --no-download          Don't download, just resolve URL
  --metadata-only        Only extract metadata
  --verbose, -v          Verbose output
  --parallel, -p N       Process N papers in parallel
```

## Validation Checklist

After intake, verify:

- [ ] PDF downloaded successfully (if from URL)
- [ ] Markdown conversion complete
- [ ] LaTeX equations preserved
- [ ] Figures referenced (even if not extracted)
- [ ] Tables converted properly
- [ ] Algorithm blocks identified
- [ ] Metadata extracted
- [ ] Context document created
