# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "rich>=13.0",
#     "pyyaml>=6.0",
#     "jinja2>=3.0",
# ]
# ///
"""
Documentation Generator - Create README, ARCHITECTURE, and module docs.

Generates comprehensive documentation from context files and code structure.

Usage:
    uv run scripts/generate_docs.py --project-dir .
    uv run scripts/generate_docs.py --project-dir . --output docs/
    uv run scripts/generate_docs.py --project-dir . --template custom_readme.md
"""

import argparse
import ast
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, FileSystemLoader, BaseLoader
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class EquationInfo:
    """Information about an equation from the paper."""

    number: int
    name: str
    latex: str
    file_path: str | None = None
    line_number: int | None = None
    test_status: str = "❓"  # ✅, ❌, ❓


@dataclass
class ModuleInfo:
    """Information about a Python module."""

    name: str
    path: Path
    docstring: str | None = None
    classes: list[str] = field(default_factory=list)
    functions: list[str] = field(default_factory=list)
    equations: list[int] = field(default_factory=list)


@dataclass
class ProjectInfo:
    """Complete project information for documentation."""

    name: str
    paper_title: str
    paper_id: str
    authors: str
    arxiv_link: str
    equations: list[EquationInfo] = field(default_factory=list)
    modules: list[ModuleInfo] = field(default_factory=list)
    config: dict = field(default_factory=dict)
    test_coverage: float = 0.0
    benchmark_results: dict = field(default_factory=dict)


# =============================================================================
# EXTRACTORS
# =============================================================================


def extract_paper_info(context_path: Path) -> dict:
    """Extract paper metadata from context file."""
    content = context_path.read_text()

    info = {
        "paper_id": context_path.stem.replace(".context", ""),
        "paper_title": "",
        "authors": "",
        "arxiv_link": "",
        "equations": [],
    }

    # Extract title (first # heading)
    title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    if title_match:
        info["paper_title"] = title_match.group(1).strip()

    # Extract arXiv link
    arxiv_match = re.search(r"(https?://arxiv\.org/abs/[\d.]+)", content)
    if arxiv_match:
        info["arxiv_link"] = arxiv_match.group(1)

    # Extract authors
    authors_match = re.search(r"Authors?:\s*(.+?)(?:\n|$)", content, re.IGNORECASE)
    if authors_match:
        info["authors"] = authors_match.group(1).strip()

    # Extract equations
    equation_pattern = r"###\s+Equation\s+(\d+):\s+(.+?)\n.*?```(?:latex|math)?\n(.+?)```"
    for match in re.finditer(equation_pattern, content, re.DOTALL):
        info["equations"].append(
            {
                "number": int(match.group(1)),
                "name": match.group(2).strip(),
                "latex": match.group(3).strip(),
            }
        )

    return info


def extract_module_info(module_path: Path) -> ModuleInfo:
    """Extract information from a Python module."""
    content = module_path.read_text()

    try:
        tree = ast.parse(content)
    except SyntaxError:
        return ModuleInfo(
            name=module_path.stem,
            path=module_path,
            docstring="(Parse error)",
        )

    # Get module docstring
    docstring = ast.get_docstring(tree)

    # Get classes and functions
    classes = []
    functions = []
    equations = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            classes.append(node.name)
        elif isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            functions.append(node.name)

    # Find equation references in comments/docstrings
    eq_pattern = r"Eq(?:uation)?\.?\s*(\d+)"
    for match in re.finditer(eq_pattern, content, re.IGNORECASE):
        eq_num = int(match.group(1))
        if eq_num not in equations:
            equations.append(eq_num)

    return ModuleInfo(
        name=module_path.stem,
        path=module_path,
        docstring=docstring,
        classes=classes,
        functions=functions,
        equations=sorted(equations),
    )


def find_equation_implementations(
    src_dir: Path, equations: list[EquationInfo]
) -> list[EquationInfo]:
    """Find where equations are implemented in source code."""
    eq_pattern = r"#.*Eq(?:uation)?\.?\s*(\d+)|Equation\s+(\d+)"

    for py_file in src_dir.rglob("*.py"):
        content = py_file.read_text()
        lines = content.split("\n")

        for i, line in enumerate(lines, 1):
            for match in re.finditer(eq_pattern, line, re.IGNORECASE):
                eq_num = int(match.group(1) or match.group(2))

                for eq in equations:
                    if eq.number == eq_num and eq.file_path is None:
                        eq.file_path = str(py_file.relative_to(src_dir.parent))
                        eq.line_number = i
                        break

    return equations


def check_equation_tests(tests_dir: Path, equations: list[EquationInfo]) -> list[EquationInfo]:
    """Check test status for each equation."""
    for eq in equations:
        test_pattern = f"test_eq{eq.number}_*.py"
        test_files = list(tests_dir.rglob(test_pattern))

        if test_files:
            eq.test_status = "✅"
        else:
            # Check for test in combined files
            for test_file in tests_dir.rglob("test_*.py"):
                content = test_file.read_text()
                if f"TestEquation{eq.number}" in content or f"test_eq{eq.number}" in content:
                    eq.test_status = "✅"
                    break
            else:
                eq.test_status = "❌"

    return equations


# =============================================================================
# TEMPLATES
# =============================================================================

README_TEMPLATE = """# {{ project.paper_title }} - Reproduction

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![uv](https://img.shields.io/badge/uv-managed-blueviolet.svg)](https://github.com/astral-sh/uv)
[![Tests](https://img.shields.io/badge/tests-{{ 'passing' if project.test_coverage > 80 else 'partial' }}-{{ 'green' if project.test_coverage > 80 else 'yellow' }}.svg)]()

> **Paper:** [{{ project.paper_title }}]({{ project.arxiv_link }})  
> **Authors:** {{ project.authors }}  
> **Reproduced:** {{ date }} using [LeCoder Research Reproduction](https://github.com/lesearch-ai/research-reproduction)

---

## 🎯 Quick Start

```bash
# Clone and setup
git clone https://github.com/{{ github_user }}/{{ project.name }}.git
cd {{ project.name }}

# Install dependencies
uv sync

# Run tests
uv run pytest tests/ -x -q

# Train small model
uv run python -m src.train --config configs/small.yaml
```

---

## 📊 Results

{% if project.benchmark_results %}
| Dataset | Metric | Paper | Ours | Δ |
|---------|--------|-------|------|---|
{% for result in project.benchmark_results.get('results', []) %}
| {{ result.name }} | {{ result.metric }} | {{ "%.2f"|format(result.paper_value) }} | {{ "%.2f"|format(result.our_value) }} | {{ "%+.1f%%"|format(result.delta_pct) }} |
{% endfor %}
{% else %}
*Run `uv run scripts/benchmark_runner.py` to generate results.*
{% endif %}

---

## 🔬 Equations Implemented

| # | Name | Location | Tests |
|---|------|----------|-------|
{% for eq in project.equations %}
| {{ eq.number }} | {{ eq.name }} | {% if eq.file_path %}`{{ eq.file_path }}:{{ eq.line_number }}`{% else %}*Not found*{% endif %} | {{ eq.test_status }} |
{% endfor %}

---

## 📁 Project Structure

```
{{ project.name }}/
├── src/                 # Source code
{% for module in project.modules %}
│   ├── {{ module.path.name }}{{ "  # " + module.docstring[:40] + "..." if module.docstring and len(module.docstring) > 40 else ("  # " + module.docstring if module.docstring else "") }}
{% endfor %}
├── tests/               # Test suite
├── configs/             # Configuration files
├── notebooks/           # Jupyter notebooks
├── papers/              # Paper context files
└── scripts/             # Utility scripts
```

---

## 🚀 Training

```bash
# Local (CPU/GPU)
uv run python -m src.train --config configs/small.yaml

# Multi-GPU
uv run torchrun --nproc_per_node=4 -m src.train --config configs/paper.yaml

# Google Colab (via LeCoder-cgpu)
lecoder-cgpu connect --variant gpu
lecoder-cgpu run "uv sync && uv run python -m src.train"
```

---

## 🧪 Testing

```bash
# Run all tests
uv run pytest tests/ -v

# Equation verification
uv run python scripts/verify_equations.py

# Benchmarks
uv run python scripts/benchmark_runner.py --checkpoint best.pt
```

---

## 📝 Citation

```bibtex
@article{ {{- project.paper_id.lower() -}} 2024,
  title={ {{- project.paper_title -}} },
  author={ {{- project.authors -}} },
  journal={arXiv preprint {{ project.arxiv_link.split('/')[-1] if project.arxiv_link else 'arXiv:XXXX.XXXXX' }}},
  year={2024}
}
```

---

## 📄 License

MIT License - see [LICENSE](./LICENSE)
"""

ARCHITECTURE_TEMPLATE = """# Architecture Documentation

## Overview

This document describes the technical architecture of the **{{ project.paper_title }}** reproduction.

Generated: {{ date }}

---

## System Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        Model                                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Input: [B, T] Token IDs                                    │
│         ↓                                                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Embedding Layer                                      │    │
│  └─────────────────────────────────────────────────────┘    │
│         ↓                                                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Transformer Blocks ×N                               │    │
{% for eq in project.equations[:5] %}
│  │   - Equation {{ eq.number }}: {{ eq.name[:30] }}{{ '...' if eq.name|length > 30 else '' }}{{ ' ' * (30 - eq.name[:30]|length) }}│    │
{% endfor %}
│  └─────────────────────────────────────────────────────┘    │
│         ↓                                                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Output Head → [B, T, V] logits                       │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Module Reference

| Module | File | Description | Equations |
|--------|------|-------------|-----------|
{% for module in project.modules %}
| `{{ module.name }}` | `{{ module.path }}` | {{ module.docstring[:50] if module.docstring else '-' }}{{ '...' if module.docstring and module.docstring|length > 50 else '' }} | {{ module.equations|join(', ') if module.equations else '-' }} |
{% endfor %}

---

## Equations Reference

{% for eq in project.equations %}
### Equation {{ eq.number }}: {{ eq.name }}

**LaTeX:**
```math
{{ eq.latex }}
```

**Implementation:** {% if eq.file_path %}`{{ eq.file_path }}:{{ eq.line_number }}`{% else %}*Not implemented*{% endif %}

**Tests:** {{ eq.test_status }}

---

{% endfor %}

## Configuration

{% if project.config %}
```yaml
{{ project.config | yaml_dump }}
```
{% else %}
See `configs/paper.yaml` for full configuration.
{% endif %}

---

## Testing Strategy

1. **Equation Tests**: Each equation has dedicated verification tests
2. **Integration Tests**: Full forward/backward pass validation
3. **Benchmark Tests**: Compare against paper-reported metrics

Coverage: {{ "%.1f"|format(project.test_coverage) }}%
"""


# =============================================================================
# GENERATOR
# =============================================================================


class DocumentationGenerator:
    """Generate project documentation."""

    def __init__(self, project_dir: Path):
        self.project_dir = project_dir
        self.project_info: ProjectInfo | None = None

    def gather_project_info(self) -> ProjectInfo:
        """Gather all project information."""
        console.print("[blue]Gathering project information...[/blue]")

        # Find context files
        papers_dir = self.project_dir / "papers"
        context_files = list(papers_dir.glob("*.context.md")) if papers_dir.exists() else []

        if not context_files:
            console.print("[yellow]No context files found in papers/[/yellow]")
            paper_info = {
                "paper_id": self.project_dir.name,
                "paper_title": self.project_dir.name.replace("-", " ").title(),
                "authors": "Unknown",
                "arxiv_link": "",
                "equations": [],
            }
        else:
            paper_info = extract_paper_info(context_files[0])
            console.print(f"[green]Found context:[/green] {context_files[0].name}")

        # Extract equations
        equations = [
            EquationInfo(
                number=eq["number"],
                name=eq["name"],
                latex=eq["latex"],
            )
            for eq in paper_info.get("equations", [])
        ]

        # Find implementations
        src_dir = self.project_dir / "src"
        if src_dir.exists():
            equations = find_equation_implementations(src_dir, equations)

        # Check tests
        tests_dir = self.project_dir / "tests"
        if tests_dir.exists():
            equations = check_equation_tests(tests_dir, equations)

        # Gather module info
        modules = []
        if src_dir.exists():
            for py_file in sorted(src_dir.rglob("*.py")):
                if py_file.name.startswith("__"):
                    continue
                modules.append(extract_module_info(py_file))

        # Load config
        config = {}
        config_path = self.project_dir / "configs" / "paper.yaml"
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f) or {}

        # Load benchmark results
        benchmark_results = {}
        results_path = self.project_dir / "benchmark_results.json"
        if results_path.exists():
            import json

            with open(results_path) as f:
                benchmark_results = json.load(f)

        # Calculate test coverage (approximate)
        tested_equations = sum(1 for eq in equations if eq.test_status == "✅")
        total_equations = len(equations) if equations else 1
        test_coverage = (tested_equations / total_equations) * 100

        self.project_info = ProjectInfo(
            name=self.project_dir.name,
            paper_title=paper_info.get("paper_title", ""),
            paper_id=paper_info.get("paper_id", ""),
            authors=paper_info.get("authors", ""),
            arxiv_link=paper_info.get("arxiv_link", ""),
            equations=equations,
            modules=modules,
            config=config,
            test_coverage=test_coverage,
            benchmark_results=benchmark_results,
        )

        return self.project_info

    def generate_readme(self, output_path: Path | None = None) -> str:
        """Generate README.md content."""
        if not self.project_info:
            self.gather_project_info()

        env = Environment(loader=BaseLoader())
        template = env.from_string(README_TEMPLATE)

        content = template.render(
            project=self.project_info,
            date=datetime.now().strftime("%Y-%m-%d"),
            github_user="your-username",
        )

        if output_path:
            output_path.write_text(content)
            console.print(f"[green]Generated:[/green] {output_path}")

        return content

    def generate_architecture(self, output_path: Path | None = None) -> str:
        """Generate ARCHITECTURE.md content."""
        if not self.project_info:
            self.gather_project_info()

        def yaml_dump(data):
            return yaml.dump(data, default_flow_style=False, sort_keys=False)

        env = Environment(loader=BaseLoader())
        env.filters["yaml_dump"] = yaml_dump
        template = env.from_string(ARCHITECTURE_TEMPLATE)

        content = template.render(
            project=self.project_info,
            date=datetime.now().strftime("%Y-%m-%d"),
        )

        if output_path:
            output_path.write_text(content)
            console.print(f"[green]Generated:[/green] {output_path}")

        return content

    def generate_all(self, output_dir: Path | None = None) -> None:
        """Generate all documentation files."""
        if output_dir is None:
            output_dir = self.project_dir

        output_dir.mkdir(parents=True, exist_ok=True)

        self.gather_project_info()

        # Generate files
        self.generate_readme(output_dir / "README.md")
        self.generate_architecture(output_dir / "ARCHITECTURE.md")

        # Create docs directory with equations reference
        docs_dir = output_dir / "docs"
        docs_dir.mkdir(exist_ok=True)

        equations_content = "# Equations Reference\n\n"
        for eq in self.project_info.equations:
            equations_content += f"## Equation {eq.number}: {eq.name}\n\n"
            equations_content += f"```math\n{eq.latex}\n```\n\n"
            if eq.file_path:
                equations_content += f"**Implementation:** `{eq.file_path}:{eq.line_number}`\n\n"
            equations_content += "---\n\n"

        (docs_dir / "equations.md").write_text(equations_content)
        console.print(f"[green]Generated:[/green] {docs_dir / 'equations.md'}")

        console.print(
            Panel(
                f"[green]Documentation generated successfully![/green]\n\n"
                f"Files:\n"
                f"  • README.md\n"
                f"  • ARCHITECTURE.md\n"
                f"  • docs/equations.md",
                title="✅ Complete",
            )
        )


def main():
    parser = argparse.ArgumentParser(description="Generate project documentation")
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path.cwd(),
        help="Project directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output directory (default: project directory)",
    )
    parser.add_argument(
        "--readme-only",
        action="store_true",
        help="Generate only README.md",
    )
    parser.add_argument(
        "--architecture-only",
        action="store_true",
        help="Generate only ARCHITECTURE.md",
    )

    args = parser.parse_args()

    console.print(
        Panel(
            "[bold blue]Documentation Generator[/bold blue]\n"
            "Creating README, ARCHITECTURE, and reference docs",
            title="📝 Research Reproduction",
        )
    )

    generator = DocumentationGenerator(args.project_dir)

    output_dir = args.output or args.project_dir

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Generating documentation...", total=None)

        if args.readme_only:
            generator.generate_readme(output_dir / "README.md")
        elif args.architecture_only:
            generator.generate_architecture(output_dir / "ARCHITECTURE.md")
        else:
            generator.generate_all(output_dir)

        progress.update(task, completed=True)


if __name__ == "__main__":
    main()
