# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "markitdown>=0.1.0",
#     "pymupdf>=1.24.0",
#     "rich>=13.0.0",
#     "typer>=0.12.0",
# ]
# ///
"""
Paper Extraction Script

Extracts content from research paper PDFs into structured markdown.
Uses markitdown as primary method, PyMuPDF as fallback.

Usage:
    uv run scripts/extract_paper.py paper.pdf --output paper.md
    uv run scripts/extract_paper.py paper.pdf --paper-id TITANS
    uv run scripts/extract_paper.py papers/ --batch  # Process directory

Features:
- Extracts text, equations, tables, figures
- Preserves LaTeX equations where possible
- Handles multi-column layouts
- Outputs structured markdown
"""

import re
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

app = typer.Typer(help="Extract research papers to structured markdown")
console = Console()


def extract_with_markitdown(pdf_path: Path) -> str:
    """Primary extraction using markitdown."""
    try:
        from markitdown import MarkItDown
        
        md = MarkItDown()
        result = md.convert(str(pdf_path))
        return result.text_content
    except Exception as e:
        console.print(f"[yellow]markitdown failed: {e}[/yellow]")
        return ""


def extract_with_pymupdf(pdf_path: Path) -> str:
    """Fallback extraction using PyMuPDF."""
    import fitz
    
    doc = fitz.open(pdf_path)
    text_blocks = []
    
    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        text_blocks.append(f"\n<!-- Page {page_num + 1} -->\n{text}")
    
    doc.close()
    return "\n".join(text_blocks)


def clean_extracted_text(text: str) -> str:
    """Clean up extracted text."""
    # Remove excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Fix common OCR issues
    text = text.replace('ﬁ', 'fi')
    text = text.replace('ﬂ', 'fl')
    text = text.replace('−', '-')
    
    # Try to preserve equation formatting
    text = re.sub(r'\$\s+', '$', text)
    text = re.sub(r'\s+\$', '$', text)
    
    return text.strip()


def identify_sections(text: str) -> dict[str, str]:
    """Identify paper sections from text."""
    sections = {}
    
    # Common section patterns
    section_patterns = [
        r'^#+\s*(Abstract)',
        r'^#+\s*(Introduction)',
        r'^#+\s*(Related Work)',
        r'^#+\s*(Background)',
        r'^#+\s*(Method|Methodology|Approach)',
        r'^#+\s*(Model|Architecture)',
        r'^#+\s*(Experiments?)',
        r'^#+\s*(Results?)',
        r'^#+\s*(Discussion)',
        r'^#+\s*(Conclusion)',
        r'^#+\s*(References?)',
        r'^#+\s*(Appendix)',
        r'^\d+\.?\s*(Abstract)',
        r'^\d+\.?\s*(Introduction)',
    ]
    
    # Find section boundaries
    lines = text.split('\n')
    current_section = "Preamble"
    current_content = []
    
    for line in lines:
        found_section = None
        for pattern in section_patterns:
            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                found_section = match.group(1)
                break
        
        if found_section:
            if current_content:
                sections[current_section] = '\n'.join(current_content)
            current_section = found_section
            current_content = [line]
        else:
            current_content.append(line)
    
    if current_content:
        sections[current_section] = '\n'.join(current_content)
    
    return sections


def extract_equations(text: str) -> list[dict]:
    """Extract equations from text."""
    equations = []
    
    # LaTeX display equations
    display_patterns = [
        r'\$\$(.*?)\$\$',  # $$ ... $$
        r'\\\[(.*?)\\\]',  # \[ ... \]
        r'\\begin\{equation\}(.*?)\\end\{equation\}',
        r'\\begin\{align\}(.*?)\\end\{align\}',
    ]
    
    eq_num = 1
    for pattern in display_patterns:
        for match in re.finditer(pattern, text, re.DOTALL):
            equations.append({
                'number': eq_num,
                'latex': match.group(1).strip(),
                'type': 'display',
                'raw': match.group(0)
            })
            eq_num += 1
    
    # Inline equations
    inline_pattern = r'(?<!\$)\$([^\$]+)\$(?!\$)'
    for match in re.finditer(inline_pattern, text):
        if len(match.group(1)) > 3:  # Skip trivial cases
            equations.append({
                'number': None,
                'latex': match.group(1).strip(),
                'type': 'inline',
                'raw': match.group(0)
            })
    
    return equations


def extract_algorithms(text: str) -> list[dict]:
    """Extract algorithm blocks from text."""
    algorithms = []
    
    # Algorithm environment patterns
    patterns = [
        r'\\begin\{algorithm\}(.*?)\\end\{algorithm\}',
        r'Algorithm\s+\d+[:\.]?\s*(.*?)(?=Algorithm\s+\d+|$)',
    ]
    
    alg_num = 1
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.DOTALL | re.IGNORECASE):
            algorithms.append({
                'number': alg_num,
                'content': match.group(1).strip() if match.lastindex else match.group(0),
                'raw': match.group(0)
            })
            alg_num += 1
    
    return algorithms


def format_output(
    text: str,
    sections: dict[str, str],
    equations: list[dict],
    algorithms: list[dict],
    paper_id: str,
    pdf_path: Path,
) -> str:
    """Format extracted content into structured markdown."""
    output = []
    
    # Header
    output.append(f"# Extracted: {paper_id}")
    output.append(f"\n**Source:** `{pdf_path.name}`")
    output.append(f"**Equations Found:** {len(equations)}")
    output.append(f"**Algorithms Found:** {len(algorithms)}")
    output.append("\n---\n")
    
    # Table of Contents
    output.append("## Table of Contents\n")
    for section_name in sections.keys():
        anchor = section_name.lower().replace(' ', '-')
        output.append(f"- [{section_name}](#{anchor})")
    output.append("\n---\n")
    
    # Equations Summary
    if equations:
        output.append("## Equations Summary\n")
        output.append("| # | Type | LaTeX Preview |")
        output.append("|---|------|---------------|")
        for eq in equations[:20]:  # Limit to first 20
            if eq['number']:
                latex_preview = eq['latex'][:50] + "..." if len(eq['latex']) > 50 else eq['latex']
                output.append(f"| {eq['number']} | {eq['type']} | `{latex_preview}` |")
        if len(equations) > 20:
            output.append(f"\n*...and {len(equations) - 20} more equations*\n")
        output.append("\n---\n")
    
    # Algorithms Summary
    if algorithms:
        output.append("## Algorithms Summary\n")
        for alg in algorithms:
            preview = alg['content'][:200] + "..." if len(alg['content']) > 200 else alg['content']
            output.append(f"### Algorithm {alg['number']}\n")
            output.append(f"```\n{preview}\n```\n")
        output.append("\n---\n")
    
    # Full Sections
    output.append("## Full Content\n")
    for section_name, content in sections.items():
        output.append(f"### {section_name}\n")
        output.append(content)
        output.append("\n")
    
    # Raw text as fallback
    output.append("---\n")
    output.append("## Raw Extracted Text\n")
    output.append("<details>")
    output.append("<summary>Click to expand raw text</summary>\n")
    output.append("```")
    output.append(text[:50000])  # Limit size
    if len(text) > 50000:
        output.append(f"\n... truncated ({len(text)} total characters)")
    output.append("```")
    output.append("</details>")
    
    return '\n'.join(output)


@app.command()
def extract(
    input_path: Path = typer.Argument(..., help="PDF file or directory"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path"),
    paper_id: str = typer.Option("PAPER", "--paper-id", "-p", help="Paper identifier"),
    batch: bool = typer.Option(False, "--batch", "-b", help="Process directory of PDFs"),
    method: str = typer.Option("auto", "--method", "-m", help="Extraction method: markitdown, pymupdf, auto"),
):
    """Extract research paper PDF to structured markdown."""
    
    if batch and input_path.is_dir():
        pdfs = list(input_path.glob("*.pdf"))
        console.print(f"[blue]Found {len(pdfs)} PDFs in {input_path}[/blue]")
        
        for pdf in pdfs:
            pid = pdf.stem.upper().replace(' ', '_').replace('-', '_')
            out = input_path / f"{pdf.stem}.md"
            _extract_single(pdf, out, pid, method)
    else:
        if not input_path.exists():
            console.print(f"[red]Error: {input_path} not found[/red]")
            raise typer.Exit(1)
        
        output_path = output or input_path.with_suffix('.md')
        _extract_single(input_path, output_path, paper_id, method)


def _extract_single(pdf_path: Path, output_path: Path, paper_id: str, method: str):
    """Extract a single PDF."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Extracting {pdf_path.name}...", total=None)
        
        # Extract text
        text = ""
        if method in ("auto", "markitdown"):
            text = extract_with_markitdown(pdf_path)
        
        if not text and method in ("auto", "pymupdf"):
            progress.update(task, description="Falling back to PyMuPDF...")
            text = extract_with_pymupdf(pdf_path)
        
        if not text:
            console.print(f"[red]Failed to extract text from {pdf_path}[/red]")
            return
        
        # Clean text
        progress.update(task, description="Cleaning extracted text...")
        text = clean_extracted_text(text)
        
        # Analyze structure
        progress.update(task, description="Analyzing structure...")
        sections = identify_sections(text)
        equations = extract_equations(text)
        algorithms = extract_algorithms(text)
        
        # Format output
        progress.update(task, description="Formatting output...")
        output = format_output(text, sections, equations, algorithms, paper_id, pdf_path)
        
        # Write output
        output_path.write_text(output)
        progress.update(task, description=f"✓ Saved to {output_path}")
    
    console.print(f"\n[green]✓ Extracted {pdf_path.name}[/green]")
    console.print(f"  Sections: {len(sections)}")
    console.print(f"  Equations: {len(equations)}")
    console.print(f"  Algorithms: {len(algorithms)}")
    console.print(f"  Output: {output_path}")


@app.command()
def equations(
    input_path: Path = typer.Argument(..., help="PDF or markdown file"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file"),
):
    """Extract only equations from a paper."""
    if input_path.suffix == '.pdf':
        text = extract_with_markitdown(input_path) or extract_with_pymupdf(input_path)
    else:
        text = input_path.read_text()
    
    equations = extract_equations(text)
    
    output_lines = ["# Extracted Equations\n"]
    for eq in equations:
        if eq['number']:
            output_lines.append(f"## Equation {eq['number']}\n")
        output_lines.append(f"**Type:** {eq['type']}\n")
        output_lines.append(f"```latex\n{eq['latex']}\n```\n")
        output_lines.append("---\n")
    
    result = '\n'.join(output_lines)
    
    if output:
        output.write_text(result)
        console.print(f"[green]✓ Saved {len(equations)} equations to {output}[/green]")
    else:
        console.print(result)


if __name__ == "__main__":
    app()
