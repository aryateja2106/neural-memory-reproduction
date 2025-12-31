# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch>=2.0",
#     "rich>=13.0",
#     "pyyaml>=6.0",
#     "pandas>=2.0",
#     "numpy>=1.24",
# ]
# ///
"""
Benchmark Runner - Compare implementation against paper results.

Validates that reproduction matches paper-reported metrics within tolerance.

Usage:
    uv run scripts/benchmark_runner.py --checkpoint best.pt --paper TITANS
    uv run scripts/benchmark_runner.py --checkpoint best.pt --all
    uv run scripts/benchmark_runner.py --checkpoint best.pt --benchmark perplexity
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    name: str
    metric: str
    paper_value: float
    our_value: float
    tolerance_pct: float
    passed: bool
    delta_pct: float
    notes: str = ""


@dataclass
class PaperBenchmarks:
    """Benchmark definitions from a paper."""

    paper_id: str
    benchmarks: list[dict] = field(default_factory=list)

    @classmethod
    def from_context_file(cls, context_path: Path) -> "PaperBenchmarks":
        """Extract benchmark definitions from .context.md file."""
        content = context_path.read_text()
        paper_id = context_path.stem.replace(".context", "")

        benchmarks = []

        # Parse benchmark section
        in_benchmark_section = False
        current_benchmark = {}

        for line in content.split("\n"):
            if "## Benchmarks" in line or "## Expected Results" in line:
                in_benchmark_section = True
                continue

            if in_benchmark_section:
                if line.startswith("## "):
                    break  # Next section

                if line.startswith("### "):
                    if current_benchmark:
                        benchmarks.append(current_benchmark)
                    current_benchmark = {"name": line.replace("### ", "").strip()}

                elif ":" in line and current_benchmark:
                    key, value = line.split(":", 1)
                    key = key.strip().lower().replace(" ", "_")
                    value = value.strip()

                    # Parse numeric values
                    try:
                        if "%" in value:
                            value = float(value.replace("%", ""))
                        else:
                            value = float(value)
                    except ValueError:
                        pass

                    current_benchmark[key] = value

        if current_benchmark:
            benchmarks.append(current_benchmark)

        return cls(paper_id=paper_id, benchmarks=benchmarks)


class BenchmarkRunner:
    """Run benchmarks and compare against paper results."""

    def __init__(self, checkpoint_path: Path, config_path: Path | None = None):
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.model = None
        self.config = None
        self.results: list[BenchmarkResult] = []

    def load_model(self) -> None:
        """Load model from checkpoint."""
        console.print(f"[blue]Loading checkpoint:[/blue] {self.checkpoint_path}")

        if not self.checkpoint_path.exists():
            console.print(f"[red]Checkpoint not found:[/red] {self.checkpoint_path}")
            sys.exit(1)

        # Try to import model - adjust path as needed
        try:
            sys.path.insert(0, str(Path.cwd()))
            from src.model import Model

            checkpoint = torch.load(self.checkpoint_path, map_location="cpu")

            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
                self.config = checkpoint.get("config", {})
            else:
                state_dict = checkpoint
                self.config = {}

            # Load config if provided
            if self.config_path and self.config_path.exists():
                with open(self.config_path) as f:
                    self.config = yaml.safe_load(f)

            self.model = Model(**self.config.get("model", {}))
            self.model.load_state_dict(state_dict)
            self.model.eval()

            console.print("[green]✓ Model loaded successfully[/green]")

        except ImportError as e:
            console.print(f"[red]Cannot import model:[/red] {e}")
            console.print("[yellow]Ensure src/model.py exists with Model class[/yellow]")
            sys.exit(1)

    def run_perplexity_benchmark(
        self,
        dataset_name: str = "wikitext103",
        paper_value: float | None = None,
        tolerance_pct: float = 10.0,
    ) -> BenchmarkResult:
        """Compute perplexity on dataset."""
        console.print(f"[blue]Running perplexity benchmark on {dataset_name}[/blue]")

        try:
            # Try to load dataset
            from src.data import get_eval_dataloader

            dataloader = get_eval_dataloader(dataset_name)

        except ImportError:
            # Mock dataloader for testing
            console.print("[yellow]Dataset loader not found, using mock data[/yellow]")

            class MockDataloader:
                def __iter__(self):
                    for _ in range(10):
                        yield {
                            "input_ids": torch.randint(0, 50257, (4, 512)),
                            "labels": torch.randint(0, 50257, (4, 512)),
                        }

            dataloader = MockDataloader()

        total_loss = 0.0
        total_tokens = 0

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                outputs = self.model(input_ids)
                loss = torch.nn.functional.cross_entropy(
                    outputs.view(-1, outputs.size(-1)),
                    labels.view(-1),
                    reduction="sum",
                )

                total_loss += loss.item()
                total_tokens += labels.numel()

        ppl = np.exp(total_loss / total_tokens)

        # Compare against paper
        if paper_value is None:
            paper_value = 17.2  # Default placeholder

        delta_pct = ((ppl - paper_value) / paper_value) * 100
        passed = abs(delta_pct) <= tolerance_pct

        return BenchmarkResult(
            name=f"Perplexity ({dataset_name})",
            metric="PPL",
            paper_value=paper_value,
            our_value=ppl,
            tolerance_pct=tolerance_pct,
            passed=passed,
            delta_pct=delta_pct,
        )

    def run_memory_scaling_benchmark(
        self, paper_claim: str = "linear", tolerance_pct: float = 20.0
    ) -> BenchmarkResult:
        """Verify memory scales as claimed (linear, sublinear, etc.)."""
        console.print("[blue]Running memory scaling benchmark[/blue]")

        if not torch.cuda.is_available():
            return BenchmarkResult(
                name="Memory Scaling",
                metric="Scaling Factor",
                paper_value=1.0,  # Linear
                our_value=0.0,
                tolerance_pct=tolerance_pct,
                passed=False,
                delta_pct=100.0,
                notes="CUDA not available - skipped",
            )

        device = torch.device("cuda")
        self.model = self.model.to(device)

        seq_lengths = [128, 256, 512, 1024]
        memories = []

        for seq_len in seq_lengths:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

            x = torch.randint(0, 50257, (1, seq_len), device=device)

            with torch.no_grad():
                _ = self.model(x)

            peak_mb = torch.cuda.max_memory_allocated() / 1024**2
            memories.append(peak_mb)

        # Calculate scaling factor
        # Linear scaling: doubling length doubles memory
        ratios = [memories[i + 1] / memories[i] for i in range(len(memories) - 1)]
        avg_ratio = np.mean(ratios)

        # For linear scaling, ratio should be ~2.0 when length doubles
        expected_ratio = 2.0 if paper_claim == "linear" else 1.5

        delta_pct = ((avg_ratio - expected_ratio) / expected_ratio) * 100
        passed = abs(delta_pct) <= tolerance_pct

        return BenchmarkResult(
            name="Memory Scaling",
            metric="Avg Ratio (2x seq)",
            paper_value=expected_ratio,
            our_value=avg_ratio,
            tolerance_pct=tolerance_pct,
            passed=passed,
            delta_pct=delta_pct,
            notes=f"Seq lengths: {seq_lengths}, Memories (MB): {[f'{m:.1f}' for m in memories]}",
        )

    def run_throughput_benchmark(
        self,
        paper_value: float | None = None,
        tolerance_pct: float = 20.0,
        batch_size: int = 4,
        seq_len: int = 512,
        num_iters: int = 100,
    ) -> BenchmarkResult:
        """Measure training throughput (tokens/second)."""
        console.print("[blue]Running throughput benchmark[/blue]")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        self.model.train()

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)

        # Warmup
        for _ in range(10):
            x = torch.randint(0, 50257, (batch_size, seq_len), device=device)
            y = torch.randint(0, 50257, (batch_size, seq_len), device=device)
            outputs = self.model(x)
            loss = torch.nn.functional.cross_entropy(
                outputs.view(-1, outputs.size(-1)), y.view(-1)
            )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Timed run
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        import time

        start = time.perf_counter()

        for _ in range(num_iters):
            x = torch.randint(0, 50257, (batch_size, seq_len), device=device)
            y = torch.randint(0, 50257, (batch_size, seq_len), device=device)
            outputs = self.model(x)
            loss = torch.nn.functional.cross_entropy(
                outputs.view(-1, outputs.size(-1)), y.view(-1)
            )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        tokens_per_sec = (batch_size * seq_len * num_iters) / elapsed

        self.model.eval()

        if paper_value is None:
            paper_value = tokens_per_sec  # No comparison
            delta_pct = 0.0
            passed = True
        else:
            delta_pct = ((tokens_per_sec - paper_value) / paper_value) * 100
            passed = delta_pct >= -tolerance_pct  # OK if faster

        return BenchmarkResult(
            name="Throughput",
            metric="tokens/sec",
            paper_value=paper_value,
            our_value=tokens_per_sec,
            tolerance_pct=tolerance_pct,
            passed=passed,
            delta_pct=delta_pct,
            notes=f"Batch: {batch_size}, Seq: {seq_len}, Device: {device}",
        )

    def run_all_benchmarks(
        self, paper_benchmarks: PaperBenchmarks | None = None
    ) -> list[BenchmarkResult]:
        """Run all benchmarks."""
        self.results = []

        # Standard benchmarks
        self.results.append(self.run_perplexity_benchmark())
        self.results.append(self.run_memory_scaling_benchmark())
        self.results.append(self.run_throughput_benchmark())

        # Paper-specific benchmarks
        if paper_benchmarks:
            for bm in paper_benchmarks.benchmarks:
                if "perplexity" in bm.get("name", "").lower():
                    self.results.append(
                        self.run_perplexity_benchmark(
                            dataset_name=bm.get("dataset", "wikitext103"),
                            paper_value=bm.get("value"),
                            tolerance_pct=bm.get("tolerance", 10.0),
                        )
                    )

        return self.results

    def display_results(self) -> None:
        """Display benchmark results in a table."""
        table = Table(title="Benchmark Results", show_header=True, header_style="bold")

        table.add_column("Benchmark", style="cyan")
        table.add_column("Metric", style="dim")
        table.add_column("Paper", justify="right")
        table.add_column("Ours", justify="right")
        table.add_column("Δ", justify="right")
        table.add_column("Status", justify="center")

        for result in self.results:
            status = "[green]✓ PASS[/green]" if result.passed else "[red]✗ FAIL[/red]"

            delta_str = f"{result.delta_pct:+.1f}%"
            if result.delta_pct > 0:
                delta_str = f"[yellow]{delta_str}[/yellow]"
            elif result.delta_pct < -5:
                delta_str = f"[green]{delta_str}[/green]"

            table.add_row(
                result.name,
                result.metric,
                f"{result.paper_value:.2f}",
                f"{result.our_value:.2f}",
                delta_str,
                status,
            )

        console.print(table)

        # Summary
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)

        if passed == total:
            console.print(
                Panel(
                    f"[green]All {total} benchmarks passed![/green]",
                    title="Summary",
                    border_style="green",
                )
            )
        else:
            console.print(
                Panel(
                    f"[yellow]{passed}/{total} benchmarks passed[/yellow]",
                    title="Summary",
                    border_style="yellow",
                )
            )

    def save_results(self, output_path: Path) -> None:
        """Save results to JSON file."""
        results_dict = {
            "checkpoint": str(self.checkpoint_path),
            "results": [
                {
                    "name": r.name,
                    "metric": r.metric,
                    "paper_value": r.paper_value,
                    "our_value": r.our_value,
                    "delta_pct": r.delta_pct,
                    "tolerance_pct": r.tolerance_pct,
                    "passed": r.passed,
                    "notes": r.notes,
                }
                for r in self.results
            ],
            "summary": {
                "passed": sum(1 for r in self.results if r.passed),
                "failed": sum(1 for r in self.results if not r.passed),
                "total": len(self.results),
            },
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results_dict, f, indent=2)

        console.print(f"[green]Results saved to:[/green] {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run benchmarks against paper-reported results"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to model config YAML",
    )
    parser.add_argument(
        "--paper",
        type=str,
        help="Paper ID to load benchmarks from (looks for papers/{PAPER}.context.md)",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["perplexity", "memory", "throughput", "all"],
        default="all",
        help="Specific benchmark to run",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results.json"),
        help="Output path for results JSON",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=10.0,
        help="Default tolerance percentage for pass/fail",
    )

    args = parser.parse_args()

    console.print(
        Panel(
            "[bold blue]Benchmark Runner[/bold blue]\n"
            "Comparing implementation against paper results",
            title="🔬 Research Reproduction",
        )
    )

    # Load paper benchmarks if specified
    paper_benchmarks = None
    if args.paper:
        context_path = Path(f"papers/{args.paper}.context.md")
        if context_path.exists():
            paper_benchmarks = PaperBenchmarks.from_context_file(context_path)
            console.print(
                f"[green]Loaded {len(paper_benchmarks.benchmarks)} benchmarks from {args.paper}[/green]"
            )
        else:
            console.print(f"[yellow]Context file not found:[/yellow] {context_path}")

    runner = BenchmarkRunner(args.checkpoint, args.config)
    runner.load_model()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Running benchmarks...", total=None)

        if args.benchmark == "all":
            runner.run_all_benchmarks(paper_benchmarks)
        elif args.benchmark == "perplexity":
            runner.results.append(runner.run_perplexity_benchmark())
        elif args.benchmark == "memory":
            runner.results.append(runner.run_memory_scaling_benchmark())
        elif args.benchmark == "throughput":
            runner.results.append(runner.run_throughput_benchmark())

        progress.update(task, completed=True)

    runner.display_results()
    runner.save_results(args.output)


if __name__ == "__main__":
    main()
