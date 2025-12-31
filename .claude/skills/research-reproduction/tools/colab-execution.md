# Colab Execution Tool (LeCoder-cgpu Integration)

## Purpose
Integrates LeCoder-cgpu CLI to execute reproduced code on Google Colab's GPU resources, managing sessions, uploading code, running experiments, and downloading results.

## Prerequisites

```bash
# Install LeCoder-cgpu CLI (published on npm)
npm install -g lecoder-cgpu

# Verify installation
lecoder-cgpu --version
```

## Resource Planning

### Colab Pro Units Budget

```
Monthly Budget: 100 units

Runtime Options:
┌─────────────┬──────────────┬─────────────┬──────────────────┐
│ GPU Type    │ Units/Hour   │ Max Hours   │ Best For         │
├─────────────┼──────────────┼─────────────┼──────────────────┤
│ T4          │ 1.96         │ ~51 hrs     │ Development      │
│ L4          │ 3.00         │ ~33 hrs     │ Medium training  │
│ A100        │ 12.00        │ ~8 hrs      │ Final benchmarks │
│ TPU v2-8    │ 1.76         │ ~56 hrs     │ Large batches    │
└─────────────┴──────────────┴─────────────┴──────────────────┘

Strategy:
1. Development/debugging: T4 (cheap, sufficient for iteration)
2. Training experiments: L4 (good balance)
3. Final benchmarks: A100 (match paper hardware when possible)
```

### Planning Execution

```python
def plan_execution(
    task: str,
    estimated_hours: float,
    gpu_preference: str = None
) -> dict:
    """
    Plan GPU allocation for a task.
    
    Returns recommendation based on task type and budget.
    """
    UNIT_RATES = {
        'T4': 1.96,
        'L4': 3.00,
        'A100': 12.00,
        'TPU_v2': 1.76,
    }
    
    task_recommendations = {
        'development': ['T4', 'TPU_v2'],
        'training_small': ['T4', 'L4'],
        'training_medium': ['L4', 'A100'],
        'training_large': ['A100'],
        'benchmark': ['A100', 'L4'],
        'inference': ['T4', 'L4'],
    }
    
    recommended = task_recommendations.get(task, ['T4'])
    
    costs = {}
    for gpu in recommended:
        rate = UNIT_RATES[gpu]
        units = rate * estimated_hours
        costs[gpu] = {
            'units': units,
            'rate': rate,
            'hours': estimated_hours,
        }
    
    return {
        'task': task,
        'recommendations': recommended,
        'costs': costs,
        'best_value': min(costs.items(), key=lambda x: x[1]['units'])[0],
    }
```

## Workflow Integration

```
┌─────────────────────────────────────────────────────────────────┐
│                    COLAB EXECUTION WORKFLOW                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  LOCAL DEVELOPMENT                                               │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  1. Code written and tested locally (CPU/MPS)            │   │
│  │  2. All equation tests pass                               │   │
│  │  3. Quality checks pass (ruff, ty)                        │   │
│  │  4. Ready for GPU training/benchmarks                     │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  PREPARE FOR UPLOAD                                              │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  uv run scripts/prepare_colab.py                         │   │
│  │  • Creates minimal upload package                         │   │
│  │  • Generates requirements.txt from pyproject.toml         │   │
│  │  • Creates setup_colab.sh script                          │   │
│  │  • Packs into colab_package.tar.gz                        │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  CONNECT TO COLAB                                                │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  lecoder-cgpu connect --variant gpu                       │   │
│  │  • Opens browser for Colab authentication                 │   │
│  │  • Creates new runtime (T4/L4/A100 based on --variant)    │   │
│  │  • Establishes terminal connection                        │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  UPLOAD & SETUP                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  lecoder-cgpu upload colab_package.tar.gz /content/       │   │
│  │  lecoder-cgpu run "cd /content && tar xzf colab_package.tar.gz"│
│  │  lecoder-cgpu run "cd /content && bash setup_colab.sh"    │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  EXECUTE                                                         │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  # Training                                               │   │
│  │  lecoder-cgpu run "cd /content && python train.py"        │   │
│  │                                                           │   │
│  │  # Or benchmarks                                          │   │
│  │  lecoder-cgpu run "cd /content && uv run scripts/benchmark_runner.py"│
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  DOWNLOAD RESULTS                                                │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  lecoder-cgpu download /content/checkpoints ./checkpoints │   │
│  │  lecoder-cgpu download /content/results ./results         │   │
│  │  lecoder-cgpu download /content/logs ./logs               │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  DISCONNECT                                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  lecoder-cgpu disconnect                                  │   │
│  │  • Releases GPU resources                                 │   │
│  │  • Saves units                                            │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Colab Package Preparation

### prepare_colab.py

```python
#!/usr/bin/env python3
"""
Prepare code package for Colab upload.

Creates a minimal package with:
- Source code (src/)
- Config files (configs/)
- Scripts (scripts/)
- Requirements
- Setup script
"""

import tarfile
from pathlib import Path
import tomllib


def prepare_colab_package(
    project_root: Path,
    output_path: Path = None,
    include_tests: bool = False
) -> Path:
    """Create tar.gz package for Colab upload."""
    
    output_path = output_path or project_root / 'colab_package.tar.gz'
    
    # Directories to include
    include_dirs = ['src', 'configs', 'scripts']
    if include_tests:
        include_dirs.append('tests')
    
    # Files to include
    include_files = [
        'pyproject.toml',
        'README.md',
    ]
    
    # Generate requirements.txt from pyproject.toml
    requirements = generate_requirements(project_root / 'pyproject.toml')
    (project_root / 'requirements.txt').write_text(requirements)
    include_files.append('requirements.txt')
    
    # Generate setup script
    setup_script = generate_setup_script()
    (project_root / 'setup_colab.sh').write_text(setup_script)
    include_files.append('setup_colab.sh')
    
    # Create tar.gz
    with tarfile.open(output_path, 'w:gz') as tar:
        for dir_name in include_dirs:
            dir_path = project_root / dir_name
            if dir_path.exists():
                tar.add(dir_path, arcname=dir_name)
        
        for file_name in include_files:
            file_path = project_root / file_name
            if file_path.exists():
                tar.add(file_path, arcname=file_name)
    
    print(f"Created Colab package: {output_path}")
    print(f"Size: {output_path.stat().st_size / 1024:.1f} KB")
    
    return output_path


def generate_requirements(pyproject_path: Path) -> str:
    """Generate requirements.txt from pyproject.toml."""
    
    with open(pyproject_path, 'rb') as f:
        config = tomllib.load(f)
    
    deps = config.get('project', {}).get('dependencies', [])
    
    # Add Colab-specific deps
    colab_deps = [
        'torch>=2.0.0',  # Ensure GPU support
        'tqdm',
        'wandb',  # Optional: for logging
    ]
    
    all_deps = list(set(deps + colab_deps))
    return '\n'.join(sorted(all_deps))


def generate_setup_script() -> str:
    """Generate setup_colab.sh for Colab initialization."""
    
    return '''#!/bin/bash
# Colab Setup Script
# Generated by prepare_colab.py

set -e

echo "🚀 Setting up Colab environment..."

# Check GPU
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')" || echo "No GPU available"

# Install uv (faster than pip)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Install dependencies with uv
uv pip install -r requirements.txt

# Verify installation
python -c "from src.model import *; print('✓ Model imports OK')"

echo "✅ Setup complete!"
'''
```

## LeCoder-cgpu Commands

### Connection

```bash
# Basic connection (auto-selects GPU)
lecoder-cgpu connect

# Specify GPU type
lecoder-cgpu connect --variant gpu      # Request GPU (T4 default)
lecoder-cgpu connect --variant premium  # Request A100
lecoder-cgpu connect --variant tpu      # Request TPU

# With startup command
lecoder-cgpu connect --startup-command "cd /content && nvidia-smi"
```

### File Transfer

```bash
# Upload single file
lecoder-cgpu upload ./model.py /content/

# Upload directory
lecoder-cgpu upload ./src/ /content/src/

# Upload tar.gz package
lecoder-cgpu upload ./colab_package.tar.gz /content/

# Download results
lecoder-cgpu download /content/results ./results/
lecoder-cgpu download /content/checkpoints ./checkpoints/
lecoder-cgpu download /content/wandb ./wandb/
```

### Execution

```bash
# Run single command
lecoder-cgpu run "python train.py"

# Run with output capture
lecoder-cgpu run "python train.py" --output train.log

# Run in background (for long training)
lecoder-cgpu run "nohup python train.py > train.log 2>&1 &"

# Check running processes
lecoder-cgpu run "ps aux | grep python"

# Monitor GPU usage
lecoder-cgpu run "nvidia-smi"

# Interactive shell
lecoder-cgpu shell
```

### Session Management

```bash
# Check status
lecoder-cgpu status

# List active sessions
lecoder-cgpu list

# Disconnect (release resources)
lecoder-cgpu disconnect

# Reconnect to existing session
lecoder-cgpu reconnect
```

## Execution Scripts

### train_colab.sh

```bash
#!/bin/bash
# Train model on Colab
# Usage: lecoder-cgpu run "bash train_colab.sh"

set -e

# Configuration
CONFIG=${1:-configs/default.yaml}
CHECKPOINT_DIR=${2:-/content/checkpoints}
LOG_DIR=${3:-/content/logs}

echo "📊 Training with config: $CONFIG"
echo "💾 Checkpoints: $CHECKPOINT_DIR"

# Create directories
mkdir -p $CHECKPOINT_DIR $LOG_DIR

# Start training with progress logging
python -u train.py \
    --config $CONFIG \
    --checkpoint-dir $CHECKPOINT_DIR \
    --log-dir $LOG_DIR \
    --log-interval 10 \
    2>&1 | tee $LOG_DIR/train.log

echo "✅ Training complete!"
echo "📁 Checkpoints saved to: $CHECKPOINT_DIR"
```

### benchmark_colab.sh

```bash
#!/bin/bash
# Run benchmarks on Colab
# Usage: lecoder-cgpu run "bash benchmark_colab.sh"

set -e

CHECKPOINT=${1:-/content/checkpoints/best.pt}
OUTPUT_DIR=${2:-/content/results}

echo "🧪 Running benchmarks..."

# Run benchmark suite
uv run scripts/benchmark_runner.py run \
    --model $CHECKPOINT \
    --spec benchmarks/paper.yaml \
    --output $OUTPUT_DIR

# Generate comparison report
uv run scripts/benchmark_runner.py report \
    --results $OUTPUT_DIR/results.json \
    --paper benchmarks/paper.yaml \
    --output $OUTPUT_DIR/benchmark_report.md

echo "✅ Benchmarks complete!"
echo "📊 Report: $OUTPUT_DIR/benchmark_report.md"
```

## Monitoring & Logging

### WandB Integration

```python
# In train.py
import wandb

def setup_wandb(config: dict, project_name: str = "research-reproduction"):
    """Initialize WandB for Colab training."""
    
    # Check if running in Colab
    try:
        import google.colab
        in_colab = True
    except ImportError:
        in_colab = False
    
    wandb.init(
        project=project_name,
        config=config,
        tags=['colab'] if in_colab else ['local'],
    )
    
    # Log system info
    if in_colab:
        import torch
        wandb.config.update({
            'gpu': torch.cuda.get_device_name(0),
            'cuda_version': torch.version.cuda,
        })


def log_training_step(step: int, metrics: dict):
    """Log training metrics."""
    wandb.log(metrics, step=step)


def save_checkpoint(model, optimizer, step: int, path: str):
    """Save checkpoint and upload to WandB."""
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    
    # Upload to WandB
    wandb.save(path)
```

### Progress Monitoring

```bash
# Monitor training progress
lecoder-cgpu run "tail -f /content/logs/train.log"

# Check GPU utilization
lecoder-cgpu run "watch -n 1 nvidia-smi"

# Check disk space
lecoder-cgpu run "df -h /content"
```

## Error Handling

### Common Issues

```python
# Colab execution wrapper with error handling
def colab_safe_execute(command: str) -> tuple[int, str, str]:
    """Execute command with Colab-specific error handling."""
    
    try:
        result = subprocess.run(
            f"lecoder-cgpu run '{command}'",
            shell=True,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
        )
        return result.returncode, result.stdout, result.stderr
    
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out after 1 hour"
    
    except Exception as e:
        return -1, "", str(e)


# Handle disconnection
def reconnect_and_resume():
    """Reconnect to Colab and resume training from checkpoint."""
    
    # Check for existing session
    status = subprocess.run(['lecoder-cgpu', 'status'], capture_output=True)
    
    if 'disconnected' in status.stdout.decode():
        # Reconnect
        subprocess.run(['lecoder-cgpu', 'connect', '--variant', 'gpu'])
        
        # Find latest checkpoint
        result = subprocess.run(
            ['lecoder-cgpu', 'run', 'ls -t /content/checkpoints/*.pt | head -1'],
            capture_output=True
        )
        latest_checkpoint = result.stdout.decode().strip()
        
        # Resume training
        if latest_checkpoint:
            subprocess.run([
                'lecoder-cgpu', 'run',
                f'python train.py --resume {latest_checkpoint}'
            ])
```

## Best Practices

### Resource Optimization

```markdown
1. **Batch Your Work**
   - Upload all files at once (tar.gz)
   - Run multiple experiments in one session
   - Download all results together

2. **Use Checkpointing**
   - Save checkpoints every N steps
   - Enable auto-resume on disconnect
   - Upload checkpoints to GDrive/WandB

3. **Monitor Resources**
   - Watch GPU memory usage
   - Kill idle processes
   - Disconnect when done

4. **Optimize Data Loading**
   - Use /content/drive for large datasets
   - Cache preprocessed data
   - Use memory-mapped files when possible
```

### Session Checklist

```markdown
Before Starting:
- [ ] Code tested locally
- [ ] Quality checks pass
- [ ] Package prepared (colab_package.tar.gz)
- [ ] Config files ready
- [ ] Checkpoint directory planned

During Session:
- [ ] GPU verified (nvidia-smi)
- [ ] Dependencies installed
- [ ] Model loads correctly
- [ ] Training/benchmark starts
- [ ] Progress monitored

After Session:
- [ ] Results downloaded
- [ ] Checkpoints saved
- [ ] Logs retrieved
- [ ] Session disconnected
- [ ] Results validated locally
```
