#!/bin/bash
# Run all tests with coverage

echo "Running all tests with coverage..."
uv run pytest tests/ -v --cov=src --cov-report=term-missing

echo ""
echo "Running ruff checks..."
uv run ruff check src/ tests/

echo ""
echo "All checks complete!"
