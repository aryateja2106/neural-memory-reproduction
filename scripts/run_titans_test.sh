#!/bin/bash
# Quick test script for TITANS implementation

echo "Running TITANS tests..."
uv run pytest tests/test_equations/test_titans_memory.py -v

echo ""
echo "Running common attention tests (used by TITANS)..."
uv run pytest tests/test_equations/test_common_attention.py -v
