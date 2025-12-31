#!/bin/bash
# Run TITANS paper test suite
#
# Tests TITANS memory equations:
# - Eq 8: Gradient-based memory update
# - Eq 9-10: Momentum-based update
# - Eq 13-14: Forgetting mechanism
#
# Paper: TITANS - Learning to Memorize at Test Time
# arXiv: 2501.00663

set -e

echo "🧠 Running TITANS Paper Test Suite"
echo "=================================="
echo ""

# Check if we're in the right directory
if [[ ! -f "pyproject.toml" ]]; then
    echo "Error: Please run from the neural-memory-reproduction directory"
    exit 1
fi

# Run TITANS-specific tests
echo "📋 Running TITANS equation tests..."
uv run pytest tests/test_equations/test_titans_memory.py -v

echo ""
echo "📋 Running common attention tests (shared with TITANS)..."
uv run pytest tests/test_equations/test_common_attention.py -v

echo ""
echo "📋 Running TITANS integration tests..."
uv run pytest tests/test_integration/test_all_papers.py -v -k "titans"

echo ""
echo "✅ All TITANS tests passed!"
echo ""
echo "TITANS Equations Implemented:"
echo "  - Eq 1-2: Standard attention (Q, K, V projections)"
echo "  - Eq 3-5: Linear attention with kernel"
echo "  - Eq 8: Gradient-based memory update"
echo "  - Eq 9-10: Momentum-based surprise accumulation"
echo "  - Eq 13-14: Forgetting mechanism"
