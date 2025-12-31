#!/bin/bash
# Run MIRAS paper test suite
#
# Tests MIRAS memory framework with three novel variants:
# - Moneta: ℓ_p attentional bias (p=3) + ℓ_q retention (q=4)
# - Yaad: Huber loss + ℓ_2 retention (robust to outliers)
# - Memora: ℓ_2 loss + KL divergence retention
#
# Paper: MIRAS - It's All Connected: A Journey Through
#        Test-Time Memorization, Attentional Bias, Retention,
#        and Online Optimization
# arXiv: 2504.13173

set -e

echo "🔄 Running MIRAS Paper Test Suite"
echo "=================================="
echo ""

# Check if we're in the right directory
if [[ ! -f "pyproject.toml" ]]; then
    echo "Error: Please run from the neural-memory-reproduction directory"
    exit 1
fi

# Run MIRAS-specific tests
echo "📋 Running MIRAS equation tests..."
uv run pytest tests/test_equations/test_miras_memory.py -v

echo ""
echo "📋 Running MIRAS integration tests..."
uv run pytest tests/test_integration/test_all_papers.py -v -k "miras"

echo ""
echo "✅ All MIRAS tests passed!"
echo ""
echo "MIRAS Equations Implemented:"
echo "  - Eq 3: Linear RNN memory update M_t = α*M_{t-1} + v_t k_t^T"
echo "  - Eq 9: Delta rule with retention"
echo "  - Eq 10-11: ℓ_p attentional bias (Moneta)"
echo "  - Eq 12: Huber loss (Yaad)"
echo "  - Eq 14: ℓ_q retention gate"
echo "  - Eq 17: KL divergence retention (Memora)"
echo ""
echo "Novel Memory Variants:"
echo "  - Moneta: ℓ_3 loss + ℓ_4 retention"
echo "  - Yaad: Huber loss + ℓ_2 retention"
echo "  - Memora: ℓ_2 loss + KL retention"
