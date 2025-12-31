# Neural Memory Reproduction: TITANS + MIRAS + NL

Reproduction of three interconnected Google Research papers on neural memory systems:

- **TITANS**: Learning to Memorize at Test Time (arXiv:2501.00663v1)
- **MIRAS**: It's All Connected (arXiv:2504.13173v1)
- **NL**: Nested Learning (NeurIPS 2025)

## Quick Start

```bash
# Install dependencies
uv venv
uv pip install -e ".[dev]"

# Run tests
uv run pytest tests/ -v

# Run specific paper tests
uv run pytest tests/test_equations/test_titans_*.py -v
uv run pytest tests/test_equations/test_miras_*.py -v
uv run pytest tests/test_equations/test_nl_*.py -v
```

## Implementation Status

| Paper | Equations | Implemented | Tests | Status |
|-------|-----------|-------------|-------|--------|
| TITANS | 35 | 3 | 3 | In Progress |
| MIRAS | 32 | 0 | 0 | Pending |
| NL | 121 | 0 | 0 | Pending |

## Papers

1. **TITANS: Learning to Memorize at Test Time**
   - Authors: Ali Behrouz, Peilin Zhong, Vahab Mirrokni
   - arXiv: https://arxiv.org/abs/2501.00663

2. **MIRAS: It's All Connected**
   - Authors: Ali Behrouz, Meisam Razaviyayn, Peilin Zhong, Vahab Mirrokni
   - arXiv: https://arxiv.org/abs/2504.13173

3. **Nested Learning: The Illusion of Deep Learning Architecture**
   - Authors: Ali Behrouz, Meisam Razaviyayn, Peilin Zhong, Vahab Mirrokni
   - Conference: NeurIPS 2025

## Project Structure

```
src/
├── common/         # Foundation (attention, embeddings)
├── titans/         # TITANS implementation
├── miras/          # MIRAS implementation
├── nl/             # NL implementation
└── utils/          # Utilities

tests/
├── test_equations/ # Equation-level tests
└── test_integration/ # Model-level tests
```

## License

Research reproduction for educational purposes.
