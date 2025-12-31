"""Shared pytest fixtures for neural memory reproduction tests."""

import pytest
import torch


@pytest.fixture
def device():
    """Get device (cuda if available, else cpu)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def batch_size():
    """Default batch size for tests."""
    return 2


@pytest.fixture
def seq_len():
    """Default sequence length for tests."""
    return 16


@pytest.fixture
def d_model():
    """Default model dimension for tests."""
    return 64


@pytest.fixture
def vocab_size():
    """Default vocabulary size for tests."""
    return 1000


@pytest.fixture
def seed():
    """Random seed for reproducibility."""
    return 42


@pytest.fixture(autouse=True)
def set_seed(seed):
    """Set random seed before each test."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
