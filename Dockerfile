# =============================================================================
# Neural Memory Reproduction - Production Dockerfile
# Multi-stage build following Docker best practices
# =============================================================================
#
# Build stages:
#   1. base      - Common base with system dependencies
#   2. builder   - Build environment with dev tools
#   3. runtime   - Minimal production image
#   4. dev       - Development image with all tools
#
# Usage:
#   Production:  docker build -t neural-memory .
#   Development: docker build --target dev -t neural-memory:dev .
#   Run tests:   docker run neural-memory pytest
#
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Base Image
# Using slim Python for smaller image size
# -----------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS base

# Prevent Python from writing bytecode and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    # Pip settings
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # App settings
    APP_HOME=/app \
    APP_USER=appuser

# Install system dependencies
# Using --no-install-recommends to minimize image size
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Required for some Python packages
    gcc \
    g++ \
    # For health checks
    curl \
    # Cleanup
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
# Running as root is a security anti-pattern
RUN groupadd --gid 1000 ${APP_USER} \
    && useradd --uid 1000 --gid 1000 --shell /bin/bash --create-home ${APP_USER}

# Set working directory
WORKDIR ${APP_HOME}

# -----------------------------------------------------------------------------
# Stage 2: Builder
# Install dependencies and build wheels
# -----------------------------------------------------------------------------
FROM base AS builder

# Install UV for faster package management
# UV is 10-100x faster than pip
RUN pip install uv

# Copy only dependency files first (better layer caching)
COPY pyproject.toml ./

# Create virtual environment and install dependencies
# This layer is cached unless pyproject.toml changes
RUN uv venv /opt/venv \
    && . /opt/venv/bin/activate \
    && uv pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && uv pip install numpy einops

# Install dev dependencies for testing
RUN . /opt/venv/bin/activate \
    && uv pip install pytest pytest-cov ruff

# -----------------------------------------------------------------------------
# Stage 3: Runtime (Production)
# Minimal image for running the application
# -----------------------------------------------------------------------------
FROM base AS runtime

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Activate virtual environment
ENV PATH="/opt/venv/bin:$PATH" \
    VIRTUAL_ENV="/opt/venv"

# Copy application code
# Using COPY instead of ADD for transparency
COPY --chown=${APP_USER}:${APP_USER} src/ ./src/
COPY --chown=${APP_USER}:${APP_USER} tests/ ./tests/
COPY --chown=${APP_USER}:${APP_USER} pyproject.toml ./
COPY --chown=${APP_USER}:${APP_USER} README.md ./

# Install the package in editable mode
RUN pip install -e . --no-deps

# Switch to non-root user
USER ${APP_USER}

# Health check - verify Python and torch are working
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; import src; print('OK')" || exit 1

# Default command: run tests
CMD ["pytest", "tests/", "-v", "--tb=short"]

# -----------------------------------------------------------------------------
# Stage 4: Development
# Full development environment with all tools
# -----------------------------------------------------------------------------
FROM runtime AS dev

# Switch back to root for installations
USER root

# Install additional development tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    vim \
    less \
    && rm -rf /var/lib/apt/lists/*

# Install development dependencies
RUN pip install \
    ipython \
    jupyter \
    black \
    isort \
    mypy

# Copy additional development files
COPY --chown=${APP_USER}:${APP_USER} notebooks/ ./notebooks/
COPY --chown=${APP_USER}:${APP_USER} scripts/ ./scripts/
COPY --chown=${APP_USER}:${APP_USER} *.md ./
COPY --chown=${APP_USER}:${APP_USER} *.sh ./

# Make scripts executable
RUN chmod +x *.sh 2>/dev/null || true

# Switch back to non-root user
USER ${APP_USER}

# Expose Jupyter port
EXPOSE 8888

# Default command for dev: interactive shell
CMD ["bash"]

# =============================================================================
# Labels for image metadata
# Following OCI Image Format Specification
# =============================================================================
LABEL org.opencontainers.image.title="Neural Memory Reproduction" \
      org.opencontainers.image.description="Reproduction of TITANS, MIRAS, and NL neural memory papers" \
      org.opencontainers.image.authors="Arya Teja Rudraraju" \
      org.opencontainers.image.source="https://github.com/AryaTejaRudraraju/neural-memory-reproduction" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.version="1.0.0"
