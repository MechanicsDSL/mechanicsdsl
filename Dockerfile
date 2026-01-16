# ============================================================================
# MechanicsDSL Docker Image
# ============================================================================
# Multi-stage build for optimized production image
#
# Build: docker build -t mechanicsdsl:2.0 .
# Run:   docker run -it mechanicsdsl:2.0
#
# GPU Build (requires nvidia-docker):
#   docker build --build-arg GPU=true -t mechanicsdsl:2.0-gpu .
# ============================================================================

# Build arguments
ARG PYTHON_VERSION=3.11
ARG GPU=false

# ============================================================================
# Stage 1: Builder
# ============================================================================
FROM python:${PYTHON_VERSION}-slim-bookworm AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# Install MechanicsDSL with all optional dependencies
COPY . /app
WORKDIR /app
RUN pip install --no-cache-dir -e ".[all]"

# ============================================================================
# Stage 2: Runtime (CPU)
# ============================================================================
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime-cpu

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 mdsl

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --from=builder /app /app
WORKDIR /app

# Set up non-root user
USER mdsl

# Default command
CMD ["python", "-c", "import mechanics_dsl; print(f'MechanicsDSL v{mechanics_dsl.__version__} ready')"]

# ============================================================================
# Stage 3: Runtime (GPU/CUDA)
# ============================================================================
FROM nvidia/cuda:12.2-runtime-ubuntu22.04 AS runtime-gpu

# Install Python and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    libopenblas0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 mdsl \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --from=builder /app /app
WORKDIR /app

# Install JAX GPU
RUN pip install --no-cache-dir "jax[cuda12]>=0.4.0" jaxlib diffrax

# Set up non-root user
USER mdsl

# Environment for JAX GPU
ENV XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
ENV JAX_PLATFORM_NAME=gpu

CMD ["python", "-c", "import mechanics_dsl; import jax; print(f'MechanicsDSL v{mechanics_dsl.__version__} with JAX GPU: {jax.devices()}')"]

# ============================================================================
# Final stage selector
# ============================================================================
FROM runtime-cpu AS final
# To use GPU image, build with: --target runtime-gpu

# Expose API server port
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import mechanics_dsl" || exit 1

# Metadata
LABEL org.opencontainers.image.title="MechanicsDSL" \
      org.opencontainers.image.description="Domain-Specific Language for Classical Mechanics" \
      org.opencontainers.image.version="2.0.0" \
      org.opencontainers.image.vendor="MechanicsDSL" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/MechanicsDSL/mechanicsdsl"
