# --- Stage 1: Builder (Install dependencies) ---
FROM python:3.12-slim-bookworm AS builder

# Install uv (The fast package manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies into a virtual environment
# --frozen ensures we use exact versions from uv.lock
# --no-dev ensures we don't install test/dev tools in production
RUN uv sync --frozen --no-dev

# --- Stage 2: Runner (Runtime environment) ---
FROM python:3.12-slim-bookworm

# Create a non-root user for security
RUN useradd -m appuser
USER appuser
WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY src/ src/
COPY data/predictions/ data/predictions/
# Note: In a real scenario, data/predictions/ might be mounted as a volume,
# but for this "Batch Serving" demo, we bake the CSV into the image.

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"
# Default API settings (can be overridden at runtime)
ENV API_HOST="0.0.0.0"
ENV API_PORT="8000"

# Expose the port
EXPOSE 8000

# Command to run the API
CMD ["python", "src/api/run.py"]
