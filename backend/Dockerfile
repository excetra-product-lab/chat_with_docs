FROM python:3.11-slim

# Install uv
RUN pip install uv

WORKDIR /app

# Copy project metadata and lockfile
COPY pyproject.toml uv.lock ./

# Sync environment (install dependencies)
RUN uv sync

# Copy application code
COPY ./app ./app

# Use uv run for our server (ensures environment is up-to-date)
CMD ["uv", "run", "uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"]
