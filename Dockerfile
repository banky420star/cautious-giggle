FROM python:3.12.8-slim

# System deps + uv (fastest installer)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl && \
    rm -rf /var/lib/apt/lists/* && \
    curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

COPY --chown=appuser:appuser requirements.txt .

# Install dependencies using uv (no cache to reduce image size)
RUN uv pip install --no-cache-dir --system -r requirements.txt torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

COPY --chown=appuser:appuser . .

RUN mkdir -p logs models && chown -R appuser:appuser logs models

EXPOSE 9090

HEALTHCHECK --interval=30s --timeout=5s \
    CMD python -c "import json,os; d=json.load(open('logs/risk_engine_state.json')); assert 'last_reset_day' in d" || exit 1

CMD ["python", "-m", "Python.Server_AGI", "--live"]
