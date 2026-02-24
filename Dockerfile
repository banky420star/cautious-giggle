FROM python:3.12-slim

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

# Install dependencies using uv
RUN uv pip install --system -r requirements.txt torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

COPY --chown=appuser:appuser . .

RUN mkdir -p logs models && chown -R appuser:appuser logs models

EXPOSE 9090

HEALTHCHECK --interval=30s --timeout=5s \
    CMD python -c "import socket; s=socket.create_connection(('127.0.0.1',9090),2); s.close()" || exit 1

CMD ["python", "-m", "Python.Server_AGI", "--production"]
