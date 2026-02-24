FROM python:3.12-slim

WORKDIR /app

# System deps for numerical libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create runtime directories
RUN mkdir -p logs models

EXPOSE 9090

# Health check — verify socket is accepting connections
HEALTHCHECK --interval=30s --timeout=5s \
    CMD python -c "import socket; s=socket.create_connection(('127.0.0.1',9090),2); s.close()" || exit 1

CMD ["python", "Python/Server_AGI.py"]
