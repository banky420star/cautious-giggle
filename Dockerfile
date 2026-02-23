FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for TA-Lib and other ML libs
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p models logs data validation

# Configure Supervisor
COPY supervisor.conf /etc/supervisor/conf.d/supervisor.conf

# Ports: 8501 (Streamlit)
EXPOSE 8502

CMD ["/usr/bin/supervisord"]
