#!/bin/bash

echo "🚀 Starting NeuroTrader v3.1 System..."

# Set Matplotlib config directory to avoid font cache warnings on Mac
export MPLCONFIGDIR=/tmp/matplotlib
mkdir -p $MPLCONFIGDIR

# Activate virtual environment
if [ -d venv ]; then
    source venv/bin/activate
elif [ -d venv_m4 ]; then
    source venv_m4/bin/activate
fi

# Load environment variables (ignoring comments)
if [ -f .env ]; then
    set -a
    source <(grep -v '^#' .env | sed 's/\r$//')
    set +a
else
    echo "⚠️ .env file not found. Copying .env.example..."
    cp -n .env.example .env
fi

# Run background data updater
echo "📊 Starting dashboard data updater..."
python3 update_dashboard_data.py &

# Start the main training and evolution server
echo "🧠 Starting Main Evolution & Trade Server..."
python3 main.py
