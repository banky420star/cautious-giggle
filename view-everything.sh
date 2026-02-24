#!/bin/bash
clear
echo "══════════════════════════════════════════════════════════════"
echo "          GROK AGI TRADING BOT — FULL MONITOR"
echo "══════════════════════════════════════════════════════════════"
echo "Time: $(date)"
echo ""

echo "AGI Server (9090): $(ss -tlnp 2>/dev/null | grep :9090 | wc -l) running"
echo "Redis: $(docker ps --filter name=redis --format "{{.Status}}" 2>/dev/null || echo 'Not running')"
echo "n8n: $(docker ps --filter name=n8n --format "{{.Status}}" 2>/dev/null || echo 'Not running')"
echo "Paper Mode: $(grep -o 'paper_mode=.*' Python/hybrid_brain.py 2>/dev/null || echo 'Unknown')"
echo ""

echo "📍 Last Trade:"
redis-cli hgetall last_trade 2>/dev/null || echo "Redis not connected"
echo ""

echo "📈 Recent Logs:"
tail -n 10 logs/agi_server*.log 2>/dev/null | tail -10 || echo "No logs"
echo "══════════════════════════════════════════════════════════════"
