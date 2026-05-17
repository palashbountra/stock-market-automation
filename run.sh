#!/bin/bash
# run.sh — convenience runner for the trading system
# Usage: ./run.sh [command]
# Commands: fetch | summary | indicators
# If no command given, defaults to fetch

COMMAND=${1:-fetch}

# Detect python — works on Mac (python3) and Linux (python or python3)
if command -v python3 &>/dev/null; then
    PY=python3
elif command -v python &>/dev/null; then
    PY=python
else
    echo "❌ Python not found. Install Python 3.11+ from https://python.org"
    exit 1
fi

echo "▶ Running: $PY main.py $COMMAND"
$PY main.py $COMMAND
