#!/usr/bin/env bash
set -e

echo "==========================================="
echo "🚀 Starting AutoShorts AI Studio"
echo "==========================================="

# Install bgutil server modules if missing
if [ -d "tools/bgutil-server" ] && [ ! -d "tools/bgutil-server/node_modules" ]; then
    echo "📦 Installing PO-Token server node_modules..."
    (cd tools/bgutil-server && npm install --silent)
fi

echo "🎬 Starting Web Dashboard on http://0.0.0.0:8000 ..."
python app.py
