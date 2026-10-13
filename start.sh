#!/usr/bin/env sh
# macOS / Linux: ./start.sh
cd "$(dirname "$0")"
docker info >/dev/null 2>&1 || { echo "Start Docker Desktop first."; exit 1; }
docker compose up -d --build || exit 1
echo "Waiting for the dashboard..."
until curl -sf http://localhost:8501/_stcore/health >/dev/null; do sleep 2; done
(open http://localhost:8501 2>/dev/null || xdg-open http://localhost:8501) >/dev/null 2>&1
echo "Dashboard: http://localhost:8501   Deck: http://localhost:8080"
