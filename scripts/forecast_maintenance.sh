#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export HARUKI_FORECAST_CONFIG="${HARUKI_FORECAST_CONFIG:-$PWD/config/forecast.production.json}"

mkdir -p state
LOCK_DIR="$PWD/state/forecast_maintenance.lock"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  if [[ -f "$LOCK_DIR/pid" ]] && kill -0 "$(cat "$LOCK_DIR/pid")" 2>/dev/null; then
    echo "forecast maintenance is already running; skip"
    exit 0
  fi
  echo "remove stale forecast maintenance lock"
  rm -rf "$LOCK_DIR"
  mkdir "$LOCK_DIR"
fi
echo "$$" > "$LOCK_DIR/pid"
trap 'rm -rf "$LOCK_DIR"' EXIT

"${UV_BIN:-/Users/deseer/.local/bin/uv}" run python scripts/forecast_ops.py \
  maintain \
  --config "$HARUKI_FORECAST_CONFIG"
