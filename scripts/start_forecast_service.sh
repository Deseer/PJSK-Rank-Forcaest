#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export HARUKI_FORECAST_CONFIG="${HARUKI_FORECAST_CONFIG:-$PWD/config/forecast.production.json}"
export HARUKI_FORECAST_HOST="${HARUKI_FORECAST_HOST:-0.0.0.0}"
export HARUKI_FORECAST_PORT="${HARUKI_FORECAST_PORT:-18746}"

exec "${UV_BIN:-/Users/deseer/.local/bin/uv}" run python scripts/forecast_service.py \
  --config "$HARUKI_FORECAST_CONFIG" \
  --host "$HARUKI_FORECAST_HOST" \
  --port "$HARUKI_FORECAST_PORT"
