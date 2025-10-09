#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/artifacts/logs"
mkdir -p "${LOG_DIR}"
cd "${ROOT_DIR}"

export PYTHONHASHSEED="${PYTHONHASHSEED:-42}"
export A22A_GLOBAL_SEED="${A22A_GLOBAL_SEED:-42}"
export A22A_LIVE_MODE="${A22A_LIVE_MODE:-0}"

TIMESTAMP="$(date -u +"%Y%m%dT%H%M%SZ")"
LOG_FILE="${LOG_DIR}/pipeline_${TIMESTAMP}.log"

STEPS=(
  "ingest"
  "features"
  "train"
  "sim"
  "meta"
  "portfolio"
  "market"
  "clv"
  "report_batch"
  "monitor"
)

{
  echo "[pipeline] started at ${TIMESTAMP} (UTC)"
  echo "[pipeline] using A22A_GLOBAL_SEED=${A22A_GLOBAL_SEED}"
  for step in "${STEPS[@]}"; do
    echo "[pipeline] running make ${step}"
    if make "${step}"; then
      echo "[pipeline] make ${step} completed"
    else
      echo "[pipeline] make ${step} failed" >&2
      exit 1
    fi
  done
  echo "[pipeline] finished at $(date -u +"%Y%m%dT%H%M%SZ")"
} 2>&1 | tee "${LOG_FILE}"

