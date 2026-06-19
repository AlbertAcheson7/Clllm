#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$HOME/CLllm}"
STAMP="${2:-$(date +%Y%m%d_%H%M%S)}"
OUT="${3:-$HOME/CLllm_inventory_$STAMP}"

mkdir -p "$OUT"
git -C "$ROOT" rev-parse HEAD > "$OUT/git_head.txt" 2>/dev/null || true
git -C "$ROOT" status --short --branch > "$OUT/git_status.txt" 2>/dev/null || true
find "$ROOT" -type f -printf '%s\t%TY-%Tm-%Td %TH:%TM\t%p\n' \
  | sort -nr > "$OUT/files.tsv"
find "$ROOT" -type f -name '*.py' -exec sha256sum {} \; \
  > "$OUT/python_sha256.txt"
find "$ROOT" \( -type d -name __pycache__ -o -type f -name '*.pyc' \) \
  > "$OUT/regenerable.txt"
find "$ROOT" -type f \( -name '*.log' -o -name '*.pt' -o -name '*.ckpt' \) \
  > "$OUT/artifacts.txt"

printf 'Inventory written to %s\n' "$OUT"
