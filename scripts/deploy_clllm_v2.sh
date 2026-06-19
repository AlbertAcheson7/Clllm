#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 user@server [remote-dir]" >&2
  exit 2
fi

SERVER="$1"
REMOTE_DIR="${2:-CLllm_v2}"

ssh "$SERVER" mkdir -p "$REMOTE_DIR"
rsync -av --delete \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  projects archive scripts .gitignore \
  "$SERVER:$REMOTE_DIR/"

cat <<EOF
Uploaded lightweight code to $SERVER:$REMOTE_DIR
No data, model, run, checkpoint, or legacy server directory was overwritten.
Run tests there before archiving the old CLllm directory.
EOF
