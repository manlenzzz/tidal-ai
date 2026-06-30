#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/mnt/nvme/622/tidal-demo}"

mkdir -p \
  "$ROOT/artifacts" \
  "$ROOT/logs" \
  "$ROOT/reports" \
  "$ROOT/recordings" \
  "$ROOT/scripts"

cat > "$ROOT/README.md" <<'EOF'
# TIDAL-AI Ascend 910B Demo Evidence

This directory stores durable materials for the Ascend 910B demo:

- `artifacts/`: JSON outputs, model/workflow summaries, copied configs.
- `logs/`: terminal logs captured with `script` or command tee.
- `reports/`: Markdown summaries and compatibility issue lists.
- `recordings/`: later demo videos or terminal recordings.
- `scripts/`: exact scripts used for reproducible runs.

Do not delete this directory during normal cleanup. It is the demo evidence bundle.
EOF

printf 'DEMO_ROOT=%s\n' "$ROOT"
find "$ROOT" -maxdepth 2 -type d | sort
