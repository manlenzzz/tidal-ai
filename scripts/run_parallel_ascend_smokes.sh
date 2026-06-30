#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-/mnt/nvme/622/tidal-ai}"
DEMO_ROOT="${2:-/mnt/nvme/622/tidal-demo}"

cd "$PROJECT_ROOT"
bash scripts/prepare_demo_workspace.sh "$DEMO_ROOT" >/dev/null

run_one() {
  local card="$1"
  local workflow="$2"
  local log="$DEMO_ROOT/logs/${workflow}_npu${card}.log"
  local json="$DEMO_ROOT/artifacts/${workflow}_npu${card}.json"
  local compat="$DEMO_ROOT/reports/${workflow}_npu${card}.md"

  (
    export ASCEND_RT_VISIBLE_DEVICES="$card"
    export PYTHONPATH="$PROJECT_ROOT"
    python scripts/ascend_npu_workflow_smoke.py \
      --device npu \
      --dtype float32 \
      --workflow "$workflow" \
      --output-json "$json" \
      --compat-md "$compat"
  ) >"$log" 2>&1 &
  echo "$! $workflow $card $log $json" >> "$DEMO_ROOT/artifacts/parallel_smoke_pids.txt"
}

rm -f "$DEMO_ROOT/artifacts/parallel_smoke_pids.txt"
run_one 0 cap
run_one 1 qpruner
run_one 2 rankadaptor

status=0
while read -r pid workflow card log json; do
  if wait "$pid"; then
    printf 'PASS %s npu%s %s\n' "$workflow" "$card" "$json"
  else
    printf 'FAIL %s npu%s %s\n' "$workflow" "$card" "$log"
    status=1
  fi
done < "$DEMO_ROOT/artifacts/parallel_smoke_pids.txt"

python scripts/aggregate_parallel_ascend_smokes.py "$DEMO_ROOT"

exit "$status"
