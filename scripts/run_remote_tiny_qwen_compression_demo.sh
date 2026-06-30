#!/usr/bin/env bash
set -euo pipefail

CONTAINER="${CONTAINER:-3ee}"
PROJECT_DIR="${PROJECT_DIR:-/mnt/nvme/622/tidal-ai}"
DEMO_ROOT="${DEMO_ROOT:-/mnt/nvme/622/tidal-demo}"
MODEL_DIR="${MODEL_DIR:-/mnt/nvme/622/models/TinyQwen3-Offline}"

docker exec -i \
  -e TIDAL_PROJECT_DIR="$PROJECT_DIR" \
  -e TIDAL_DEMO_ROOT="$DEMO_ROOT" \
  -e TIDAL_MODEL_DIR="$MODEL_DIR" \
  "$CONTAINER" bash -s <<'CONTAINER_SCRIPT'
set -euo pipefail

cd "$TIDAL_PROJECT_DIR"
export PYTHONPATH="$TIDAL_PROJECT_DIR"

bash scripts/prepare_demo_workspace.sh "$TIDAL_DEMO_ROOT" >/dev/null
python scripts/create_tiny_qwen_fixture.py \
  --output-dir "$TIDAL_MODEL_DIR" \
  --vocab-size 256 \
  --hidden-size 64 \
  --intermediate-size 128
python scripts/tiny_qwen_compression_benchmark.py \
  --demo-root "$TIDAL_DEMO_ROOT" \
  --model-id TinyQwen3-Offline \
  --model-path "$TIDAL_MODEL_DIR" \
  --device npu \
  --dtype float16 \
  --seq-len 16 \
  --batch-size 1 \
  --iters 3 \
  --warmup 1 \
  --cap-budget 8192 \
  --cap-max-iter 4 \
  --cap-policy-steps 1 \
  --cap-samples-per-step 1 \
  --qpruner-average-bits 4.0 \
  --run-label tiny_qwen3_compression_npu
python scripts/write_demo_progress_report.py --demo-root "$TIDAL_DEMO_ROOT"
python - <<'PY'
from pathlib import Path
import json
import os

root = Path(os.environ["TIDAL_DEMO_ROOT"])
for rel in (
    "artifacts/tiny_qwen_compression_tiny_qwen3_compression_npu.json",
    "reports/tiny-qwen-compression-tiny_qwen3_compression_npu.md",
    "reports/ascend-910b-demo-progress.md",
):
    path = root / rel
    size = path.stat().st_size if path.exists() else 0
    print(f"REMOTE_FILE {path} exists={path.exists()} size={size}")

report = json.loads((root / "artifacts/tiny_qwen_compression_tiny_qwen3_compression_npu.json").read_text())
print("REMOTE_COMPRESSION_STATUS", report.get("status"))
print("REMOTE_CAP_RATIO", report.get("cap", {}).get("targeted_compression_ratio"))
print("REMOTE_QPRUNER_BITS", report.get("qpruner", {}).get("average_bits"))
print("REMOTE_BASELINE_LATENCY", report.get("baseline", {}).get("latency_ms"))
PY
CONTAINER_SCRIPT
