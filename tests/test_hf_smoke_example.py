import subprocess
import sys
from pathlib import Path


def test_hf_smoke_help_runs_without_model_download():
    script = Path("examples/model_support/hf_smoke.py")
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--model-id" in result.stdout
    assert "--max-modules" in result.stdout
