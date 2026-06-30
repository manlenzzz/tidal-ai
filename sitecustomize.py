"""Project-local Python startup hooks for opt-in demo compatibility patches."""
from __future__ import annotations

import os
import sys


def _install_vllm_ascend_metadata_shim() -> None:
    try:
        from tidal.integrations.ascend_runtime_path import apply_acl_pythonpath_to_env, apply_acl_pythonpath_to_sys_path

        apply_acl_pythonpath_to_env(os.environ)
        apply_acl_pythonpath_to_sys_path(os.environ)
        from tidal.integrations.vllm_ascend_attention_metadata_shim import install_auto

        install_auto()
    except Exception as exc:
        if os.environ.get("TIDAL_VLLM_ASCEND_METADATA_SHIM_DEBUG") == "1":
            print(f"[tidal] metadata shim startup install failed: {exc!r}", file=sys.stderr)


if os.environ.get("TIDAL_VLLM_ASCEND_METADATA_SHIM") == "1":
    _install_vllm_ascend_metadata_shim()
