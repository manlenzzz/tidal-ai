#!/usr/bin/env python
"""Benchmark vLLM-Ascend generate on baseline/CAP/QPruner serving exports."""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from ascend_inference_benchmark import (  # noqa: E402
    DEFAULT_PROMPTS,
    count_vllm_output_tokens,
    tokens_per_second,
)
from tidal.integrations.ascend_runtime_path import apply_acl_pythonpath_to_env


METHODS = ("baseline", "cap", "qpruner")
METADATA_SHIM_ENV = "TIDAL_VLLM_ASCEND_METADATA_SHIM"
POSITIVE_NUMBER_RE = re.compile(r"(\d+(?:\.\d+)?)")


def export_path(demo_root: Path, export_run_label: str, method: str) -> Path:
    return demo_root / "serving_exports" / export_run_label / method


def fmt(value: Any) -> str:
    if value is None:
        return "missing"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def synchronize_backend() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if hasattr(torch, "npu") and hasattr(torch.npu, "synchronize"):
        try:
            torch.npu.synchronize()
        except Exception:
            pass


def reset_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    if hasattr(torch, "npu") and hasattr(torch.npu, "reset_peak_memory_stats"):
        try:
            torch.npu.reset_peak_memory_stats()
        except Exception:
            pass


def peak_memory_mb() -> float | None:
    if torch.cuda.is_available():
        return round(torch.cuda.max_memory_allocated() / 1024**2, 1)
    if hasattr(torch, "npu") and hasattr(torch.npu, "max_memory_allocated"):
        try:
            return round(torch.npu.max_memory_allocated() / 1024**2, 1)
        except Exception:
            return None
    return None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _positive_mb(value: Any) -> float | None:
    number = _as_float(value)
    if number is None or number <= 0.0:
        return None
    return round(number, 1)


def _first_number(cell: str) -> float | None:
    match = POSITIVE_NUMBER_RE.search(cell)
    if not match:
        return None
    return float(match.group(1))


def _integer_cell(cell: str) -> int | None:
    stripped = cell.strip()
    if not re.fullmatch(r"\d+", stripped):
        return None
    return int(stripped)


def _process_row_card_memory(cells: list[str]) -> tuple[str, float] | None:
    if len(cells) < 4:
        return None
    card: str | None = None
    pid: int | None = None
    if re.fullmatch(r"\d+\s+\d+", cells[0].strip()):
        card = cells[0].strip().split()[0]
        pid = _integer_cell(cells[1])
    elif len(cells) >= 5 and _integer_cell(cells[0]) is not None and _integer_cell(cells[1]) is not None:
        card = cells[0].strip()
        pid = _integer_cell(cells[2])
    elif _integer_cell(cells[0]) is not None:
        card = cells[0].strip()
        pid = _integer_cell(cells[1])
    if card is None or pid is None or pid <= 0:
        return None
    memory = _first_number(cells[-1])
    return (card, round(memory, 1)) if memory is not None else None


def parse_npu_smi_process_memory_mb(text: str, cards: set[str] | None = None) -> float | None:
    values: list[float] = []
    for line in text.splitlines():
        if "Process memory" in line or "No running processes" in line:
            continue
        if not line.lstrip().startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        parsed = _process_row_card_memory(cells)
        if parsed is None:
            continue
        card, memory = parsed
        if cards is not None and card not in cards:
            continue
        if memory is not None and memory > 0.0:
            values.append(memory)
    if not values:
        return None
    return round(max(values), 1)


def visible_npu_cards_from_env(env: dict[str, str] | None = None) -> set[str] | None:
    source = env if env is not None else os.environ
    raw = source.get("ASCEND_RT_VISIBLE_DEVICES") or source.get("ASCEND_VISIBLE_DEVICES")
    if not raw:
        return None
    cards = {part.strip() for part in raw.split(",") if part.strip() and part.strip().lower() != "all"}
    return cards or None


def npu_smi_process_memory_mb(timeout_s: float = 5.0) -> float | None:
    try:
        completed = subprocess.run(
            ["npu-smi", "info"],
            check=False,
            text=True,
            capture_output=True,
            timeout=timeout_s,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    return parse_npu_smi_process_memory_mb(completed.stdout + completed.stderr, cards=visible_npu_cards_from_env())


def select_memory_measurement(torch_peak_mb: Any, npu_smi_process_mb: Any) -> dict[str, Any]:
    torch_peak = _as_float(torch_peak_mb)
    npu_smi_peak = _as_float(npu_smi_process_mb)
    selected_npu_smi = _positive_mb(npu_smi_peak)
    selected_torch = _positive_mb(torch_peak)
    if selected_npu_smi is not None:
        peak_mem_mb = selected_npu_smi
        source = "npu-smi process table"
    elif selected_torch is not None:
        peak_mem_mb = selected_torch
        source = "torch allocator"
    else:
        peak_mem_mb = None
        source = "unavailable"
    return {
        "peak_mem_mb": peak_mem_mb,
        "memory_measurement_source": source,
        "torch_peak_mem_mb": round(torch_peak, 1) if torch_peak is not None else None,
        "npu_smi_process_mem_mb": round(npu_smi_peak, 1) if npu_smi_peak is not None else None,
    }


def empty_backend_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "npu") and hasattr(torch.npu, "empty_cache"):
        try:
            torch.npu.empty_cache()
        except Exception:
            pass


def _prepend_pythonpath(env: dict[str, str], path: Path) -> None:
    path_text = str(path)
    entries = [entry for entry in env.get("PYTHONPATH", "").split(os.pathsep) if entry]
    if path_text not in entries:
        entries.insert(0, path_text)
    env["PYTHONPATH"] = os.pathsep.join(entries)


def metadata_shim_environment(enabled: bool) -> dict[str, str]:
    env = os.environ.copy()
    if enabled:
        env[METADATA_SHIM_ENV] = "1"
        _prepend_pythonpath(env, REPO_ROOT)
        apply_acl_pythonpath_to_env(env)
    return env


def enable_metadata_shim_environment(enabled: bool) -> None:
    if enabled:
        os.environ[METADATA_SHIM_ENV] = "1"
        _prepend_pythonpath(os.environ, REPO_ROOT)  # type: ignore[arg-type]
        apply_acl_pythonpath_to_env(os.environ)  # type: ignore[arg-type]


def import_vllm_ascend_patch(enabled: bool) -> dict[str, Any]:
    if not enabled:
        return {"enabled": False, "status": "SKIPPED"}
    try:
        import vllm_ascend.patch.platform  # noqa: F401

        return {"enabled": True, "status": "IMPORTED"}
    except Exception as exc:
        return {
            "enabled": True,
            "status": "FAIL",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def install_selector_shim(enabled: bool) -> dict[str, Any]:
    if not enabled:
        return {"enabled": False, "status": "SKIPPED"}
    try:
        from tidal.integrations.vllm_ascend_selector_shim import install

        result = install()
        result["enabled"] = True
        return result
    except Exception as exc:
        return {
            "enabled": True,
            "status": "FAIL",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def install_metadata_shim(enabled: bool) -> dict[str, Any]:
    if not enabled:
        return {"enabled": False, "status": "SKIPPED"}
    try:
        from tidal.integrations.vllm_ascend_attention_metadata_shim import install

        result = install()
        result["enabled"] = True
        return result
    except Exception as exc:
        return {
            "enabled": True,
            "status": "FAIL",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def llm_kwargs(
    *,
    path: Path,
    dtype_name: str,
    gpu_memory_utilization: float,
    max_model_len: int,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": str(path),
        "trust_remote_code": True,
        "max_model_len": max_model_len,
        "gpu_memory_utilization": gpu_memory_utilization,
        "enforce_eager": True,
    }
    if dtype_name:
        kwargs["dtype"] = dtype_name
    return kwargs


def benchmark_export(
    *,
    path: Path,
    method: str,
    dtype_name: str,
    prompts: Sequence[str],
    max_new_tokens: int,
    iters: int,
    warmup: int,
    gpu_memory_utilization: float,
    max_model_len: int,
) -> dict[str, Any]:
    if not path.exists():
        return {
            "status": "MISSING",
            "method": method,
            "exists": False,
            "path": str(path),
            "vllm_load": "MISSING",
        }

    from vllm import LLM, SamplingParams

    llm = None
    try:
        load_start = time.perf_counter()
        llm = LLM(
            **llm_kwargs(
                path=path,
                dtype_name=dtype_name,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
            )
        )
        synchronize_backend()
        load_time_s = time.perf_counter() - load_start
        reset_peak_memory()
        sampling_params = SamplingParams(max_tokens=max_new_tokens, temperature=0.0)
        prompt_list = list(prompts)
        for _ in range(warmup):
            llm.generate(prompt_list, sampling_params)
        synchronize_backend()
        start = time.perf_counter()
        outputs = None
        for _ in range(iters):
            outputs = llm.generate(prompt_list, sampling_params)
        synchronize_backend()
        latency_ms = (time.perf_counter() - start) / max(1, iters) * 1000.0
        generated_tokens = count_vllm_output_tokens(outputs or [])
        memory_measurement = select_memory_measurement(
            torch_peak_mb=peak_memory_mb(),
            npu_smi_process_mb=npu_smi_process_memory_mb(),
        )
        return {
            "status": "PASS",
            "method": method,
            "exists": True,
            "path": str(path),
            "vllm_load": "PASS",
            "load_time_s": round(load_time_s, 3),
            "batch_size": len(prompt_list),
            "generated_tokens": generated_tokens,
            "latency_ms": latency_ms,
            "tokens_per_s": tokens_per_second(generated_tokens=generated_tokens, latency_ms=latency_ms),
            **memory_measurement,
        }
    except Exception as exc:
        return {
            "status": "FAIL",
            "method": method,
            "exists": True,
            "path": str(path),
            "vllm_load": "FAIL",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    finally:
        if llm is not None:
            del llm
        gc.collect()
        empty_backend_cache()


def missing_report(
    *,
    demo_root: Path,
    export_run_label: str,
    run_label: str,
    dtype_name: str,
    gpu_memory_utilization: float,
    preload_patch: bool,
    preload_selector_shim: bool,
    preload_metadata_shim: bool,
) -> dict[str, Any]:
    exports = {
        method: {
            "status": "MISSING",
            "method": method,
            "exists": False,
            "path": str(export_path(demo_root, export_run_label, method)),
            "vllm_load": "MISSING",
        }
        for method in METHODS
    }
    return {
        "status": "MODEL_MISSING",
        "run_label": run_label,
        "export_run_label": export_run_label,
        "backend": "vllm_ascend_generate",
        "dtype": dtype_name,
        "gpu_memory_utilization": gpu_memory_utilization,
        "preload_vllm_ascend_patch": preload_patch,
        "preload_vllm_ascend_selector_shim": preload_selector_shim,
        "preload_vllm_ascend_metadata_shim": preload_metadata_shim,
        "exports": exports,
        "summary": empty_summary(),
        "download_attempted": False,
        "platform": platform.platform(),
    }


def empty_summary() -> dict[str, Any]:
    return {
        "best_method": None,
        "best_tokens_per_s": None,
        "cap_vs_baseline_speedup": None,
        "qpruner_vs_baseline_speedup": None,
        "cap_vs_qpruner_speedup": None,
    }


def _speedup(numerator: Any, denominator: Any) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return round(float(numerator) / float(denominator), 3)


def summarize(exports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    passing = {
        method: payload
        for method, payload in exports.items()
        if payload.get("status") == "PASS" and payload.get("tokens_per_s") is not None
    }
    if not passing:
        return empty_summary()
    best_method, best_payload = max(passing.items(), key=lambda item: float(item[1]["tokens_per_s"]))
    baseline_tps = exports.get("baseline", {}).get("tokens_per_s")
    cap_tps = exports.get("cap", {}).get("tokens_per_s")
    qpruner_tps = exports.get("qpruner", {}).get("tokens_per_s")
    return {
        "best_method": best_method,
        "best_tokens_per_s": best_payload.get("tokens_per_s"),
        "cap_vs_baseline_speedup": _speedup(cap_tps, baseline_tps),
        "qpruner_vs_baseline_speedup": _speedup(qpruner_tps, baseline_tps),
        "cap_vs_qpruner_speedup": _speedup(cap_tps, qpruner_tps),
    }


def run_benchmark(
    *,
    demo_root: Path,
    export_run_label: str,
    run_label: str,
    dtype_name: str,
    max_new_tokens: int,
    iters: int,
    warmup: int,
    gpu_memory_utilization: float,
    preload_vllm_ascend_patch: bool,
    preload_vllm_ascend_selector_shim: bool,
    preload_vllm_ascend_metadata_shim: bool,
    max_model_len: int = 128,
    subprocess_per_method: bool = True,
    parallel_methods: bool = False,
    method_cards: Sequence[int] | None = None,
) -> dict[str, Any]:
    paths = {method: export_path(demo_root, export_run_label, method) for method in METHODS}
    if not any(path.exists() for path in paths.values()):
        return missing_report(
            demo_root=demo_root,
            export_run_label=export_run_label,
            run_label=run_label,
            dtype_name=dtype_name,
            gpu_memory_utilization=gpu_memory_utilization,
            preload_patch=preload_vllm_ascend_patch,
            preload_selector_shim=preload_vllm_ascend_selector_shim,
            preload_metadata_shim=preload_vllm_ascend_metadata_shim,
        )

    patch_status = import_vllm_ascend_patch(preload_vllm_ascend_patch)
    selector_shim = install_selector_shim(preload_vllm_ascend_selector_shim)
    metadata_shim = install_metadata_shim(preload_vllm_ascend_metadata_shim)
    card_list = list(method_cards or [])
    if subprocess_per_method and parallel_methods:
        exports = benchmark_exports_parallel(
            paths=paths,
            dtype_name=dtype_name,
            max_new_tokens=max_new_tokens,
            iters=iters,
            warmup=warmup,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            preload_vllm_ascend_patch=preload_vllm_ascend_patch,
            preload_vllm_ascend_selector_shim=preload_vllm_ascend_selector_shim,
            preload_vllm_ascend_metadata_shim=preload_vllm_ascend_metadata_shim,
            method_cards=card_list,
        )
    elif subprocess_per_method:
        exports = {
            method: benchmark_export_subprocess(
                path=path,
                method=method,
                dtype_name=dtype_name,
                max_new_tokens=max_new_tokens,
                iters=iters,
                warmup=warmup,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                preload_vllm_ascend_patch=preload_vllm_ascend_patch,
                preload_vllm_ascend_selector_shim=preload_vllm_ascend_selector_shim,
                preload_vllm_ascend_metadata_shim=preload_vllm_ascend_metadata_shim,
            )
            for method, path in paths.items()
        }
    else:
        exports = {
            method: benchmark_export(
                path=path,
                method=method,
                dtype_name=dtype_name,
                prompts=DEFAULT_PROMPTS,
                max_new_tokens=max_new_tokens,
                iters=iters,
                warmup=warmup,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
            )
            for method, path in paths.items()
        }
    status = "PASS" if all(payload.get("status") == "PASS" for payload in exports.values()) else "FAIL"
    return {
        "status": status,
        "run_label": run_label,
        "export_run_label": export_run_label,
        "backend": "vllm_ascend_generate",
        "dtype": dtype_name,
        "max_new_tokens": max_new_tokens,
        "iters": iters,
        "warmup": warmup,
        "gpu_memory_utilization": gpu_memory_utilization,
        "preload_vllm_ascend_patch": preload_vllm_ascend_patch,
        "preload_vllm_ascend_patch_status": patch_status,
        "preload_vllm_ascend_selector_shim": preload_vllm_ascend_selector_shim,
        "selector_shim": selector_shim,
        "preload_vllm_ascend_metadata_shim": preload_vllm_ascend_metadata_shim,
        "metadata_shim": metadata_shim,
        "max_model_len": max_model_len,
        "subprocess_per_method": subprocess_per_method,
        "parallel_methods": parallel_methods if subprocess_per_method else False,
        "method_cards": card_list if subprocess_per_method and parallel_methods else None,
        "exports": exports,
        "summary": summarize(exports),
        "download_attempted": False,
        "platform": platform.platform(),
    }


def extract_method_payload(stdout: str) -> dict[str, Any] | None:
    prefix = "VLLM_METHOD_BENCHMARK "
    for line in reversed(stdout.splitlines()):
        if line.startswith(prefix):
            return json.loads(line[len(prefix) :])
    return None


def method_command(
    *,
    path: Path,
    method: str,
    dtype_name: str,
    max_new_tokens: int,
    iters: int,
    warmup: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    preload_vllm_ascend_patch: bool,
    preload_vllm_ascend_selector_shim: bool,
    preload_vllm_ascend_metadata_shim: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--single-method",
        method,
        "--method-path",
        str(path),
        "--dtype",
        dtype_name,
        "--max-new-tokens",
        str(max_new_tokens),
        "--iters",
        str(iters),
        "--warmup",
        str(warmup),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--max-model-len",
        str(max_model_len),
    ]
    if preload_vllm_ascend_patch:
        command.append("--preload-vllm-ascend-patch")
    if preload_vllm_ascend_selector_shim:
        command.append("--preload-vllm-ascend-selector-shim")
    if preload_vllm_ascend_metadata_shim:
        command.append("--preload-vllm-ascend-metadata-shim")
    return command


def method_environment(*, preload_vllm_ascend_metadata_shim: bool, visible_card: int | None = None) -> dict[str, str]:
    env = metadata_shim_environment(preload_vllm_ascend_metadata_shim)
    if visible_card is not None:
        env["ASCEND_RT_VISIBLE_DEVICES"] = str(visible_card)
    return env


def payload_from_completed_process(
    *,
    method: str,
    path: Path,
    returncode: int,
    stdout: str,
    stderr: str,
    visible_card: int | None = None,
) -> dict[str, Any]:
    payload = extract_method_payload(stdout)
    if payload is not None:
        payload["subprocess_returncode"] = returncode
        if visible_card is not None:
            payload["visible_card"] = visible_card
        if returncode != 0 and payload.get("status") == "PASS":
            payload["status"] = "FAIL"
        if returncode != 0 or payload.get("status") != "PASS":
            payload["stdout_tail"] = stdout[-4000:]
            payload["stderr_tail"] = stderr[-4000:]
        return payload
    payload = {
        "status": "FAIL",
        "method": method,
        "exists": True,
        "path": str(path),
        "vllm_load": "FAIL",
        "error_type": "SubprocessError",
        "error": "Missing VLLM_METHOD_BENCHMARK payload",
        "subprocess_returncode": returncode,
        "stdout_tail": stdout[-4000:],
        "stderr_tail": stderr[-4000:],
    }
    if visible_card is not None:
        payload["visible_card"] = visible_card
    return payload


def benchmark_export_subprocess(
    *,
    path: Path,
    method: str,
    dtype_name: str,
    max_new_tokens: int,
    iters: int,
    warmup: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    preload_vllm_ascend_patch: bool,
    preload_vllm_ascend_selector_shim: bool,
    preload_vllm_ascend_metadata_shim: bool,
) -> dict[str, Any]:
    if not path.exists():
        return {
            "status": "MISSING",
            "method": method,
            "exists": False,
            "path": str(path),
            "vllm_load": "MISSING",
        }
    command = method_command(
        path=path,
        method=method,
        dtype_name=dtype_name,
        max_new_tokens=max_new_tokens,
        iters=iters,
        warmup=warmup,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        preload_vllm_ascend_patch=preload_vllm_ascend_patch,
        preload_vllm_ascend_selector_shim=preload_vllm_ascend_selector_shim,
        preload_vllm_ascend_metadata_shim=preload_vllm_ascend_metadata_shim,
    )
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        capture_output=True,
        env=method_environment(preload_vllm_ascend_metadata_shim=preload_vllm_ascend_metadata_shim),
    )
    return payload_from_completed_process(
        method=method,
        path=path,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _card_for_method(method_cards: Sequence[int], index: int) -> int | None:
    if index >= len(method_cards):
        return None
    return int(method_cards[index])


def benchmark_exports_parallel(
    *,
    paths: dict[str, Path],
    dtype_name: str,
    max_new_tokens: int,
    iters: int,
    warmup: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    preload_vllm_ascend_patch: bool,
    preload_vllm_ascend_selector_shim: bool,
    preload_vllm_ascend_metadata_shim: bool,
    method_cards: Sequence[int],
) -> dict[str, dict[str, Any]]:
    running: dict[str, tuple[subprocess.Popen[str], Path, int | None]] = {}
    exports: dict[str, dict[str, Any]] = {}
    for index, method in enumerate(METHODS):
        path = paths[method]
        if not path.exists():
            exports[method] = {
                "status": "MISSING",
                "method": method,
                "exists": False,
                "path": str(path),
                "vllm_load": "MISSING",
            }
            continue
        visible_card = _card_for_method(method_cards, index)
        command = method_command(
            path=path,
            method=method,
            dtype_name=dtype_name,
            max_new_tokens=max_new_tokens,
            iters=iters,
            warmup=warmup,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            preload_vllm_ascend_patch=preload_vllm_ascend_patch,
            preload_vllm_ascend_selector_shim=preload_vllm_ascend_selector_shim,
            preload_vllm_ascend_metadata_shim=preload_vllm_ascend_metadata_shim,
        )
        process = subprocess.Popen(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=method_environment(
                preload_vllm_ascend_metadata_shim=preload_vllm_ascend_metadata_shim,
                visible_card=visible_card,
            ),
        )
        running[method] = (process, path, visible_card)
    for method in METHODS:
        if method in exports:
            continue
        process, path, visible_card = running[method]
        stdout, stderr = process.communicate()
        exports[method] = payload_from_completed_process(
            method=method,
            path=path,
            returncode=int(process.returncode or 0),
            stdout=stdout,
            stderr=stderr,
            visible_card=visible_card,
        )
    return exports


def row(method: str, payload: dict[str, Any]) -> str:
    return (
        "| {method} | {status} | {exists} | {load} | {load_time} | {latency} | {tps} | {peak} | {source} | {npu_smi} | {torch_peak} |"
    ).format(
        method=method,
        status=payload.get("status", "missing"),
        exists=payload.get("exists", "missing"),
        load=payload.get("vllm_load", "missing"),
        load_time=fmt(payload.get("load_time_s")),
        latency=fmt(payload.get("latency_ms")),
        tps=fmt(payload.get("tokens_per_s")),
        peak=fmt(payload.get("peak_mem_mb")),
        source=fmt(payload.get("memory_measurement_source")),
        npu_smi=fmt(payload.get("npu_smi_process_mem_mb")),
        torch_peak=fmt(payload.get("torch_peak_mem_mb")),
    )


def method_label(method: str) -> str:
    return {"baseline": "baseline", "cap": "CAP", "qpruner": "QPruner"}.get(method, method)


def _first_root_cause_line(text: str) -> str | None:
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        for marker in ("AttributeError:", "RuntimeError:", "ValueError:", "TypeError:", "ImportError:"):
            if marker in line:
                return line[line.index(marker) :]
    return None


def failure_summary(exports: dict[str, Any]) -> str:
    failures = []
    root_cause = None
    for method in METHODS:
        payload = exports.get(method, {}) if isinstance(exports.get(method), dict) else {}
        if payload.get("status") == "PASS":
            continue
        if not payload:
            continue
        failures.append((method_label(method), payload.get("error_type") or payload.get("status") or "unknown"))
        if root_cause is None:
            root_cause = _first_root_cause_line(
                "\n".join(
                    str(payload.get(key) or "")
                    for key in ("stdout_tail", "stderr_tail", "error")
                )
            )
    if not failures:
        return "no failing methods"
    grouped = ", ".join(label for label, _ in failures)
    error_types = {error_type for _, error_type in failures}
    error_text = failures[0][1] if len(error_types) == 1 else ", ".join(
        f"{label}={error_type}" for label, error_type in failures
    )
    summary = f"{grouped}: {error_text}"
    if root_cause:
        summary += f"; {root_cause}"
    return summary


def markdown_report(report: dict[str, Any]) -> str:
    exports = report.get("exports", {})
    summary = report.get("summary", {})
    root_cause = failure_summary(exports)
    lines = [
        "# vLLM-Ascend Serving Benchmark",
        "",
        "vLLM generate benchmark for dense HF serving exports, using the project-local selector shim when enabled.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Status | {report.get('status')} |",
        f"| Backend | {report.get('backend')} |",
        f"| Export run label | {report.get('export_run_label')} |",
        f"| Dtype | {report.get('dtype')} |",
        f"| GPU memory utilization | {report.get('gpu_memory_utilization')} |",
        f"| preload_vllm_ascend_patch | {report.get('preload_vllm_ascend_patch')} |",
        f"| preload_vllm_ascend_selector_shim | {report.get('preload_vllm_ascend_selector_shim')} |",
        f"| preload_vllm_ascend_metadata_shim | {report.get('preload_vllm_ascend_metadata_shim')} |",
        f"| selector_shim_status | {(report.get('selector_shim') or {}).get('status')} |",
        f"| metadata_shim_status | {(report.get('metadata_shim') or {}).get('status')} |",
        f"| best_method | {summary.get('best_method')} |",
        f"| best_tokens_per_s | {summary.get('best_tokens_per_s')} |",
        f"| cap_vs_baseline_speedup | {summary.get('cap_vs_baseline_speedup')} |",
        f"| qpruner_vs_baseline_speedup | {summary.get('qpruner_vs_baseline_speedup')} |",
        f"| cap_vs_qpruner_speedup | {summary.get('cap_vs_qpruner_speedup')} |",
        f"| Root cause | {root_cause} |",
        "",
        "| Method | Status | Exists | vLLM load | Load time s | Latency ms | Tokens/s | Peak MB | Peak source | NPU process MB | Torch peak MB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|",
        row("Baseline", exports.get("baseline", {})),
        row("CAP", exports.get("cap", {})),
        row("QPruner", exports.get("qpruner", {})),
    ]
    return "\n".join(lines) + "\n"


def write_outputs(report: dict[str, Any], demo_root: Path, run_label: str) -> None:
    artifacts = demo_root / "artifacts"
    reports = demo_root / "reports"
    artifacts.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    (artifacts / f"vllm_serving_benchmark_{run_label}.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    (reports / f"vllm-serving-benchmark-{run_label}.md").write_text(markdown_report(report))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark vLLM-Ascend generate on serving exports")
    parser.add_argument("--demo-root", default="/mnt/nvme/622/tidal-demo")
    parser.add_argument("--export-run-label", default="tiny_qwen3_serving_export_npu")
    parser.add_argument("--run-label", default="tiny_qwen3_serving_vllm_selector_shim_npu")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.05)
    parser.add_argument("--max-model-len", type=int, default=128)
    parser.add_argument("--preload-vllm-ascend-patch", action="store_true")
    parser.add_argument("--preload-vllm-ascend-selector-shim", action="store_true")
    parser.add_argument("--preload-vllm-ascend-metadata-shim", action="store_true")
    parser.add_argument("--single-method", choices=METHODS)
    parser.add_argument("--method-path")
    parser.add_argument("--output-json")
    parser.add_argument(
        "--in-process",
        action="store_true",
        help="Benchmark all methods in the current Python process instead of one subprocess per method.",
    )
    parser.add_argument(
        "--parallel-methods",
        action="store_true",
        help="Run baseline/CAP/QPruner method subprocesses concurrently.",
    )
    parser.add_argument(
        "--method-cards",
        default="0,1,2",
        help="Comma-separated ASCEND_RT_VISIBLE_DEVICES card ids for baseline,CAP,QPruner when --parallel-methods is set.",
    )
    return parser.parse_args(argv)


def parse_method_cards(value: str) -> list[int]:
    cards: list[int] = []
    for part in value.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        cards.append(int(stripped))
    return cards


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.single_method:
        if not args.method_path:
            print("--method-path is required with --single-method", file=sys.stderr)
            return 2
        enable_metadata_shim_environment(args.preload_vllm_ascend_metadata_shim)
        patch_status = import_vllm_ascend_patch(args.preload_vllm_ascend_patch)
        selector_shim = install_selector_shim(args.preload_vllm_ascend_selector_shim)
        metadata_shim = install_metadata_shim(args.preload_vllm_ascend_metadata_shim)
        payload = benchmark_export(
            path=Path(args.method_path),
            method=args.single_method,
            dtype_name=args.dtype,
            prompts=DEFAULT_PROMPTS,
            max_new_tokens=args.max_new_tokens,
            iters=args.iters,
            warmup=args.warmup,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
        )
        payload["preload_vllm_ascend_patch_status"] = patch_status
        payload["selector_shim"] = selector_shim
        payload["metadata_shim"] = metadata_shim
        if args.output_json:
            output_path = Path(args.output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        print("VLLM_METHOD_BENCHMARK " + json.dumps(payload, sort_keys=True))
        return 0 if payload.get("status") == "PASS" else 1

    report = run_benchmark(
        demo_root=Path(args.demo_root),
        export_run_label=args.export_run_label,
        run_label=args.run_label,
        dtype_name=args.dtype,
        max_new_tokens=args.max_new_tokens,
        iters=args.iters,
        warmup=args.warmup,
        gpu_memory_utilization=args.gpu_memory_utilization,
        preload_vllm_ascend_patch=args.preload_vllm_ascend_patch,
        preload_vllm_ascend_selector_shim=args.preload_vllm_ascend_selector_shim,
        preload_vllm_ascend_metadata_shim=args.preload_vllm_ascend_metadata_shim,
        max_model_len=args.max_model_len,
        subprocess_per_method=not args.in_process,
        parallel_methods=args.parallel_methods,
        method_cards=parse_method_cards(args.method_cards),
    )
    write_outputs(report, Path(args.demo_root), args.run_label)
    print("VLLM_SERVING_BENCHMARK " + json.dumps(report, sort_keys=True))
    return 0 if report.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
