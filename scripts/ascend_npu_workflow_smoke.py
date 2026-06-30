from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch
from torch import nn

from tidal.device import resolve_device, resolve_dtype
from tidal.methods.global_rank_sparsity.torch import CAPPackedLinear
from tidal.methods.qpruner.torch import QuantizedLinear
from tidal.workflows.adaptation import rankadaptor_adapt
from tidal.workflows.compression import cap_compress, qpruner_compress


WORKFLOWS = ("cap", "qpruner", "rankadaptor")


class TinyWorkflowModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(16, 16, bias=False)
        self.k_proj = nn.Linear(16, 16, bias=False)
        self.v_proj = nn.Linear(16, 16, bias=False)
        self.o_proj = nn.Linear(16, 16, bias=False)
        self.gate_proj = nn.Linear(16, 32, bias=False)
        self.down_proj = nn.Linear(32, 16, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.o_proj(self.q_proj(x) + self.k_proj(x) + self.v_proj(x))
        return self.down_proj(self.gate_proj(attn))


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "npu":
        torch.npu.synchronize()


def timed(label: str, fn, device: torch.device) -> tuple[object, dict[str, object]]:
    synchronize(device)
    start = time.perf_counter()
    try:
        result = fn()
        synchronize(device)
        return result, {"status": "PASS", "seconds": round(time.perf_counter() - start, 4)}
    except Exception as exc:
        synchronize(device)
        return None, {
            "status": "FAIL",
            "seconds": round(time.perf_counter() - start, 4),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "label": label,
        }


def module_device_types(model: nn.Module) -> list[str]:
    devices = {param.device.type for param in model.parameters(recurse=True)}
    devices.update(buffer.device.type for buffer in model.buffers(recurse=True))
    return sorted(devices)


def select_workflows(workflow: str) -> tuple[str, ...]:
    normalized = workflow.strip().lower()
    if normalized == "all":
        return WORKFLOWS
    if normalized in WORKFLOWS:
        return (normalized,)
    raise ValueError(f"unknown workflow: {workflow!r}; expected all, cap, qpruner, or rankadaptor")


def main() -> int:
    parser = argparse.ArgumentParser(description="TIDAL Ascend NPU workflow smoke")
    parser.add_argument("--device", default="npu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--workflow", default="all", help="Workflow to run: all, cap, qpruner, or rankadaptor.")
    parser.add_argument("--cap-rpca-backend", default="numpy", choices=("numpy", "torch"))
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--compat-md", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    torch_dtype = dtype if isinstance(dtype, torch.dtype) else torch.float32

    report: dict[str, object] = {
        "status": "STARTED",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch_version": torch.__version__,
        "device": str(device),
        "dtype": str(torch_dtype),
        "seed": args.seed,
        "requested_workflow": args.workflow,
        "cap_rpca_backend": args.cap_rpca_backend,
    }

    if device.type == "npu":
        import torch_npu  # noqa: F401

        report["torch_npu_import"] = True
        report["npu_available"] = bool(torch.npu.is_available())
        report["npu_count"] = int(torch.npu.device_count())

    def run_cap() -> dict[str, object]:
        model = TinyWorkflowModel()
        result = cap_compress(
            model=model,
            budget=768,
            target_roles=None,
            name_filter=lambda name: name.endswith("_proj"),
            device=device,
            dtype=torch_dtype,
            max_iter=8,
            policy_steps=2,
            samples_per_step=2,
            rpca_backend=args.cap_rpca_backend,
        )
        packed_count = sum(isinstance(m, CAPPackedLinear) for m in result.model.modules())
        x = torch.randn(2, 16, device=device, dtype=torch_dtype)
        y = next(m for m in result.model.modules() if isinstance(m, CAPPackedLinear))(x)
        return {
            "summary": result.summary,
            "packed_count": packed_count,
            "module_devices": module_device_types(result.model),
            "forward_output_device": y.device.type,
        }

    workflows = select_workflows(args.workflow)
    report["workflows"] = list(workflows)

    if "cap" in workflows:
        cap_result, cap_status = timed("cap", run_cap, device)
        report["cap"] = cap_status | (cap_result or {})

    def run_qpruner() -> dict[str, object]:
        model = TinyWorkflowModel()
        importances = {
            "q_proj": 1.0,
            "k_proj": 0.8,
            "v_proj": 0.8,
            "o_proj": 1.0,
            "gate_proj": 0.6,
            "down_proj": 0.7,
        }
        result = qpruner_compress(
            model=model,
            importances=importances,
            candidate_bits=(2, 4, 8),
            max_average_bits=4.0,
            target_roles=None,
            name_filter=lambda name: name.endswith("_proj"),
            device=device,
            dtype=torch_dtype,
        )
        quantized_count = sum(isinstance(m, QuantizedLinear) for m in result.model.modules())
        sample = next(m for m in result.model.modules() if isinstance(m, QuantizedLinear))
        x = torch.randn(2, sample.in_features, device=device, dtype=torch_dtype)
        y = sample(x)
        return {
            "summary": result.summary,
            "quantized_count": quantized_count,
            "module_devices": module_device_types(result.model),
            "forward_output_device": y.device.type,
        }

    if "qpruner" in workflows:
        qpruner_result, qpruner_status = timed("qpruner", run_qpruner, device)
        report["qpruner"] = qpruner_status | (qpruner_result or {})

    def run_rankadaptor() -> dict[str, object]:
        model = TinyWorkflowModel()
        result = rankadaptor_adapt(
            model=model,
            sensitivities={
                "q_proj": 1.0,
                "k_proj": 0.5,
                "v_proj": 0.5,
                "o_proj": 1.0,
                "gate_proj": 0.8,
                "down_proj": 0.9,
            },
            budget=512,
            min_rank=1,
            max_rank=4,
            rank_step=1,
            target_roles=None,
            name_filter=lambda name: name.endswith("_proj"),
            apply_peft=False,
            device=device,
            dtype=torch_dtype,
        )
        return {
            "summary": result.summary,
            "profile_count": len(result.profiles),
            "module_devices": module_device_types(result.model),
            "peft_applied": False,
        }

    if "rankadaptor" in workflows:
        rank_result, rank_status = timed("rankadaptor", run_rankadaptor, device)
        report["rankadaptor"] = rank_status | (rank_result or {})

    workflow_statuses = [report[name]["status"] for name in workflows]
    report["status"] = "PASS" if all(status == "PASS" for status in workflow_statuses) else "FAIL"

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True))

    compat = Path(args.compat_md)
    compat.parent.mkdir(parents=True, exist_ok=True)
    issues = []
    if not report.get("npu_available", device.type != "npu"):
        issues.append("- NPU was not available through `torch.npu.is_available()`.")
    if "rankadaptor" in workflows and report["rankadaptor"]["status"] == "PASS":
        issues.append("- RankAdaptor smoke used `apply_peft=False` because the current container lacks `peft`.")
    for name in workflows:
        section = report[name]
        if section["status"] != "PASS":
            issues.append(f"- {name} failed: `{section.get('error_type')}` {section.get('error')}")
    if not issues:
        issues.append("- No compatibility issues found in the tiny workflow smoke.")

    compat.write_text(
        "# Ascend 910B Tiny Workflow Compatibility Notes\n\n"
        f"- Overall status: `{report['status']}`\n"
        f"- Device: `{report['device']}`\n"
        f"- Dtype: `{report['dtype']}`\n"
        f"- Torch: `{report['torch_version']}`\n"
        f"- NPU count: `{report.get('npu_count', 'unknown')}`\n\n"
        "## Issues\n\n"
        + "\n".join(issues)
        + "\n"
    )

    print("ASCEND_NPU_WORKFLOW_SMOKE " + json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
