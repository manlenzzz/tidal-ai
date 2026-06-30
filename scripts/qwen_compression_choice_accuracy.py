#!/usr/bin/env python
"""Small fixed-choice accuracy table for Qwen compression demos."""
from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from io import StringIO
from pathlib import Path
from typing import Any

import torch
from torch import nn

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from qwen_compression_generate_benchmark import limited_target_filter, select_target_names
from tiny_qwen_compression_benchmark import load_model, load_tokenizer, targeted_parameter_count
from tidal.device import resolve_device, resolve_dtype
from tidal.workflows.compression import cap_compress, qpruner_compress


DEFAULT_SAMPLES = [
    {
        "id": "arithmetic_2_plus_3",
        "question": "Question: What is 2 + 3?\nA. 4\nB. 5\nC. 6\nD. 7\nAnswer:",
        "choices": {"A": " A", "B": " B", "C": " C", "D": " D"},
        "answer": "B",
    },
    {
        "id": "capital_france",
        "question": "Question: Which city is the capital of France?\nA. Berlin\nB. Madrid\nC. Paris\nD. Rome\nAnswer:",
        "choices": {"A": " A", "B": " B", "C": " C", "D": " D"},
        "answer": "C",
    },
    {
        "id": "compression_memory",
        "question": "Question: In model compression, which metric directly reflects memory footprint?\nA. Parameter storage\nB. Prompt wording\nC. Dataset name\nD. File path\nAnswer:",
        "choices": {"A": " A", "B": " B", "C": " C", "D": " D"},
        "answer": "A",
    },
    {
        "id": "qpruner_quantization",
        "question": "Question: QPruner mainly reduces selected linear-layer weight storage by using what?\nA. More training epochs\nB. Lower-bit quantized weights\nC. Longer prompts\nD. Larger hidden states\nAnswer:",
        "choices": {"A": " A", "B": " B", "C": " C", "D": " D"},
        "answer": "B",
    },
]
METHOD_ORDER = ("baseline", "cap", "qpruner")


def parse_methods(value: str) -> tuple[str, ...]:
    methods = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    if not methods:
        raise argparse.ArgumentTypeError("at least one method must be selected")
    unknown = [method for method in methods if method not in METHOD_ORDER]
    if unknown:
        raise argparse.ArgumentTypeError(
            "unknown method(s): {unknown}; expected one or more of {expected}".format(
                unknown=", ".join(unknown),
                expected=", ".join(METHOD_ORDER),
            )
        )
    if len(set(methods)) != len(methods):
        raise argparse.ArgumentTypeError("methods must not contain duplicates")
    return methods


def ordered_report_methods(report: dict[str, Any]) -> list[str]:
    methods = report.get("methods", {})
    if not isinstance(methods, dict):
        return []
    ordered = [method for method in METHOD_ORDER if method in methods]
    ordered.extend(method for method in methods if method not in METHOD_ORDER)
    return ordered


def _normalize_sample(raw: dict[str, Any], *, index: int) -> dict[str, Any]:
    required = ("question", "choices", "answer")
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"sample {index} missing required fields: {', '.join(missing)}")
    choices = raw["choices"]
    if not isinstance(choices, dict) or not choices:
        raise ValueError(f"sample {index} choices must be a non-empty object")
    answer = str(raw["answer"])
    if answer not in choices:
        raise ValueError(f"sample {index} answer {answer!r} is not one of the choices")
    return {
        "id": str(raw.get("id") or f"sample_{index}"),
        "task": str(raw.get("task") or "default"),
        "question": str(raw["question"]),
        "choices": {str(label): str(value) for label, value in choices.items()},
        "answer": answer,
    }


def load_samples(path: Path | None, *, sample_limit: int) -> list[dict[str, Any]]:
    if path is None:
        return [_normalize_sample(sample, index=index) for index, sample in enumerate(DEFAULT_SAMPLES[: max(1, sample_limit)])]
    samples: list[dict[str, Any]] = []
    for line_no, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no} is not valid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_no} must be a JSON object")
        samples.append(_normalize_sample(payload, index=len(samples)))
        if len(samples) >= max(1, sample_limit):
            break
    if not samples:
        raise ValueError(f"{path} did not contain any samples")
    return samples


def fmt_pct(value: Any) -> str:
    if value is None:
        return "missing"
    return f"{float(value) * 100.0:.3f}%"


def encode_text(tokenizer: Any, text: str, *, device: torch.device) -> torch.Tensor:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        eos = getattr(tokenizer, "eos_token_id", None)
        pad = getattr(tokenizer, "pad_token_id", None)
        token_ids = [eos if eos is not None else pad if pad is not None else 0]
    return torch.tensor(token_ids, dtype=torch.long, device=device)


def continuation_score(
    model: nn.Module,
    tokenizer: Any,
    *,
    prompt: str,
    continuation: str,
    device: torch.device,
    max_length: int,
) -> float:
    return continuation_scores(
        model,
        tokenizer,
        [(prompt, continuation)],
        device=device,
        max_length=max_length,
    )[0]


def continuation_scores(
    model: nn.Module,
    tokenizer: Any,
    prompt_continuations: list[tuple[str, str]],
    *,
    device: torch.device,
    max_length: int,
) -> list[float]:
    encoded: list[tuple[torch.Tensor, int]] = []
    for prompt, continuation in prompt_continuations:
        prompt_ids = encode_text(tokenizer, prompt, device=device)
        continuation_ids = encode_text(tokenizer, continuation, device=device)
        input_ids = torch.cat([prompt_ids, continuation_ids], dim=0)
        if input_ids.numel() > max_length:
            input_ids = input_ids[-max_length:]
            prompt_length = max(1, input_ids.numel() - continuation_ids.numel())
        else:
            prompt_length = int(prompt_ids.numel())
        encoded.append((input_ids, prompt_length))

    if not encoded:
        return []

    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is None:
        pad_token_id = 0

    max_seq_len = max(int(input_ids.numel()) for input_ids, _ in encoded)
    batch = torch.full((len(encoded), max_seq_len), int(pad_token_id), dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(batch)
    lengths: list[int] = []
    for row, (input_ids, _) in enumerate(encoded):
        length = int(input_ids.numel())
        lengths.append(length)
        batch[row, :length] = input_ids
        attention_mask[row, :length] = 1

    with torch.no_grad():
        output = model(input_ids=batch, attention_mask=attention_mask)
        logits = output.logits
        log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
        target_ids = batch[:, 1:]
        token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

    scores: list[float] = []
    for row, ((_, prompt_length), length) in enumerate(zip(encoded, lengths)):
        if length < 2:
            scores.append(float("-inf"))
            continue
        start = max(0, prompt_length - 1)
        end = length - 1
        selected = token_log_probs[row, start:end]
        if selected.numel() == 0:
            scores.append(float("-inf"))
        else:
            scores.append(float(selected.sum().detach().cpu()))
    return scores


def evaluate_choice_accuracy(
    model: nn.Module,
    tokenizer: Any,
    samples: list[dict[str, Any]],
    *,
    device: torch.device,
    max_length: int,
) -> dict[str, Any]:
    prompts: list[tuple[str, str]] = []
    score_entries: list[tuple[int, str]] = []
    for sample_index, sample in enumerate(samples):
        for label, choice in sample["choices"].items():
            prompts.append((str(sample["question"]), str(choice)))
            score_entries.append((sample_index, label))

    flat_scores = continuation_scores(model, tokenizer, prompts, device=device, max_length=max_length)
    scores_by_sample: list[dict[str, float]] = [{} for _ in samples]
    for (sample_index, label), score in zip(score_entries, flat_scores):
        scores_by_sample[sample_index][label] = score

    items: list[dict[str, Any]] = []
    for sample, scores in zip(samples, scores_by_sample):
        predicted = max(scores, key=scores.get)
        answer = str(sample["answer"])
        task = str(sample.get("task") or "default")
        items.append(
            {
                "id": sample.get("id", "unknown"),
                "task": task,
                "answer": answer,
                "predicted": predicted,
                "is_correct": predicted == answer,
                "choice_scores": {key: round(value, 6) for key, value in scores.items()},
            }
        )
    correct = sum(1 for item in items if item["is_correct"])
    total = len(items)
    task_accuracy: dict[str, dict[str, Any]] = {}
    for task in sorted({str(item.get("task") or "default") for item in items}):
        task_items = [item for item in items if str(item.get("task") or "default") == task]
        task_correct = sum(1 for item in task_items if item["is_correct"])
        task_total = len(task_items)
        task_accuracy[task] = {
            "status": "PASS",
            "accuracy": round(task_correct / task_total, 6) if task_total else None,
            "correct": task_correct,
            "total": task_total,
        }
    return {
        "status": "PASS",
        "accuracy": round(correct / total, 6) if total else None,
        "correct": correct,
        "total": total,
        "task_accuracy": task_accuracy,
        "items": items,
    }


def method_section(
    *,
    model: nn.Module,
    tokenizer: Any,
    samples: list[dict[str, Any]],
    device: torch.device,
    max_length: int,
    compression_time_s: float | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = evaluate_choice_accuracy(model, tokenizer, samples, device=device, max_length=max_length)
    if compression_time_s is not None:
        result["compression_time_s"] = round(compression_time_s, 6)
    if extra:
        result.update(extra)
    return result


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    torch_dtype = dtype if isinstance(dtype, torch.dtype) else torch.float32
    model_path = Path(args.model_path)
    tokenizer = load_tokenizer(model_path)
    samples = load_samples(args.samples_jsonl, sample_limit=args.sample_limit)

    baseline_model = load_model(model_path, device, torch_dtype)
    target_names, targeted_layers_total = select_target_names(
        baseline_model,
        args.target_layer_limit,
        args.target_layer_pattern,
    )
    if not target_names:
        raise ValueError("model does not contain matching Qwen projection layers")
    targeted_params = targeted_parameter_count(baseline_model, target_names)
    methods: dict[str, Any] = {}
    if "baseline" in args.methods:
        methods["baseline"] = method_section(
            model=baseline_model,
            tokenizer=tokenizer,
            samples=samples,
            device=device,
            max_length=args.max_length,
            extra={"targeted_layers": len(target_names), "targeted_params": targeted_params},
        )

    if "cap" in args.methods:
        start = time.perf_counter()
        cap_result = cap_compress(
            model=load_model(model_path, device, torch_dtype),
            model_id=args.model_id,
            budget=args.cap_budget,
            target_roles=None,
            name_filter=limited_target_filter(set(target_names)),
            device=device,
            dtype=torch_dtype,
            max_iter=args.cap_max_iter,
            policy_steps=args.cap_policy_steps,
            samples_per_step=args.cap_samples_per_step,
            seed=args.seed,
            inplace=bool(args.inplace_compression),
            rpca_backend=args.cap_rpca_backend,
        )
        cap_time = time.perf_counter() - start
        methods["cap"] = method_section(
            model=cap_result.model,
            tokenizer=tokenizer,
            samples=samples,
            device=device,
            max_length=args.max_length,
            compression_time_s=cap_time,
            extra={
                "targeted_layers": len(target_names),
                "targeted_compression_ratio": round(
                    targeted_params / max(1, targeted_parameter_count(cap_result.model, target_names)),
                    3,
                ),
            },
        )

    if "qpruner" in args.methods:
        start = time.perf_counter()
        qpruner_result = qpruner_compress(
            model=load_model(model_path, device, torch_dtype),
            model_id=args.model_id,
            importances={name: 1.0 for name in target_names},
            candidate_bits=(2, 4, 8),
            max_average_bits=args.qpruner_average_bits,
            target_roles=None,
            name_filter=limited_target_filter(set(target_names)),
            device=device,
            dtype=torch_dtype,
            seed=args.seed,
            inplace=bool(args.inplace_compression),
        )
        qpruner_time = time.perf_counter() - start
        methods["qpruner"] = method_section(
            model=qpruner_result.model,
            tokenizer=tokenizer,
            samples=samples,
            device=device,
            max_length=args.max_length,
            compression_time_s=qpruner_time,
            extra={
                "targeted_layers": len(target_names),
                "average_bits": qpruner_result.summary.get("average_bits"),
                "memory_bits": qpruner_result.summary.get("memory_bits"),
            },
        )

    return {
        "status": "PASS",
        "backend": "choice_log_likelihood",
        "model_id": args.model_id,
        "model_path": str(model_path),
        "device": str(device),
        "dtype": str(torch_dtype),
        "target_layer_limit": args.target_layer_limit,
        "targeted_layers_total": targeted_layers_total,
        "target_layer_pattern": args.target_layer_pattern,
        "max_length": args.max_length,
        "sample_source": str(args.samples_jsonl) if args.samples_jsonl else "default_builtin",
        "task_count": len({str(sample.get("task") or "default") for sample in samples}),
        "samples": [
            {"id": sample["id"], "task": sample.get("task", "default"), "answer": sample["answer"]}
            for sample in samples
        ],
        "methods": methods,
        "platform": platform.platform(),
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Qwen Compression Choice Accuracy",
        "",
        "This fixed prompt-choice table is a lightweight external accuracy check for the demo. It scores each answer option by continuation log-likelihood.",
        "",
        "| Method | Status | Accuracy | Correct |",
        "|---|---:|---:|---:|",
    ]
    for method in ordered_report_methods(report):
        payload = report.get("methods", {}).get(method, {})
        lines.append(
            "| {method} | {status} | {accuracy} | {correct} / {total} |".format(
                method=method,
                status=payload.get("status", "missing"),
                accuracy=fmt_pct(payload.get("accuracy")),
                correct=payload.get("correct", "missing"),
                total=payload.get("total", "missing"),
            )
        )
    lines.extend(
        [
            "",
            "## Metadata",
            "",
            f"- Model: `{report.get('model_id')}`",
            f"- Device: `{report.get('device')}`",
            f"- Target layer limit: `{report.get('target_layer_limit')} / {report.get('targeted_layers_total')}`",
            f"- Sample source: `{report.get('sample_source', 'default_builtin')}`",
            f"- Task count: `{report.get('task_count', 'missing')}`",
            f"- Backend: `{report.get('backend')}`",
            "",
            "## Task Accuracy",
            "",
            "| Method | Task | Accuracy | Correct |",
            "|---|---|---:|---:|",
        ]
    )
    for method in ordered_report_methods(report):
        task_accuracy = report.get("methods", {}).get(method, {}).get("task_accuracy", {})
        if not isinstance(task_accuracy, dict):
            continue
        for task, payload in sorted(task_accuracy.items()):
            lines.append(
                "| {method} | {task} | {accuracy} | {correct} / {total} |".format(
                    method=method,
                    task=task,
                    accuracy=fmt_pct(payload.get("accuracy")),
                    correct=payload.get("correct", "missing"),
                    total=payload.get("total", "missing"),
                )
            )
    lines.extend(
        [
            "",
            "## Per-Sample Predictions",
            "",
            "| Method | Task | Sample | Answer | Prediction | Correct |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for method in ordered_report_methods(report):
        for item in report.get("methods", {}).get(method, {}).get("items", []):
            lines.append(
                "| {method} | {task} | {sample} | {answer} | {predicted} | {correct} |".format(
                    method=method,
                    task=item.get("task", "default"),
                    sample=item.get("id", "unknown"),
                    answer=item.get("answer", "missing"),
                    predicted=item.get("predicted", "missing"),
                    correct=item.get("is_correct", "missing"),
                )
            )
    return "\n".join(lines) + "\n"


def csv_report(report: dict[str, Any]) -> str:
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["method", "task", "status", "accuracy", "correct", "total"])
    for method in ordered_report_methods(report):
        payload = report.get("methods", {}).get(method, {})
        writer.writerow(
            [
                method,
                "__overall__",
                payload.get("status"),
                payload.get("accuracy"),
                payload.get("correct"),
                payload.get("total"),
            ]
        )
        task_accuracy = payload.get("task_accuracy", {})
        if not isinstance(task_accuracy, dict):
            continue
        for task, task_payload in sorted(task_accuracy.items()):
            writer.writerow(
                [
                    method,
                    task,
                    task_payload.get("status"),
                    task_payload.get("accuracy"),
                    task_payload.get("correct"),
                    task_payload.get("total"),
                ]
            )
    return output.getvalue()


def write_outputs(report: dict[str, Any], demo_root: Path, run_label: str) -> None:
    artifacts = demo_root / "artifacts"
    reports = demo_root / "reports"
    artifacts.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    (artifacts / f"qwen_compression_choice_accuracy_{run_label}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )
    (reports / f"qwen-compression-choice-accuracy-{run_label}.md").write_text(markdown_report(report))
    (reports / f"qwen-compression-choice-accuracy-{run_label}.csv").write_text(csv_report(report))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen compression choice accuracy")
    parser.add_argument("--demo-root", default="/mnt/nvme/622/tidal-demo")
    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--model-path", default="/mnt/nvme/622/models/Qwen3-0.6B")
    parser.add_argument("--device", default="npu")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--target-layer-limit", type=int, default=64)
    parser.add_argument("--target-layer-pattern", default="self_attn\\.(q_proj|k_proj)$")
    parser.add_argument("--cap-budget", type=int, default=1048576)
    parser.add_argument("--cap-max-iter", type=int, default=1)
    parser.add_argument("--cap-policy-steps", type=int, default=1)
    parser.add_argument("--cap-samples-per-step", type=int, default=1)
    parser.add_argument("--cap-rpca-backend", default="numpy")
    parser.add_argument("--qpruner-average-bits", type=float, default=8.0)
    parser.add_argument("--inplace-compression", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--sample-limit", type=int, default=len(DEFAULT_SAMPLES))
    parser.add_argument("--samples-jsonl", type=Path)
    parser.add_argument("--methods", type=parse_methods, default=METHOD_ORDER)
    parser.add_argument("--run-label", default="qwen3_06b_choice_accuracy_npu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = run_benchmark(args)
    except Exception as exc:
        report = {
            "status": "FAIL",
            "backend": "choice_log_likelihood",
            "model_id": args.model_id,
            "model_path": args.model_path,
            "device": args.device,
            "target_layer_limit": args.target_layer_limit,
            "error_type": type(exc).__name__,
            "error": str(exc)[:2000],
            "platform": platform.platform(),
        }
    write_outputs(report, Path(args.demo_root), args.run_label)
    print("QWEN_COMPRESSION_CHOICE_ACCURACY " + json.dumps(report, sort_keys=True))
    return 0 if report.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
