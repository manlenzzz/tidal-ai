#!/usr/bin/env python
"""Render a dependency-free SVG frontier chart from QPruner quality-memory CSV."""
from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path
from typing import Any


WIDTH = 1120
HEIGHT = 720
LEFT = 92
RIGHT = 300
TOP = 72
BOTTOM = 96
COLORS = {
    4: "#1f77b4",
    8: "#2ca02c",
    16: "#d62728",
}


def _float(value: Any) -> float:
    return float(value)


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in csv.DictReader(path.read_text().splitlines()):
        if row.get("status") != "PASS":
            continue
        rows.append(
            {
                "label": row["label"],
                "status": row["status"],
                "target_layer_limit": int(row["target_layer_limit"]),
                "targeted_layers_total": int(row["targeted_layers_total"]),
                "requested_bits": _float(row["requested_bits"]),
                "actual_bits": _float(row["actual_bits"]),
                "target_memory_reduction_pct": _float(row["target_memory_reduction_pct"]),
                "loss_delta": _float(row["loss_delta"]),
                "tokens_per_s": _float(row["tokens_per_s"]),
                "bitwidth_mix": row.get("bitwidth_mix", ""),
            }
        )
    return rows


def best_memory_quality_point(rows: list[dict[str, Any]], *, max_loss_delta: float) -> dict[str, Any] | None:
    candidates = [
        row
        for row in rows
        if row["loss_delta"] <= max_loss_delta and row["target_memory_reduction_pct"] is not None
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda row: (row["target_memory_reduction_pct"], -row["loss_delta"]))


def _domain(values: list[float], *, pad_ratio: float, floor: float | None = None, ceil: float | None = None) -> tuple[float, float]:
    lo = min(values)
    hi = max(values)
    if lo == hi:
        lo -= 1.0
        hi += 1.0
    pad = (hi - lo) * pad_ratio
    lo -= pad
    hi += pad
    if floor is not None:
        lo = min(lo, floor)
    if ceil is not None:
        hi = max(hi, ceil)
    return lo, hi


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def _point_radius(row: dict[str, Any]) -> float:
    return 5.0 + min(8.0, max(0.0, row["actual_bits"] - 2.0) * 0.9)


def render_svg(
    rows: list[dict[str, Any]],
    *,
    best: dict[str, Any] | None,
    max_loss_delta: float,
) -> str:
    if not rows:
        raise ValueError("no passing rows to plot")
    plot_w = WIDTH - LEFT - RIGHT
    plot_h = HEIGHT - TOP - BOTTOM
    x_values = [row["target_memory_reduction_pct"] for row in rows]
    y_values = [row["loss_delta"] for row in rows] + [max_loss_delta, 0.0]
    x_min, x_max = _domain(x_values, pad_ratio=0.08)
    y_min, y_max = _domain(y_values, pad_ratio=0.12, floor=0.0, ceil=max_loss_delta)

    def x(value: float) -> float:
        return LEFT + ((value - x_min) / (x_max - x_min)) * plot_w

    def y(value: float) -> float:
        return TOP + (1.0 - ((value - y_min) / (y_max - y_min))) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-label="QPruner quality-memory frontier">',
        "<style>",
        "text{font-family:Arial,Helvetica,sans-serif;fill:#111827} .muted{fill:#6b7280} .grid{stroke:#e5e7eb;stroke-width:1} .axis{stroke:#111827;stroke-width:1.5} .threshold{stroke:#ef4444;stroke-width:2;stroke-dasharray:8 6} .frontier{stroke:#111827;stroke-width:2.5;fill:none} .point{stroke:#ffffff;stroke-width:2} .best{stroke:#111827;stroke-width:4}",
        "</style>",
        '<rect x="0" y="0" width="1120" height="720" fill="#ffffff"/>',
        f"<!-- loss delta <= {_fmt(max_loss_delta)} -->",
        '<text x="92" y="42" font-size="28" font-weight="700">QPruner quality-memory frontier</text>',
        '<text x="92" y="66" font-size="14" class="muted">Ascend 910B Qwen3-0.6B sweep: memory reduction vs held-out loss delta</text>',
    ]

    for tick in range(6):
        ratio = tick / 5
        tx = LEFT + ratio * plot_w
        value = x_min + ratio * (x_max - x_min)
        parts.extend(
            [
                f'<line class="grid" x1="{tx:.1f}" y1="{TOP}" x2="{tx:.1f}" y2="{TOP + plot_h}"/>',
                f'<text x="{tx:.1f}" y="{TOP + plot_h + 28}" font-size="12" text-anchor="middle" class="muted">{value:.1f}%</text>',
            ]
        )
    for tick in range(6):
        ratio = tick / 5
        ty = TOP + ratio * plot_h
        value = y_max - ratio * (y_max - y_min)
        parts.extend(
            [
                f'<line class="grid" x1="{LEFT}" y1="{ty:.1f}" x2="{LEFT + plot_w}" y2="{ty:.1f}"/>',
                f'<text x="{LEFT - 14}" y="{ty + 4:.1f}" font-size="12" text-anchor="end" class="muted">{value:.2f}</text>',
            ]
        )

    threshold_y = y(max_loss_delta)
    parts.extend(
        [
            f'<line class="threshold" x1="{LEFT}" y1="{threshold_y:.1f}" x2="{LEFT + plot_w}" y2="{threshold_y:.1f}"/>',
            f'<text x="{LEFT + plot_w - 4}" y="{threshold_y - 8:.1f}" font-size="13" text-anchor="end" fill="#b91c1c">loss delta &lt;= {_fmt(max_loss_delta)}</text>',
            f'<line class="axis" x1="{LEFT}" y1="{TOP + plot_h}" x2="{LEFT + plot_w}" y2="{TOP + plot_h}"/>',
            f'<line class="axis" x1="{LEFT}" y1="{TOP}" x2="{LEFT}" y2="{TOP + plot_h}"/>',
            f'<text x="{LEFT + plot_w / 2:.1f}" y="{HEIGHT - 36}" font-size="16" text-anchor="middle">Target memory reduction</text>',
            f'<text x="28" y="{TOP + plot_h / 2:.1f}" font-size="16" text-anchor="middle" transform="rotate(-90 28 {TOP + plot_h / 2:.1f})">Loss delta</text>',
        ]
    )

    for layer_limit in sorted({row["target_layer_limit"] for row in rows}):
        layer_rows = sorted(
            [row for row in rows if row["target_layer_limit"] == layer_limit],
            key=lambda row: row["target_memory_reduction_pct"],
        )
        points = " ".join(
            f'{x(row["target_memory_reduction_pct"]):.1f},{y(row["loss_delta"]):.1f}' for row in layer_rows
        )
        parts.append(
            f'<polyline class="frontier" points="{points}" stroke="{COLORS.get(layer_limit, "#6b7280")}" opacity="0.55"/>'
        )

    for row in rows:
        color = COLORS.get(row["target_layer_limit"], "#6b7280")
        cx = x(row["target_memory_reduction_pct"])
        cy = y(row["loss_delta"])
        classes = "point"
        if best is not None and row["label"] == best["label"]:
            classes += " best"
        title = (
            f"{row['label']}: target layers {row['target_layer_limit']}, "
            f"{row['target_memory_reduction_pct']:.3f}% memory reduction, "
            f"loss delta {row['loss_delta']:.3f}, actual bits {row['actual_bits']:.3f}"
        )
        parts.extend(
            [
                f'<circle class="{classes}" data-label="{html.escape(row["label"])}" cx="{cx:.1f}" cy="{cy:.1f}" r="{_point_radius(row):.1f}" fill="{color}">',
                f"<title>{html.escape(title)}</title>",
                "</circle>",
            ]
        )

    legend_x = WIDTH - RIGHT + 42
    legend_y = TOP + 12
    parts.extend(
        [
            f'<rect x="{legend_x - 18}" y="{legend_y - 28}" width="238" height="260" fill="#f9fafb" stroke="#e5e7eb"/>',
            f'<text x="{legend_x}" y="{legend_y}" font-size="16" font-weight="700">Legend</text>',
        ]
    )
    for index, layer_limit in enumerate(sorted({row["target_layer_limit"] for row in rows})):
        y0 = legend_y + 34 + index * 28
        color = COLORS.get(layer_limit, "#6b7280")
        parts.extend(
            [
                f'<circle cx="{legend_x + 7}" cy="{y0 - 5}" r="7" fill="{color}" stroke="#ffffff" stroke-width="2"/>',
                f'<text x="{legend_x + 24}" y="{y0}" font-size="14">target layers {layer_limit}</text>',
            ]
        )
    parts.extend(
        [
            f'<line x1="{legend_x}" y1="{legend_y + 126}" x2="{legend_x + 58}" y2="{legend_y + 126}" class="threshold"/>',
            f'<text x="{legend_x + 70}" y="{legend_y + 131}" font-size="14">loss delta &lt;= {_fmt(max_loss_delta)}</text>',
        ]
    )
    if best is not None:
        parts.extend(
            [
                f'<text x="{legend_x}" y="{legend_y + 178}" font-size="15" font-weight="700">Best: {html.escape(best["label"])}</text>',
                f'<text x="{legend_x}" y="{legend_y + 202}" font-size="13" class="muted">{best["target_memory_reduction_pct"]:.3f}% memory reduction</text>',
                f'<text x="{legend_x}" y="{legend_y + 222}" font-size="13" class="muted">loss delta {best["loss_delta"]:.3f}, actual bits {best["actual_bits"]:.3f}</text>',
            ]
        )

    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def write_markdown(path: Path, image_path: Path, *, best: dict[str, Any] | None) -> None:
    rel = image_path.name
    lines = [
        "# Qwen QPruner Quality-Memory Frontier Chart",
        "",
        f"![QPruner quality-memory frontier]({rel})",
        "",
    ]
    if best is not None:
        lines.append(
            "Best point: `{label}` with {memory:.3f}% target memory reduction, loss delta {delta:.3f}, actual bits {bits:.3f}.".format(
                label=best["label"],
                memory=best["target_memory_reduction_pct"],
                delta=best["loss_delta"],
                bits=best["actual_bits"],
            )
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Render QPruner quality-memory frontier SVG")
    parser.add_argument("--csv", type=Path, default=Path("/mnt/nvme/622/tidal-demo/reports/qwen-qpruner-quality-memory-sweep.csv"))
    parser.add_argument("--output", type=Path, default=Path("/mnt/nvme/622/tidal-demo/reports/qwen-qpruner-quality-memory-frontier.svg"))
    parser.add_argument("--markdown", type=Path, default=Path("/mnt/nvme/622/tidal-demo/reports/qwen-qpruner-quality-memory-frontier.md"))
    parser.add_argument("--max-loss-delta", type=float, default=0.5)
    args = parser.parse_args()

    rows = load_rows(args.csv)
    best = best_memory_quality_point(rows, max_loss_delta=args.max_loss_delta)
    svg = render_svg(rows, best=best, max_loss_delta=args.max_loss_delta)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(args.markdown, args.output, best=best)
    print(f"QPRUNER_FRONTIER_CHART {args.output} {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
