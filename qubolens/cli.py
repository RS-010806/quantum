"""Command-line entry point for repeatable local and CI experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .data import load_tabular_dataset, make_demo
from .pipeline import optimize_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qubolens",
        description="Visual feature selection with a built-in optimizer.",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--file",
        type=Path,
        help="CSV, TSV, TXT, JSON, JSONL, or XLSX data file.",
    )
    source.add_argument(
        "--csv",
        type=Path,
        help="CSV data file (kept as a backward-compatible alias).",
    )
    source.add_argument(
        "--demo",
        choices=("edge-failure", "cloud-cost"),
        default="edge-failure",
        help="Built-in seeded dataset (default: edge-failure).",
    )
    parser.add_argument("--target", help="Target column; required with a data file.")
    parser.add_argument(
        "--task",
        choices=("auto", "classification", "regression"),
        default="auto",
    )
    parser.add_argument("-k", "--features", type=int, default=6)
    parser.add_argument(
        "--redundancy", type=float, default=0.65, help="Pairwise penalty weight."
    )
    parser.add_argument(
        "--quality", choices=("fast", "balanced", "deep"), default="balanced"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-o", "--output", type=Path, help="Write full JSON result.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        upload_path = args.file or args.csv
        if upload_path:
            if not args.target:
                raise ValueError("--target is required when a file is used.")
            dataset = load_tabular_dataset(
                upload_path.read_bytes(),
                filename=upload_path.name,
                target_name=args.target,
                task=args.task,
                name=upload_path.stem,
            )
        else:
            dataset = make_demo(args.demo)
        result = optimize_dataset(
            dataset,
            k=args.features,
            redundancy_weight=args.redundancy,
            quality=args.quality,
            seed=args.seed,
        )
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        selection = result["selection"]
        benchmark = result["benchmark"]["qubo"]
        print(f"Dataset: {result['dataset']['name']}")
        print(f"Selected ({selection['k']}): {', '.join(selection['names'])}")
        print(
            f"{benchmark['score_label']}: {benchmark['score']:.4f} · "
            f"feature reduction: {selection['feature_reduction']:.1f}%"
        )
        print(result["insight"]["finding"])
    return 0
