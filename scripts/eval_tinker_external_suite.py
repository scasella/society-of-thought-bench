from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

from society_of_thought_bench.external_benchmarks import (
    CORE_EXTERNAL_BENCHMARKS,
    core_benchmark_order,
    ensure_required_env_vars,
    evaluate_external_suite_with_tinker,
)
from society_of_thought_bench.training_data import OUTPUT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name")
    parser.add_argument("--model-path")
    parser.add_argument("--trace-mode", choices=["debate", "monologue"], required=True)
    parser.add_argument("--benchmarks", nargs="*", default=list(CORE_EXTERNAL_BENCHMARKS))
    parser.add_argument("--num-examples", type=int, default=20)
    parser.add_argument("--rollouts-per-example", type=int, default=1)
    parser.add_argument("--max-concurrent", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--save-lock-file", type=Path)
    parser.add_argument("--replay-lock-file", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if bool(args.model_name) == bool(args.model_path):
        raise SystemExit("Provide exactly one of --model-name or --model-path")
    if args.save_lock_file and args.replay_lock_file:
        raise SystemExit("Use either --save-lock-file or --replay-lock-file, not both")

    benchmarks = core_benchmark_order(args.benchmarks)
    resolved = {
        "model_name": args.model_name,
        "model_path": args.model_path,
        "trace_mode": args.trace_mode,
        "benchmarks": benchmarks,
        "num_examples": args.num_examples,
        "rollouts_per_example": args.rollouts_per_example,
        "max_concurrent": args.max_concurrent,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "save_lock_file": str(args.save_lock_file) if args.save_lock_file else None,
        "replay_lock_file": str(args.replay_lock_file) if args.replay_lock_file else None,
        "output_dir": str(args.output_dir or default_output_dir(args.trace_mode, args.model_name or args.model_path or "checkpoint")),
    }
    if args.dry_run:
        print(json.dumps(resolved, indent=2))
        return

    require_env("TINKER_API_KEY")
    ensure_required_env_vars(benchmarks)

    output_dir = Path(resolved["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = asyncio.run(
        evaluate_external_suite_with_tinker(
            model_name=args.model_name,
            model_path=args.model_path,
            benchmarks=benchmarks,
            trace_mode=args.trace_mode,
            num_examples=args.num_examples,
            rollouts_per_example=args.rollouts_per_example,
            max_concurrent=args.max_concurrent,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            save_lock_file=args.save_lock_file,
            replay_lock_file=args.replay_lock_file,
        )
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def default_output_dir(trace_mode: str, model_ref: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = model_ref.replace("/", "--").replace(":", "--")
    return OUTPUT_ROOT / "external_evals" / f"{stamp}-{trace_mode}-{slug}"


def require_env(name: str) -> None:
    if os.environ.get(name):
        return
    raise SystemExit(f"{name} is required for non-dry-run Tinker commands.")


if __name__ == "__main__":
    main()
