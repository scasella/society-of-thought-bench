from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

from society_of_thought_bench.external_benchmarks import (
    CORE_EXTERNAL_BENCHMARKS,
    compare_external_suite_aggregates,
    core_benchmark_order,
    ensure_required_env_vars,
    evaluate_external_acceptance,
    evaluate_external_suite_with_tinker,
)
from society_of_thought_bench.training_data import OUTPUT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--debate-model-name")
    parser.add_argument("--debate-model-path")
    parser.add_argument("--monologue-model-name")
    parser.add_argument("--monologue-model-path")
    parser.add_argument("--base-model-name")
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

    if bool(args.debate_model_name) == bool(args.debate_model_path):
        raise SystemExit("Provide exactly one of --debate-model-name or --debate-model-path")
    if bool(args.monologue_model_name) == bool(args.monologue_model_path) and (args.monologue_model_name or args.monologue_model_path):
        raise SystemExit("Provide exactly one of --monologue-model-name or --monologue-model-path when using a monologue control")
    if args.save_lock_file and args.replay_lock_file:
        raise SystemExit("Use either --save-lock-file or --replay-lock-file, not both")

    benchmarks = core_benchmark_order(args.benchmarks)
    resolved = {
        "debate_model_name": args.debate_model_name,
        "debate_model_path": args.debate_model_path,
        "monologue_model_name": args.monologue_model_name,
        "monologue_model_path": args.monologue_model_path,
        "base_model_name": args.base_model_name,
        "benchmarks": benchmarks,
        "num_examples": args.num_examples,
        "rollouts_per_example": args.rollouts_per_example,
        "max_concurrent": args.max_concurrent,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "save_lock_file": str(args.save_lock_file) if args.save_lock_file else None,
        "replay_lock_file": str(args.replay_lock_file) if args.replay_lock_file else None,
        "output_dir": str(args.output_dir or default_output_dir(args.debate_model_name or args.debate_model_path or "checkpoint")),
    }
    if args.dry_run:
        print(json.dumps(resolved, indent=2))
        return

    require_env("TINKER_API_KEY")
    ensure_required_env_vars(benchmarks)

    output_dir = Path(resolved["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = asyncio.run(run_comparisons(args, benchmarks))
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


async def run_comparisons(args, benchmarks: list[str]) -> dict[str, object]:
    paired_lock_file = args.replay_lock_file or args.save_lock_file
    same_checkpoint_debate = await evaluate_external_suite_with_tinker(
        model_name=args.debate_model_name,
        model_path=args.debate_model_path,
        benchmarks=benchmarks,
        trace_mode="debate",
        num_examples=args.num_examples,
        rollouts_per_example=args.rollouts_per_example,
        max_concurrent=args.max_concurrent,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        save_lock_file=args.save_lock_file if args.replay_lock_file is None else None,
        replay_lock_file=args.replay_lock_file,
    )
    same_checkpoint_monologue = await evaluate_external_suite_with_tinker(
        model_name=args.debate_model_name,
        model_path=args.debate_model_path,
        benchmarks=benchmarks,
        trace_mode="monologue",
        num_examples=args.num_examples,
        rollouts_per_example=args.rollouts_per_example,
        max_concurrent=args.max_concurrent,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        replay_lock_file=paired_lock_file,
    )
    same_checkpoint = compare_external_suite_aggregates(
        same_checkpoint_debate,
        same_checkpoint_monologue,
        left_label="debate",
        right_label="monologue",
    )
    same_checkpoint["acceptance"] = evaluate_external_acceptance(same_checkpoint, threshold_key="same_checkpoint")

    summary: dict[str, object] = {
        "same_checkpoint": same_checkpoint,
        "same_checkpoint_runs": {
            "debate": same_checkpoint_debate,
            "monologue": same_checkpoint_monologue,
        },
    }

    if args.monologue_model_name or args.monologue_model_path:
        control_monologue = await evaluate_external_suite_with_tinker(
            model_name=args.monologue_model_name,
            model_path=args.monologue_model_path,
            benchmarks=benchmarks,
            trace_mode="monologue",
            num_examples=args.num_examples,
            rollouts_per_example=args.rollouts_per_example,
            max_concurrent=args.max_concurrent,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            replay_lock_file=paired_lock_file,
        )
        debate_vs_control = compare_external_suite_aggregates(
            same_checkpoint_debate,
            control_monologue,
            left_label="debate_checkpoint",
            right_label="monologue_control",
        )
        debate_vs_control["acceptance"] = evaluate_external_acceptance(
            debate_vs_control,
            threshold_key="debate_vs_control",
        )
        summary["debate_vs_control"] = debate_vs_control
        summary["monologue_control_run"] = control_monologue

    if args.base_model_name:
        base_debate = await evaluate_external_suite_with_tinker(
            model_name=args.base_model_name,
            model_path=None,
            benchmarks=benchmarks,
            trace_mode="debate",
            num_examples=args.num_examples,
            rollouts_per_example=args.rollouts_per_example,
            max_concurrent=args.max_concurrent,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            replay_lock_file=paired_lock_file,
        )
        base_monologue = await evaluate_external_suite_with_tinker(
            model_name=args.base_model_name,
            model_path=None,
            benchmarks=benchmarks,
            trace_mode="monologue",
            num_examples=args.num_examples,
            rollouts_per_example=args.rollouts_per_example,
            max_concurrent=args.max_concurrent,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            replay_lock_file=paired_lock_file,
        )
        summary["base_calibration"] = {
            "comparison": compare_external_suite_aggregates(
                base_debate,
                base_monologue,
                left_label="debate",
                right_label="monologue",
            ),
            "runs": {"debate": base_debate, "monologue": base_monologue},
        }
    return summary


def default_output_dir(model_ref: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = model_ref.replace("/", "--").replace(":", "--")
    return OUTPUT_ROOT / "external_comparisons" / f"{stamp}-{slug}"


def require_env(name: str) -> None:
    if os.environ.get(name):
        return
    raise SystemExit(f"{name} is required for non-dry-run Tinker commands.")


if __name__ == "__main__":
    main()
