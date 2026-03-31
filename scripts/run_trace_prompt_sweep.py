from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from society_of_thought_bench.diagnostics import DEFAULT_OUTPUTS_DIR, analyze_results

MODELS = (
    "qwen/qwen3-30b-a3b-thinking-2507",
    "qwen/qwen3-235b-a22b-thinking-2507",
)

RUN_MATRIX = (
    {"trace_prompt_variant": "official"},
    {"trace_prompt_variant": "trace_minimal"},
    {"trace_prompt_variant": "trace_tail_block"},
    {"trace_prompt_variant": "debug_minimal"},
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-examples", type=int, default=8)
    parser.add_argument("--rollouts", type=int, default=1)
    parser.add_argument("--family", default="countdown")
    parser.add_argument("--difficulty", default="easy")
    parser.add_argument("--trace-mode", default="debate")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    workspace_root = Path(__file__).resolve().parents[3]
    sweep_root = (
        Path(__file__).resolve().parents[1]
        / "outputs"
        / "trace_prompt_sweeps"
        / datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    sweep_root.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, Any]] = []
    for model in MODELS:
        for variant in RUN_MATRIX:
            runs.append(
                run_one(
                    workspace_root=workspace_root,
                    sweep_root=sweep_root,
                    model=model,
                    num_examples=args.num_examples,
                    rollouts=args.rollouts,
                    family=args.family,
                    difficulty=args.difficulty,
                    trace_mode=args.trace_mode,
                    seed=args.seed,
                    trace_prompt_variant=variant["trace_prompt_variant"],
                    dry_run=args.dry_run,
                )
            )

    ranked_runs = sorted(runs, key=_rank_key)
    summary = {
        "sweep_root": str(sweep_root),
        "num_examples": args.num_examples,
        "rollouts": args.rollouts,
        "family": args.family,
        "difficulty": args.difficulty,
        "trace_mode": args.trace_mode,
        "seed": args.seed,
        "models": list(MODELS),
        "matrix": list(RUN_MATRIX),
        "runs": runs,
        "ranked_runs": [
            {
                "model": run["model"],
                "trace_prompt_variant": run["trace_prompt_variant"],
                "joint_contract_valid_rate": run.get("joint_contract_valid_rate", 0.0),
                "task_and_joint_valid_rate": run.get("task_and_joint_valid_rate", 0.0),
                "trace_format_valid_rate": run.get("trace_format_valid_rate", 0.0),
                "answer_format_valid_rate": run.get("answer_format_valid_rate", 0.0),
                "near_miss_rate": run.get("near_miss_rate", 0.0),
                "results_path": run.get("results_path"),
            }
            for run in ranked_runs
        ],
        "decision": _decide(runs),
    }
    (sweep_root / "summary.json").write_text(json.dumps(summary, indent=2))
    (sweep_root / "report.md").write_text(render_report(summary))
    print(json.dumps(summary, indent=2))


def run_one(
    *,
    workspace_root: Path,
    sweep_root: Path,
    model: str,
    num_examples: int,
    rollouts: int,
    family: str,
    difficulty: str,
    trace_mode: str,
    seed: int,
    trace_prompt_variant: str,
    dry_run: bool,
) -> dict[str, Any]:
    run_label = f"{_slugify_model(model)}--{trace_prompt_variant}"
    run_dir = sweep_root / run_label
    run_dir.mkdir(parents=True, exist_ok=True)

    env_args = {
        "family": family,
        "difficulty": difficulty,
        "trace_mode": trace_mode,
        "seed": seed,
        "trace_prompt_variant": trace_prompt_variant,
    }
    command = [
        "prime",
        "eval",
        "run",
        "society-of-thought-bench",
        "-m",
        model,
        "-n",
        str(num_examples),
        "-r",
        str(rollouts),
        "-a",
        json.dumps(env_args, separators=(",", ":")),
    ]
    before_paths = set(DEFAULT_OUTPUTS_DIR.glob("**/results.jsonl"))

    stdout = ""
    stderr = ""
    returncode = 0
    if not dry_run:
        completed = subprocess.run(
            command,
            cwd=workspace_root,
            text=True,
            capture_output=True,
            check=False,
        )
        stdout = completed.stdout
        stderr = completed.stderr
        returncode = completed.returncode

    (run_dir / "command.txt").write_text(shlex.join(command) + "\n")
    (run_dir / "stdout.txt").write_text(stdout)
    (run_dir / "stderr.txt").write_text(stderr)

    after_paths = set(DEFAULT_OUTPUTS_DIR.glob("**/results.jsonl"))
    results_path = _detect_results_path(before_paths, after_paths)
    run: dict[str, Any] = {
        "model": model,
        "trace_prompt_variant": trace_prompt_variant,
        "command": command,
        "command_shell": shlex.join(command),
        "returncode": returncode,
        "results_path": str(results_path) if results_path else None,
        "stdout_path": str(run_dir / "stdout.txt"),
        "stderr_path": str(run_dir / "stderr.txt"),
        "command_path": str(run_dir / "command.txt"),
    }
    if results_path and results_path.exists():
        analysis = analyze_results(results_path, only_invalid=False, limit=num_examples)
        run.update({
            "examples_total": analysis["examples_total"],
            "invalid_examples": analysis["invalid_examples"],
            "joint_contract_valid_rate": analysis["joint_contract_valid_rate"],
            "task_and_joint_valid_rate": analysis["task_and_joint_valid_rate"],
            "near_miss_rate": analysis["near_miss_rate"],
            "trace_format_valid_rate": analysis["trace_format_valid_rate"],
            "answer_format_valid_rate": analysis["answer_format_valid_rate"],
            "protocol_valid_rate": analysis["protocol_valid_rate"],
            "average_task_score": analysis["average_task_score"],
            "average_reward": analysis["average_reward"],
            "primary_error_counts": analysis["primary_error_counts"],
        })
    return run


def _detect_results_path(before_paths: set[Path], after_paths: set[Path]) -> Path | None:
    new_paths = list(after_paths - before_paths)
    if not new_paths:
        return None
    return max(new_paths, key=lambda path: path.stat().st_mtime)


def _rank_key(run: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -float(run.get("joint_contract_valid_rate", 0.0) or 0.0),
        -float(run.get("task_and_joint_valid_rate", 0.0) or 0.0),
        -float(run.get("trace_format_valid_rate", 0.0) or 0.0),
        -float(run.get("answer_format_valid_rate", 0.0) or 0.0),
        float(run.get("near_miss_rate", 1.0) or 1.0),
        str(run.get("model", "")),
        str(run.get("trace_prompt_variant", "")),
    )


def _decide(runs: list[dict[str, Any]]) -> dict[str, Any]:
    promotion_candidates: list[dict[str, Any]] = []
    for prompt_variant in {run["trace_prompt_variant"] for run in runs}:
        variant_runs = [run for run in runs if run["trace_prompt_variant"] == prompt_variant]
        if len(variant_runs) != len(MODELS):
            continue
        if all(
            float(run.get("joint_contract_valid_rate", 0.0) or 0.0) >= 0.25
            and float(run.get("task_and_joint_valid_rate", 0.0) or 0.0) >= 0.20
            for run in variant_runs
        ):
            promotion_candidates.append({
                "trace_prompt_variant": prompt_variant,
                "avg_joint_contract_valid_rate": _avg(run.get("joint_contract_valid_rate", 0.0) for run in variant_runs),
                "avg_task_and_joint_valid_rate": _avg(run.get("task_and_joint_valid_rate", 0.0) for run in variant_runs),
                "avg_trace_format_valid_rate": _avg(run.get("trace_format_valid_rate", 0.0) for run in variant_runs),
                "avg_answer_format_valid_rate": _avg(run.get("answer_format_valid_rate", 0.0) for run in variant_runs),
            })
    promotion_candidates.sort(
        key=lambda item: (
            -item["avg_joint_contract_valid_rate"],
            -item["avg_task_and_joint_valid_rate"],
            -item["avg_trace_format_valid_rate"],
            -item["avg_answer_format_valid_rate"],
        )
    )
    return {
        "promote_trace_prompt_variant": promotion_candidates[0]["trace_prompt_variant"] if promotion_candidates else None,
        "keep_public_trace_prompt_variant": "official",
        "promotion_candidates": promotion_candidates,
    }


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Trace Prompt Sweep",
        "",
        f"- Sweep root: `{summary['sweep_root']}`",
        f"- Task setup: `{summary['family']}` / `{summary['difficulty']}` / `{summary['trace_mode']}`",
        f"- Examples per run: `{summary['num_examples']}`",
        "",
        "| Model | Prompt Variant | Joint Valid | Task+Joint | Trace Valid | Answer Valid | Near Miss |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in summary["ranked_runs"]:
        lines.append(
            f"| {run['model']} | {run['trace_prompt_variant']} | {run['joint_contract_valid_rate']:.3f} | {run['task_and_joint_valid_rate']:.3f} | {run['trace_format_valid_rate']:.3f} | {run['answer_format_valid_rate']:.3f} | {run['near_miss_rate']:.3f} |"
        )
    lines.extend([
        "",
        "## Decision",
        "",
        f"- Keep public prompt variant: `{summary['decision']['keep_public_trace_prompt_variant']}`",
        f"- Promote prompt variant: `{summary['decision']['promote_trace_prompt_variant']}`",
    ])
    return "\n".join(lines) + "\n"


def _avg(values: Any) -> float:
    values = [float(value or 0.0) for value in values]
    if not values:
        return 0.0
    return sum(values) / len(values)


def _slugify_model(model: str) -> str:
    return model.replace("/", "--")


if __name__ == "__main__":
    main()
