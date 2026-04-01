from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from society_of_thought_bench.checkpoint_chat import BASE_MODEL, sample_checkpoint_async
from society_of_thought_bench.release_hardening import (
    _compute_release_reward,
    build_audit_note,
    build_trace_audit,
    classify_trace_issue,
    render_trace_audit_markdown,
)
from society_of_thought_bench.scoring import SocietyOfThoughtScorer
from society_of_thought_bench.parser import SocietyOfThoughtParser
from society_of_thought_bench.training_data import DATA_ROOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", default=BASE_MODEL)
    parser.add_argument(
        "--input-file",
        type=Path,
        default=DATA_ROOT / "disagreement_gold_v1" / "lock_eval.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    summary = {
        "model_path": args.model_path,
        "model_name": args.model_name,
        "input_file": str(args.input_file),
        "output_dir": str(args.output_dir),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }
    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return

    require_env("TINKER_API_KEY")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(args.input_file)
    results = asyncio.run(
        evaluate_rows_async(
            rows,
            model_path=args.model_path,
            model_name=args.model_name,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )
    )
    audit = build_trace_audit(results)
    output_summary = summarize_results(results, args)
    (args.output_dir / "samples.json").write_text(json.dumps(results, indent=2))
    (args.output_dir / "summary.json").write_text(json.dumps(output_summary, indent=2))
    (args.output_dir / "trace_audit.json").write_text(json.dumps(audit, indent=2))
    (args.output_dir / "TRACE_AUDIT.md").write_text(render_trace_audit_markdown(audit))
    print(json.dumps(output_summary, indent=2))


async def evaluate_rows_async(
    rows: list[dict[str, Any]],
    *,
    model_path: str,
    model_name: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> list[dict[str, Any]]:
    parser = SocietyOfThoughtParser(trace_mode="debate")
    scorer = SocietyOfThoughtScorer(max_personas=4, max_debate_turns=10, objective_profile="debate_primary")
    samples: list[dict[str, Any]] = []
    for row in rows:
        prompt_messages = row["messages"][:-1]
        response = await sample_checkpoint_async(
            prompt_messages,
            model_path=model_path,
            model_name=model_name,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        state = {
            "completion": [response.parsed_message],
            "info": {
                "family": row["family"],
                "difficulty": row["difficulty"],
                "trace_mode": "debate",
                "objective_profile": "debate_primary",
                "task": row["oracle_task"],
            },
        }
        metrics = {key: float(value) for key, value in scorer._ensure_metrics(state, parser).items()}
        parsed = state["_sot_trace"]
        reward = _compute_release_reward(metrics)
        sample = {
            "name": row["gold_set_name"],
            "family": row["family"],
            "difficulty": row["difficulty"],
            "institution": row["institution"],
            "challenge_pattern": row["challenge_pattern"],
            "trace_dialect": row["trace_dialect"],
            "prompt": prompt_messages,
            "raw_output": response.raw_output,
            "thinking_trace": response.thinking_trace,
            "visible_answer": response.visible_answer,
            "parsed_message": response.parsed_message,
            "reward": reward,
            "metrics": metrics,
            "parsed": {
                "personas": [persona.id for persona in parsed.personas],
                "turn_acts": [turn.act for turn in parsed.turns],
                "error_codes": parsed.error_codes,
            },
        }
        sample["label"] = classify_trace_issue(metrics, parsed)
        sample["note"] = build_audit_note(sample)
        samples.append(sample)
    return samples


def summarize_results(results: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    label_counts: dict[str, int] = {}
    pattern_counts: dict[str, int] = {}
    for sample in results:
        label_counts[sample["label"]] = label_counts.get(sample["label"], 0) + 1
        pattern = sample["challenge_pattern"]
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
    return {
        "model_path": args.model_path,
        "model_name": args.model_name,
        "input_file": str(args.input_file),
        "examples": len(results),
        "average_reward": sum(sample["reward"] for sample in results) / max(1, len(results)),
        "average_disagreement_quality": sum(sample["metrics"].get("disagreement_quality", 0.0) for sample in results) / max(1, len(results)),
        "average_interaction_score": sum(sample["metrics"].get("interaction_score", 0.0) for sample in results) / max(1, len(results)),
        "average_joint_valid": sum(sample["metrics"].get("format_valid", 0.0) for sample in results) / max(1, len(results)),
        "weak_disagreement_count": sum(1 for sample in results if sample["label"] == "weak_disagreement"),
        "label_counts": label_counts,
        "pattern_counts": pattern_counts,
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def require_env(name: str) -> None:
    if os.environ.get(name):
        return
    raise SystemExit(f"{name} is required for non-dry-run Tinker commands.")


if __name__ == "__main__":
    main()
