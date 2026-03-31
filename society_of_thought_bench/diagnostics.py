from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .parser import SocietyOfThoughtParser

DEFAULT_OUTPUTS_DIR = Path(__file__).resolve().parents[1] / "outputs" / "evals"


def latest_results_path(base_dir: Path | None = None) -> Path:
    search_root = base_dir or DEFAULT_OUTPUTS_DIR
    candidates = list(search_root.glob("**/results.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"No results.jsonl files found under {search_root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_results(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def analyze_results(
    path: Path,
    *,
    only_invalid: bool = False,
    limit: int = 10,
) -> dict[str, Any]:
    rows = load_results(path)
    analyses = [_analyze_row(row) for row in rows]
    selected = [
        analysis
        for analysis in analyses
        if not only_invalid or analysis["primary_error_code"] is not None
    ]
    selected = selected[: max(0, limit)]

    examples_total = len(analyses)
    invalid_examples = sum(1 for analysis in analyses if analysis["primary_error_code"] is not None)
    joint_contract_valid_examples = sum(int(analysis["joint_contract_valid"]) for analysis in analyses)
    task_and_joint_valid_examples = sum(int(analysis["task_and_joint_valid"]) for analysis in analyses)
    near_miss_examples = sum(int(analysis["near_miss"]) for analysis in analyses)

    return {
        "results_path": str(path),
        "examples_total": examples_total,
        "invalid_examples": invalid_examples,
        "joint_contract_valid_examples": joint_contract_valid_examples,
        "joint_contract_valid_rate": _safe_rate(joint_contract_valid_examples, examples_total),
        "task_and_joint_valid_examples": task_and_joint_valid_examples,
        "task_and_joint_valid_rate": _safe_rate(task_and_joint_valid_examples, examples_total),
        "near_miss_examples": near_miss_examples,
        "near_miss_rate": _safe_rate(near_miss_examples, examples_total),
        "protocol_valid_rate": _mean(analysis["protocol_valid"] for analysis in analyses),
        "trace_format_valid_rate": _mean(analysis["trace_valid"] for analysis in analyses),
        "answer_format_valid_rate": _mean(analysis["answer_valid"] for analysis in analyses),
        "average_task_score": _mean(analysis["task_score"] for analysis in analyses),
        "average_reward": _mean(analysis["reward"] for analysis in analyses),
        "primary_error_counts": dict(
            sorted(Counter(analysis["primary_error_code"] or "valid" for analysis in analyses).items())
        ),
        "trace_source_counts": dict(
            sorted(Counter(analysis["trace_source"] or "missing" for analysis in analyses).items())
        ),
        "trace_prompt_variant_counts": dict(
            sorted(Counter(analysis["trace_prompt_variant"] for analysis in analyses).items())
        ),
        "examples": selected,
    }


def render_analysis_report(summary: dict[str, Any]) -> str:
    lines = [
        f"Results: {summary['results_path']}",
        f"Examples: {summary['examples_total']}",
        f"Invalid examples: {summary['invalid_examples']}",
        f"Joint contract valid rate: {summary['joint_contract_valid_rate']:.3f}",
        f"Task and joint valid rate: {summary['task_and_joint_valid_rate']:.3f}",
        f"Near miss rate: {summary['near_miss_rate']:.3f}",
        f"Trace format valid rate: {summary['trace_format_valid_rate']:.3f}",
        f"Answer format valid rate: {summary['answer_format_valid_rate']:.3f}",
        "Primary error counts:",
    ]
    for code, count in summary["primary_error_counts"].items():
        lines.append(f"  {code}: {count}")
    lines.append("Trace sources:")
    for source, count in summary["trace_source_counts"].items():
        lines.append(f"  {source}: {count}")
    lines.append("Prompt variants:")
    for variant, count in summary["trace_prompt_variant_counts"].items():
        lines.append(f"  {variant}: {count}")
    lines.append("Examples:")
    for analysis in summary["examples"]:
        lines.extend(
            [
                (
                    f"  example_id={analysis['example_id']} reward={analysis['reward']:.3f} "
                    f"task_score={analysis['task_score']:.3f} joint_valid={analysis['joint_contract_valid']:.1f} "
                    f"protocol_valid={analysis['protocol_valid']:.1f} trace_valid={analysis['trace_valid']:.1f} "
                    f"answer_valid={analysis['answer_valid']:.1f}"
                ),
                (
                    f"    variant={analysis['trace_prompt_variant']} trace_source={analysis['trace_source'] or 'missing'}"
                ),
                f"    primary_error={analysis['primary_error_code'] or 'valid'}",
                f"    protocol_codes={','.join(analysis['protocol_error_codes']) or '-'}",
                f"    trace_codes={','.join(analysis['trace_error_codes']) or '-'}",
                f"    answer_codes={','.join(analysis['answer_error_codes']) or '-'}",
                f"    answer_excerpt={analysis['answer_excerpt']}",
                f"    reasoning_excerpt={analysis['reasoning_excerpt']}",
            ]
        )
    return "\n".join(lines)


def _analyze_row(row: dict[str, Any]) -> dict[str, Any]:
    info = row.get("info", {})
    if isinstance(info, str):
        info = json.loads(info)
    trace_mode = info.get("trace_mode", "debate")
    parser = SocietyOfThoughtParser(trace_mode=trace_mode)
    parsed = parser.parse_completion(row.get("completion", []))
    metrics = row.get("metrics", {})
    task_score = float(metrics.get("task_score", row.get("task_score", 0.0)) or 0.0)
    reward = float(row.get("reward", 0.0) or 0.0)
    joint_contract_valid = 1.0 if parsed.protocol_valid and parsed.trace_valid and parsed.answer_valid else 0.0
    task_and_joint_valid = 1.0 if joint_contract_valid == 1.0 and task_score >= 0.999 else 0.0
    near_miss = float(
        metrics.get(
            "task_correct_but_protocol_invalid",
            1.0 if task_score >= 0.999 and joint_contract_valid == 0.0 else 0.0,
        )
        or 0.0
    )
    return {
        "example_id": row.get("example_id"),
        "reward": reward,
        "task_score": task_score,
        "protocol_valid": 1.0 if parsed.protocol_valid else 0.0,
        "trace_valid": 1.0 if parsed.trace_valid else 0.0,
        "answer_valid": 1.0 if parsed.answer_valid else 0.0,
        "joint_contract_valid": joint_contract_valid,
        "task_and_joint_valid": task_and_joint_valid,
        "near_miss": near_miss,
        "primary_error_code": parsed.primary_error_code,
        "protocol_error_codes": parsed.protocol_error_codes,
        "trace_error_codes": parsed.trace_error_codes,
        "answer_error_codes": parsed.answer_error_codes,
        "trace_source": parsed.trace_source,
        "trace_prompt_variant": info.get("trace_prompt_variant", "official"),
        "answer_excerpt": _clip(parsed.answer_text or _completion_content(row), 160),
        "reasoning_excerpt": _clip(parsed.reasoning_text, 220),
    }


def _completion_content(row: dict[str, Any]) -> str:
    completion = row.get("completion", [])
    if isinstance(completion, list) and completion:
        message = completion[-1]
        if isinstance(message, dict):
            content = message.get("content", "")
            return content if isinstance(content, str) else json.dumps(content)
    return ""


def _clip(text: str, width: int) -> str:
    compact = " ".join(text.split())
    if not compact:
        return "-"
    if len(compact) <= width:
        return compact
    return compact[: width - 3] + "..."


def _mean(values: Any) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(float(value) for value in values) / len(values)


def _safe_rate(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return count / total
