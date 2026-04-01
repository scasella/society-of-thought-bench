from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

from .checkpoint_chat import BASE_MODEL, BEST_CHECKPOINT, sample_checkpoint_async
from .core import ParsedTrace
from .families import inspect_example
from .parser import SocietyOfThoughtParser
from .scoring import PROFILE_CONFIGS, SocietyOfThoughtScorer, resolve_debate_weights

PUBLIC_CHECKPOINT = BEST_CHECKPOINT
PUBLIC_BASE_MODEL = BASE_MODEL

FAILURE_TAXONOMY = (
    "shallow_debate",
    "weak_disagreement",
    "premature_reconcile",
    "format_break",
    "answer_drift",
    "redundant_personas",
)

RELEASE_ACCEPTANCE_THRESHOLDS = {
    "reward_delta": 0.35,
    "joint_valid_delta": 0.20,
    "disagreement_quality_delta": 0.40,
    "easy_answer_format_valid_rate": 0.95,
    "easy_joint_contract_valid_rate": 0.85,
    "hard_interaction_score": 0.65,
    "hard_disagreement_quality": 0.45,
}

CONFIRMATION_RUNS = {
    "debate_vs_monologue_medium_100": {"suite": "debate_vs_monologue", "num_examples": 100},
    "protocol_easy_40": {"suite": "protocol_easy_gate", "num_examples": 40},
    "debate_hard_40": {"suite": "debate_hard_gate", "num_examples": 40},
}

DEFAULT_DEMO_PROMPT_SPECS: tuple[dict[str, Any], ...] = (
    {"name": "countdown_easy_01", "family": "countdown", "difficulty": "easy", "institution": "flat", "seed": 11},
    {"name": "countdown_easy_02", "family": "countdown", "difficulty": "easy", "institution": "flat", "seed": 23},
    {"name": "evidence_easy_01", "family": "evidence", "difficulty": "easy", "institution": "flat", "seed": 17},
    {"name": "evidence_easy_02", "family": "evidence", "difficulty": "easy", "institution": "flat", "seed": 29},
    {"name": "countdown_medium_01", "family": "countdown", "difficulty": "medium", "institution": "auto", "seed": 101},
    {"name": "countdown_medium_02", "family": "countdown", "difficulty": "medium", "institution": "auto", "seed": 113},
    {"name": "countdown_medium_03", "family": "countdown", "difficulty": "medium", "institution": "auto", "seed": 127},
    {"name": "countdown_medium_04", "family": "countdown", "difficulty": "medium", "institution": "auto", "seed": 139},
    {"name": "evidence_medium_01", "family": "evidence", "difficulty": "medium", "institution": "auto", "seed": 103},
    {"name": "evidence_medium_02", "family": "evidence", "difficulty": "medium", "institution": "auto", "seed": 107},
    {"name": "evidence_medium_03", "family": "evidence", "difficulty": "medium", "institution": "auto", "seed": 131},
    {"name": "evidence_medium_04", "family": "evidence", "difficulty": "medium", "institution": "auto", "seed": 149},
    {"name": "countdown_hard_01", "family": "countdown", "difficulty": "hard", "institution": "auto", "seed": 211},
    {"name": "countdown_hard_02", "family": "countdown", "difficulty": "hard", "institution": "auto", "seed": 223},
    {"name": "countdown_hard_03", "family": "countdown", "difficulty": "hard", "institution": "auto", "seed": 227},
    {"name": "countdown_hard_04", "family": "countdown", "difficulty": "hard", "institution": "auto", "seed": 229},
    {"name": "evidence_hard_01", "family": "evidence", "difficulty": "hard", "institution": "auto", "seed": 233},
    {"name": "evidence_hard_02", "family": "evidence", "difficulty": "hard", "institution": "auto", "seed": 239},
    {"name": "evidence_hard_03", "family": "evidence", "difficulty": "hard", "institution": "auto", "seed": 241},
    {"name": "evidence_hard_04", "family": "evidence", "difficulty": "hard", "institution": "auto", "seed": 251},
)


def default_demo_prompt_specs() -> list[dict[str, Any]]:
    return [dict(spec) for spec in DEFAULT_DEMO_PROMPT_SPECS]


def evaluate_release_acceptance(
    debate_vs_monologue_summary: dict[str, Any],
    easy_summary: dict[str, Any],
    hard_summary: dict[str, Any],
) -> dict[str, Any]:
    checks = [
        _acceptance_check(
            "reward_delta",
            float(debate_vs_monologue_summary.get("reward_delta", 0.0)),
            RELEASE_ACCEPTANCE_THRESHOLDS["reward_delta"],
        ),
        _acceptance_check(
            "joint_valid_delta",
            float(debate_vs_monologue_summary.get("joint_valid_delta", 0.0)),
            RELEASE_ACCEPTANCE_THRESHOLDS["joint_valid_delta"],
        ),
        _acceptance_check(
            "disagreement_quality_delta",
            float(debate_vs_monologue_summary.get("disagreement_quality_delta", 0.0)),
            RELEASE_ACCEPTANCE_THRESHOLDS["disagreement_quality_delta"],
        ),
        _acceptance_check(
            "easy_answer_format_valid_rate",
            float(easy_summary.get("answer_format_valid_rate", 0.0)),
            RELEASE_ACCEPTANCE_THRESHOLDS["easy_answer_format_valid_rate"],
        ),
        _acceptance_check(
            "easy_joint_contract_valid_rate",
            float(easy_summary.get("joint_contract_valid_rate", 0.0)),
            RELEASE_ACCEPTANCE_THRESHOLDS["easy_joint_contract_valid_rate"],
        ),
        _acceptance_check(
            "hard_interaction_score",
            float(hard_summary.get("interaction_score", 0.0)),
            RELEASE_ACCEPTANCE_THRESHOLDS["hard_interaction_score"],
        ),
        _acceptance_check(
            "hard_disagreement_quality",
            float(hard_summary.get("disagreement_quality", 0.0)),
            RELEASE_ACCEPTANCE_THRESHOLDS["hard_disagreement_quality"],
        ),
    ]
    return {"passed": all(check["passed"] for check in checks), "checks": checks}


async def sample_demo_prompt_pack_async(
    *,
    model_path: str = PUBLIC_CHECKPOINT,
    model_name: str = PUBLIC_BASE_MODEL,
    prompt_specs: Sequence[dict[str, Any]] | None = None,
    temperature: float = 0.6,
    top_p: float = 0.95,
    max_tokens: int = 1024,
) -> list[dict[str, Any]]:
    specs = list(prompt_specs or default_demo_prompt_specs())
    parser = SocietyOfThoughtParser(trace_mode="debate")
    scorer = SocietyOfThoughtScorer(max_personas=4, max_debate_turns=10, objective_profile="debate_primary")

    samples: list[dict[str, Any]] = []
    for spec in specs:
        example = inspect_example(
            family=spec["family"],
            difficulty=spec["difficulty"],
            institution=spec.get("institution", "auto"),
            seed=int(spec["seed"]),
            max_personas=4,
            max_debate_turns=10,
            trace_mode="debate",
            trace_prompt_variant="official",
            countdown_reward_profile="benchmark",
            objective_profile="debate_primary",
        )
        prompt = example["prompt"]
        conversation = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
        response = await sample_checkpoint_async(
            conversation,
            model_path=model_path,
            model_name=model_name,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        assistant_message = response.parsed_message
        state = {
            "completion": [assistant_message],
            "prompt": prompt,
            "task": example["oracle"],
            "info": {
                **example["meta"],
                "task": example["oracle"],
                "family": spec["family"],
                "difficulty": spec["difficulty"],
                "trace_mode": "debate",
                "objective_profile": "debate_primary",
            },
        }
        metrics = {key: float(value) for key, value in scorer._ensure_metrics(state, parser).items()}
        parsed = state.get("_sot_trace")
        if not isinstance(parsed, ParsedTrace):
            parsed = parser.parse_completion([assistant_message])
        reward = _compute_release_reward(metrics)
        sample = {
            "name": spec["name"],
            "family": spec["family"],
            "difficulty": spec["difficulty"],
            "institution": spec.get("institution", "auto"),
            "seed": int(spec["seed"]),
            "prompt": prompt,
            "meta": example["meta"],
            "oracle": example["oracle"],
            "model_name": response.model_name,
            "model_path": response.model_path,
            "renderer_name": response.renderer_name,
            "raw_output": response.raw_output,
            "thinking_trace": response.thinking_trace,
            "visible_answer": response.visible_answer,
            "parsed_message": response.parsed_message,
            "reward": reward,
            "metrics": metrics,
            "parsed": _parsed_to_dict(parsed),
            "label": classify_trace_issue(metrics, parsed),
        }
        sample["note"] = build_audit_note(sample)
        samples.append(sample)
    return samples


def classify_trace_issue(metrics: dict[str, Any], parsed: ParsedTrace | dict[str, Any]) -> str:
    if isinstance(parsed, ParsedTrace):
        parsed_error_codes = parsed.error_codes
        answer_error_codes = parsed.answer_error_codes
        trace_error_codes = parsed.trace_error_codes
    else:
        parsed_error_codes = list(parsed.get("error_codes", []))
        answer_error_codes = list(parsed.get("answer_error_codes", []))
        trace_error_codes = list(parsed.get("trace_error_codes", []))

    if answer_error_codes or "answer_block_invalid" in parsed_error_codes or "answer_unexpected_text" in parsed_error_codes:
        return "answer_drift"
    if trace_error_codes or "missing_reasoning_trace" in parsed_error_codes:
        return "format_break"

    persona_diversity = float(metrics.get("persona_diversity", 0.0))
    interaction_score = float(metrics.get("interaction_score", 0.0))
    disagreement_quality = float(metrics.get("disagreement_quality", 0.0))
    reconcile_count = float(metrics.get("reconcile_link_count", 0.0))
    alternative_path_count = float(metrics.get("alternative_path_count", 0.0))
    conflict_count = float(metrics.get("conflict_of_perspectives_count", 0.0))
    challenge_pairs = float(metrics.get("challenge_response_pair_count", 0.0))

    if persona_diversity < 0.70:
        return "redundant_personas"
    if conflict_count >= 1.0 and challenge_pairs >= 1.0 and reconcile_count < 1.0:
        return "premature_reconcile"
    if disagreement_quality < 0.25 and interaction_score >= 0.45:
        return "weak_disagreement"
    if interaction_score < 0.45 and alternative_path_count < 1.0:
        return "shallow_debate"
    if disagreement_quality < 0.40:
        return "weak_disagreement"
    return "premature_reconcile" if reconcile_count < 1.0 else "shallow_debate"


def build_trace_audit(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if len(samples) < 12:
        raise ValueError("Trace audit requires at least 12 sampled prompts.")

    strong = [sample for sample in samples if _bucket(sample) == "strong"]
    borderline = [sample for sample in samples if _bucket(sample) == "borderline"]
    failures = [sample for sample in samples if _bucket(sample) == "failure"]

    strong_selected = _take_best(strong, count=4, reverse=True, fallback=samples)
    failure_selected = _take_best(failures, count=4, reverse=False, fallback=samples)
    used = {sample["name"] for sample in [*strong_selected, *failure_selected]}
    borderline_selected = _take_middle(borderline, count=4, fallback=[sample for sample in samples if sample["name"] not in used])
    used.update(sample["name"] for sample in borderline_selected)
    strong_selected = _fill_to_count(strong_selected, target=4, fallback=samples, used_names=used - {sample["name"] for sample in strong_selected})
    failure_selected = _fill_to_count(failure_selected, target=4, fallback=samples, used_names=used - {sample["name"] for sample in failure_selected})
    borderline_selected = _fill_to_count(borderline_selected, target=4, fallback=samples, used_names=used - {sample["name"] for sample in borderline_selected})

    selected = {
        "strong_successes": strong_selected,
        "borderline_examples": borderline_selected,
        "failures": failure_selected,
    }
    return {
        "counts": {key: len(value) for key, value in selected.items()},
        "examples": selected,
        "taxonomy": list(FAILURE_TAXONOMY),
    }


def render_trace_audit_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Trace Audit",
        "",
        "This pack is meant to make the checkpoint inspectable by a skeptical reader. It includes strong cases, ordinary cases, and failures.",
        "",
    ]
    section_titles = {
        "strong_successes": "Strong Successes",
        "borderline_examples": "Ordinary Or Borderline Examples",
        "failures": "Failures",
    }
    for key in ("strong_successes", "borderline_examples", "failures"):
        lines.extend([f"## {section_titles[key]}", ""])
        for example in audit["examples"][key]:
            lines.extend(
                [
                    f"### {example['name']}",
                    "",
                    f"- Family: `{example['family']}`",
                    f"- Difficulty: `{example['difficulty']}`",
                    f"- Label: `{example['label']}`",
                    f"- Reward: `{example['reward']:.3f}`",
                    f"- Joint validity: `{example['metrics'].get('format_valid', 0.0):.3f}`",
                    f"- Task score: `{example['metrics'].get('task_score', 0.0):.3f}`",
                    f"- Interaction: `{example['metrics'].get('interaction_score', 0.0):.3f}`",
                    f"- Disagreement quality: `{example['metrics'].get('disagreement_quality', 0.0):.3f}`",
                    "",
                    example["note"],
                    "",
                    "#### Prompt",
                    "",
                    "```json" if not isinstance(example["prompt"], str) else "```text",
                    _prompt_text(example["prompt"]),
                    "```",
                    "",
                    "#### Raw Thinking Trace",
                    "",
                    "```text",
                    example["thinking_trace"] or "[none]",
                    "```",
                    "",
                    "#### Final Answer",
                    "",
                    "```text",
                    example["visible_answer"] or "[none]",
                    "```",
                    "",
                ]
            )
    return "\n".join(lines).strip() + "\n"


def render_usage_guidance() -> str:
    return (
        "# How To Use This Checkpoint\n\n"
        "- Use it when you want to inspect a paper-style multi-voice hidden discussion, not when you need a broad claim about general benchmark superiority.\n"
        "- It is strongest on the benchmark’s medium-difficulty tasks, where the voices usually stay distinct and the answer channel stays clean.\n"
        "- Read the hidden discussion as a diagnostic artifact. It shows how the model is organizing the work, but it is not a guarantee that every step is true.\n"
        "- If the task is very constrained, watch the final answer channel more than the hidden discussion. The current model sometimes keeps the discussion clean while still missing the final answer.\n"
        "- If you want a quick inspection path, start with the 20-prompt demo pack and the 12-example audit pack before drawing broader conclusions.\n"
        "- The strongest evidence is still benchmark-local. Use the outside benchmark characterization as background context, not as the headline claim.\n"
    )


def render_confirmation_markdown(
    *,
    debate_vs_monologue_summary: dict[str, Any],
    easy_summary: dict[str, Any],
    hard_summary: dict[str, Any],
    acceptance: dict[str, Any],
    outside_summary: dict[str, Any] | None = None,
) -> str:
    lines = [
        "# Public Checkpoint Confirmation",
        "",
        f"- Checkpoint: `{PUBLIC_CHECKPOINT}`",
        f"- Model: `{PUBLIC_BASE_MODEL}`",
        "",
        "## Internal Confirmation",
        "",
        "| Check | Result |",
        "| --- | ---: |",
        f"| Medium reward delta | {debate_vs_monologue_summary.get('reward_delta', 0.0):.3f} |",
        f"| Medium task score delta | {debate_vs_monologue_summary.get('task_score_delta', 0.0):.3f} |",
        f"| Medium joint-valid delta | {debate_vs_monologue_summary.get('joint_valid_delta', 0.0):.3f} |",
        f"| Medium disagreement delta | {debate_vs_monologue_summary.get('disagreement_quality_delta', 0.0):.3f} |",
        f"| Easy joint-valid rate | {easy_summary.get('joint_contract_valid_rate', 0.0):.3f} |",
        f"| Easy answer-valid rate | {easy_summary.get('answer_format_valid_rate', 0.0):.3f} |",
        f"| Hard interaction score | {hard_summary.get('interaction_score', 0.0):.3f} |",
        f"| Hard disagreement quality | {hard_summary.get('disagreement_quality', 0.0):.3f} |",
        f"| Hard persona diversity | {hard_summary.get('persona_diversity', 0.0):.3f} |",
        "",
        "## Release Acceptance",
        "",
    ]
    for check in acceptance["checks"]:
        status = "pass" if check["passed"] else "fail"
        lines.append(
            f"- `{check['metric']}`: `{check['actual']:.3f}` vs target `{check['threshold']:.3f}` -> {status}"
        )
    if outside_summary is not None:
        macro = outside_summary.get("same_checkpoint", {}).get("macro_average", {})
        lines.extend(
            [
                "",
                "## Background External Characterization",
                "",
                f"- Macro native score delta: `{float(macro.get('native_score_delta', 0.0)):.3f}`",
                f"- Macro visible-answer-valid delta: `{float(macro.get('visible_answer_valid_delta', 0.0)):.3f}`",
                f"- Macro reasoning-contract-valid delta: `{float(macro.get('reasoning_contract_valid_delta', 0.0)):.3f}`",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def build_release_manifest(
    *,
    output_dir: Path,
    confirmation_paths: dict[str, str],
    demo_prompt_pack_path: str,
    demo_samples_path: str,
    audit_path: str,
    usage_path: str,
    outside_summary_path: str | None = None,
) -> dict[str, Any]:
    return {
        "output_dir": _display_path(output_dir),
        "public_checkpoint": PUBLIC_CHECKPOINT,
        "public_model": PUBLIC_BASE_MODEL,
        "confirmation_paths": {key: _display_path(value) for key, value in confirmation_paths.items()},
        "demo_prompt_pack_path": _display_path(demo_prompt_pack_path),
        "demo_samples_path": _display_path(demo_samples_path),
        "audit_path": _display_path(audit_path),
        "usage_path": _display_path(usage_path),
        "outside_summary_path": _display_path(outside_summary_path),
    }


def _display_path(path: str | Path | None) -> str | None:
    if not path:
        return None
    path_obj = Path(path).resolve()
    try:
        return str(path_obj.relative_to(Path.cwd().resolve()))
    except ValueError:
        return path_obj.as_posix()


def _acceptance_check(metric: str, actual: float, threshold: float) -> dict[str, Any]:
    return {
        "metric": metric,
        "actual": float(actual),
        "threshold": float(threshold),
        "passed": float(actual) >= float(threshold),
    }


def _parsed_to_dict(parsed: ParsedTrace) -> dict[str, Any]:
    return {
        "trace_mode": parsed.trace_mode,
        "raw_trace": parsed.raw_trace,
        "raw_answer": parsed.raw_answer,
        "personas": [asdict(persona) for persona in parsed.personas],
        "turns": [asdict(turn) for turn in parsed.turns],
        "final_answer": parsed.final_answer,
        "support": list(parsed.support),
        "group_solution": parsed.group_solution,
        "reasoning_text": parsed.reasoning_text,
        "answer_text": parsed.answer_text,
        "trace_source": parsed.trace_source,
        "protocol_error_codes": list(parsed.protocol_error_codes),
        "trace_error_codes": list(parsed.trace_error_codes),
        "answer_error_codes": list(parsed.answer_error_codes),
        "error_codes": list(parsed.error_codes),
    }


def _bucket(sample: dict[str, Any]) -> str:
    metrics = sample["metrics"]
    reward = float(sample.get("reward", 0.0))
    joint_valid = 1.0 if float(metrics.get("format_valid", 0.0)) >= 0.999 else 0.0
    interaction = float(metrics.get("interaction_score", 0.0))
    disagreement = float(metrics.get("disagreement_quality", 0.0))
    persona_diversity = float(metrics.get("persona_diversity", 0.0))
    if joint_valid >= 0.999 and reward >= 0.65 and interaction >= 0.60 and persona_diversity >= 0.75:
        return "strong"
    if joint_valid < 0.999 or reward < 0.45 or disagreement < 0.10:
        return "failure"
    return "borderline"


def _quality_key(sample: dict[str, Any]) -> float:
    metrics = sample["metrics"]
    return float(mean([
        float(sample.get("reward", 0.0)),
        float(metrics.get("task_score", 0.0)),
        float(metrics.get("interaction_score", 0.0)),
        float(metrics.get("persona_diversity", 0.0)),
        float(metrics.get("disagreement_quality", 0.0)),
    ]))


def _take_best(
    samples: Sequence[dict[str, Any]],
    *,
    count: int,
    reverse: bool,
    fallback: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    pool = list(samples) if samples else list(fallback)
    ranked = sorted(pool, key=_quality_key, reverse=reverse)
    seen: set[str] = set()
    picked: list[dict[str, Any]] = []
    for sample in ranked:
        if sample["name"] in seen:
            continue
        seen.add(sample["name"])
        picked.append(sample)
        if len(picked) >= count:
            break
    return picked


def _take_middle(
    samples: Sequence[dict[str, Any]],
    *,
    count: int,
    fallback: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    pool = list(samples) if samples else list(fallback)
    ranked = sorted(pool, key=_quality_key, reverse=True)
    if len(ranked) <= count:
        return ranked[:count]
    start = max(0, (len(ranked) // 2) - (count // 2))
    picked = ranked[start : start + count]
    if len(picked) < count:
        picked.extend(ranked[: count - len(picked)])
    return picked[:count]


def _fill_to_count(
    picked: list[dict[str, Any]],
    *,
    target: int,
    fallback: Sequence[dict[str, Any]],
    used_names: set[str],
) -> list[dict[str, Any]]:
    if len(picked) >= target:
        return picked[:target]
    result = list(picked)
    local_used = {sample["name"] for sample in result} | set(used_names)
    for sample in fallback:
        if sample["name"] in local_used:
            continue
        result.append(sample)
        local_used.add(sample["name"])
        if len(result) >= target:
            break
    return result[:target]


def _prompt_text(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    return json_dumps(prompt)


def build_audit_note(sample: dict[str, Any]) -> str:
    label = sample["label"]
    reward = float(sample.get("reward", 0.0))
    metrics = sample["metrics"]
    task_score = float(metrics.get("task_score", 0.0))
    interaction = float(metrics.get("interaction_score", 0.0))
    disagreement = float(metrics.get("disagreement_quality", 0.0))

    if label == "format_break":
        return (
            "The model loses the clean contract here. The hidden discussion or the answer block breaks in a way that makes the example less trustworthy, even before we judge whether the answer itself is right."
        )
    if label == "answer_drift":
        return (
            "The hidden discussion stays recognizable, but the final answer channel drifts away from the clean target format or from the task itself. This is usable for inspection but not reliable enough for deployment."
        )
    if label == "redundant_personas":
        return (
            "The trace looks multi-voice on the surface, but the voices collapse into near-duplicates. The result is readable, yet it does not buy much real deliberation."
        )
    if label == "premature_reconcile":
        return (
            "The model moves to consensus too early. It resolves the discussion before the challenge path is worked through carefully enough, which weakens the value of the trace."
        )
    if label == "weak_disagreement":
        return (
            "The exchange is orderly and usually readable, but the pushback is mild. The voices do not pressure-test the main route strongly enough before settling on an answer."
        )
    if reward >= 0.65 and task_score >= 0.65 and interaction >= 0.60:
        return (
            "This is a strong example of the current checkpoint. The answer channel stays separate, the voices stay distinct, and the trace shows real back-and-forth rather than filler."
        )
    if disagreement >= 0.10 and interaction >= 0.50:
        return (
            "This is an ordinary example: the basic structure is there and the discussion is usable, but the debate is still more scripted than fully searching."
        )
    return (
        "This is a shallow example. The checkpoint keeps some of the intended paper-style shell, but the discussion does not add enough pressure or branching to be convincing."
    )


def json_dumps(value: Any) -> str:
    import json

    return json.dumps(value, indent=2, sort_keys=True)


def _compute_release_reward(metrics: dict[str, float]) -> float:
    profile = PROFILE_CONFIGS["debate_primary"]
    debate = resolve_debate_weights("debate_primary", debate_reward_weight=None, debate_metric_weights=None)
    reward = (
        profile["format_valid"] * metrics.get("format_valid", 0.0)
        + profile["task_score"] * metrics.get("task_score", 0.0)
        + debate["behavior_coverage"] * metrics.get("behavior_coverage", 0.0)
        + debate["persona_diversity"] * metrics.get("persona_diversity", 0.0)
        + debate["interaction_score"] * metrics.get("interaction_score", 0.0)
        + debate["debate_relevance"] * metrics.get("debate_relevance", 0.0)
        + debate["disagreement_quality"] * metrics.get("disagreement_quality", 0.0)
        - 0.05 * metrics.get("efficiency_penalty", 0.0)
    )
    if metrics.get("protocol_valid", 0.0) < 0.999:
        return 0.0
    return max(0.0, min(1.0, float(reward)))
