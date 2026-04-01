from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence

import verifiers as vf
from datasets import Dataset, load_dataset
from verifiers.utils.data_utils import BOXED_SYSTEM_PROMPT, extract_boxed_answer

from .core import ordered_unique
from .families import _wrap_trace_payload, build_example
from .parser import SocietyOfThoughtParser
from .tinker_renderers import get_recommended_renderer_name, get_renderer
from .training_data import (
    DEFAULT_COUNTDOWN_REWARD_PROFILE,
    DEFAULT_OBJECTIVE_PROFILE,
    DEFAULT_TRACE_PROMPT_VARIANT,
    DEFAULT_TRACE_MODE,
    EXTERNAL_DEBATE_CURRICULUM_PROFILES,
    EXTERNAL_ACCEPTANCE_THRESHOLDS,
    external_dpo_component_counts,
    external_sft_source_counts,
    select_trace_dialect,
    warmup_mix_counts,
)

CORE_EXTERNAL_BENCHMARKS = ("gsm8k", "mmlu_pro", "gpqa", "ifeval")
DEFAULT_EXTERNAL_MODEL = "Qwen/Qwen3-30B-A3B"
STEM_CATEGORIES = {"biology", "chemistry", "computer science", "engineering", "math", "physics"}
BOXED_MC_OPTIONS = set("ABCDEFGHIJ")
BOXED_GPQA_OPTIONS = set("ABCD")


@dataclass(frozen=True, slots=True)
class ExternalBenchmarkSpec:
    key: str
    env_id: str
    answer_kind: str
    needs_hf_token: bool = False


SPECS: dict[str, ExternalBenchmarkSpec] = {
    "gsm8k": ExternalBenchmarkSpec(
        key="gsm8k",
        env_id="primeintellect/gsm8k",
        answer_kind="boxed_numeric",
    ),
    "mmlu_pro": ExternalBenchmarkSpec(
        key="mmlu_pro",
        env_id="primeintellect/mmlu-pro",
        answer_kind="boxed_mcq",
    ),
    "gpqa": ExternalBenchmarkSpec(
        key="gpqa",
        env_id="primeintellect/gpqa",
        answer_kind="boxed_gpqa",
        needs_hf_token=True,
    ),
    "ifeval": ExternalBenchmarkSpec(
        key="ifeval",
        env_id="primeintellect/ifeval",
        answer_kind="free_text",
    ),
}


def normalize_benchmark_name(name: str) -> str:
    value = name.strip().lower().replace("-", "_").replace("/", "_")
    if value in SPECS:
        return value
    if value.endswith("gsm8k"):
        return "gsm8k"
    if value.endswith("mmlu_pro") or value.endswith("mmlu"):
        return "mmlu_pro"
    if value.endswith("gpqa"):
        return "gpqa"
    if value.endswith("ifeval"):
        return "ifeval"
    raise KeyError(f"Unknown benchmark: {name}")


def benchmark_spec(name: str) -> ExternalBenchmarkSpec:
    return SPECS[normalize_benchmark_name(name)]


def core_benchmark_order(names: Sequence[str] | None = None) -> list[str]:
    values = list(names or CORE_EXTERNAL_BENCHMARKS)
    return [normalize_benchmark_name(value) for value in values]


_EXTERNAL_DEBATE_SYSTEM = (
    "Use the reasoning stream as hidden scratch space. Inside it, write one paper-style society-of-thought "
    "discussion with exactly one <cast_of_characters> block, one <conversation> block, and one <group_solution> block. "
    "Allow dialect flexibility inside that structure: persona tags like <persona1>...</persona1> or equivalent "
    "<character ...>...</character> / <character .../> entries are both acceptable, and numbered turns like "
    "<think1>...</think1>, equivalent ordered turn tags like <step ...>...</step>, named speaker tags, or simple "
    "speaker-prefixed lines are all acceptable. "
    "Make the hidden discussion genuinely deliberate: propose a route, challenge it, answer the challenge, and verify the visible answer format. "
    "Keep the visible answer outside the reasoning stream, do not duplicate it inside the reasoning stream, and follow the task's requested answer format exactly."
)

_EXTERNAL_MONOLOGUE_SYSTEM = (
    "Use the reasoning stream as hidden scratch space. Reason in a single voice. Do not invent personas, cast blocks, "
    "conversation tags, or dialogue. Keep the visible answer outside the reasoning stream and follow the task's "
    "requested answer format exactly."
)


def build_external_system_prompt(
    benchmark: str,
    *,
    trace_mode: str,
    original_system_prompt: str | None = None,
) -> str:
    if trace_mode not in {"debate", "monologue"}:
        raise ValueError("trace_mode must be debate or monologue")
    base = _EXTERNAL_DEBATE_SYSTEM if trace_mode == "debate" else _EXTERNAL_MONOLOGUE_SYSTEM
    reminder = "Return only the final answer in the benchmark-native format requested by the task."
    if benchmark == "gsm8k":
        reminder = (
            "Return the final answer inside \\boxed{} and do not add extra visible text. "
            "The hidden discussion should pressure-test the arithmetic before committing to the boxed number."
        )
    elif benchmark in {"mmlu_pro", "gpqa"}:
        reminder = (
            "Return only the final boxed answer letter required by the question. "
            "The hidden discussion should compare the leading option against the strongest distractor before boxing the letter."
        )
    elif benchmark == "ifeval":
        reminder = (
            "Return only the requested user-facing response after the reasoning stream. "
            "The hidden discussion should enumerate the instruction constraints and challenge any likely violation."
        )
    structure = ""
    if trace_mode == "debate":
        structure = (
            "Preferred hidden structure:\n"
            "<think>\n"
            "<cast_of_characters>\n"
            "<persona1>Role: ... Personality: ... Expertise: ... Style: ...</persona1>\n"
            "<persona2>...</persona2>\n"
            "</cast_of_characters>\n"
            "<conversation>\n"
            "<think1>...</think1>\n"
            "<think2>...</think2>\n"
            "</conversation>\n"
            "<group_solution>...</group_solution>\n"
            "</think>\n"
            "Equivalent character tags, step tags, named speaker tags, or simple speaker lines are allowed if the cast, turn order, and final group solution stay clear."
        )
    parts = [base, structure, reminder] if structure else [base, reminder]
    if original_system_prompt:
        parts.append(f"Original answer-format instruction: {original_system_prompt}")
    return "\n\n".join(parts)


def ensure_required_env_vars(benchmarks: Iterable[str]) -> None:
    needs_hf = any(benchmark_spec(name).needs_hf_token for name in benchmarks)
    if needs_hf and not os.environ.get("HF_TOKEN"):
        raise SystemExit("HF_TOKEN is required for GPQA-based external evaluations.")


def save_external_suite_lock_file(
    path: str | os.PathLike[str],
    *,
    rows: Sequence[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "metadata": metadata or {},
        "rows": [json.loads(json.dumps(row)) for row in rows],
    }
    destination.write_text(json.dumps(payload, indent=2))
    return destination


def load_external_suite_lock_file(path: str | os.PathLike[str]) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict) or "rows" not in payload:
        raise ValueError(f"Invalid external lock file: {path}")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"Invalid external lock rows in: {path}")
    return payload


def load_external_eval_env(
    benchmark: str,
    *,
    trace_mode: str,
    num_examples: int = -1,
) -> vf.Environment:
    key = normalize_benchmark_name(benchmark)
    spec = benchmark_spec(key)
    if spec.needs_hf_token and not os.environ.get("HF_TOKEN"):
        raise SystemExit("HF_TOKEN is required for GPQA-based external evaluations.")

    if key == "gsm8k":
        system_prompt = build_external_system_prompt(key, trace_mode=trace_mode, original_system_prompt=BOXED_SYSTEM_PROMPT)
        return vf.load_environment(
            spec.env_id,
            system_prompt=system_prompt,
            num_train_examples=1,
            num_eval_examples=num_examples,
        )
    if key == "mmlu_pro":
        return vf.load_environment(
            spec.env_id,
            dataset_split="test",
            system_prompt=build_external_system_prompt(key, trace_mode=trace_mode),
        )
    if key == "gpqa":
        return vf.load_environment(
            spec.env_id,
            system_prompt=build_external_system_prompt(key, trace_mode=trace_mode),
            verifier="exact-match",
            diamond=True,
        )
    return vf.load_environment(
        spec.env_id,
        mode="strict",
        system_prompt=build_external_system_prompt(key, trace_mode=trace_mode),
    )


def extract_visible_answer_text(completion: Any) -> str:
    assistant = _assistant_message(completion)
    if assistant is None:
        return ""
    text = _content_to_text(_field(assistant, "content", ""))
    _, visible = _split_reasoning_and_visible_text(text)
    if "<think>" in text.lower():
        return visible
    return visible or text.strip()


def extract_reasoning_text(completion: Any) -> str:
    assistant = _assistant_message(completion)
    if assistant is None:
        return ""
    reasoning = _field(assistant, "reasoning_content", "") or ""
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning.strip()
    text = _content_to_text(_field(assistant, "content", ""))
    extracted, _ = _split_reasoning_and_visible_text(text)
    return extracted


def visible_answer_is_valid(benchmark: str, answer_text: str) -> bool:
    key = normalize_benchmark_name(benchmark)
    spec = benchmark_spec(key)
    stripped = answer_text.strip()
    if not stripped:
        return False
    if spec.answer_kind == "boxed_numeric":
        boxed = _extract_boxed_literal(stripped)
        return bool(boxed and re.fullmatch(r"[-+]?\d+(?:\.\d+)?", boxed))
    if spec.answer_kind == "boxed_mcq":
        boxed = (_extract_boxed_literal(stripped) or "").upper()
        return boxed in BOXED_MC_OPTIONS
    if spec.answer_kind == "boxed_gpqa":
        boxed = (_extract_boxed_literal(stripped) or "").upper()
        return boxed in BOXED_GPQA_OPTIONS
    return bool(stripped)


def reasoning_contract_is_valid(trace_mode: str, reasoning_text: str) -> bool:
    stripped = reasoning_text.strip()
    if not stripped:
        return False
    if trace_mode == "monologue":
        lowered = stripped.lower()
        return "<cast_of_characters>" not in lowered and "<conversation>" not in lowered
    parser = SocietyOfThoughtParser(trace_mode=trace_mode)
    parsed = parser.parse(f"<think>{stripped}</think>\n<answer>stub</answer>")
    return parsed.trace_valid


def summarize_external_generate_outputs(
    results: dict[str, Any],
    *,
    benchmark: str,
    trace_mode: str,
) -> dict[str, Any]:
    outputs = list(results.get("outputs", []))
    rewards = [float(item.get("reward", 0.0)) for item in outputs]
    metric_names = sorted({name for item in outputs for name in item.get("metrics", {}).keys()})
    metric_summary = {
        name: _safe_mean([float(item.get("metrics", {}).get(name, 0.0)) for item in outputs])
        for name in metric_names
    }
    visible_valid = []
    reasoning_valid = []
    for item in outputs:
        completion = item.get("completion") or []
        answer_text = extract_visible_answer_text(completion)
        reasoning_text = extract_reasoning_text(completion)
        visible_valid.append(1.0 if visible_answer_is_valid(benchmark, answer_text) else 0.0)
        reasoning_valid.append(1.0 if reasoning_contract_is_valid(trace_mode, reasoning_text) else 0.0)
    return {
        "benchmark": normalize_benchmark_name(benchmark),
        "env_id": benchmark_spec(benchmark).env_id,
        "trace_mode": trace_mode,
        "examples": len(outputs),
        "native_score": _safe_mean(rewards),
        "average_reward": _safe_mean(rewards),
        "visible_answer_valid_rate": _safe_mean(visible_valid),
        "reasoning_contract_valid_rate": _safe_mean(reasoning_valid),
        "metrics": metric_summary,
    }


def aggregate_external_suite(summaries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_name = {summary["benchmark"]: summary for summary in summaries}
    ordered = [by_name[name] for name in CORE_EXTERNAL_BENCHMARKS if name in by_name]
    metric_names = sorted({name for summary in ordered for name in summary.get("metrics", {}).keys()})
    macro_metrics = {
        name: _safe_mean([float(summary.get("metrics", {}).get(name, 0.0)) for summary in ordered])
        for name in metric_names
    }
    return {
        "benchmarks": ordered,
        "macro_average": {
            "native_score": _safe_mean([summary["native_score"] for summary in ordered]),
            "visible_answer_valid_rate": _safe_mean([summary["visible_answer_valid_rate"] for summary in ordered]),
            "reasoning_contract_valid_rate": _safe_mean(
                [summary["reasoning_contract_valid_rate"] for summary in ordered]
            ),
            **macro_metrics,
        },
    }


def compare_external_suite_aggregates(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    left_label: str,
    right_label: str,
) -> dict[str, Any]:
    left_by_name = {summary["benchmark"]: summary for summary in left["benchmarks"]}
    right_by_name = {summary["benchmark"]: summary for summary in right["benchmarks"]}
    benchmark_rows = []
    non_negative_count = 0
    for name in CORE_EXTERNAL_BENCHMARKS:
        if name not in left_by_name or name not in right_by_name:
            continue
        lhs = left_by_name[name]
        rhs = right_by_name[name]
        native_delta = float(lhs["native_score"] - rhs["native_score"])
        if native_delta >= 0.0:
            non_negative_count += 1
        benchmark_rows.append(
            {
                "benchmark": name,
                left_label: lhs,
                right_label: rhs,
                "native_score_delta": native_delta,
                "visible_answer_valid_delta": float(
                    lhs["visible_answer_valid_rate"] - rhs["visible_answer_valid_rate"]
                ),
                "reasoning_contract_valid_delta": float(
                    lhs["reasoning_contract_valid_rate"] - rhs["reasoning_contract_valid_rate"]
                ),
            }
        )
    macro = {
        "native_score_delta": float(
            left["macro_average"]["native_score"] - right["macro_average"]["native_score"]
        ),
        "visible_answer_valid_delta": float(
            left["macro_average"]["visible_answer_valid_rate"]
            - right["macro_average"]["visible_answer_valid_rate"]
        ),
        "reasoning_contract_valid_delta": float(
            left["macro_average"]["reasoning_contract_valid_rate"]
            - right["macro_average"]["reasoning_contract_valid_rate"]
        ),
        "non_negative_native_score_benchmarks": non_negative_count,
    }
    return {
        "benchmarks": benchmark_rows,
        "macro_average": macro,
        "left_label": left_label,
        "right_label": right_label,
    }


def evaluate_external_acceptance(comparison: dict[str, Any], *, threshold_key: str) -> dict[str, Any]:
    threshold = EXTERNAL_ACCEPTANCE_THRESHOLDS[threshold_key]
    checks = [
        {
            "metric": "native_score_delta",
            "actual": comparison["macro_average"]["native_score_delta"],
            "threshold": threshold["native_score_delta"],
            "passed": comparison["macro_average"]["native_score_delta"] >= threshold["native_score_delta"],
        },
        {
            "metric": "non_negative_native_score_benchmarks",
            "actual": comparison["macro_average"]["non_negative_native_score_benchmarks"],
            "threshold": threshold["non_negative_native_score_benchmarks"],
            "passed": comparison["macro_average"]["non_negative_native_score_benchmarks"]
            >= threshold["non_negative_native_score_benchmarks"],
        },
        {
            "metric": "reasoning_contract_valid_delta",
            "actual": comparison["macro_average"]["reasoning_contract_valid_delta"],
            "threshold": threshold["reasoning_contract_valid_delta"],
            "passed": comparison["macro_average"]["reasoning_contract_valid_delta"]
            >= threshold["reasoning_contract_valid_delta"],
        },
        {
            "metric": "visible_answer_valid_delta",
            "actual": comparison["macro_average"]["visible_answer_valid_delta"],
            "threshold": threshold["visible_answer_valid_delta"],
            "passed": comparison["macro_average"]["visible_answer_valid_delta"]
            >= threshold["visible_answer_valid_delta"],
        },
    ]
    return {
        "threshold_key": threshold_key,
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
    }


def build_external_sft_rows(
    *,
    total: int,
    curriculum_profile: str,
    split: str,
    seed_start: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    source_counts = external_sft_source_counts(total, curriculum_profile=curriculum_profile)
    trace_mode = "debate" if curriculum_profile in EXTERNAL_DEBATE_CURRICULUM_PROFILES else "monologue"
    rows: list[dict[str, Any]] = []
    seed = seed_start
    if source_counts.get("internal_sot", 0):
        rows.extend(
            _build_internal_sft_rows(
                source_counts["internal_sot"],
                split=split,
                seed_start=seed,
                trace_mode=trace_mode,
                curriculum_profile=curriculum_profile,
            )
        )
        seed += source_counts["internal_sot"]
    if source_counts.get("gsm8k_train", 0):
        rows.extend(_build_gsm8k_sft_rows(source_counts["gsm8k_train"], seed_start=seed, trace_mode=trace_mode))
        seed += source_counts["gsm8k_train"]
    if source_counts.get("mmlu_pro_non_eval", 0):
        rows.extend(_build_mmlu_sft_rows(source_counts["mmlu_pro_non_eval"], seed_start=seed, trace_mode=trace_mode, stem_only=False))
        seed += source_counts["mmlu_pro_non_eval"]
    if source_counts.get("mmlu_pro_stem", 0):
        rows.extend(_build_mmlu_sft_rows(source_counts["mmlu_pro_stem"], seed_start=seed, trace_mode=trace_mode, stem_only=True))
        seed += source_counts["mmlu_pro_stem"]
    if source_counts.get("ifeval_synthetic", 0):
        rows.extend(_build_ifeval_synthetic_sft_rows(source_counts["ifeval_synthetic"], seed_start=seed, trace_mode=trace_mode))
    return rows, source_counts


def build_external_dpo_rows(
    *,
    total: int,
    curriculum_profile: str,
    split: str,
    seed_start: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    component_counts = external_dpo_component_counts(total, curriculum_profile=curriculum_profile)
    trace_mode = "debate" if curriculum_profile in EXTERNAL_DEBATE_CURRICULUM_PROFILES else "monologue"
    rows: list[dict[str, Any]] = []
    seed = seed_start
    if curriculum_profile in EXTERNAL_DEBATE_CURRICULUM_PROFILES:
        if component_counts.get("internal_debate_quality", 0):
            rows.extend(
                _build_internal_dpo_rows(
                    component_counts["internal_debate_quality"],
                    split=split,
                    seed_start=seed,
                    trace_mode="debate",
                    curriculum_profile=curriculum_profile,
                )
            )
            seed += component_counts["internal_debate_quality"]
        if component_counts.get("external_answer_discipline", 0):
            rows.extend(_build_external_answer_discipline_pairs(component_counts["external_answer_discipline"], seed_start=seed, trace_mode=trace_mode))
            seed += component_counts["external_answer_discipline"]
        if component_counts.get("external_reasoning_structure", 0):
            rows.extend(_build_external_reasoning_structure_pairs(component_counts["external_reasoning_structure"], seed_start=seed, trace_mode=trace_mode))
    else:
        if component_counts.get("internal_monologue_structure", 0):
            rows.extend(
                _build_internal_dpo_rows(
                    component_counts["internal_monologue_structure"],
                    split=split,
                    seed_start=seed,
                    trace_mode="monologue",
                    curriculum_profile=curriculum_profile,
                )
            )
            seed += component_counts["internal_monologue_structure"]
        if component_counts.get("external_answer_discipline", 0):
            rows.extend(_build_external_answer_discipline_pairs(component_counts["external_answer_discipline"], seed_start=seed, trace_mode=trace_mode))
            seed += component_counts["external_answer_discipline"]
        if component_counts.get("external_monologue_structure", 0):
            rows.extend(_build_external_monologue_structure_pairs(component_counts["external_monologue_structure"], seed_start=seed))
    return rows, component_counts


def _build_internal_sft_rows(
    count: int,
    *,
    split: str,
    seed_start: int,
    trace_mode: str,
    curriculum_profile: str,
) -> list[dict[str, Any]]:
    from .training_data import build_warmup_example, build_parser_completion

    if trace_mode == "debate":
        counts = warmup_mix_counts(count, curriculum_profile=curriculum_profile)
        order = [
            ("countdown", "easy"),
            ("countdown", "medium"),
            ("countdown", "hard"),
            ("evidence", "easy"),
            ("evidence", "medium"),
            ("evidence", "hard"),
        ]
        rows = []
        offset = 0
        for family, difficulty in order:
            block_count = counts[(family, difficulty)]
            for index in range(block_count):
                example = build_warmup_example(
                    family=family,
                    difficulty=difficulty,
                    seed=seed_start + offset + index,
                    split=split,
                    example_id=index,
                )
                completion = build_parser_completion(example)
                remapped = {
                    **example,
                    "messages": [
                        *example["messages"][:-1],
                        _assistant_message_structured(completion["reasoning_content"], completion["content"]),
                    ],
                }
                parser = SocietyOfThoughtParser(trace_mode="debate")
                parsed = parser.parse_completion([build_parser_completion(remapped)])
                if not (parsed.protocol_valid and parsed.trace_valid and parsed.answer_valid):
                    raise ValueError(f"Invalid internal debate example: {parsed.error_codes}")
                rows.append(remapped)
            offset += block_count
        return rows

    counts = warmup_mix_counts(count, curriculum_profile=curriculum_profile)
    rows = []
    offset = 0
    for family, difficulty in [
        ("countdown", "easy"),
        ("countdown", "medium"),
        ("countdown", "hard"),
        ("evidence", "easy"),
        ("evidence", "medium"),
        ("evidence", "hard"),
    ]:
        block_count = counts[(family, difficulty)]
        for index in range(block_count):
            row = build_example(
                family=family,
                difficulty=difficulty,
                institution="auto",
                seed=seed_start + offset + index,
                index=index,
                max_personas=4,
                max_debate_turns=10,
                trace_mode="monologue",
                split=split,
                trace_prompt_variant=DEFAULT_TRACE_PROMPT_VARIANT,
                countdown_reward_profile=DEFAULT_COUNTDOWN_REWARD_PROFILE,
                objective_profile=DEFAULT_OBJECTIVE_PROFILE,
            )
            info = json.loads(row["info"])
            reasoning = _internal_monologue_reasoning(family=family, task=info["task"], difficulty=difficulty)
            answer_text = _internal_answer_text(family=family, task=info["task"])
            example = {
                "messages": [*row["prompt"], _assistant_message_structured(reasoning, answer_text)],
                "family": family,
                "difficulty": difficulty,
                "seed": seed_start + offset + index,
                "source": "internal_sot",
                "trace_mode": "monologue",
            }
            parser = SocietyOfThoughtParser(trace_mode="monologue")
            parsed = parser.parse_completion([build_parser_completion(example)])
            if not (parsed.protocol_valid and parsed.trace_valid and parsed.answer_valid):
                raise ValueError(f"Invalid internal monologue example: {parsed.error_codes}")
            rows.append(example)
        offset += block_count
    return rows


def _build_internal_dpo_rows(
    count: int,
    *,
    split: str,
    seed_start: int,
    trace_mode: str,
    curriculum_profile: str,
) -> list[dict[str, Any]]:
    from .training_data import available_pair_types, build_dpo_pair_example, build_parser_completion

    rows = []
    order = [
        ("countdown", "easy"),
        ("countdown", "medium"),
        ("countdown", "hard"),
        ("evidence", "easy"),
        ("evidence", "medium"),
        ("evidence", "hard"),
    ]
    counts = warmup_mix_counts(count, curriculum_profile=curriculum_profile)
    offset = 0
    if trace_mode == "debate":
        for family, difficulty in order:
            block_count = counts[(family, difficulty)]
            pair_types = available_pair_types(difficulty)
            for index in range(block_count):
                example = build_dpo_pair_example(
                    family=family,
                    difficulty=difficulty,
                    seed=seed_start + offset + index,
                    split=split,
                    pair_type=pair_types[index % len(pair_types)],
                    example_id=index,
                )
                example["completion_A"] = _remap_completion_messages(example["completion_A"], build_parser_completion)
                example["completion_B"] = _remap_completion_messages(example["completion_B"], build_parser_completion)
                rows.append(example)
            offset += block_count
        return rows

    for family, difficulty in order:
        block_count = counts[(family, difficulty)]
        for index in range(block_count):
            seed = seed_start + offset + index
            chosen = _build_internal_monologue_single(family=family, difficulty=difficulty, split=split, seed=seed, example_id=index)
            rejected = _degrade_monologue_completion(chosen["messages"][-1], extra_suffix="\nI am adding filler instead of staying concise.")
            rows.append(
                {
                    "prompt_messages": chosen["messages"][:-1],
                    "completion_A": [chosen["messages"][-1]],
                    "completion_B": [rejected],
                    "label": "A",
                    "source": "internal_monologue_structure",
                    "family": family,
                    "difficulty": difficulty,
                    "trace_mode": "monologue",
                }
            )
        offset += block_count
    return rows


def _build_gsm8k_sft_rows(count: int, *, seed_start: int, trace_mode: str) -> list[dict[str, Any]]:
    dataset = load_dataset("gsm8k", "main", split="train")
    rows = []
    for offset, item in enumerate(_sample_rows(dataset, count=count, seed=seed_start)):
        prompt = _rewrite_prompt(
            [
                {"role": "system", "content": BOXED_SYSTEM_PROMPT},
                {"role": "user", "content": item["question"]},
            ],
            benchmark="gsm8k",
            trace_mode=trace_mode,
        )
        answer_value = str(item["answer"]).split("####")[-1].strip()
        answer_text = f"\\boxed{{{answer_value}}}"
        reasoning = _external_reasoning_text(
            "gsm8k",
            {"question": item["question"], "answer": answer_value},
            answer_text=answer_text,
            trace_mode=trace_mode,
            seed=seed_start + offset,
        )
        rows.append(
            {
                "messages": [*prompt, _assistant_message_structured(reasoning, answer_text)],
                "source": "gsm8k_train",
                "benchmark": "gsm8k",
                "trace_mode": trace_mode,
                "question": item["question"],
                "answer": answer_value,
                "native_answer_text": answer_text,
            }
        )
    return rows


def _build_mmlu_sft_rows(count: int, *, seed_start: int, trace_mode: str, stem_only: bool) -> list[dict[str, Any]]:
    dataset = load_dataset("TIGER-Lab/MMLU-Pro", "default", split="validation")
    rows_data = []
    for row in dataset:
        category = str(row.get("category", "")).lower()
        if stem_only and category not in STEM_CATEGORIES:
            continue
        if not stem_only or category in STEM_CATEGORIES:
            question = (
                "Please reason step by step, then ONLY give the letter of the correct answer within \\boxed{}.\n\n"
                + row["question"]
                + "\n\n"
                + "\n".join([f"{chr(65 + i)}. {option}" for i, option in enumerate(row["options"])])
            )
            rows_data.append(
                {
                    "prompt": [{"role": "user", "content": question}],
                    "question": question,
                    "answer": row["answer"],
                    "options": list(row["options"]),
                    "info": {"category": row.get("category", "")},
                }
            )
    source_name = "mmlu_pro_stem" if stem_only else "mmlu_pro_non_eval"
    rows = []
    for offset, item in enumerate(_sample_rows(rows_data, count=count, seed=seed_start)):
        prompt = _rewrite_prompt(item["prompt"], benchmark="mmlu_pro", trace_mode=trace_mode)
        answer_text = f"\\boxed{{{item['answer']}}}"
        reasoning = _external_reasoning_text("mmlu_pro", item, answer_text=answer_text, trace_mode=trace_mode, seed=seed_start + offset)
        rows.append(
            {
                "messages": [*prompt, _assistant_message_structured(reasoning, answer_text)],
                "source": source_name,
                "benchmark": "mmlu_pro",
                "trace_mode": trace_mode,
                "question": item["question"],
                "answer": item["answer"],
                "options": list(item["options"]),
                "info": dict(item.get("info", {})),
                "native_answer_text": answer_text,
            }
        )
    return rows


def _build_ifeval_synthetic_sft_rows(count: int, *, seed_start: int, trace_mode: str) -> list[dict[str, Any]]:
    rows = []
    for offset in range(count):
        prompt_text, answer_text, meta = _synthetic_ifeval_example(seed_start + offset)
        prompt = _rewrite_prompt([{"role": "user", "content": prompt_text}], benchmark="ifeval", trace_mode=trace_mode)
        reasoning = _external_reasoning_text("ifeval", meta, answer_text=answer_text, trace_mode=trace_mode, seed=seed_start + offset)
        rows.append(
            {
                "messages": [*prompt, _assistant_message_structured(reasoning, answer_text)],
                "source": "ifeval_synthetic",
                "benchmark": "ifeval",
                "trace_mode": trace_mode,
                "question": prompt_text,
                "answer": answer_text,
                "info": dict(meta),
                "native_answer_text": answer_text,
            }
        )
    return rows


def _build_external_answer_discipline_pairs(count: int, *, seed_start: int, trace_mode: str) -> list[dict[str, Any]]:
    sources = _mixed_external_sft_sources(count=count, seed_start=seed_start, trace_mode=trace_mode)
    rows = []
    for item in sources:
        chosen = item["messages"][-1]
        chosen_reasoning, chosen_answer_text = _split_assistant_content(chosen)
        wrong_text = _degrade_answer_format(item["benchmark"], chosen_answer_text, seed=seed_start + len(rows))
        rows.append(
            {
                "prompt_messages": item["messages"][:-1],
                "completion_A": [chosen],
                "completion_B": [_assistant_message_structured(chosen_reasoning, wrong_text)],
                "label": "A",
                "source": "external_answer_discipline",
                "benchmark": item["benchmark"],
                "trace_mode": trace_mode,
            }
        )
    return rows


def _build_external_reasoning_structure_pairs(count: int, *, seed_start: int, trace_mode: str) -> list[dict[str, Any]]:
    sources = _mixed_external_sft_sources(count=count, seed_start=seed_start, trace_mode=trace_mode)
    rows = []
    rejection_variants = (
        "shallow_valid",
        "redundant_cast",
        "weak_challenge",
        "premature_reconcile",
    )
    for item in sources:
        chosen = item["messages"][-1]
        chosen_reasoning, answer_text = _split_assistant_content(chosen)
        rejection_variant = rejection_variants[len(rows) % len(rejection_variants)]
        rejected_reasoning = _degrade_external_reasoning_structure(
            item["benchmark"],
            item,
            chosen_reasoning=chosen_reasoning,
            answer_text=answer_text,
            seed=seed_start + len(rows),
            variant=rejection_variant,
        )
        rows.append(
            {
                "prompt_messages": item["messages"][:-1],
                "completion_A": [chosen],
                "completion_B": [_assistant_message_structured(rejected_reasoning, answer_text)],
                "label": "A",
                "source": "external_reasoning_structure",
                "benchmark": item["benchmark"],
                "trace_mode": trace_mode,
                "rejection_variant": rejection_variant,
            }
        )
    return rows


def _build_external_monologue_structure_pairs(count: int, *, seed_start: int) -> list[dict[str, Any]]:
    sources = _mixed_external_sft_sources(count=count, seed_start=seed_start, trace_mode="monologue")
    rows = []
    for item in sources:
        chosen = item["messages"][-1]
        _, answer_text = _split_assistant_content(chosen)
        dialogue_reasoning = _external_reasoning_text(item["benchmark"], {"question": "", "answer": answer_text}, answer_text=answer_text, trace_mode="debate", seed=seed_start)
        rows.append(
            {
                "prompt_messages": item["messages"][:-1],
                "completion_A": [chosen],
                "completion_B": [_assistant_message_structured(dialogue_reasoning, answer_text)],
                "label": "A",
                "source": "external_monologue_structure",
                "benchmark": item["benchmark"],
                "trace_mode": "monologue",
            }
        )
    return rows


def _mixed_external_sft_sources(*, count: int, seed_start: int, trace_mode: str) -> list[dict[str, Any]]:
    counts = {"gsm8k_train": max(1, count // 3), "mmlu_pro_non_eval": max(1, count // 3), "ifeval_synthetic": count - 2 * max(1, count // 3)}
    rows: list[dict[str, Any]] = []
    rows.extend(_build_gsm8k_sft_rows(counts["gsm8k_train"], seed_start=seed_start, trace_mode=trace_mode))
    rows.extend(_build_mmlu_sft_rows(counts["mmlu_pro_non_eval"], seed_start=seed_start + 5_000, trace_mode=trace_mode, stem_only=False))
    rows.extend(_build_ifeval_synthetic_sft_rows(counts["ifeval_synthetic"], seed_start=seed_start + 10_000, trace_mode=trace_mode))
    return rows[:count]


def _build_internal_monologue_single(*, family: str, difficulty: str, split: str, seed: int, example_id: int) -> dict[str, Any]:
    row = build_example(
        family=family,
        difficulty=difficulty,
        institution="auto",
        seed=seed,
        index=example_id,
        max_personas=4,
        max_debate_turns=10,
        trace_mode="monologue",
        split=split,
        trace_prompt_variant=DEFAULT_TRACE_PROMPT_VARIANT,
        countdown_reward_profile=DEFAULT_COUNTDOWN_REWARD_PROFILE,
        objective_profile=DEFAULT_OBJECTIVE_PROFILE,
    )
    info = json.loads(row["info"])
    reasoning = _internal_monologue_reasoning(family=family, task=info["task"], difficulty=difficulty)
    answer_text = _internal_answer_text(family=family, task=info["task"])
    return {
        "messages": [*row["prompt"], _assistant_message_structured(reasoning, answer_text)],
        "family": family,
        "difficulty": difficulty,
        "seed": seed,
        "source": "internal_sot",
        "trace_mode": "monologue",
    }


def _internal_monologue_reasoning(*, family: str, task: dict[str, Any], difficulty: str) -> str:
    if family == "countdown":
        expression = task["oracle_expression"]
        target = task["target"]
        if difficulty == "hard":
            return (
                f"First isolate a clean exact path to the target {target}. "
                f"A tempting branch can be discarded because it breaks legal input use. "
                f"The surviving path is {expression}, which evaluates to {target} exactly."
            )
        return (
            f"Use the legal arithmetic path that reaches the target exactly. "
            f"The candidate {expression} respects the allowed inputs and evaluates to {target}."
        )
    support = ", ".join(task["oracle_support"])
    verdict = task["oracle_verdict"]
    return (
        f"Read the evidence chronologically, separate decisive support from distractors, and keep only the grounded verdict. "
        f"The evidence IDs {support} are sufficient for the final verdict {verdict}."
    )


def _internal_answer_text(*, family: str, task: dict[str, Any]) -> str:
    if family == "countdown":
        return f"<answer>{task['oracle_expression']}</answer>"
    return f"<answer>{task['oracle_verdict']}</answer>\n<support>{','.join(task['oracle_support'])}</support>"


def _assistant_message_structured(reasoning_text: str, answer_text: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": _compose_assistant_content(reasoning_text, answer_text),
    }


def _rewrite_prompt(prompt: Sequence[dict[str, Any]], *, benchmark: str, trace_mode: str) -> list[dict[str, Any]]:
    rewritten = [dict(message) for message in prompt]
    if rewritten and rewritten[0].get("role") == "system":
        original = str(rewritten[0].get("content", ""))
        rewritten[0]["content"] = build_external_system_prompt(benchmark, trace_mode=trace_mode, original_system_prompt=original)
        return rewritten
    return [{"role": "system", "content": build_external_system_prompt(benchmark, trace_mode=trace_mode)}, *rewritten]


def _external_reasoning_text(
    benchmark: str,
    row: dict[str, Any],
    *,
    answer_text: str,
    trace_mode: str,
    seed: int,
    quality: str = "rich",
) -> str:
    if trace_mode == "monologue":
        focus = _question_focus_snippet(benchmark, row)
        format_rule = _visible_answer_rule(benchmark, answer_text=answer_text, row=row)
        return (
            f"Focus on the real task first: {focus}. "
            f"Reject any path that would violate the answer discipline. {format_rule} "
            f"The final answer is {answer_text}."
        )
    if quality == "shallow":
        return _shallow_external_debate_reasoning(benchmark, row, answer_text=answer_text, seed=seed)
    payload = _external_debate_trace_payload(benchmark, row, answer_text=answer_text, seed=seed)
    return _wrap_trace_payload(payload)


def _persona_specs_for_benchmark(
    benchmark: str,
    rng: random.Random,
) -> list[tuple[str, str, str, str]]:
    bank = {
        "gsm8k": [
            ("planner", "high_openness", "problem decomposition", "Frames candidate arithmetic routes before committing."),
            ("skeptic", "low_agreeableness", "error checking", "Pushes on dropped numbers, illegal shortcuts, and sign errors."),
            ("verifier", "high_conscientiousness", "exact arithmetic", "Recomputes the surviving route and checks the boxed number."),
            ("synthesizer", "high_agreeableness", "decision making", "Collapses the surviving route into the final visible answer."),
        ],
        "mmlu_pro": [
            ("analyst", "high_openness", "concept selection", "Names the leading option and the clue that supports it."),
            ("contrarian", "low_agreeableness", "distractor analysis", "Pushes on the strongest nearby option before committing."),
            ("verifier", "high_conscientiousness", "evidence checking", "Checks the surviving option against the prompt details."),
            ("editor", "high_agreeableness", "format discipline", "Ensures the final output is only one boxed letter."),
        ],
        "gpqa": [
            ("analyst", "high_openness", "scientific reasoning", "Finds the leading answer and the key differentiating fact."),
            ("contrarian", "low_agreeableness", "distractor analysis", "Pushes on the strongest competing option."),
            ("verifier", "high_conscientiousness", "consistency checking", "Checks that the surviving option still matches the question."),
            ("editor", "high_agreeableness", "format discipline", "Ensures the visible answer is only one boxed letter."),
        ],
        "ifeval": [
            ("planner", "high_openness", "instruction planning", "Lists the explicit constraints before drafting."),
            ("violation_hunter", "low_agreeableness", "constraint auditing", "Looks for the easiest way the response could fail."),
            ("verifier", "high_conscientiousness", "rule checking", "Checks the draft against each instruction one by one."),
            ("editor", "high_agreeableness", "response polishing", "Keeps the visible answer minimal while preserving compliance."),
        ],
    }[benchmark][:]
    rng.shuffle(bank)
    return bank


def _external_debate_trace_payload(
    benchmark: str,
    row: dict[str, Any],
    *,
    answer_text: str,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    persona_specs = _persona_specs_for_benchmark(benchmark, rng)
    persona_count = 4 if benchmark == "ifeval" or seed % 3 == 0 else 3
    selected_personas = persona_specs[:persona_count]
    personas = [
        {
            "id": f"P{index}",
            "role": role,
            "personality": personality,
            "expertise": expertise,
            "style": style,
        }
        for index, (role, personality, expertise, style) in enumerate(selected_personas, start=1)
    ]
    turns = _external_turns_for_benchmark(
        benchmark,
        row=row,
        personas=personas,
        answer_text=answer_text,
        seed=seed,
    )
    return {
        "dialect": select_trace_dialect(seed),
        "personas": personas,
        "debate": turns,
        "group_solution": _group_solution_line(benchmark, answer_text=answer_text, row=row),
    }


def _question_focus_snippet(benchmark: str, row: dict[str, Any]) -> str:
    if benchmark == "ifeval":
        prompt = str(row.get("question", "") or row.get("prompt", ""))
        first_sentence = re.split(r"(?<=[.!?])\s+", prompt.strip(), maxsplit=1)[0]
        return _squash_ws(first_sentence)[:160]
    question = str(row.get("question", "") or row.get("prompt", ""))
    lines = [line.strip() for line in question.splitlines() if line.strip()]
    if not lines:
        return benchmark
    focus = lines[0]
    if focus.lower().startswith("please reason step by step") and len(lines) > 1:
        focus = lines[1]
    return _squash_ws(focus)[:160]


def _external_turns_for_benchmark(
    benchmark: str,
    *,
    row: dict[str, Any],
    personas: list[dict[str, Any]],
    answer_text: str,
    seed: int,
) -> list[dict[str, str]]:
    if benchmark == "gsm8k":
        return _gsm8k_turns(row=row, personas=personas, seed=seed)
    if benchmark in {"mmlu_pro", "gpqa"}:
        return _mcq_turns(benchmark=benchmark, row=row, personas=personas, seed=seed)
    return _ifeval_turns(row=row, personas=personas, seed=seed, answer_text=answer_text)


def _visible_answer_rule(benchmark: str, *, answer_text: str, row: dict[str, Any]) -> str:
    if benchmark == "gsm8k":
        return f"Keep the visible answer to one boxed number only: {answer_text}."
    if benchmark in {"mmlu_pro", "gpqa"}:
        return f"Keep the visible answer to one boxed letter only: {answer_text}."
    constraints = _ifeval_constraints(str(row.get("question", "") or row.get("prompt", "")))
    if constraints:
        return "Keep the visible answer constrained by: " + "; ".join(constraints[:4]) + "."
    return "Keep the visible answer limited to the requested instruction-following response."


def _gsm8k_turns(*, row: dict[str, Any], personas: list[dict[str, Any]], seed: int) -> list[dict[str, str]]:
    focus = _question_focus_snippet("gsm8k", row)
    quantities = _extract_numeric_tokens(str(row.get("question", "")))
    cited = ", ".join(quantities[:4]) if quantities else "the cited quantities"
    dropped = quantities[-1] if quantities else "a remaining quantity"
    question = f"Question: Which quantity-preserving route for '{focus}' uses {cited} without inventing new values?"
    proposals = [
        f"Primary proposal: Start from the quantities {cited} and combine them in the story order before simplifying.",
        f"Primary proposal: Test the branch that keeps {cited} visible all the way to the last operation.",
        f"Primary proposal: Try the route that groups the largest quantity first but still keeps every cited value in play.",
    ]
    challenges = [
        f"Challenge: That branch could quietly drop {dropped} or change the order of operations in a way the story does not support.",
        f"Challenge: A shortcut might reach a neat number while still reusing one quantity or skipping {dropped}.",
        f"Challenge: The leading route only survives if each step still accounts for {cited} in a legal way.",
    ]
    responses = [
        "Response: Compare it with an alternative branch that keeps every quantity explicit and avoids the shortcut failure.",
        "Alternative path: Keep the untouched quantity available longer and reject any branch that hides a dropped term.",
        "Response: The safer branch is the one that still makes each quantity traceable before the final arithmetic check.",
    ]
    verifications = [
        "Verification: Keep only the branch that uses the cited quantities legally and then emit one boxed number.",
        "Verification: Recompute the surviving route from the listed quantities and preserve the boxed-number format.",
        "Verification: The visible answer should be a single boxed number after the legal route is checked once more.",
    ]
    reconciles = [
        "Reconcile: Reject the shortcut branch and keep the quantity-preserving route for the visible answer.",
        "Reconcile: Keep the checked branch that survives the dropped-quantity challenge.",
        "Reconcile: Use the route that still respects every cited quantity and nothing else.",
    ]
    return _assemble_external_turns(
        personas,
        question=question,
        propose=proposals[seed % len(proposals)],
        challenge=challenges[(seed + 1) % len(challenges)],
        response=responses[(seed + 2) % len(responses)],
        verify=verifications[(seed + 3) % len(verifications)],
        reconcile=reconciles[(seed + 4) % len(reconciles)],
    )


def _mcq_turns(
    *,
    benchmark: str,
    row: dict[str, Any],
    personas: list[dict[str, Any]],
    seed: int,
) -> list[dict[str, str]]:
    focus = _question_focus_snippet(benchmark, row)
    correct_label = str(row.get("answer", "")).strip().upper()
    distractor_label = _strong_distractor_label(row) or "B"
    lead_text = _option_snippet(row, correct_label)
    distractor_text = _option_snippet(row, distractor_label)
    question = f"Question: Which option best matches the decisive clue in '{focus}'?"
    proposals = [
        f"Primary proposal: Test the option summarized as '{lead_text}' against the stem before looking at nearby distractors.",
        f"Primary proposal: Start from the option whose wording '{lead_text}' seems to explain the question most directly.",
        f"Primary proposal: Use the stem clue to evaluate the option paraphrased as '{lead_text}' first.",
    ]
    challenges = [
        f"Challenge: The distractor '{distractor_text}' can sound close unless one detail in the stem rules it out.",
        f"Challenge: A nearby answer like '{distractor_text}' could steal the lead if the key clue is read too loosely.",
        f"Challenge: The leading reading only survives if it beats the distractor '{distractor_text}' on the decisive clue.",
    ]
    responses = [
        "Response: Keep the interpretation that explains the key clue and drop the distractor that only matches superficially.",
        "Alternative path: Re-read the stem through the distractor, then return to the leading option if the clue still points away from it.",
        "Response: The surviving option is the one that still fits after the strongest distractor has been pressure-tested.",
    ]
    verifications = [
        "Verification: After the distractor check, emit only one boxed letter in the visible answer.",
        "Verification: The visible answer should be a single boxed letter and nothing else.",
        "Verification: Keep the final output to the surviving boxed letter only.",
    ]
    reconciles = [
        "Reconcile: Keep the option that survives the distractor check and discard the tempting nearby answer.",
        "Reconcile: Settle on the stem-consistent option and keep the visible answer to one boxed letter.",
        "Reconcile: Use the option that still fits the decisive clue after the comparison.",
    ]
    return _assemble_external_turns(
        personas,
        question=question,
        propose=proposals[seed % len(proposals)],
        challenge=challenges[(seed + 1) % len(challenges)],
        response=responses[(seed + 2) % len(responses)],
        verify=verifications[(seed + 3) % len(verifications)],
        reconcile=reconciles[(seed + 4) % len(reconciles)],
    )


def _ifeval_turns(
    *,
    row: dict[str, Any],
    personas: list[dict[str, Any]],
    seed: int,
    answer_text: str,
) -> list[dict[str, str]]:
    prompt = str(row.get("question", "") or row.get("prompt", ""))
    constraints = _ifeval_constraints(prompt)
    head = constraints[:4] if constraints else ["the explicit formatting constraints"]
    first = head[0]
    second = head[1] if len(head) > 1 else head[0]
    third = head[2] if len(head) > 2 else head[-1]
    question = f"Question: Which response plan satisfies {first} while still preserving {second}?"
    proposals = [
        f"Primary proposal: Draft only the sections needed to satisfy {first} and {second} before polishing anything else.",
        f"Primary proposal: Build the visible response around {first} first, then check {second} before adding wording.",
        f"Primary proposal: Use the smallest response plan that still satisfies {first} and keeps the structure explicit.",
    ]
    challenges = [
        f"Challenge: That draft can still fail by violating {third} or by adding extra text after the requested response.",
        f"Challenge: A cleaner-sounding draft is still wrong if it breaks {third} or leaves stray prose around the answer.",
        f"Challenge: The plan only survives if it respects {third} while keeping the visible answer minimal.",
    ]
    responses = [
        "Response: Switch to the safer draft that preserves the named constraints even if it sounds plainer.",
        "Alternative path: Trim the draft harder so the user-facing response carries only the requested structure.",
        "Response: Keep the minimal compliant draft and drop anything that looks like an explanation.",
    ]
    verifications = [
        "Verification: Check each named instruction once more and leave only the user-facing response.",
        "Verification: The final output should satisfy the listed constraints without any extra note or wrapper.",
        "Verification: Keep the visible response minimal and make sure no explanatory tail remains.",
    ]
    reconciles = [
        "Reconcile: Prefer the stricter compliant draft over the more expressive one.",
        "Reconcile: Keep the constraint-satisfying draft and remove any extra sentence.",
        "Reconcile: Use the minimal compliant response plan for the visible answer.",
    ]
    turns = _assemble_external_turns(
        personas,
        question=question,
        propose=proposals[seed % len(proposals)],
        challenge=challenges[(seed + 1) % len(challenges)],
        response=responses[(seed + 2) % len(responses)],
        verify=verifications[(seed + 3) % len(verifications)],
        reconcile=reconciles[(seed + 4) % len(reconciles)],
    )
    if answer_text and seed % 2 == 0:
        turns[-1]["content"] += " Keep the visible output separate from the hidden discussion."
    return turns


def _assemble_external_turns(
    personas: list[dict[str, Any]],
    *,
    question: str,
    propose: str,
    challenge: str,
    response: str,
    verify: str,
    reconcile: str,
) -> list[dict[str, str]]:
    ids = [persona["id"] for persona in personas]
    if len(ids) >= 4:
        return [
            {"speaker": ids[0], "act": "question", "content": question},
            {"speaker": ids[1], "act": "propose", "content": propose},
            {"speaker": ids[2], "act": "challenge", "content": challenge},
            {"speaker": ids[1], "act": "shift", "content": response},
            {"speaker": ids[3], "act": "verify", "content": verify},
            {"speaker": ids[0], "act": "reconcile", "content": reconcile},
        ]
    return [
        {"speaker": ids[0], "act": "question", "content": question},
        {"speaker": ids[1], "act": "propose", "content": propose},
        {"speaker": ids[2], "act": "challenge", "content": challenge},
        {"speaker": ids[1], "act": "verify", "content": response},
        {"speaker": ids[2], "act": "reconcile", "content": verify + " " + reconcile},
    ]


def _extract_numeric_tokens(text: str) -> list[str]:
    return ordered_unique(re.findall(r"[-+]?\d+(?:\.\d+)?", text or ""))


def _option_snippet(row: dict[str, Any], label: str) -> str:
    options = row.get("options")
    if not isinstance(options, list):
        return "the leading reading"
    index = ord(label.upper()) - ord("A")
    if index < 0 or index >= len(options):
        return "the nearby reading"
    words = _squash_ws(str(options[index])).split()
    return " ".join(words[:8]) if words else "the nearby reading"


def _candidate_claim(benchmark: str, *, answer_text: str, row: dict[str, Any]) -> str:
    focus = _question_focus_snippet(benchmark, row)
    if benchmark == "gsm8k":
        return f"Anchor on the quantities in '{focus}' and find an exact route that lands on {answer_text}."
    if benchmark in {"mmlu_pro", "gpqa"}:
        return f"Start from the leading option {answer_text} for '{focus}' and justify why it beats the nearest distractor."
    constraints = _ifeval_constraints(str(row.get('question', '') or row.get('prompt', '')))
    if constraints:
        return "Start from a response plan that satisfies " + "; ".join(constraints[:3]) + "."
    return f"Start from the response plan that best matches '{focus}'."


def _challenge_line(benchmark: str, *, row: dict[str, Any]) -> str:
    focus = _question_focus_snippet(benchmark, row)
    if benchmark == "gsm8k":
        return f"A shortcut for '{focus}' could quietly drop a quantity, reuse one twice, or change the arithmetic order."
    if benchmark in {"mmlu_pro", "gpqa"}:
        distractor = _strong_distractor_label(row)
        if distractor:
            return f"The tempting distractor {distractor} could look plausible unless we isolate the one clue that rules it out."
        return "A nearby distractor could look plausible unless we isolate the one clue that rules it out."
    constraints = _ifeval_constraints(str(row.get("question", "") or row.get("prompt", "")))
    if constraints:
        return "The draft could fail by violating: " + "; ".join(constraints[:2]) + "."
    return "The draft could sound right while still breaking one of the requested constraints."


def _response_line(benchmark: str, *, answer_text: str, row: dict[str, Any]) -> str:
    focus = _question_focus_snippet(benchmark, row)
    if benchmark == "gsm8k":
        return f"Discard the shortcut and keep only the route that uses the stated quantities from '{focus}' legally before boxing {answer_text}."
    if benchmark in {"mmlu_pro", "gpqa"}:
        return f"The alternative falls away once we check the prompt details carefully, so the leading option {answer_text} survives."
    constraints = _ifeval_constraints(str(row.get("question", "") or row.get("prompt", "")))
    if constraints:
        return "Use the safer wording path that satisfies every named constraint, even if it is less expressive."
    return "Use the safer wording path that satisfies the prompt instead of the flashier but riskier one."


def _verification_line(benchmark: str, *, answer_text: str, format_rule: str) -> str:
    if benchmark == "gsm8k":
        return f"Recompute the final arithmetic once more and then emit exactly {answer_text}. {format_rule}"
    if benchmark in {"mmlu_pro", "gpqa"}:
        return f"Check that the surviving choice is still correct and then emit exactly {answer_text}. {format_rule}"
    return format_rule + " Remove any extra note, explanation, or trailing text."


def _reconcile_line(benchmark: str, *, answer_text: str, row: dict[str, Any]) -> str:
    if benchmark == "ifeval":
        return "Prefer the compliant response plan over the more verbose one and keep only the user-facing answer."
    return f"Keep the route that survived the challenge, reject the tempting alternative, and commit to {answer_text}."


def _group_solution_line(benchmark: str, *, answer_text: str, row: dict[str, Any]) -> str:
    if benchmark == "gsm8k":
        return "Use the checked arithmetic route and emit one boxed number only."
    if benchmark in {"mmlu_pro", "gpqa"}:
        return "Use the surviving option and emit one boxed letter only."
    return "Use the compliant response plan and emit only the requested user-facing answer."


def _shallow_external_debate_reasoning(
    benchmark: str,
    row: dict[str, Any],
    *,
    answer_text: str,
    seed: int,
) -> str:
    rng = random.Random(seed)
    selected = _persona_specs_for_benchmark(benchmark, rng)[:3]
    personas = [
        {
            "id": f"P{index}",
            "role": role,
            "personality": personality,
            "expertise": expertise,
            "style": style,
        }
        for index, (role, personality, expertise, style) in enumerate(selected, start=1)
    ]
    focus = _question_focus_snippet(benchmark, row)
    payload = {
        "dialect": select_trace_dialect(seed),
        "personas": personas,
        "debate": [
            {
                "speaker": "P1",
                "act": "question",
                "content": f"Question: We should answer '{focus}' while keeping the visible output concise.",
            },
            {
                "speaker": "P2",
                "act": "propose",
                "content": "Primary proposal: Keep the first plausible route and avoid a long side discussion.",
            },
            {
                "speaker": "P3",
                "act": "verify",
                "content": "Verification: Keep the visible answer in the native format and do not add extra text.",
            },
        ],
        "group_solution": "Keep the concise route and emit only the visible answer.",
    }
    return _wrap_trace_payload(payload)


def _degrade_external_reasoning_structure(
    benchmark: str,
    row: dict[str, Any],
    *,
    chosen_reasoning: str,
    answer_text: str,
    seed: int,
    variant: str,
) -> str:
    if variant == "shallow_valid":
        return _external_reasoning_text(
            benchmark,
            row,
            answer_text=answer_text,
            trace_mode="debate",
            seed=seed,
            quality="shallow",
        )
    if variant == "redundant_cast":
        payload = _soft_external_debate_trace_payload(
            benchmark,
            row=row,
            answer_text=answer_text,
            seed=seed,
            variant=variant,
        )
        return _wrap_trace_payload(payload)
    if variant == "weak_challenge":
        payload = _soft_external_debate_trace_payload(
            benchmark,
            row=row,
            answer_text=answer_text,
            seed=seed,
            variant=variant,
        )
        return _wrap_trace_payload(payload)
    if variant == "premature_reconcile":
        payload = _soft_external_debate_trace_payload(
            benchmark,
            row=row,
            answer_text=answer_text,
            seed=seed,
            variant=variant,
        )
        return _wrap_trace_payload(payload)
    raise KeyError(f"Unknown external reasoning rejection variant: {variant}")


def _soft_external_debate_trace_payload(
    benchmark: str,
    row: dict[str, Any],
    *,
    answer_text: str,
    seed: int,
    variant: str,
) -> dict[str, Any]:
    personas = _soft_external_personas(benchmark, seed=seed, variant=variant)
    turns = _soft_external_turns(
        benchmark,
        row=row,
        personas=personas,
        answer_text=answer_text,
        seed=seed,
        variant=variant,
    )
    return {
        "dialect": select_trace_dialect(seed),
        "personas": personas,
        "debate": turns,
        "group_solution": _group_solution_line(benchmark, answer_text=answer_text, row=row),
    }


def _soft_external_personas(
    benchmark: str,
    *,
    seed: int,
    variant: str,
) -> list[dict[str, Any]]:
    if variant == "redundant_cast":
        labels_by_benchmark = {
            "gsm8k": ("solver", "solver_2", "solver_3"),
            "mmlu_pro": ("reader", "reader_2", "reader_3"),
            "gpqa": ("analyst", "analyst_2", "analyst_3"),
            "ifeval": ("drafter", "drafter_2", "drafter_3"),
        }
        expertise_by_benchmark = {
            "gsm8k": "arithmetic checking",
            "mmlu_pro": "option reading",
            "gpqa": "option reading",
            "ifeval": "instruction checking",
        }
        labels = labels_by_benchmark[benchmark]
        expertise = expertise_by_benchmark[benchmark]
        return [
            {
                "id": f"P{index}",
                "role": label,
                "personality": "high_openness",
                "expertise": expertise,
                "style": "Restates the route with only minor variation.",
            }
            for index, label in enumerate(labels, start=1)
        ]
    rng = random.Random(seed)
    selected = _persona_specs_for_benchmark(benchmark, rng)[:3]
    return [
        {
            "id": f"P{index}",
            "role": role,
            "personality": personality,
            "expertise": expertise,
            "style": style,
        }
        for index, (role, personality, expertise, style) in enumerate(selected, start=1)
    ]


def _soft_external_turns(
    benchmark: str,
    *,
    row: dict[str, Any],
    personas: list[dict[str, Any]],
    answer_text: str,
    seed: int,
    variant: str,
) -> list[dict[str, str]]:
    focus = _question_focus_snippet(benchmark, row)
    format_rule = _soft_visible_answer_rule(benchmark, row=row)
    if benchmark == "gsm8k":
        anchor = ", ".join(_extract_numeric_tokens(str(row.get("question", "")))[:4]) or "the stated quantities"
        if variant == "redundant_cast":
            return [
                {"speaker": personas[0]["id"], "act": "question", "content": f"Question: Which route for '{focus}' keeps {anchor} visible?"},
                {"speaker": personas[1]["id"], "act": "propose", "content": f"Proposal: Keep the route that seems to preserve {anchor}."},
                {"speaker": personas[2]["id"], "act": "verify", "content": "Verification: Recheck the arithmetic once before the boxed number."},
                {"speaker": personas[0]["id"], "act": "reconcile", "content": "Reconcile: Use the route that already looks consistent."},
            ]
        if variant == "weak_challenge":
            return [
                {"speaker": personas[0]["id"], "act": "question", "content": f"Question: Which arithmetic path best answers '{focus}'?"},
                {"speaker": personas[1]["id"], "act": "propose", "content": f"Proposal: Start from the branch that keeps {anchor} in view."},
                {"speaker": personas[2]["id"], "act": "challenge", "content": "Challenge: We should double-check a detail before committing."},
                {"speaker": personas[1]["id"], "act": "verify", "content": f"Verification: {format_rule}"},
                {"speaker": personas[0]["id"], "act": "reconcile", "content": "Reconcile: The original route still seems acceptable after a quick check."},
            ]
        return [
            {"speaker": personas[0]["id"], "act": "question", "content": f"Question: Which route best matches '{focus}' while keeping {anchor} legal?"},
            {"speaker": personas[1]["id"], "act": "propose", "content": f"Proposal: Start from the branch that keeps {anchor} explicit."},
            {"speaker": personas[2]["id"], "act": "challenge", "content": "Challenge: A shortcut could still be risky, but the leading route already looks strong."},
            {"speaker": personas[0]["id"], "act": "reconcile", "content": f"Reconcile: Keep the leading route and move to the visible answer. {format_rule}"},
        ]
    if benchmark in {"mmlu_pro", "gpqa"}:
        lead = _option_snippet(row, str(row.get("answer", "")).strip().upper())
        distractor = _option_snippet(row, _strong_distractor_label(row) or "B")
        if variant == "redundant_cast":
            return [
                {"speaker": personas[0]["id"], "act": "question", "content": f"Question: Which option best matches '{focus}'?"},
                {"speaker": personas[1]["id"], "act": "propose", "content": f"Proposal: Keep the option summarized as '{lead}' in front."},
                {"speaker": personas[2]["id"], "act": "verify", "content": "Verification: Keep the visible output to one boxed letter only."},
                {"speaker": personas[0]["id"], "act": "reconcile", "content": "Reconcile: Stay with the leading option unless something obviously breaks it."},
            ]
        if variant == "weak_challenge":
            return [
                {"speaker": personas[0]["id"], "act": "question", "content": f"Question: Which option best fits '{focus}'?"},
                {"speaker": personas[1]["id"], "act": "propose", "content": f"Proposal: Start from the option summarized as '{lead}'."},
                {"speaker": personas[2]["id"], "act": "challenge", "content": f"Challenge: The nearby answer '{distractor}' is worth a quick glance."},
                {"speaker": personas[1]["id"], "act": "verify", "content": "Verification: Emit one boxed letter only."},
                {"speaker": personas[0]["id"], "act": "reconcile", "content": "Reconcile: Keep the leading option after the light distractor check."},
            ]
        return [
            {"speaker": personas[0]["id"], "act": "question", "content": f"Question: Which option still survives for '{focus}'?"},
            {"speaker": personas[1]["id"], "act": "propose", "content": f"Proposal: The leading option is still '{lead}'."},
            {"speaker": personas[2]["id"], "act": "challenge", "content": f"Challenge: The distractor '{distractor}' is close, but not close enough to force a branch change."},
            {"speaker": personas[0]["id"], "act": "reconcile", "content": "Reconcile: Keep the leading option and box the letter."},
        ]
    constraints = _ifeval_constraints(str(row.get("question", "") or row.get("prompt", "")))
    first = constraints[0] if constraints else "the explicit instructions"
    second = constraints[1] if len(constraints) > 1 else first
    if variant == "redundant_cast":
        return [
            {"speaker": personas[0]["id"], "act": "question", "content": f"Question: Which draft best preserves {first}?"},
            {"speaker": personas[1]["id"], "act": "propose", "content": f"Proposal: Keep the smallest draft that satisfies {first}."},
            {"speaker": personas[2]["id"], "act": "verify", "content": "Verification: Leave only the requested user-facing response."},
            {"speaker": personas[0]["id"], "act": "reconcile", "content": "Reconcile: Use the compact compliant draft."},
        ]
    if variant == "weak_challenge":
        return [
            {"speaker": personas[0]["id"], "act": "question", "content": f"Question: Which draft satisfies {first} and {second}?"},
            {"speaker": personas[1]["id"], "act": "propose", "content": f"Proposal: Draft around {first} first."},
            {"speaker": personas[2]["id"], "act": "challenge", "content": "Challenge: Check one more instruction before we stop."},
            {"speaker": personas[1]["id"], "act": "verify", "content": "Verification: Keep the visible output minimal and instruction-only."},
            {"speaker": personas[0]["id"], "act": "reconcile", "content": "Reconcile: The first compliant draft is probably good enough."},
        ]
    return [
        {"speaker": personas[0]["id"], "act": "question", "content": f"Question: Which draft best fits '{focus}'?"},
        {"speaker": personas[1]["id"], "act": "propose", "content": f"Proposal: Use the draft that satisfies {first} first."},
        {"speaker": personas[2]["id"], "act": "challenge", "content": "Challenge: Another wording might sound nicer, but the current draft is already close to compliant."},
        {"speaker": personas[0]["id"], "act": "reconcile", "content": "Reconcile: Keep the first draft and move on to the visible response."},
    ]


def _soft_visible_answer_rule(benchmark: str, *, row: dict[str, Any]) -> str:
    if benchmark == "gsm8k":
        return "Keep the visible answer to one boxed number only."
    if benchmark in {"mmlu_pro", "gpqa"}:
        return "Keep the visible answer to one boxed letter only."
    constraints = _ifeval_constraints(str(row.get("question", "") or row.get("prompt", "")))
    if constraints:
        return "Keep the visible response constrained by: " + "; ".join(constraints[:3]) + "."
    return "Keep the visible response limited to the requested instruction-following answer."


def _strong_distractor_label(row: dict[str, Any]) -> str | None:
    answer = str(row.get("answer", "")).strip().upper()
    options = row.get("options")
    if not isinstance(options, list):
        return None
    labels = [chr(65 + index) for index in range(len(options))]
    for label in labels:
        if label != answer:
            return label
    return None


def _ifeval_constraints(prompt: str) -> list[str]:
    lowered = prompt.lower()
    constraints: list[str] = []
    if "exactly three short sections" in lowered:
        constraints.append("write exactly three short sections")
    match = re.search(r"end with the exact word ([A-Z]+)", prompt)
    if match:
        constraints.append(f"end with the exact word {match.group(1)}")
    if "single asterisks" in lowered:
        constraints.append("wrap each section title in single asterisks")
    if "no commas" in lowered:
        constraints.append("use no commas")
    return constraints


def _squash_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _synthetic_ifeval_example(seed: int) -> tuple[str, str, dict[str, Any]]:
    rng = random.Random(seed)
    topic = rng.choice(["lighthouses", "urban gardens", "paper maps", "astronomy clubs"])
    adjective = rng.choice(["brief", "clear", "compact", "structured"])
    title_one = rng.choice(["Overview", "Why it matters", "Key steps"])
    title_two = rng.choice(["Use cases", "Common mistakes", "Takeaway"])
    title_three = rng.choice(["Examples", "Checklist", "Final note"])
    prompt = (
        f"Write exactly three short sections about {topic}. Each section title must be wrapped in single asterisks, "
        f"for example *{title_one}*. Use no commas. End with the exact word DONE."
    )
    answer = (
        f"*{title_one}*\n{topic.title()} are {adjective} to explain in plain language.\n\n"
        f"*{title_two}*\nThey reward careful wording and simple structure for the reader.\n\n"
        f"*{title_three}*\nA short compliant response is easier to verify and compare.\nDONE"
    )
    return prompt, answer, {"question": prompt, "answer": answer, "benchmark": "ifeval"}


def _degrade_answer_format(benchmark: str, answer_text: str, *, seed: int) -> str:
    key = normalize_benchmark_name(benchmark)
    rng = random.Random(seed)
    if key == "gsm8k":
        boxed = extract_boxed_answer(answer_text) or answer_text
        variants = [
            boxed,
            f"The answer is {boxed}.",
            f"{answer_text}\nBecause that is the computed value.",
        ]
        return rng.choice(variants)
    if key in {"mmlu_pro", "gpqa"}:
        boxed = extract_boxed_answer(answer_text) or answer_text
        variants = [
            boxed,
            f"I choose {boxed} because it seems best.",
            f"{answer_text} and the runner-up was close.",
        ]
        return rng.choice(variants)
    variants = [
        answer_text + "\nThis extra note should not be here.",
        "Here is the response:\n" + answer_text,
        answer_text + "\nP.S. I double-checked it.",
    ]
    return rng.choice(variants)


def _degrade_monologue_completion(message: dict[str, Any], *, extra_suffix: str) -> dict[str, Any]:
    reasoning_text, answer_text = _split_assistant_content(message)
    return {"role": "assistant", "content": _compose_assistant_content(reasoning_text + extra_suffix, answer_text)}


def _compose_assistant_content(reasoning_text: str, answer_text: str) -> str:
    return f"<think>\n{reasoning_text.strip()}\n</think>\n{answer_text.strip()}".strip()


def _split_assistant_content(message: dict[str, Any]) -> tuple[str, str]:
    content = _field(message, "content", "")
    if isinstance(content, list):
        reasoning_text = ""
        answer_text = ""
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "thinking" and isinstance(part.get("thinking"), str):
                reasoning_text = part["thinking"].strip()
            elif part.get("type") == "text" and isinstance(part.get("text"), str):
                answer_text = part["text"].strip()
        return reasoning_text, answer_text
    text = _content_to_text(content)
    if "<think>" in text and "</think>" in text:
        _, after_open = text.split("<think>", 1)
        reasoning_text, answer_text = after_open.split("</think>", 1)
        return reasoning_text.strip(), answer_text.strip()
    return "", text.strip()


def _remap_completion_messages(messages: list[dict[str, Any]], build_parser_completion) -> list[dict[str, Any]]:
    remapped = [dict(message) for message in messages]
    if not remapped or remapped[-1].get("role") != "assistant":
        return remapped
    assistant = remapped[-1]
    if not isinstance(assistant.get("content"), list):
        return remapped
    completion = build_parser_completion({"messages": remapped})
    remapped[-1] = _assistant_message_structured(completion["reasoning_content"], completion["content"])
    return remapped


def _sample_rows(rows: Sequence[Any] | Dataset, *, count: int, seed: int) -> list[Any]:
    materialized = list(rows)
    if not materialized:
        return []
    rng = random.Random(seed)
    indices = list(range(len(materialized)))
    rng.shuffle(indices)
    if count <= len(indices):
        return [materialized[index] for index in indices[:count]]
    repeated = []
    while len(repeated) < count:
        rng.shuffle(indices)
        repeated.extend(materialized[index] for index in indices)
    return repeated[:count]


def _assistant_message(completion: Any) -> Any:
    if isinstance(completion, list):
        for message in reversed(completion):
            if _field(message, "role", "") == "assistant":
                return message
    return None


def _field(message: Any, key: str, default: Any = None) -> Any:
    if isinstance(message, dict):
        return message.get(key, default)
    return getattr(message, key, default)


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    text = part.get("text")
                    if isinstance(text, str):
                        chunks.append(text)
                continue
            text_attr = getattr(part, "text", None)
            if isinstance(text_attr, str):
                chunks.append(text_attr)
        return " ".join(chunk for chunk in chunks if chunk).strip()
    return ""


def _split_reasoning_and_visible_text(text: str) -> tuple[str, str]:
    stripped = text.strip()
    if not stripped:
        return "", ""
    lowered = stripped.lower()
    if "<think>" not in lowered:
        return "", stripped
    if "</think>" in lowered:
        reasoning, visible = stripped.split("</think>", 1)
        return reasoning.strip(), visible.strip()
    group_solution_close = re.search(r"</group_solution\s*>", stripped, flags=re.IGNORECASE)
    if group_solution_close:
        reasoning = stripped[: group_solution_close.end()].strip()
        visible = stripped[group_solution_close.end() :].strip()
        return reasoning, visible.lstrip("</think>").strip()
    return stripped, ""


def _extract_boxed_literal(text: str) -> str | None:
    match = re.search(r"\\boxed\{([^{}]+)\}", text)
    if not match:
        return None
    return match.group(1).strip()


def _safe_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(mean(values))


async def evaluate_external_suite_with_tinker(
    *,
    model_name: str | None,
    model_path: str | None,
    benchmarks: Sequence[str],
    trace_mode: str,
    num_examples: int,
    rollouts_per_example: int,
    max_concurrent: int,
    max_tokens: int,
    temperature: float,
    save_lock_file: str | os.PathLike[str] | None = None,
    replay_lock_file: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    from verifiers.clients import OpenAIChatCompletionsClient

    import tinker
    from tinker_cookbook.recipes.verifiers_rl.tinker_openai import TinkerAsyncOpenAIClient
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    service = tinker.ServiceClient()

    if model_path is not None:
        rest_client = service.create_rest_client()
        training_run = await rest_client.get_training_run_by_tinker_path_async(model_path)
        if model_name and model_name != training_run.base_model:
            raise ValueError(
                f"Model name {model_name} does not match checkpoint base model {training_run.base_model}"
            )
        model_name = training_run.base_model
    if model_name is None:
        raise ValueError("model_name or model_path must be provided")

    tokenizer = get_tokenizer(model_name)
    if model_path is not None:
        renderer_name = getattr(training_run, "renderer_name", None) or getattr(training_run, "renderer", None)
    else:
        renderer_name = None
    if renderer_name is None:
        renderer_name = get_recommended_renderer_name(model_name)
    renderer = get_renderer(renderer_name, tokenizer)
    if model_path:
        sampling = service.create_sampling_client(model_path=model_path, base_model=model_name)
    else:
        sampling = service.create_sampling_client(base_model=model_name)
    client = OpenAIChatCompletionsClient(TinkerAsyncOpenAIClient(sampling, renderer, tokenizer))

    if save_lock_file is not None and replay_lock_file is not None:
        raise ValueError("Use either save_lock_file or replay_lock_file, not both")

    replay_rows_by_benchmark: dict[str, list[dict[str, Any]]] = {}
    lock_payload: dict[str, Any] | None = None
    if replay_lock_file is not None:
        lock_payload = load_external_suite_lock_file(replay_lock_file)
        replay_rows_by_benchmark = _lock_rows_by_benchmark(lock_payload.get("rows", []))

    benchmark_summaries = []
    saved_lock_rows: list[dict[str, Any]] = []
    for benchmark in core_benchmark_order(benchmarks):
        if replay_rows_by_benchmark:
            env = _load_external_eval_env_from_lock_rows(
                benchmark,
                trace_mode=trace_mode,
                lock_rows=replay_rows_by_benchmark.get(benchmark, []),
            )
            requested_examples = len(replay_rows_by_benchmark.get(benchmark, []))
        else:
            env = load_external_eval_env(benchmark, trace_mode=trace_mode, num_examples=num_examples)
            requested_examples = num_examples
        raw_results = env.evaluate_sync(
            client=client,
            model=model_name,
            num_examples=requested_examples,
            rollouts_per_example=rollouts_per_example,
            max_concurrent=max_concurrent,
            sampling_args={"max_tokens": max_tokens, "temperature": temperature},
        )
        if save_lock_file is not None:
            saved_lock_rows.extend(
                _lock_rows_from_outputs(raw_results.get("outputs", []), benchmark=benchmark)
            )
        benchmark_summaries.append(
            summarize_external_generate_outputs(raw_results, benchmark=benchmark, trace_mode=trace_mode)
        )
    aggregate = aggregate_external_suite(benchmark_summaries)
    aggregate["model_name"] = model_name
    aggregate["model_path"] = model_path
    aggregate["trace_mode"] = trace_mode
    aggregate["renderer_name"] = renderer_name
    aggregate["num_examples"] = num_examples
    aggregate["rollouts_per_example"] = rollouts_per_example
    if replay_lock_file is not None:
        aggregate["paired_rows_source"] = "replay_lock"
        aggregate["lock_row_count"] = sum(len(rows) for rows in replay_rows_by_benchmark.values())
    if save_lock_file is not None:
        save_external_suite_lock_file(
            save_lock_file,
            rows=saved_lock_rows,
            metadata={
                "model_name": model_name,
                "model_path": model_path,
                "trace_mode": trace_mode,
                "benchmarks": list(core_benchmark_order(benchmarks)),
                "num_examples": num_examples,
                "rollouts_per_example": rollouts_per_example,
            },
        )
        aggregate["paired_rows_source"] = "saved_lock"
        aggregate["lock_row_count"] = len(saved_lock_rows)
    return aggregate


def _lock_rows_from_outputs(outputs: Sequence[dict[str, Any]], *, benchmark: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(outputs):
        prompt = item.get("prompt")
        rows.append(
            {
                "benchmark": normalize_benchmark_name(benchmark),
                "example_id": int(item.get("example_id", index)),
                "prompt": _to_json_safe(prompt),
                "answer": item.get("answer", ""),
                "task": item.get("task", ""),
                "info": _normalize_lock_info(item.get("info", {})),
                "question": _extract_lock_question(prompt, task=item.get("task", "")),
            }
        )
    return rows


def _normalize_lock_info(info: Any) -> Any:
    normalized = _to_json_safe(info)
    if isinstance(normalized, str):
        try:
            return json.loads(normalized)
        except json.JSONDecodeError:
            return normalized
    return normalized


def _to_json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_json_safe(item) for key, item in value.items()}
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return _to_json_safe(value.model_dump())
    if hasattr(value, "dict") and callable(value.dict):
        return _to_json_safe(value.dict())
    return str(value)


def _lock_rows_by_benchmark(rows: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = normalize_benchmark_name(str(row.get("benchmark", "")))
        grouped.setdefault(key, []).append(row)
    for key, value in grouped.items():
        value.sort(key=lambda item: int(item.get("example_id", 0)))
    return grouped


def _extract_lock_question(prompt: Any, *, task: Any) -> str:
    if isinstance(prompt, list):
        for message in reversed(prompt):
            if _field(message, "role") == "user":
                text = _content_to_text(_field(message, "content", ""))
                if text.strip():
                    return text.strip()
    if isinstance(task, str) and task.strip():
        return task.strip()
    return ""


def _load_external_eval_env_from_lock_rows(
    benchmark: str,
    *,
    trace_mode: str,
    lock_rows: Sequence[dict[str, Any]],
) -> vf.Environment:
    from datasets import Dataset

    env = load_external_eval_env(benchmark, trace_mode=trace_mode, num_examples=len(lock_rows))
    replay_rows = []
    for index, row in enumerate(lock_rows):
        replay_rows.append(
            {
                "question": str(row.get("question") or _extract_lock_question(row.get("prompt"), task=row.get("task", ""))),
                "answer": row.get("answer", ""),
                "info": _normalize_lock_info(row.get("info", {})),
                "example_id": int(row.get("example_id", index)),
            }
        )
    env.eval_dataset = Dataset.from_list(replay_rows)
    env.eval_dataset = env._ensure_example_id(env.eval_dataset)
    env.eval_dataset = env._ensure_prompt(
        env.eval_dataset,
        system_prompt=env.system_prompt,
        few_shot=env.few_shot,
        map_kwargs=env.map_kwargs,
    )
    return env
