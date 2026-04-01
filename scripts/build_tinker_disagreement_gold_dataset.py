from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from society_of_thought_bench.parser import SocietyOfThoughtParser
from society_of_thought_bench.scoring import SocietyOfThoughtScorer
from society_of_thought_bench.training_data import (
    DATA_ROOT,
    DEFAULT_OBJECTIVE_PROFILE,
    _assistant_completion_message,
    _build_personas,
    _build_teacher_answer_payload,
    _wrap_trace_payload,
    build_parser_completion,
    build_warmup_example,
)

ALLOWED_PATTERNS = (
    "assumption_challenge",
    "alternative_route_challenge",
    "stale_evidence_challenge",
    "verification_pushback",
)
ALLOWED_DIALECTS = (
    "persona_think",
    "character_step",
    "named_tag",
    "speaker_lines",
)
SPLIT_NAMES = ("train", "val", "lock")
FAMILY_DIFFICULTY_ORDER = (
    ("countdown", "medium"),
    ("countdown", "hard"),
    ("evidence", "medium"),
    ("evidence", "hard"),
)
TUNE_INTERNAL_TRAIN_TOTAL = 20
TUNE_INTERNAL_VAL_TOTAL = 7
TUNE_INTERNAL_ORDER = FAMILY_DIFFICULTY_ORDER


@dataclass(frozen=True, slots=True)
class GoldSpec:
    name: str
    family: str
    difficulty: str
    institution: str
    split: str
    seed: int
    pattern: str
    dialect: str
    example_index: int


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-file",
        type=Path,
        default=DATA_ROOT / "disagreement_gold_v1" / "source.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_ROOT / "disagreement_gold_v1",
    )
    parser.add_argument("--internal-train-total", type=int, default=TUNE_INTERNAL_TRAIN_TOTAL)
    parser.add_argument("--internal-val-total", type=int, default=TUNE_INTERNAL_VAL_TOTAL)
    args = parser.parse_args()

    source = json.loads(args.source_file.read_text())
    specs = expand_specs(source)
    validate_source_specs(specs)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    parser_obj = SocietyOfThoughtParser(trace_mode="debate")
    scorer = SocietyOfThoughtScorer(max_personas=4, max_debate_turns=10, objective_profile=DEFAULT_OBJECTIVE_PROFILE)

    rows_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLIT_NAMES}
    for spec in specs:
        row = build_gold_row(spec, parser_obj=parser_obj, scorer=scorer)
        rows_by_split[spec.split].append(row)

    internal_train_rows = build_internal_mix_rows(
        total=args.internal_train_total,
        split="train",
        seed_start=510_000,
    )
    internal_val_rows = build_internal_mix_rows(
        total=args.internal_val_total,
        split="eval",
        seed_start=520_000,
    )

    tune_train_rows = rows_by_split["train"] + internal_train_rows
    tune_val_rows = rows_by_split["val"] + internal_val_rows

    train_path = args.output_dir / "sft_train.jsonl"
    val_path = args.output_dir / "sft_val.jsonl"
    lock_path = args.output_dir / "lock_eval.jsonl"
    tune_train_path = args.output_dir / "tune_train.jsonl"
    tune_val_path = args.output_dir / "tune_val.jsonl"
    manifest_path = args.output_dir / "manifest.json"

    write_jsonl(train_path, rows_by_split["train"])
    write_jsonl(val_path, rows_by_split["val"])
    write_jsonl(lock_path, rows_by_split["lock"])
    write_jsonl(tune_train_path, tune_train_rows)
    write_jsonl(tune_val_path, tune_val_rows)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_file": _relative_display_path(args.source_file),
        "objective_profile": DEFAULT_OBJECTIVE_PROFILE,
        "gold_counts": {split: len(rows) for split, rows in rows_by_split.items()},
        "family_difficulty_counts": summarize_family_difficulty_counts(specs),
        "pattern_counts": summarize_pattern_counts(specs),
        "dialect_counts": summarize_dialect_counts(specs),
        "train_path": _relative_display_path(train_path),
        "val_path": _relative_display_path(val_path),
        "lock_eval_path": _relative_display_path(lock_path),
        "tune_train_path": _relative_display_path(tune_train_path),
        "tune_val_path": _relative_display_path(tune_val_path),
        "tune_mix": {
            "gold_train_examples": len(rows_by_split["train"]),
            "gold_val_examples": len(rows_by_split["val"]),
            "internal_train_examples": len(internal_train_rows),
            "internal_val_examples": len(internal_val_rows),
            "train_gold_ratio": len(rows_by_split["train"]) / max(1, len(tune_train_rows)),
            "val_gold_ratio": len(rows_by_split["val"]) / max(1, len(tune_val_rows)),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


def _relative_display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return path.as_posix()


def expand_specs(source: dict[str, Any]) -> list[GoldSpec]:
    if source.get("version") != 1:
        raise ValueError("disagreement gold source must declare version=1")
    dialect_cycle = source.get("dialect_cycle", list(ALLOWED_DIALECTS))
    if not isinstance(dialect_cycle, list) or not dialect_cycle:
        raise ValueError("dialect_cycle must be a non-empty list")
    for dialect in dialect_cycle:
        if dialect not in ALLOWED_DIALECTS:
            raise ValueError(f"Unsupported dialect in source: {dialect}")

    specs: list[GoldSpec] = []
    for block in source.get("blocks", []):
        family = str(block["family"])
        difficulty = str(block["difficulty"])
        institution = str(block["institution"])
        seed_start = int(block["seed_start"])
        name_prefix = f"{family}_{difficulty}"
        offset = 0
        for split in SPLIT_NAMES:
            raw_patterns = block[f"{split}_patterns"]
            if not isinstance(raw_patterns, list):
                raise ValueError(f"{family} {difficulty} {split}_patterns must be a list")
            for index, pattern in enumerate(raw_patterns):
                dialect = dialect_cycle[(offset + index) % len(dialect_cycle)]
                specs.append(
                    GoldSpec(
                        name=f"{name_prefix}_{split}_{index + 1:02d}",
                        family=family,
                        difficulty=difficulty,
                        institution=institution,
                        split=split,
                        seed=seed_start + offset + index,
                        pattern=str(pattern),
                        dialect=dialect,
                        example_index=index,
                    )
                )
            offset += len(raw_patterns)
    return specs


def validate_source_specs(specs: list[GoldSpec]) -> None:
    for spec in specs:
        if spec.pattern not in ALLOWED_PATTERNS:
            raise ValueError(f"Unsupported disagreement pattern: {spec.pattern}")
    if len(specs) != 80:
        raise ValueError(f"Expected 80 disagreement gold specs, found {len(specs)}")
    split_counts = Counter(spec.split for spec in specs)
    expected_split_counts = {"train": 48, "val": 16, "lock": 16}
    if dict(split_counts) != expected_split_counts:
        raise ValueError(f"Unexpected split counts: {dict(split_counts)}")

    family_difficulty_counts = Counter((spec.family, spec.difficulty) for spec in specs)
    for key in FAMILY_DIFFICULTY_ORDER:
        if family_difficulty_counts[key] != 20:
            raise ValueError(f"Expected 20 specs for {key}, found {family_difficulty_counts[key]}")

    pattern_counts = Counter(spec.pattern for spec in specs)
    for pattern in ALLOWED_PATTERNS:
        if pattern_counts[pattern] != 20:
            raise ValueError(f"Expected 20 specs for pattern={pattern}, found {pattern_counts[pattern]}")

    seen_names: set[str] = set()
    seen_seeds: set[tuple[str, str, int]] = set()
    for spec in specs:
        if spec.name in seen_names:
            raise ValueError(f"Duplicate gold example name: {spec.name}")
        seen_names.add(spec.name)
        seed_key = (spec.family, spec.difficulty, spec.seed)
        if seed_key in seen_seeds:
            raise ValueError(f"Duplicate seed within family/difficulty block: {seed_key}")
        seen_seeds.add(seed_key)


def build_gold_row(
    spec: GoldSpec,
    *,
    parser_obj: SocietyOfThoughtParser,
    scorer: SocietyOfThoughtScorer,
) -> dict[str, Any]:
    split = "train" if spec.split == "train" else "eval"
    row = build_warmup_example(
        family=spec.family,
        difficulty=spec.difficulty,
        institution=spec.institution,
        seed=spec.seed,
        split=split,
        example_id=spec.example_index,
        curriculum_profile="debate_primary",
    )
    task = row["oracle_task"]
    trace_payload = build_gold_trace_payload(
        family=spec.family,
        difficulty=spec.difficulty,
        task=task,
        institution=spec.institution,
        seed=spec.seed,
        pattern=spec.pattern,
    )
    trace_payload["dialect"] = spec.dialect
    _validate_gold_payload(trace_payload, family=spec.family, difficulty=spec.difficulty)

    answer_text = _build_teacher_answer_payload(spec.family, task)
    assistant = collapse_message(_assistant_completion_message(_wrap_trace_payload(trace_payload), answer_text))
    row["messages"][-1] = assistant
    row["challenge_pattern"] = spec.pattern
    row["gold_set_name"] = spec.name
    row["gold_set_split"] = spec.split
    row["gold_set_version"] = "disagreement_gold_v1"
    row["trace_dialect"] = spec.dialect

    completion = build_parser_completion(row)
    parsed = parser_obj.parse_completion([completion])
    if not (parsed.protocol_valid and parsed.trace_valid and parsed.answer_valid):
        raise ValueError(f"{spec.name} violates the benchmark contract: {parsed.error_codes}")

    state = {
        "completion": [completion],
        "info": {
            "family": spec.family,
            "difficulty": spec.difficulty,
            "trace_mode": "debate",
            "objective_profile": DEFAULT_OBJECTIVE_PROFILE,
            "task": task,
        },
    }
    metrics = scorer._ensure_metrics(state, parser_obj)
    if not any(turn.act == "challenge" for turn in parsed.turns):
        raise ValueError(f"{spec.name} is missing a challenge turn")
    if not any(turn.act == "verify" for turn in parsed.turns):
        raise ValueError(f"{spec.name} is missing a verification turn")
    if not any(turn.act == "reconcile" for turn in parsed.turns):
        raise ValueError(f"{spec.name} is missing a reconciliation turn")
    if metrics["challenge_response_pair_count"] < 1.0:
        raise ValueError(f"{spec.name} is missing a cross-speaker challenge response pair")
    if spec.difficulty == "hard" and metrics["alternative_path_count"] < 1.0:
        raise ValueError(f"{spec.name} must include an explicit alternative path on hard difficulty")
    if spec.family == "evidence" and "<support>" not in assistant["content"]:
        raise ValueError(f"{spec.name} is missing a separate support block")
    if "<answer>" not in assistant["content"]:
        raise ValueError(f"{spec.name} is missing a separate answer block")
    return row


def build_gold_trace_payload(
    *,
    family: str,
    difficulty: str,
    task: dict[str, Any],
    institution: str,
    seed: int,
    pattern: str,
) -> dict[str, Any]:
    if family == "countdown":
        return _countdown_gold_trace_payload(task, difficulty, institution, seed, pattern)
    return _evidence_gold_trace_payload(task, difficulty, institution, seed, pattern)


def _countdown_gold_trace_payload(
    task: dict[str, Any],
    difficulty: str,
    institution: str,
    seed: int,
    pattern: str,
) -> dict[str, Any]:
    refs = [item["id"] for item in task["numbers"]]
    target = task["target"]
    expression = task["oracle_expression"]
    pair_refs = refs[:2] if len(refs) >= 2 else refs
    third_ref = refs[2] if len(refs) >= 3 else refs[-1]
    personas = _build_personas("countdown", difficulty, institution, seed)
    speaker_by_role = {persona["role"]: persona["id"] for persona in personas}
    brainstormer = speaker_by_role.get("brainstormer", personas[0]["id"])
    devils_advocate = speaker_by_role.get("devils_advocate", personas[-1]["id"])
    verifier = speaker_by_role.get("verifier", personas[-1]["id"])
    synthesizer = speaker_by_role.get("synthesizer", verifier)
    refs_with_target = refs + ["T"]

    challenge_text, response_text, shift_text, reconcile_text = _countdown_pattern_text(
        pattern=pattern,
        target=target,
        expression=expression,
        pair_refs=pair_refs,
        third_ref=third_ref,
    )
    if difficulty == "hard":
        debate = [
            {
                "id": "t1",
                "speaker": brainstormer,
                "act": "question",
                "reply_to": [],
                "refs": refs_with_target,
                "content": f"Which exact route reaches T={target}, and which competing branch should we inspect before we settle?",
            },
            {
                "id": "t2",
                "speaker": brainstormer,
                "act": "propose",
                "reply_to": ["t1"],
                "refs": refs_with_target,
                "content": f"Initial proposal: {expression} is the clean exact-hit route.",
            },
            {
                "id": "t3",
                "speaker": devils_advocate,
                "act": "challenge",
                "reply_to": ["t2"],
                "refs": refs_with_target,
                "content": challenge_text,
            },
            {
                "id": "t4",
                "speaker": verifier,
                "act": "shift",
                "reply_to": ["t3"],
                "refs": refs_with_target,
                "content": shift_text,
            },
            {
                "id": "t5",
                "speaker": brainstormer,
                "act": "verify",
                "reply_to": ["t3", "t4"],
                "refs": refs_with_target,
                "content": response_text,
            },
            {
                "id": "t6",
                "speaker": synthesizer,
                "act": "reconcile",
                "reply_to": ["t3", "t5"],
                "refs": refs_with_target,
                "content": reconcile_text,
            },
        ]
    else:
        debate = [
            {
                "id": "t1",
                "speaker": brainstormer,
                "act": "question",
                "reply_to": [],
                "refs": refs_with_target,
                "content": f"What exact route reaches T={target} from the numbered inputs without hiding any illegal step?",
            },
            {
                "id": "t2",
                "speaker": brainstormer,
                "act": "propose",
                "reply_to": ["t1"],
                "refs": refs_with_target,
                "content": f"Initial proposal: {expression}.",
            },
            {
                "id": "t3",
                "speaker": devils_advocate,
                "act": "challenge",
                "reply_to": ["t2"],
                "refs": refs_with_target,
                "content": challenge_text,
            },
            {
                "id": "t4",
                "speaker": verifier,
                "act": "verify",
                "reply_to": ["t3"],
                "refs": refs_with_target,
                "content": response_text,
            },
            {
                "id": "t5",
                "speaker": brainstormer if institution == "flat" else synthesizer,
                "act": "reconcile",
                "reply_to": ["t3", "t4"],
                "refs": refs_with_target,
                "content": reconcile_text,
            },
        ]
    return {"personas": personas, "debate": debate, "group_solution": expression}


def _evidence_gold_trace_payload(
    task: dict[str, Any],
    difficulty: str,
    institution: str,
    seed: int,
    pattern: str,
) -> dict[str, Any]:
    verdict = task["oracle_verdict"]
    support = task["oracle_support"]
    support_str = ", ".join(support)
    evidence_ids = [entry["id"] for entry in task["evidence"]]
    refs = list(dict.fromkeys(support + evidence_ids[:3]))
    stale_refs = [entry_id for entry_id in evidence_ids if entry_id not in support][:2] or refs[:2]
    personas = _build_personas("evidence", difficulty, institution, seed)
    speaker_by_role = {persona["role"]: persona["id"] for persona in personas}
    brainstormer = speaker_by_role.get("brainstormer", personas[0]["id"])
    devils_advocate = speaker_by_role.get("devils_advocate", personas[-1]["id"])
    verifier = speaker_by_role.get("verifier", personas[-1]["id"])
    synthesizer = speaker_by_role.get("synthesizer", verifier)

    challenge_text, response_text, shift_text, reconcile_text = _evidence_pattern_text(
        pattern=pattern,
        claim=task["claim"],
        verdict=verdict,
        support_str=support_str,
        support=support,
        stale_refs=stale_refs,
    )
    group_solution = f"{verdict} from {support_str}"
    if difficulty == "hard":
        debate = [
            {
                "id": "t1",
                "speaker": brainstormer,
                "act": "question",
                "reply_to": [],
                "refs": refs,
                "content": f"Which current snippets decide the claim, and which competing reading should we inspect before we settle: {task['claim']}",
            },
            {
                "id": "t2",
                "speaker": brainstormer,
                "act": "propose",
                "reply_to": ["t1"],
                "refs": support,
                "content": f"Initial proposal: {verdict}, anchored on {support_str}.",
            },
            {
                "id": "t3",
                "speaker": devils_advocate,
                "act": "challenge",
                "reply_to": ["t2"],
                "refs": refs,
                "content": challenge_text,
            },
            {
                "id": "t4",
                "speaker": verifier,
                "act": "shift",
                "reply_to": ["t3"],
                "refs": refs,
                "content": shift_text,
            },
            {
                "id": "t5",
                "speaker": brainstormer,
                "act": "verify",
                "reply_to": ["t3", "t4"],
                "refs": refs,
                "content": response_text,
            },
            {
                "id": "t6",
                "speaker": synthesizer,
                "act": "reconcile",
                "reply_to": ["t3", "t5"],
                "refs": support,
                "content": reconcile_text,
            },
        ]
    else:
        debate = [
            {
                "id": "t1",
                "speaker": brainstormer,
                "act": "question",
                "reply_to": [],
                "refs": refs,
                "content": f"Which snippets control the present truth of the claim: {task['claim']}",
            },
            {
                "id": "t2",
                "speaker": brainstormer,
                "act": "propose",
                "reply_to": ["t1"],
                "refs": support,
                "content": f"Initial proposal: {verdict}, supported by {support_str}.",
            },
            {
                "id": "t3",
                "speaker": devils_advocate,
                "act": "challenge",
                "reply_to": ["t2"],
                "refs": refs,
                "content": challenge_text,
            },
            {
                "id": "t4",
                "speaker": verifier,
                "act": "verify",
                "reply_to": ["t3"],
                "refs": refs,
                "content": response_text,
            },
            {
                "id": "t5",
                "speaker": brainstormer if institution == "flat" else synthesizer,
                "act": "reconcile",
                "reply_to": ["t3", "t4"],
                "refs": support,
                "content": reconcile_text,
            },
        ]
    return {"personas": personas, "debate": debate, "group_solution": group_solution}


def _countdown_pattern_text(
    *,
    pattern: str,
    target: int,
    expression: str,
    pair_refs: list[str],
    third_ref: str,
) -> tuple[str, str, str, str]:
    joined_pair = " and ".join(pair_refs)
    if pattern == "assumption_challenge":
        return (
            f"Challenge: hitting T={target} is not enough on its own. Recheck whether {expression} reuses any numbered ID or smuggles in a constant outside {joined_pair} and {third_ref}.",
            f"Verification: {expression} still reaches {target} exactly, every operand comes from the numbered IDs, and the legality check does not uncover reuse or a hidden constant.",
            f"Alternative path: start from {joined_pair} first, then compare that branch back to T={target}; it is grounded, but it still does not beat the certified route {expression}.",
            f"Reconcile on {expression} because the assumption challenge forced a legality check, and that check now supports the exact route instead of weakening it.",
        )
    if pattern == "alternative_route_challenge":
        return (
            f"Challenge: the first exact hit is not automatically the clearest route, so compare it to a branch that starts with {joined_pair}.",
            f"Verification: after comparing the competing branch to the main line, {expression} remains the clean exact route to T={target}.",
            f"Alternative path: combine {joined_pair} first and only then chase T={target}; that branch is real, but it is longer and less direct than {expression}.",
            f"Reconcile on {expression} because the alternative route was inspected rather than ignored, and it still loses the direct comparison.",
        )
    if pattern == "stale_evidence_challenge":
        return (
            f"Challenge: do not trust the earlier route just because it looked plausible on the first pass. Recompute it directly against T={target} and the numbered IDs.",
            f"Verification: a fresh recomputation still lands on T={target}, so the proposal survives the stale-route challenge instead of coasting on an unchecked first impression.",
            f"Alternative path: start from {third_ref} with one of {joined_pair} to see whether the first-pass route missed a cleaner branch; it does not beat {expression}.",
            f"Reconcile on {expression} because the stale-route objection was answered with a fresh check, not with a shrug.",
        )
    if pattern == "verification_pushback":
        return (
            f"Challenge: do not settle yet. Push back until we explicitly recompute {expression} and confirm the target value from {joined_pair}, {third_ref}, and T={target}.",
            f"Verification: the recomputation is explicit now, and {expression} still reaches {target} without violating the one-use rule.",
            f"Alternative path: test a competing route built from {joined_pair} before the final answer; it stays grounded but still trails the verified line {expression}.",
            f"Reconcile only after the explicit check: {expression} remains the best grounded route to T={target}.",
        )
    raise ValueError(f"Unknown pattern: {pattern}")


def _evidence_pattern_text(
    *,
    pattern: str,
    claim: str,
    verdict: str,
    support_str: str,
    support: list[str],
    stale_refs: list[str],
) -> tuple[str, str, str, str]:
    stale_text = ", ".join(stale_refs)
    if pattern == "assumption_challenge":
        return (
            f"Challenge: the first cited support does not settle the claim by itself. Test whether {stale_text} changes the current reading of: {claim}",
            f"Verification: after checking the conflicting snippets directly, {support_str} still carries the current decision, so the grounded verdict remains {verdict}.",
            f"Competing reading: give {stale_text} the strongest possible reading first, then compare it back to {support_str}; that branch still loses the current-timeline check.",
            f"Reconcile on {verdict} from {support_str} because the assumption challenge is resolved with a direct comparison rather than with a polite agreement.",
        )
    if pattern == "alternative_route_challenge":
        return (
            f"Challenge: the first reading is not automatically the final one, so follow a competing reading built around {stale_text}.",
            f"Verification: the alternative reading was examined, but {support_str} still decides the present claim and keeps the verdict at {verdict}.",
            f"Competing reading: treat {stale_text} as if it were decisive, then compare that branch directly against the support chain {support_str}.",
            f"Reconcile on {verdict} from {support_str} because the competing reading was checked seriously and still fails the current-evidence test.",
        )
    if pattern == "stale_evidence_challenge":
        return (
            f"Challenge: {stale_text} may be stale evidence that only looks decisive because it came first in the discussion.",
            f"Verification: the stale-evidence check keeps {support_str} as the current support, so the verdict remains {verdict} after the timeline is cleaned up.",
            f"Competing reading: let the stale snippets speak first, then compare that branch to {support_str}; the stale branch does not survive the current-timeline review.",
            f"Reconcile on {verdict} from {support_str} because the stale-evidence objection was answered with a timeline check, not with a vague summary.",
        )
    if pattern == "verification_pushback":
        return (
            f"Challenge: do not settle yet. Push back until we explicitly verify that {support_str} still outranks {stale_text} for the claim: {claim}",
            f"Verification: the explicit check still favors {support_str}, so the grounded verdict remains {verdict} instead of drifting with the noisy snippets.",
            f"Competing reading: test the noisy branch from {stale_text} first, then compare it back to {support_str}; it does not overturn the current support chain.",
            f"Reconcile only after the explicit check: {verdict} from {support_str} is still the best grounded conclusion.",
        )
    raise ValueError(f"Unknown pattern: {pattern}")


def _validate_gold_payload(payload: dict[str, Any], *, family: str, difficulty: str) -> None:
    debate = payload["debate"]
    turns_by_id = {turn["id"]: turn for turn in debate}
    challenge_turns = [turn for turn in debate if turn["act"] == "challenge"]
    verify_turns = [turn for turn in debate if turn["act"] == "verify"]
    reconcile_turns = [turn for turn in debate if turn["act"] == "reconcile"]
    if not challenge_turns or not verify_turns or not reconcile_turns:
        raise ValueError(f"{family} {difficulty} payload is missing challenge/verify/reconcile")
    first_challenge = debate.index(challenge_turns[0])
    first_verify = debate.index(verify_turns[0])
    first_reconcile = debate.index(reconcile_turns[0])
    if not (first_challenge < first_verify < first_reconcile):
        raise ValueError(f"{family} {difficulty} payload does not preserve challenge -> verify -> reconcile order")
    if difficulty == "hard" and not any(turn["act"] == "shift" for turn in debate):
        raise ValueError(f"{family} {difficulty} payload must include a shift turn")
    for turn in challenge_turns:
        if not any(parent_id in turns_by_id and turns_by_id[parent_id]["act"] == "propose" for parent_id in turn["reply_to"]):
            raise ValueError(f"{family} {difficulty} challenge must reply to a proposal")
    for turn in reconcile_turns:
        if not turn["reply_to"]:
            raise ValueError(f"{family} {difficulty} reconcile turn must reply to prior disagreement")


def build_internal_mix_rows(*, total: int, split: str, seed_start: int) -> list[dict[str, Any]]:
    counts = split_evenly(total, len(TUNE_INTERNAL_ORDER))
    rows: list[dict[str, Any]] = []
    offset = 0
    for (family, difficulty), count in zip(TUNE_INTERNAL_ORDER, counts, strict=True):
        for index in range(count):
            seed = seed_start + offset + index
            row = build_warmup_example(
                family=family,
                difficulty=difficulty,
                seed=seed,
                split=split,
                example_id=index,
                curriculum_profile="debate_primary",
            )
            row["messages"] = [collapse_message(message) for message in row["messages"]]
            row["source_mix"] = "benchmark_internal"
            rows.append(row)
        offset += count
    return rows


def split_evenly(total: int, buckets: int) -> list[int]:
    base = total // buckets
    counts = [base] * buckets
    for index in range(total - (base * buckets)):
        counts[index] += 1
    return counts


def collapse_message(message: dict[str, Any]) -> dict[str, Any]:
    content = message.get("content")
    if not isinstance(content, list):
        return message
    thinking_parts: list[str] = []
    text_parts: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "thinking" and isinstance(part.get("thinking"), str):
            thinking_parts.append(part["thinking"].strip())
        if part.get("type") == "text" and isinstance(part.get("text"), str):
            text_parts.append(part["text"].strip())
    rendered = ""
    if thinking_parts:
        rendered += "<think>" + "\n\n".join(part for part in thinking_parts if part) + "</think>"
    if text_parts:
        rendered += "\n" + "\n".join(part for part in text_parts if part) if rendered else "\n".join(part for part in text_parts if part)
    return {**message, "content": rendered.strip()}


def summarize_family_difficulty_counts(specs: list[GoldSpec]) -> dict[str, int]:
    counts = Counter(f"{spec.family}_{spec.difficulty}_{spec.split}" for spec in specs)
    return dict(sorted(counts.items()))


def summarize_pattern_counts(specs: list[GoldSpec]) -> dict[str, int]:
    counts = Counter(spec.pattern for spec in specs)
    return dict(sorted(counts.items()))


def summarize_dialect_counts(specs: list[GoldSpec]) -> dict[str, int]:
    counts = Counter(spec.dialect for spec in specs)
    return dict(sorted(counts.items()))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
