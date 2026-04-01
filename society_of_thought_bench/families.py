from __future__ import annotations

import ast
import json
import random
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from datasets import Dataset

from .core import (
    EXPERTISE_BY_FAMILY,
    STYLE_BY_ROLE,
    TASK_NAME_BY_FAMILY,
    TRACE_PROMPT_VARIANTS,
    clamp,
    difficulty_turn_range,
    ordered_unique,
    resolve_institution,
)

COUNTDOWN_OPS = ("+", "-", "*", "/")

ENTITY_PREFIXES = (
    "Orion",
    "Cinder",
    "Atlas",
    "Nova",
    "Summit",
    "Harbor",
    "Lattice",
    "Beacon",
)
ENTITY_SUFFIXES = (
    "Museum",
    "Festival",
    "Project",
    "Committee",
    "Initiative",
    "Archive",
    "Network",
    "Program",
)
PEOPLE = (
    "Mira Chen",
    "Jonah Reed",
    "Aisha Patel",
    "Elena Soto",
    "Ravi Iyer",
    "Lena Park",
    "Noah Briggs",
    "Talia Ross",
)
LOCATIONS = (
    "Riverside Hall",
    "North Annex",
    "Crescent Center",
    "Pier Pavilion",
    "East Atrium",
    "Summit Court",
)
STATUSES = (
    "approved",
    "delayed",
    "postponed",
    "cancelled",
    "rescheduled",
)


@dataclass(slots=True)
class CountdownEvaluation:
    expression_valid: float
    number_usage_valid: float
    exact_hit: float
    distance: float
    task_score: float
    used_ids: list[str]


@dataclass(slots=True)
class EvidenceEvaluation:
    verdict_correct: float
    support_f1: float
    task_score: float


def build_dataset(
    family: str,
    difficulty: str,
    institution: str,
    seed: int,
    num_examples: int,
    max_personas: int,
    max_debate_turns: int,
    trace_mode: str,
    split: str,
    trace_prompt_variant: str,
    countdown_reward_profile: str = "benchmark",
    objective_profile: str = "debate_primary",
) -> Dataset:
    rows = [
        build_example(
            family=family,
            difficulty=difficulty,
            institution=institution,
            seed=seed,
            index=index,
            max_personas=max_personas,
            max_debate_turns=max_debate_turns,
            trace_mode=trace_mode,
            split=split,
            trace_prompt_variant=trace_prompt_variant,
            countdown_reward_profile=countdown_reward_profile,
            objective_profile=objective_profile,
        )
        for index in range(num_examples)
    ]
    return Dataset.from_list(rows)


def build_example(
    family: str,
    difficulty: str,
    institution: str,
    seed: int,
    index: int,
    max_personas: int,
    max_debate_turns: int,
    trace_mode: str,
    split: str,
    trace_prompt_variant: str = "official",
    forced_verdict: str | None = None,
    countdown_reward_profile: str = "benchmark",
    objective_profile: str = "debate_primary",
) -> dict[str, Any]:
    if trace_prompt_variant not in TRACE_PROMPT_VARIANTS:
        raise ValueError(
            f"trace_prompt_variant must be one of: {', '.join(TRACE_PROMPT_VARIANTS)}"
        )

    example_seed = seed + (index * 9973) + (0 if split == "train" else 100_003)
    resolved_institution = resolve_institution(difficulty, institution)
    if family == "countdown":
        task = _make_countdown_task(example_seed, difficulty, countdown_reward_profile)
    else:
        task = _make_evidence_task(example_seed, difficulty, forced_verdict=forced_verdict)

    info = {
        "family": family,
        "difficulty": difficulty,
        "institution": resolved_institution,
        "trace_mode": trace_mode,
        "trace_prompt_variant": trace_prompt_variant,
        "max_personas": max_personas,
        "max_debate_turns": max_debate_turns,
        "task": task,
        "objective_profile": objective_profile,
    }
    prompt = _build_prompt(
        family=family,
        task=task,
        difficulty=difficulty,
        institution=resolved_institution,
        trace_mode=trace_mode,
        max_personas=max_personas,
        max_debate_turns=max_debate_turns,
        trace_prompt_variant=trace_prompt_variant,
        countdown_reward_profile=countdown_reward_profile,
        objective_profile=objective_profile,
    )
    return {
        "prompt": prompt,
        "info": json.dumps(info),
        "example_id": index,
        "task": TASK_NAME_BY_FAMILY[family],
    }


def inspect_example(
    family: str,
    difficulty: str,
    institution: str,
    seed: int,
    max_personas: int,
    max_debate_turns: int,
    trace_mode: str,
    trace_prompt_variant: str = "official",
    countdown_reward_profile: str = "benchmark",
    objective_profile: str = "debate_primary",
) -> dict[str, Any]:
    row = build_example(
        family=family,
        difficulty=difficulty,
        institution=institution,
        seed=seed,
        index=0,
        max_personas=max_personas,
        max_debate_turns=max_debate_turns,
        trace_mode=trace_mode,
        split="eval",
        trace_prompt_variant=trace_prompt_variant,
        countdown_reward_profile=countdown_reward_profile,
        objective_profile=objective_profile,
    )
    info = json.loads(row["info"])
    return {
        "prompt": row["prompt"],
        "oracle": info["task"],
        "meta": {
            "family": info["family"],
            "difficulty": info["difficulty"],
            "institution": info["institution"],
            "trace_mode": info["trace_mode"],
            "trace_prompt_variant": info["trace_prompt_variant"],
            "objective_profile": info.get("objective_profile", "debate_primary"),
        },
        "contracts": {
            "reasoning_trace": _reasoning_contract_example(
                family=family,
                trace_mode=trace_mode,
            ),
            "visible_answer": _answer_example(family),
        },
    }


def evaluate_countdown_expression(expression: str, task: dict[str, Any]) -> CountdownEvaluation:
    if not expression.strip():
        return CountdownEvaluation(0.0, 0.0, 0.0, -1.0, 0.0, [])

    value, values_used, is_valid = _safe_eval_expression(expression)
    if not is_valid or value is None:
        return CountdownEvaluation(0.0, 0.0, 0.0, -1.0, 0.0, [])

    value_to_id = {entry["value"]: entry["id"] for entry in task["numbers"]}
    expected_values = sorted(value_to_id.keys())
    used_values = sorted(values_used)
    number_usage_valid = (
        1.0
        if len(used_values) == len(set(used_values)) and _multiset_subset(used_values, expected_values)
        else 0.0
    )
    used_ids = [value_to_id[number] for number in used_values if number in value_to_id]

    if number_usage_valid == 0.0:
        return CountdownEvaluation(1.0, 0.0, 0.0, -1.0, 0.35, ordered_unique(used_ids))

    target = Fraction(task["target"], 1)
    distance = float(abs(value - target))
    exact_hit = 1.0 if value == target else 0.0
    if exact_hit == 1.0:
        return CountdownEvaluation(1.0, 1.0, 1.0, 0.0, 1.0, ordered_unique(used_ids))

    closeness = max(0.0, 1.0 - (distance / max(10.0, float(abs(target)))))
    reward_profile = task.get("reward_profile", "benchmark")
    if reward_profile == "exact_emphasis":
        task_score = min(0.70, 0.15 * 1.0 + 0.15 * 1.0 + 0.40 * closeness)
    else:
        task_score = min(0.95, 0.35 * 1.0 + 0.25 * 1.0 + 0.40 * closeness)
    return CountdownEvaluation(
        expression_valid=1.0,
        number_usage_valid=1.0,
        exact_hit=0.0,
        distance=distance,
        task_score=task_score,
        used_ids=ordered_unique(used_ids),
    )


def evaluate_evidence_verdict(
    verdict: str,
    support: list[str],
    task: dict[str, Any],
) -> EvidenceEvaluation:
    oracle_verdict = task["oracle_verdict"]
    oracle_support = set(task["oracle_support"])
    predicted_support = set(support)
    verdict_correct = 1.0 if verdict == oracle_verdict else 0.0

    overlap = len(predicted_support & oracle_support)
    precision = overlap / len(predicted_support) if predicted_support else 0.0
    recall = overlap / len(oracle_support) if oracle_support else 0.0
    support_f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    task_score = 0.7 * verdict_correct + 0.3 * support_f1
    return EvidenceEvaluation(
        verdict_correct=verdict_correct,
        support_f1=support_f1,
        task_score=task_score,
    )


def _reasoning_contract_example(family: str, trace_mode: str) -> str:
    payload = _trace_payload_example(family, trace_mode)
    rendered = _wrap_trace_payload(payload)
    if trace_mode == "debate":
        return f"<think>\n{rendered}\n</think>"
    return f"<think>{rendered}</think>"


def _trace_payload_example(family: str, trace_mode: str) -> dict[str, Any]:
    if trace_mode == "debate":
        refs = ["N1", "N2", "T"] if family == "countdown" else ["E2"]
        return {
            "personas": [
                {
                    "id": "P1",
                    "role": "brainstormer",
                    "personality": "high_openness",
                    "expertise": EXPERTISE_BY_FAMILY[family][0],
                    "style": STYLE_BY_ROLE["brainstormer"],
                },
                {
                    "id": "P2",
                    "role": "devils_advocate",
                    "personality": "low_agreeableness",
                    "expertise": EXPERTISE_BY_FAMILY[family][1],
                    "style": STYLE_BY_ROLE["devils_advocate"],
                },
                {
                    "id": "P3",
                    "role": "verifier",
                    "personality": "high_conscientiousness",
                    "expertise": EXPERTISE_BY_FAMILY[family][-1],
                    "style": STYLE_BY_ROLE["verifier"],
                },
            ],
            "debate": [
                {
                    "speaker": "P1",
                    "content": f"What is the decisive route using {', '.join(refs)}?",
                },
                {
                    "speaker": "P2",
                    "content": f"But we should challenge the first path before trusting it, especially around {refs[-1]}.",
                },
                {
                    "speaker": "P3",
                    "content": f"Verification passes after checking the cited IDs directly: {', '.join(refs)}.",
                },
            ],
            "group_solution": "Concise grounded solution.",
        }
    refs = ["N1", "N2", "T"] if family == "countdown" else ["E2"]
    return {
        "analysis": [
            {
                "content": f"Compare the decisive refs {', '.join(refs)} before deciding.",
            },
            {
                "content": "Verify the final candidate before answering.",
            },
        ]
    }


def _answer_example(family: str) -> str:
    if family == "countdown":
        return "<answer>(9*5)-10</answer>"
    return "<answer>TRUE</answer>\n<support>E2</support>"


def _wrap_trace_payload(payload: dict[str, Any]) -> str:
    if "analysis" in payload:
        return "\n".join(step["content"] for step in payload["analysis"])

    dialect = payload.get("dialect", "persona_think")
    if dialect == "character_step":
        return _render_character_step_trace(payload)
    if dialect == "named_tag":
        return _render_named_tag_trace(payload)
    if dialect == "speaker_lines":
        return _render_speaker_line_trace(payload)
    return _render_persona_think_trace(payload)


def _render_persona_think_trace(payload: dict[str, Any]) -> str:
    persona_lines = []
    for index, persona in enumerate(payload["personas"], start=1):
        persona_lines.append(
            f"<persona{index}>Role: {persona['role']}\n"
            f"Personality: {persona['personality']}\n"
            f"Expertise: {persona['expertise']}\n"
            f"Style: {persona.get('style') or STYLE_BY_ROLE.get(persona['role'], '')}</persona{index}>"
        )
    ordinal_by_speaker = {persona["id"]: index for index, persona in enumerate(payload["personas"], start=1)}
    conversation_lines = []
    for turn in payload["debate"]:
        ordinal = ordinal_by_speaker.get(turn["speaker"], 1)
        conversation_lines.append(f"<think{ordinal}>{turn['content']}</think{ordinal}>")
    return _wrap_common_trace("\n".join(persona_lines), "\n".join(conversation_lines), payload["group_solution"])


def _render_character_step_trace(payload: dict[str, Any]) -> str:
    speaker_labels = _speaker_labels(payload["personas"])
    persona_lines = []
    for persona in payload["personas"]:
        label = speaker_labels[persona["id"]]
        persona_lines.append(
            f'<character name="{label}" role="{persona["role"]}" personality="{persona["personality"]}" '
            f'expertise="{persona["expertise"]}" style="{persona.get("style") or STYLE_BY_ROLE.get(persona["role"], "")}"/>'
        )
    conversation_lines = []
    for index, turn in enumerate(payload["debate"], start=1):
        speaker = speaker_labels.get(turn["speaker"], f"voice_{index}")
        conversation_lines.append(
            f'<step speaker="{speaker}" step="{index}" action="{turn.get("act", "propose")}">{turn["content"]}</step>'
        )
    return _wrap_common_trace("\n".join(persona_lines), "\n".join(conversation_lines), payload["group_solution"])


def _render_named_tag_trace(payload: dict[str, Any]) -> str:
    speaker_labels = _speaker_labels(payload["personas"])
    persona_lines = []
    for persona in payload["personas"]:
        label = speaker_labels[persona["id"]]
        persona_lines.append(
            f'<character name="{label}">Role: {persona["role"]}. Personality: {persona["personality"]}. '
            f'Expertise: {persona["expertise"]}. Style: {persona.get("style") or STYLE_BY_ROLE.get(persona["role"], "")}.</character>'
        )
    conversation_lines = []
    for turn in payload["debate"]:
        label = speaker_labels.get(turn["speaker"], turn["speaker"])
        tag = _tagify_label(label)
        conversation_lines.append(f"<{tag}>{turn['content']}</{tag}>")
    return _wrap_common_trace("\n".join(persona_lines), "\n".join(conversation_lines), payload["group_solution"])


def _render_speaker_line_trace(payload: dict[str, Any]) -> str:
    speaker_labels = _speaker_labels(payload["personas"])
    persona_lines = []
    for persona in payload["personas"]:
        label = speaker_labels[persona["id"]]
        descriptor = f"{persona['role']} | {persona['expertise']} | {persona.get('style') or STYLE_BY_ROLE.get(persona['role'], '')}"
        persona_lines.append(f"{label}: {descriptor}")
    conversation_lines = []
    for turn in payload["debate"]:
        label = speaker_labels.get(turn["speaker"], turn["speaker"])
        conversation_lines.append(f"{label}: {turn['content']}")
    return _wrap_common_trace("\n".join(persona_lines), "\n".join(conversation_lines), payload["group_solution"])


def _wrap_common_trace(cast_text: str, conversation_text: str, group_solution: str) -> str:
    return (
        "<cast_of_characters>\n"
        + cast_text
        + "\n</cast_of_characters>\n"
        + "<conversation>\n"
        + conversation_text
        + "\n</conversation>\n"
        + f"<group_solution>{group_solution}</group_solution>"
    )


def _speaker_labels(personas: list[dict[str, Any]]) -> dict[str, str]:
    labels: dict[str, str] = {}
    role_names = {
        "brainstormer": "solver",
        "planner": "planner",
        "analyst": "analyst",
        "devils_advocate": "skeptic",
        "skeptic": "skeptic",
        "contrarian": "contrarian",
        "violation_hunter": "auditor",
        "verifier": "checker",
        "synthesizer": "synthesizer",
        "editor": "editor",
    }
    used: set[str] = set()
    for index, persona in enumerate(personas, start=1):
        base = role_names.get(persona["role"], f"voice_{index}")
        label = base
        suffix = 2
        while label in used:
            label = f"{base}_{suffix}"
            suffix += 1
        used.add(label)
        labels[persona["id"]] = label
    return labels


def _tagify_label(label: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in label).strip("_")
    if not cleaned:
        return "voice"
    if cleaned[0].isdigit():
        cleaned = f"voice_{cleaned}"
    return cleaned


def _prompt_variant_header(trace_prompt_variant: str) -> str:
    if trace_prompt_variant == "official":
        return "This benchmark scores the exposed reasoning trace. Solve the task without explaining the benchmark."
    if trace_prompt_variant == "trace_minimal":
        return "Solve the task. Put the paper-style conversation in exposed reasoning and keep the visible answer separate."
    if trace_prompt_variant == "trace_tail_block":
        return "Solve the task. The paper-style reasoning conversation must be the final thing inside the exposed reasoning trace."
    return "Do this exactly: one paper-style reasoning conversation and one visible answer block."


def _prompt_variant_reasoning_rules(trace_prompt_variant: str) -> str:
    if trace_prompt_variant == "official":
        return (
            "Reasoning trace rules:\n"
            "- Use the exposed reasoning stream / <think> as scratch space.\n"
            "- Inside that stream, include exactly one <cast_of_characters> block, one <conversation> block, and one <group_solution> block.\n"
            "- Inside those blocks, any clear multi-voice dialect is acceptable: persona tags, character tags, step tags, named speaker tags, or simple speaker lines.\n"
            "- Do not write any extra tags outside that paper-style structure.\n"
            "- Keep the reasoning task-focused and concise."
        )
    if trace_prompt_variant == "trace_minimal":
        return (
            "Reasoning trace rules:\n"
            "- Use one paper-style cast block, one conversation block, and one group solution block in the reasoning stream.\n"
            "- Keep the reasoning short and task-specific."
        )
    if trace_prompt_variant == "trace_tail_block":
        return (
            "Reasoning trace rules:\n"
            "- Any scratch reasoning must stay inside the final paper-style conversation structure.\n"
            "- End reasoning with the group solution inside the same thought block.\n"
            "- Do not narrate the benchmark or schema."
        )
    return (
        "Reasoning trace rules:\n"
        "- Write only the paper-style cast, conversation, and group solution inside the exposed reasoning stream.\n"
        "- No benchmark commentary. No schema narration."
    )


def _prompt_variant_answer_rules(family: str) -> str:
    if family == "countdown":
        return (
            "Visible answer rules:\n"
            "- Return only one <answer>...</answer> block.\n"
            "- Do not include JSON, markdown, prose, or any other tags.\n"
            "- Put only the final arithmetic expression inside <answer>."
        )
    return (
        "Visible answer rules:\n"
        "- Return only one <answer>...</answer> block and one <support>...</support> block.\n"
        "- Do not include JSON, markdown, prose, or any other tags.\n"
        "- Put only the final verdict inside <answer> and the decisive evidence IDs inside <support>."
    )


def _objective_profile_block(objective_profile: str) -> str:
    if objective_profile == "debate_primary":
        return (
            "Scoring prioritizes diverse, grounded debate inside the exposed reasoning trace. "
            "Keep the final answer grounded, but the debate quality matters more than exact task accuracy."
        )
    return (
        "Scoring balances grounded debate quality with final task correctness. "
        "Keep both the reasoning trace and the final answer clean."
    )


def _build_prompt(
    family: str,
    task: dict[str, Any],
    difficulty: str,
    institution: str,
    trace_mode: str,
    max_personas: int,
    max_debate_turns: int,
    trace_prompt_variant: str,
    countdown_reward_profile: str = "benchmark",
    objective_profile: str = "debate_primary",
) -> list[dict[str, str]]:
    debate_behavior_line = {
        "easy": "Use at least 2 personas and keep the conversation concise.",
        "medium": "Use at least 3 personas and include real disagreement before convergence.",
        "hard": "Prefer 4 personas and include both an alternative path and a reconciliation.",
    }[difficulty]
    monologue_behavior_line = {
        "easy": "Keep the monologue concise and grounded.",
        "medium": "Show a genuine reconsideration or self-correction.",
        "hard": "Show at least one alternative path before the final answer.",
    }[difficulty]
    institution_line = (
        {
            "flat": "All personas deliberate as peers.",
            "hierarchical": "Worker personas deliberate first, then the synthesizer resolves the final group solution.",
        }[institution]
        if trace_mode == "debate"
        else "Use a single-voice chain of thought inside the reasoning stream."
    )
    allowed_expertise = ", ".join(EXPERTISE_BY_FAMILY[family])
    turn_low, turn_high = difficulty_turn_range(difficulty, max_debate_turns)
    reasoning_contract = _reasoning_contract_example(
        family=family,
        trace_mode=trace_mode,
    )
    answer_contract = _answer_example(family)
    role_metadata = (
        "- Use at least two distinct voices with meaningfully different perspectives.\n"
        f"- Family-relevant expertise is encouraged, especially around: {allowed_expertise}\n"
        "- Voice metadata is flexible. You may use explicit Role / Personality / Expertise / Style fields, lighter character labels, or short speaker descriptors.\n"
        if trace_mode == "debate"
        else ""
    )
    task_block = _build_task_block(family, task, objective_profile)
    user_message = (
        f"{_prompt_variant_header(trace_prompt_variant)}\n\n"
        f"{_objective_profile_block(objective_profile)}\n\n"
        f"{_prompt_variant_reasoning_rules(trace_prompt_variant)}\n\n"
        f"{_prompt_variant_answer_rules(family)}\n\n"
        "Structured trace requirements:\n"
        f"- Trace mode: {trace_mode}\n"
        f"- Difficulty: {difficulty}\n"
        f"- Institution: {institution}\n"
        f"- {debate_behavior_line if trace_mode == 'debate' else monologue_behavior_line}\n"
        f"- {institution_line}\n"
        f"- Use no more than {max_personas} personas and no more than {max_debate_turns} turns.\n"
        f"- Target conversation length for this difficulty is roughly {turn_low}-{turn_high} turns.\n"
        f"{role_metadata}"
        "- Cite the real task IDs naturally in the conversation when you refer to inputs or evidence.\n\n"
        "Reasoning trace example:\n"
        f"{reasoning_contract}\n\n"
        "Visible answer example:\n"
        f"{answer_contract}\n\n"
        f"{task_block}"
    )
    system_message = (
        "Solve the task with task-focused reasoning. "
        "The reasoning trace should look like paper-style internal conversation, and the visible answer must stay separate."
    )
    if trace_prompt_variant == "debug_minimal":
        system_message = (
            "Follow the paper-style format exactly. "
            "One thought conversation, one visible answer block."
        )
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]


def _build_task_block(family: str, task: dict[str, Any], objective_profile: str) -> str:
    if family == "countdown":
        numbers = ", ".join(f'{item["id"]}={item["value"]}' for item in task["numbers"])
        return (
            "Task family: countdown_debate\n"
            + f"Numbers: {numbers}\n"
            + f"Target: T={task['target']}\n"
            + (
                "Goal: use the conversation to explore grounded arithmetic paths. Return only the final arithmetic expression in <answer>.\n"
                if objective_profile == "debate_primary"
                else "Goal: produce a valid arithmetic expression in <answer> that reaches the target.\n"
            )
            + "Use only the provided numbers and use each number at most once."
        )

    evidence_lines = "\n".join(
        f'{entry["id"]}: {entry["text"]}' for entry in task["evidence"]
    )
    return (
        "Task family: evidence_verdict_debate\n"
        + f'Claim: {task["claim"]}\n'
        + "Evidence snippets:\n"
        + f"{evidence_lines}\n"
        + (
            "Goal: debate the evidence, challenge stale or conflicting snippets, then return the verdict in <answer> and decisive evidence IDs in <support>."
            if objective_profile == "debate_primary"
            else "Goal: classify the claim in <answer> and list decisive evidence IDs in <support>."
        )
    )


def _make_countdown_task(seed: int, difficulty: str, countdown_reward_profile: str = "benchmark") -> dict[str, Any]:
    rng = random.Random(seed)
    num_count = {"easy": 3, "medium": 4, "hard": 5}[difficulty]
    ops_by_difficulty = {
        "easy": ("+", "-", "*"),
        "medium": COUNTDOWN_OPS,
        "hard": COUNTDOWN_OPS,
    }

    for _ in range(200):
        numbers = rng.sample(range(2, 16), num_count)
        target, expression = _build_expression(numbers, rng, ops_by_difficulty[difficulty], difficulty)
        if target is None or expression is None:
            continue
        if difficulty == "hard" and not _has_near_miss(numbers, target):
            continue
        if difficulty == "easy" and target > 120:
            continue
        return {
            "numbers": [
                {"id": f"N{idx + 1}", "value": value}
                for idx, value in enumerate(numbers)
            ],
            "target": target,
            "oracle_expression": expression,
            "oracle_support": [f"N{idx + 1}" for idx in range(len(numbers))] + ["T"],
            "reward_profile": countdown_reward_profile,
        }

    raise RuntimeError(f"Failed to generate countdown task for difficulty={difficulty}")


def _build_expression(
    numbers: list[int],
    rng: random.Random,
    allowed_ops: tuple[str, ...],
    difficulty: str,
) -> tuple[int | None, str | None]:
    values = [Fraction(number, 1) for number in numbers]
    expressions = [str(number) for number in numbers]
    while len(values) > 1:
        i, j = sorted(rng.sample(range(len(values)), 2), reverse=True)
        left_value = values.pop(i)
        right_value = values.pop(j)
        left_expr = expressions.pop(i)
        right_expr = expressions.pop(j)
        candidates: list[tuple[Fraction, str]] = []
        for op in allowed_ops:
            if op == "+":
                candidates.append((left_value + right_value, f"({left_expr}+{right_expr})"))
            elif op == "-":
                if left_value != right_value:
                    candidates.append((left_value - right_value, f"({left_expr}-{right_expr})"))
                    candidates.append((right_value - left_value, f"({right_expr}-{left_expr})"))
            elif op == "*":
                candidates.append((left_value * right_value, f"({left_expr}*{right_expr})"))
            elif op == "/":
                if right_value != 0 and left_value / right_value == int(left_value / right_value):
                    candidates.append((left_value / right_value, f"({left_expr}/{right_expr})"))
                if left_value != 0 and right_value / left_value == int(right_value / left_value):
                    candidates.append((right_value / left_value, f"({right_expr}/{left_expr})"))
        rng.shuffle(candidates)
        chosen_value = None
        chosen_expr = None
        for value, expr in candidates:
            if value.denominator != 1:
                continue
            if value <= 0 or value > 200:
                continue
            if difficulty == "easy" and "/" in expr:
                continue
            chosen_value = value
            chosen_expr = expr
            break
        if chosen_value is None or chosen_expr is None:
            return None, None
        values.append(chosen_value)
        expressions.append(chosen_expr)
    final_value = values[0]
    if final_value.denominator != 1:
        return None, None
    return int(final_value), expressions[0]


def _has_near_miss(numbers: list[int], target: int) -> bool:
    for i, first in enumerate(numbers):
        for j, second in enumerate(numbers):
            if i == j:
                continue
            for candidate in (
                first + second,
                abs(first - second),
                first * second,
            ):
                if candidate != target and abs(candidate - target) <= 3:
                    return True
    return False


def _multiset_subset(used_values: list[int], available_values: list[int]) -> bool:
    remaining = list(available_values)
    for value in used_values:
        if value not in remaining:
            return False
        remaining.remove(value)
    return True


def _safe_eval_expression(expression: str) -> tuple[Fraction | None, list[int], bool]:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return None, [], False

    values_used: list[int] = []

    def _eval(node: ast.AST) -> Fraction:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            values_used.append(int(node.value))
            return Fraction(int(node.value), 1)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -_eval(node.operand)
        if isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                if right == 0:
                    raise ValueError("division by zero")
                return left / right
        raise ValueError("unsupported expression")

    try:
        value = _eval(tree)
    except ValueError:
        return None, [], False
    return value, values_used, True


def _make_evidence_task(
    seed: int,
    difficulty: str,
    forced_verdict: str | None = None,
) -> dict[str, Any]:
    rng = random.Random(seed)
    verdict = forced_verdict or rng.choice(("TRUE", "FALSE", "INSUFFICIENT"))
    template_type = rng.choice(("leader", "venue", "status"))

    entity = f"{rng.choice(ENTITY_PREFIXES)} {rng.choice(ENTITY_SUFFIXES)}"
    date1 = _make_date(rng, 1)
    date2 = _make_date(rng, 2)
    date3 = _make_date(rng, 3)

    if template_type == "leader":
        old_value, new_value, rumor_value = rng.sample(PEOPLE, 3)
        subject = "director"
    elif template_type == "venue":
        old_value, new_value, rumor_value = rng.sample(LOCATIONS, 3)
        subject = "venue"
    else:
        old_value, new_value, rumor_value = rng.sample(STATUSES, 3)
        subject = "status"

    evidence: list[dict[str, str]]
    if verdict == "TRUE":
        claim_value = new_value
        claim = f'The current {subject} for {entity} is "{claim_value}".'
        evidence = [
            {
                "id": "E1",
                "text": f"{date1} official directory entry: {entity} listed {old_value} as its {subject}.",
            },
            {
                "id": "E2",
                "text": f"{date2} official update: {entity} changed its {subject} to {new_value}.",
            },
            {
                "id": "E3",
                "text": f"{date3} rumor post: an unverified account claims the {subject} is {rumor_value}.",
            },
            {
                "id": "E4",
                "text": f"{date3} visitor note: the snack kiosk at {entity} closed early.",
            },
        ]
        oracle_support = ["E2"]
    elif verdict == "FALSE":
        claim_value = old_value
        claim = f'The current {subject} for {entity} is "{claim_value}".'
        evidence = [
            {
                "id": "E1",
                "text": f"{date1} official directory entry: {entity} listed {old_value} as its {subject}.",
            },
            {
                "id": "E2",
                "text": f"{date2} official correction: {entity} now names {new_value} as its {subject}.",
            },
            {
                "id": "E3",
                "text": f"{date3} second-hand recap: one attendee still repeats the old {subject} value {old_value}.",
            },
            {
                "id": "E4",
                "text": f"{date3} facilities bulletin: the east entrance opens at 08:00.",
            },
        ]
        oracle_support = ["E2"]
    else:
        claim_value = rumor_value
        claim = f'The current {subject} for {entity} is "{claim_value}".'
        evidence = [
            {
                "id": "E1",
                "text": f"{date1} draft planning memo: staff proposed {old_value} as the {subject}, pending confirmation.",
            },
            {
                "id": "E2",
                "text": f"{date2} official FAQ: {entity} says the {subject} is still to be announced.",
            },
            {
                "id": "E3",
                "text": f"{date3} rumor post: an unverified account claims the {subject} is {rumor_value}.",
            },
            {
                "id": "E4",
                "text": f"{date3} sponsor note: vendor booths will open thirty minutes later than usual.",
            },
        ]
        oracle_support = ["E2"]

    if difficulty == "hard":
        evidence.append(
            {
                "id": "E5",
                "text": f"{date2} partial transcript: one speaker questions whether {old_value} might still appear in stale documents.",
            }
        )
        evidence.append(
            {
                "id": "E6",
                "text": f"{date3} volunteer message: a volunteer guessed {rumor_value} but cited no official source.",
            }
        )
    elif difficulty == "medium":
        evidence.append(
            {
                "id": "E5",
                "text": f"{date2} attendee recap: someone mentioned hearing both {old_value} and {new_value} in hallway chatter.",
            }
        )

    return {
        "claim": claim,
        "evidence": evidence,
        "oracle_verdict": verdict,
        "oracle_support": oracle_support,
        "subject": subject,
        "entity": entity,
    }


def _make_date(rng: random.Random, month: int) -> str:
    day = rng.randint(1, 27)
    return f"2025-{month:02d}-{day:02d}"
