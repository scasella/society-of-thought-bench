from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

ALLOWED_ACTS = (
    "question",
    "propose",
    "challenge",
    "verify",
    "shift",
    "reconcile",
)

ALLOWED_ROLES = (
    "brainstormer",
    "devils_advocate",
    "verifier",
    "synthesizer",
)

ALLOWED_PERSONALITIES = (
    "high_openness",
    "low_agreeableness",
    "high_conscientiousness",
    "high_agreeableness",
    "high_extraversion",
    "high_neuroticism",
)

EXPERTISE_BY_FAMILY = {
    "countdown": (
        "arithmetic",
        "search",
        "error_checking",
        "verification",
    ),
    "evidence": (
        "fact_extraction",
        "timeline_reasoning",
        "contradiction_checking",
        "synthesis",
    ),
}

STYLE_BY_ROLE = {
    "brainstormer": "Generates candidate approaches and explores options.",
    "devils_advocate": "Pushes on weak assumptions and stresses alternatives.",
    "verifier": "Checks calculations, evidence, and consistency.",
    "synthesizer": "Resolves disagreement and writes the final consensus.",
}

TASK_NAME_BY_FAMILY = {
    "countdown": "countdown_debate",
    "evidence": "evidence_verdict_debate",
}

PERSONA_MINIMUMS = {
    "easy": 2,
    "medium": 3,
    "hard": 4,
}

PREFERRED_TURN_RANGES = {
    "easy": (3, 5),
    "medium": (5, 7),
    "hard": (7, 10),
}

THINK_TAG = "think"
ANSWER_TAG = "answer"
SUPPORT_TAG = "support"
CAST_TAG = "cast_of_characters"
CONVERSATION_TAG = "conversation"
GROUP_SOLUTION_TAG = "group_solution"
TRACE_PROMPT_VARIANTS = (
    "official",
    "trace_minimal",
    "trace_tail_block",
    "debug_minimal",
)
OBJECTIVE_PROFILES = (
    "debate_primary",
    "balanced",
)


@dataclass(slots=True)
class Persona:
    id: str
    role: str
    personality: str
    expertise: str
    style: str = ""
    ordinal: int = 0


@dataclass(slots=True)
class TraceTurn:
    id: str
    act: str
    refs: list[str]
    content: str
    speaker: str | None = None
    reply_to: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ParsedTrace:
    trace_mode: str
    raw_trace: dict[str, Any] | None
    raw_answer: dict[str, Any] | None
    personas: list[Persona]
    turns: list[TraceTurn]
    final_answer: str
    support: list[str]
    group_solution: str | None
    reasoning_text: str = ""
    answer_text: str = ""
    trace_source: str | None = None
    protocol_error_codes: list[str] = field(default_factory=list)
    protocol_errors: list[str] = field(default_factory=list)
    trace_error_codes: list[str] = field(default_factory=list)
    trace_errors: list[str] = field(default_factory=list)
    answer_error_codes: list[str] = field(default_factory=list)
    answer_errors: list[str] = field(default_factory=list)

    @property
    def errors(self) -> list[str]:
        return [
            *self.protocol_errors,
            *self.trace_errors,
            *self.answer_errors,
        ]

    @property
    def error_codes(self) -> list[str]:
        return [
            *self.protocol_error_codes,
            *self.trace_error_codes,
            *self.answer_error_codes,
        ]

    @property
    def primary_error_code(self) -> str | None:
        if self.protocol_error_codes:
            return self.protocol_error_codes[0]
        if self.trace_error_codes:
            return self.trace_error_codes[0]
        if self.answer_error_codes:
            return self.answer_error_codes[0]
        return None

    @property
    def is_valid(self) -> bool:
        return not self.errors

    @property
    def protocol_valid(self) -> bool:
        return not self.errors

    @property
    def trace_valid(self) -> bool:
        return self.raw_trace is not None and not self.trace_errors

    @property
    def answer_valid(self) -> bool:
        return self.raw_answer is not None and not self.answer_errors


def required_acts(difficulty: str) -> tuple[str, ...]:
    if difficulty == "easy":
        return ("question", "propose", "verify")
    if difficulty == "medium":
        return ("question", "challenge", "verify", "reconcile")
    return ("question", "challenge", "shift", "verify", "reconcile")


def resolve_institution(difficulty: str, institution: str) -> str:
    if institution == "auto":
        return "hierarchical" if difficulty == "hard" else "flat"
    return institution


def difficulty_turn_range(difficulty: str, max_debate_turns: int) -> tuple[int, int]:
    low, high = PREFERRED_TURN_RANGES[difficulty]
    return low, min(high, max_debate_turns)


def normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def word_count(text: str) -> int:
    return len([token for token in text.strip().split() if token])


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered
