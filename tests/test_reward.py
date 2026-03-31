from __future__ import annotations

import asyncio

from society_of_thought_bench.parser import SocietyOfThoughtParser
from society_of_thought_bench.scoring import ClippedRubric, SocietyOfThoughtScorer, build_rubric


STYLE = {
    "brainstormer": "Generates candidate approaches and explores options.",
    "devils_advocate": "Pushes on weak assumptions and stresses alternatives.",
    "verifier": "Checks calculations, evidence, and consistency.",
    "synthesizer": "Resolves disagreement and writes the final consensus.",
}


def _completion(answer: str, reasoning: str | None = None) -> list[dict[str, str]]:
    message: dict[str, str] = {"role": "assistant", "content": answer}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return [message]


def _state(info: dict, answer: str, reasoning: str | None = None) -> dict:
    return {
        "info": info,
        "completion": _completion(answer, reasoning),
        "prompt": [],
        "task": info["family"],
        "trajectory": [],
        "timing": {"total_ms": 0.0},
    }


def _countdown_info(difficulty: str = "medium") -> dict:
    return {
        "family": "countdown",
        "difficulty": difficulty,
        "trace_mode": "debate",
        "objective_profile": "debate_primary",
        "task": {
            "numbers": [
                {"id": "N1", "value": 4},
                {"id": "N2", "value": 6},
                {"id": "N3", "value": 8},
                {"id": "N4", "value": 3},
                {"id": "N5", "value": 5},
            ],
            "target": 10,
            "oracle_support": ["N1", "N2", "T"],
        },
    }


def _evidence_info() -> dict:
    return {
        "family": "evidence",
        "difficulty": "medium",
        "trace_mode": "debate",
        "objective_profile": "debate_primary",
        "task": {
            "claim": 'The current venue is "Riverside Hall".',
            "evidence": [
                {"id": "E1", "text": "Old official entry says East Atrium."},
                {"id": "E2", "text": "Current official correction says Riverside Hall."},
                {"id": "E3", "text": "Rumor post repeats East Atrium."},
            ],
            "oracle_verdict": "TRUE",
            "oracle_support": ["E2"],
        },
    }


def _paper_trace(payload: dict) -> str:
    persona_lines = []
    for idx, persona in enumerate(payload["personas"], start=1):
        persona_lines.append(
            f"<persona{idx}>Role: {persona['role']}\n"
            f"Personality: {persona['personality']}\n"
            f"Expertise: {persona['expertise']}\n"
            f"Style: {persona['style']}</persona{idx}>"
        )
    turn_lines = []
    for turn in payload["debate"]:
        ordinal = int(turn["speaker"].replace("P", ""))
        turn_lines.append(f"<think{ordinal}>{turn['content']}</think{ordinal}>")
    return (
        "<cast_of_characters>\n"
        + "\n".join(persona_lines)
        + "\n</cast_of_characters>\n"
        + "<conversation>\n"
        + "\n".join(turn_lines)
        + "\n</conversation>\n"
        + f"<group_solution>{payload['group_solution']}</group_solution>"
    )


def _medium_payload() -> dict:
    return {
        "personas": [
            {"id": "P1", "role": "brainstormer", "personality": "high_openness", "expertise": "arithmetic", "style": STYLE["brainstormer"]},
            {"id": "P2", "role": "devils_advocate", "personality": "low_agreeableness", "expertise": "search", "style": STYLE["devils_advocate"]},
            {"id": "P3", "role": "verifier", "personality": "high_conscientiousness", "expertise": "verification", "style": STYLE["verifier"]},
        ],
        "debate": [
            {"speaker": "P1", "content": "Which legal path hits N1 N2 T?"},
            {"speaker": "P1", "content": "Primary proposal: 4+6 reaches N1 N2 T exactly."},
            {"speaker": "P2", "content": "But challenge the proposal first: confirm there is no hidden reuse and that 4+6=10 really hits T."},
            {"speaker": "P3", "content": "Verification: 4+6=10 uses N1 and N2 once each, so the candidate survives the challenge and reaches T exactly."},
            {"speaker": "P1", "content": "Reconcile on 4+6 because the challenge improved confidence without changing the verified route to N1 N2 T."},
        ],
        "group_solution": "Use 4+6.",
    }


def _medium_shallow_payload() -> dict:
    return {
        "personas": _medium_payload()["personas"],
        "debate": [
            {"speaker": "P1", "content": "Which legal path hits N1 N2 T?"},
            {"speaker": "P1", "content": "Primary proposal: 4+6 reaches N1 N2 T exactly."},
            {"speaker": "P2", "content": "I agree with the proposal and do not see any real problem."},
            {"speaker": "P3", "content": "Verification: 4+6=10 reaches T exactly."},
            {"speaker": "P1", "content": "Final agreement on 4+6."},
        ],
        "group_solution": "Use 4+6.",
    }


def _hard_payload() -> dict:
    return {
        "personas": [
            {"id": "P1", "role": "brainstormer", "personality": "high_openness", "expertise": "arithmetic", "style": STYLE["brainstormer"]},
            {"id": "P2", "role": "devils_advocate", "personality": "low_agreeableness", "expertise": "search", "style": STYLE["devils_advocate"]},
            {"id": "P3", "role": "verifier", "personality": "high_conscientiousness", "expertise": "verification", "style": STYLE["verifier"]},
            {"id": "P4", "role": "synthesizer", "personality": "high_agreeableness", "expertise": "error_checking", "style": STYLE["synthesizer"]},
        ],
        "debate": [
            {"speaker": "P1", "content": "Which route and alternative branch should we compare for N1 N2 N3 T?"},
            {"speaker": "P1", "content": "Main proposal: 4+6 reaches N1 N2 T exactly."},
            {"speaker": "P2", "content": "But challenge the main path and compare it against an alternative branch that starts from N1 and N3 before trusting T."},
            {"speaker": "P3", "content": "Alternative path: start with N1+N3 and compare that branch against the main route to T, even though it looks less direct."},
            {"speaker": "P3", "content": "Verification: 4+6=10 still beats the alternative branch and survives the challenge on legality and exactness for N1 N2 T."},
            {"speaker": "P4", "content": "Reconcile on 4+6 and record the alternative branch as checked but inferior for reaching T exactly."},
        ],
        "group_solution": "Use 4+6.",
    }


def _hard_single_path_payload() -> dict:
    payload = _hard_payload()
    payload["debate"][3] = {"speaker": "P3", "content": "Verification: stay on the original route and skip any real alternative path for T."}
    return payload


def _evidence_payload() -> dict:
    return {
        "personas": [
            {"id": "P1", "role": "brainstormer", "personality": "high_openness", "expertise": "fact_extraction", "style": STYLE["brainstormer"]},
            {"id": "P2", "role": "devils_advocate", "personality": "low_agreeableness", "expertise": "contradiction_checking", "style": STYLE["devils_advocate"]},
            {"id": "P3", "role": "verifier", "personality": "high_conscientiousness", "expertise": "timeline_reasoning", "style": STYLE["verifier"]},
        ],
        "debate": [
            {"speaker": "P1", "content": "Which evidence decides the current claim among E1 E2 E3?"},
            {"speaker": "P1", "content": "Initial proposal: TRUE because E2 is the current official correction."},
            {"speaker": "P2", "content": "But challenge that proposal: can the stale entry E1 or rumor E3 overturn the current evidence in E2?"},
            {"speaker": "P3", "content": "Verification: E2 is current, while E1 is stale and E3 is unsupported, so the timeline still favors TRUE."},
            {"speaker": "P1", "content": "Reconcile on TRUE with E2 as the decisive current support after resolving the conflict."},
        ],
        "group_solution": "TRUE from E2",
    }


def test_grounded_reasoning_trace_beats_ungrounded_trace() -> None:
    parser = SocietyOfThoughtParser(trace_mode="debate")
    scorer = SocietyOfThoughtScorer(max_personas=4, max_debate_turns=10)
    info = _countdown_info()
    answer = "<answer>4+6</answer>"
    grounded_score = scorer._ensure_metrics(_state(info, answer, _paper_trace(_medium_payload())), parser)["debate_relevance"]
    ungrounded = {
        "personas": _medium_payload()["personas"],
        "debate": [
            {"speaker": "P1", "content": "Maybe intuition is enough."},
            {"speaker": "P2", "content": "I agree without checking anything."},
        ],
        "group_solution": "Guess.",
    }
    ungrounded_score = scorer._ensure_metrics(_state(info, answer, _paper_trace(ungrounded)), parser)["debate_relevance"]
    assert grounded_score > ungrounded_score


def test_missing_reasoning_trace_marks_protocol_invalid() -> None:
    parser = SocietyOfThoughtParser(trace_mode="debate")
    scorer = SocietyOfThoughtScorer(max_personas=4, max_debate_turns=10)
    state = _state(_countdown_info(), "<answer>4+6</answer>", None)
    metrics = scorer._ensure_metrics(state, parser)
    assert metrics["protocol_valid"] == 0.0
    assert metrics["reasoning_trace_present"] == 0.0
    assert metrics["protocol_error_count"] == 1.0


def test_protocol_invalid_reward_is_zero() -> None:
    parser = SocietyOfThoughtParser(trace_mode="debate")
    scorer = SocietyOfThoughtScorer(max_personas=4, max_debate_turns=10)
    rubric = build_rubric(parser=parser, scorer=scorer, objective_profile="debate_primary")
    state = _state(_countdown_info(), "<answer>4+6</answer>", None)
    assert isinstance(rubric, ClippedRubric)
    asyncio.run(rubric.score_rollout(state))
    assert state["reward"] == 0.0


def test_correct_answer_with_bad_trace_sets_near_miss_metric() -> None:
    parser = SocietyOfThoughtParser(trace_mode="debate")
    scorer = SocietyOfThoughtScorer(max_personas=4, max_debate_turns=10)
    state = _state(
        _countdown_info(),
        "<answer>4+6</answer>",
        "I checked N1, N2, and T but I am only narrating the debate informally.",
    )
    metrics = scorer._ensure_metrics(state, parser)
    assert metrics["task_score"] == 1.0
    assert metrics["protocol_valid"] == 0.0
    assert metrics["task_correct_but_protocol_invalid"] == 1.0


def test_debate_bonuses_zero_out_when_answer_is_invalid() -> None:
    parser = SocietyOfThoughtParser(trace_mode="debate")
    scorer = SocietyOfThoughtScorer(max_personas=4, max_debate_turns=10)
    state = _state(
        _countdown_info(),
        "<answer>4+6</answer> trailing text",
        _paper_trace(_medium_payload()),
    )
    metrics = scorer._ensure_metrics(state, parser)
    assert metrics["trace_format_valid"] == 1.0
    assert metrics["answer_format_valid"] == 0.0
    assert metrics["behavior_coverage"] == 0.0
    assert metrics["disagreement_quality"] == 0.0


def test_debate_primary_keeps_relevance_when_task_score_is_low() -> None:
    parser = SocietyOfThoughtParser(trace_mode="debate")
    info = _evidence_info()
    answer = "<answer>FALSE</answer>\n<support>E1</support>"
    primary = SocietyOfThoughtScorer(max_personas=4, max_debate_turns=10, objective_profile="debate_primary")
    balanced = SocietyOfThoughtScorer(max_personas=4, max_debate_turns=10, objective_profile="balanced")
    primary_metrics = primary._ensure_metrics(_state(info, answer, _paper_trace(_evidence_payload())), parser)
    balanced_info = {**info, "objective_profile": "balanced"}
    balanced_metrics = balanced._ensure_metrics(_state(balanced_info, answer, _paper_trace(_evidence_payload())), parser)
    assert primary_metrics["task_score"] < 0.2
    assert primary_metrics["debate_relevance"] > 0.0
    assert balanced_metrics["debate_relevance"] > 0.0


def test_medium_disagreement_quality_beats_shallow_agreement() -> None:
    parser = SocietyOfThoughtParser(trace_mode="debate")
    scorer = SocietyOfThoughtScorer(max_personas=4, max_debate_turns=10)
    answer = "<answer>4+6</answer>"
    strong = scorer._ensure_metrics(_state(_countdown_info(), answer, _paper_trace(_medium_payload())), parser)
    shallow = scorer._ensure_metrics(_state(_countdown_info(), answer, _paper_trace(_medium_shallow_payload())), parser)
    assert strong["challenge_response_pair_count"] > 0.0
    assert strong["disagreement_quality"] > shallow["disagreement_quality"]


def test_hard_trace_with_alternative_path_scores_higher_than_single_path() -> None:
    parser = SocietyOfThoughtParser(trace_mode="debate")
    scorer = SocietyOfThoughtScorer(max_personas=4, max_debate_turns=10)
    info = _countdown_info(difficulty="hard")
    answer = "<answer>4+6</answer>"
    hard_metrics = scorer._ensure_metrics(_state(info, answer, _paper_trace(_hard_payload())), parser)
    single_path_metrics = scorer._ensure_metrics(_state(info, answer, _paper_trace(_hard_single_path_payload())), parser)
    assert hard_metrics["alternative_path_count"] == 1.0
    assert hard_metrics["disagreement_quality"] > single_path_metrics["disagreement_quality"]
