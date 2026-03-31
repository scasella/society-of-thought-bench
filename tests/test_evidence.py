from __future__ import annotations

from society_of_thought_bench.families import build_example, evaluate_evidence_verdict


def test_evidence_true_case_scores_full_verdict_credit() -> None:
    row = build_example(
        family="evidence",
        difficulty="easy",
        institution="auto",
        seed=3,
        index=0,
        max_personas=4,
        max_debate_turns=10,
        trace_mode="debate",
        split="eval",
        forced_verdict="TRUE",
    )
    task = __import__("json").loads(row["info"])["task"]
    result = evaluate_evidence_verdict("TRUE", task["oracle_support"], task)
    assert result.verdict_correct == 1.0
    assert result.support_f1 == 1.0


def test_evidence_incorrect_support_lowers_f1() -> None:
    row = build_example(
        family="evidence",
        difficulty="medium",
        institution="auto",
        seed=7,
        index=0,
        max_personas=4,
        max_debate_turns=10,
        trace_mode="debate",
        split="eval",
        forced_verdict="FALSE",
    )
    task = __import__("json").loads(row["info"])["task"]
    result = evaluate_evidence_verdict("FALSE", ["E1"], task)
    assert result.verdict_correct == 1.0
    assert result.support_f1 < 1.0
