from __future__ import annotations

from society_of_thought_bench.families import evaluate_countdown_expression, inspect_example


def test_countdown_oracle_expression_scores_full_credit() -> None:
    example = inspect_example(
        family="countdown",
        difficulty="easy",
        institution="auto",
        seed=11,
        max_personas=4,
        max_debate_turns=10,
        trace_mode="debate",
    )
    task = example["oracle"]
    result = evaluate_countdown_expression(task["oracle_expression"], task)
    assert result.expression_valid == 1.0
    assert result.number_usage_valid == 1.0
    assert result.exact_hit == 1.0
    assert result.task_score == 1.0


def test_countdown_reused_number_is_not_fully_valid() -> None:
    task = {
        "numbers": [
            {"id": "N1", "value": 4},
            {"id": "N2", "value": 6},
            {"id": "N3", "value": 8},
        ],
        "target": 10,
        "oracle_support": ["N1", "N2", "T"],
    }
    result = evaluate_countdown_expression("4+4+2", task)
    assert result.expression_valid == 1.0
    assert result.number_usage_valid == 0.0


def test_countdown_exact_emphasis_reduces_non_exact_partial_credit() -> None:
    task = {
        "numbers": [
            {"id": "N1", "value": 4},
            {"id": "N2", "value": 6},
            {"id": "N3", "value": 8},
        ],
        "target": 20,
        "oracle_support": ["N1", "N2", "N3", "T"],
    }
    benchmark = evaluate_countdown_expression("8+6", task)
    exact_emphasis = evaluate_countdown_expression("8+6", {**task, "reward_profile": "exact_emphasis"})
    assert benchmark.expression_valid == 1.0
    assert benchmark.number_usage_valid == 1.0
    assert benchmark.exact_hit == 0.0
    assert exact_emphasis.task_score < benchmark.task_score
    assert exact_emphasis.task_score <= 0.70
