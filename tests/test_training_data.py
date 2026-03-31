from __future__ import annotations

import json

import pytest

from society_of_thought_bench.parser import SocietyOfThoughtParser
from society_of_thought_bench.training_data import (
    available_pair_types,
    build_dpo_pair_example,
    build_parser_completion,
    build_warmup_example,
    compare_mode_summaries,
    dpo_mix_counts,
    evaluate_gate,
    select_training_institution,
    summarize_generate_outputs,
    warmup_mix_counts,
)


def test_countdown_hard_warmup_example_contains_shift_and_reconcile() -> None:
    example = build_warmup_example(
        family="countdown",
        difficulty="hard",
        seed=10_000,
        split="train",
    )
    parser = SocietyOfThoughtParser(trace_mode="debate")
    parsed = parser.parse_completion([build_parser_completion(example)])
    acts = {turn.act for turn in parsed.turns}
    assert parsed.protocol_valid
    assert parsed.trace_valid
    assert parsed.answer_valid
    assert {"challenge", "shift", "reconcile"} <= acts
    assert len(parsed.personas) == 4


def test_evidence_medium_warmup_example_contains_challenge() -> None:
    example = build_warmup_example(
        family="evidence",
        difficulty="medium",
        seed=20_000,
        split="eval",
    )
    parser = SocietyOfThoughtParser(trace_mode="debate")
    parsed = parser.parse_completion([build_parser_completion(example)])
    acts = {turn.act for turn in parsed.turns}
    assert parsed.protocol_valid
    assert parsed.trace_valid
    assert parsed.answer_valid
    assert "challenge" in acts
    assert parsed.final_answer == example["oracle_task"]["oracle_verdict"]


def test_warmup_mix_counts_match_debate_first_defaults() -> None:
    assert warmup_mix_counts(8_000) == {
        ("countdown", "easy"): 800,
        ("countdown", "medium"): 2_000,
        ("countdown", "hard"): 1_200,
        ("evidence", "easy"): 800,
        ("evidence", "medium"): 2_000,
        ("evidence", "hard"): 1_200,
    }
    assert warmup_mix_counts(1_000) == {
        ("countdown", "easy"): 100,
        ("countdown", "medium"): 250,
        ("countdown", "hard"): 150,
        ("evidence", "easy"): 100,
        ("evidence", "medium"): 250,
        ("evidence", "hard"): 150,
    }


def test_dpo_mix_counts_match_defaults() -> None:
    assert dpo_mix_counts(12_000)[("countdown", "medium")] == 3_000
    assert dpo_mix_counts(12_000)[("evidence", "hard")] == 1_800


def test_protocol_bootcamp_mix_counts_shift_toward_easy_and_medium() -> None:
    assert warmup_mix_counts(1_000, curriculum_profile="protocol_bootcamp") == {
        ("countdown", "easy"): 150,
        ("countdown", "medium"): 300,
        ("countdown", "hard"): 50,
        ("evidence", "easy"): 150,
        ("evidence", "medium"): 300,
        ("evidence", "hard"): 50,
    }
    assert dpo_mix_counts(1_000, curriculum_profile="protocol_bootcamp")[("evidence", "medium")] == 300


def test_build_dpo_pair_example_produces_valid_pair() -> None:
    example = build_dpo_pair_example(
        family="evidence",
        difficulty="hard",
        seed=30_000,
        split="train",
        pair_type="single_path",
    )
    parser = SocietyOfThoughtParser(trace_mode="debate")
    chosen = parser.parse_completion(example["completion_A"])
    rejected = parser.parse_completion(example["completion_B"])
    chosen_acts = {turn.act for turn in chosen.turns}
    rejected_acts = {turn.act for turn in rejected.turns}
    assert chosen.protocol_valid and rejected.protocol_valid
    assert chosen.answer_valid and rejected.answer_valid
    assert "shift" in chosen_acts
    assert "shift" not in rejected_acts


def test_protocol_bootcamp_pair_can_use_invalid_trace_but_keep_clean_answer() -> None:
    example = build_dpo_pair_example(
        family="countdown",
        difficulty="medium",
        seed=31_000,
        split="train",
        pair_type="missing_block",
        curriculum_profile="protocol_bootcamp",
    )
    parser = SocietyOfThoughtParser(trace_mode="debate")
    chosen = parser.parse_completion(example["completion_A"])
    rejected = parser.parse_completion(example["completion_B"])
    assert chosen.protocol_valid and chosen.trace_valid and chosen.answer_valid
    assert rejected.answer_valid
    assert not rejected.protocol_valid
    assert not rejected.trace_valid


def test_generate_outputs_summary_and_gate_include_debate_metrics() -> None:
    results = {
        "reward": [0.8, 0.6],
        "metrics": {
            "protocol_valid": [1.0, 1.0],
            "trace_format_valid": [1.0, 1.0],
            "answer_format_valid": [1.0, 1.0],
            "task_score": [0.2, 0.4],
            "task_correct_but_protocol_invalid": [0.0, 0.0],
            "behavior_coverage": [1.0, 0.8],
            "persona_diversity": [0.9, 0.8],
            "interaction_score": [0.7, 0.3],
            "debate_relevance": [0.9, 0.7],
            "disagreement_quality": [0.8, 0.6],
            "challenge_response_pair_count": [1.0, 1.0],
            "alternative_path_count": [1.0, 0.0],
            "reconcile_link_count": [1.0, 1.0],
        },
    }
    summary = summarize_generate_outputs(results)
    assert summary["joint_contract_valid_rate"] == 1.0
    assert summary["behavior_coverage"] == pytest.approx(0.9)
    assert summary["disagreement_quality"] == pytest.approx(0.7)

    gate = evaluate_gate(summary, suite="debate_medium_gate", stage="debate_primary")
    assert gate.passed is False
    assert any(check["metric"] == "interaction_score" for check in gate.checks)


def test_compare_mode_summaries_reports_disagreement_delta() -> None:
    debate = {
        "average_reward": 0.62,
        "average_task_score": 0.22,
        "joint_contract_valid_rate": 0.98,
        "disagreement_quality": 0.74,
        "persona_diversity": 0.88,
        "interaction_score": 0.58,
        "behavior_coverage": 0.94,
        "debate_relevance": 0.82,
    }
    monologue = {
        "average_reward": 0.46,
        "average_task_score": 0.20,
        "joint_contract_valid_rate": 0.99,
        "disagreement_quality": 0.05,
        "persona_diversity": 0.0,
        "interaction_score": 0.0,
        "behavior_coverage": 0.61,
        "debate_relevance": 0.55,
    }
    comparison = compare_mode_summaries(debate, monologue)
    assert comparison["reward_delta"] == pytest.approx(0.16)
    assert comparison["disagreement_quality_delta"] == pytest.approx(0.69)


def test_institution_sampler_matches_requested_distribution_edges() -> None:
    assert select_training_institution("easy", 1) == "flat"
    assert select_training_institution("medium", 0) in {"flat", "hierarchical"}
    assert select_training_institution("hard", 0) in {"flat", "hierarchical"}


def test_pair_type_availability_expands_with_difficulty() -> None:
    assert "single_path" not in available_pair_types("medium")
    assert "single_path" in available_pair_types("hard")
    assert "missing_block" in available_pair_types("easy", curriculum_profile="protocol_bootcamp")
    assert "single_path" not in available_pair_types("medium", curriculum_profile="protocol_bootcamp")
