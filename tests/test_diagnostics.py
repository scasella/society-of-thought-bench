from __future__ import annotations

import json
import os
from pathlib import Path

from society_of_thought_bench.diagnostics import analyze_results, latest_results_path


def _debate_trace() -> str:
    return (
        "<cast_of_characters>\n"
        "<persona1>Role: brainstormer\n"
        "Personality: high_openness\n"
        "Expertise: arithmetic\n"
        "Style: Generates candidate approaches and explores options.</persona1>\n"
        "<persona2>Role: verifier\n"
        "Personality: high_conscientiousness\n"
        "Expertise: verification\n"
        "Style: Checks calculations, evidence, and consistency.</persona2>\n"
        "</cast_of_characters>\n"
        "<conversation>\n"
        "<think1>Which legal path uses N1, N2, and T?</think1>\n"
        "<think2>Verification: 4+6=10 reaches N1 N2 T exactly.</think2>\n"
        "</conversation>\n"
        "<group_solution>Use 4+6.</group_solution>"
    )


def test_analyze_results_computes_rates_for_paper_contract(tmp_path: Path) -> None:
    results_path = tmp_path / "results.jsonl"
    rows = [
        {
            "example_id": 0,
            "reward": 0.0,
            "task_score": 1.0,
            "metrics": {"task_score": 1.0, "task_correct_but_protocol_invalid": 1.0},
            "info": {"trace_mode": "debate", "trace_prompt_variant": "official"},
            "completion": [
                {
                    "role": "assistant",
                    "content": "<answer>4+6</answer>",
                    "reasoning_content": "I checked N1, N2, and T but I am only narrating the debate informally.",
                }
            ],
        },
        {
            "example_id": 1,
            "reward": 1.0,
            "task_score": 1.0,
            "metrics": {"task_score": 1.0},
            "info": {"trace_mode": "monologue", "trace_prompt_variant": "debug_minimal"},
            "completion": [
                {
                    "role": "assistant",
                    "content": "<think>Use E1 as the current official evidence.</think>\n<answer>TRUE</answer>\n<support>E1</support>",
                }
            ],
        },
    ]
    with results_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    summary = analyze_results(results_path, only_invalid=False, limit=10)
    assert summary["examples_total"] == 2
    assert summary["invalid_examples"] == 1
    assert summary["joint_contract_valid_rate"] == 0.5
    assert summary["task_and_joint_valid_rate"] == 0.5
    assert summary["near_miss_rate"] == 0.5
    assert summary["trace_format_valid_rate"] == 0.5
    assert summary["answer_format_valid_rate"] == 1.0
    assert summary["trace_prompt_variant_counts"]["official"] == 1
    assert summary["examples"][0]["primary_error_code"] == "cast_block_invalid"
    assert summary["examples"][1]["joint_contract_valid"] == 1.0


def test_latest_results_path_picks_newest_results_file(tmp_path: Path) -> None:
    older = tmp_path / "a" / "results.jsonl"
    newer = tmp_path / "b" / "results.jsonl"
    older.parent.mkdir(parents=True)
    newer.parent.mkdir(parents=True)
    older.write_text("{}\n")
    newer.write_text("{}\n")
    os.utime(older, (1, 1))
    os.utime(newer, (2, 2))
    assert latest_results_path(tmp_path) == newer
