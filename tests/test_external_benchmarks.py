from __future__ import annotations

import re

from society_of_thought_bench.external_benchmarks import (
    _build_gsm8k_sft_rows,
    _load_external_eval_env_from_lock_rows,
    _split_assistant_content,
    build_external_system_prompt,
    build_external_dpo_rows,
    compare_external_suite_aggregates,
    evaluate_external_acceptance,
    extract_reasoning_text,
    extract_visible_answer_text,
    reasoning_contract_is_valid,
    load_external_suite_lock_file,
    save_external_suite_lock_file,
    visible_answer_is_valid,
)
from society_of_thought_bench.training_data import (
    external_dpo_component_counts,
    external_sft_source_counts,
)


def test_external_source_counts_match_profile_mix() -> None:
    assert external_sft_source_counts(100, curriculum_profile="external_core_debate") == {
        "internal_sot": 40,
        "gsm8k_train": 20,
        "mmlu_pro_non_eval": 20,
        "mmlu_pro_stem": 10,
        "ifeval_synthetic": 10,
    }
    assert external_sft_source_counts(100, curriculum_profile="external_bridge_debate") == {
        "internal_sot": 55,
        "gsm8k_train": 10,
        "mmlu_pro_non_eval": 15,
        "mmlu_pro_stem": 10,
        "ifeval_synthetic": 10,
    }
    assert external_sft_source_counts(100, curriculum_profile="external_proxy_focus_debate") == {
        "internal_sot": 30,
        "gsm8k_train": 5,
        "mmlu_pro_non_eval": 10,
        "mmlu_pro_stem": 35,
        "ifeval_synthetic": 20,
    }
    assert external_dpo_component_counts(20, curriculum_profile="external_core_debate") == {
        "internal_debate_quality": 9,
        "external_answer_discipline": 2,
        "external_reasoning_structure": 9,
    }
    assert external_dpo_component_counts(100, curriculum_profile="external_core_monologue") == {
        "internal_monologue_structure": 30,
        "external_answer_discipline": 40,
        "external_monologue_structure": 30,
    }


def test_external_system_prompt_keeps_native_answer_discipline() -> None:
    gsm_prompt = build_external_system_prompt(
        "gsm8k",
        trace_mode="debate",
        original_system_prompt="Please reason step by step, and put your final answer within \\boxed{}.",
    )
    mmlu_prompt = build_external_system_prompt("mmlu_pro", trace_mode="monologue")
    assert "society-of-thought" in gsm_prompt
    assert "\\boxed{}" in gsm_prompt
    assert "simple speaker lines are allowed" in gsm_prompt
    assert "single voice" in mmlu_prompt
    assert "boxed answer letter" in mmlu_prompt


def test_visible_answer_validity_is_benchmark_specific() -> None:
    assert visible_answer_is_valid("gsm8k", r"\boxed{72}")
    assert not visible_answer_is_valid("gsm8k", "72")
    assert visible_answer_is_valid("mmlu_pro", r"\boxed{I}")
    assert not visible_answer_is_valid("mmlu_pro", r"\boxed{11}")
    assert visible_answer_is_valid("gpqa", r"\boxed{B}")
    assert not visible_answer_is_valid("gpqa", r"\boxed{I}")
    assert visible_answer_is_valid("ifeval", "Short compliant response")


def test_reasoning_contract_checks_debate_vs_monologue() -> None:
    debate_reasoning = (
        "<cast_of_characters>\n"
        "<persona1>Role: solver\nPersonality: high_openness\nExpertise: arithmetic\nStyle: direct</persona1>\n"
        "<persona2>Role: verifier\nPersonality: high_conscientiousness\nExpertise: checking\nStyle: precise</persona2>\n"
        "</cast_of_characters>\n<conversation>\n<think1>Propose an exact route.</think1>\n"
        "<think2>Challenge the route.</think2>\n</conversation>\n"
        "<group_solution>Keep only the exact answer.</group_solution>"
    )
    monologue_reasoning = "Work through the constraints in one voice and keep the final response concise."
    assert reasoning_contract_is_valid("debate", debate_reasoning)
    assert not reasoning_contract_is_valid("debate", monologue_reasoning)
    assert reasoning_contract_is_valid("monologue", monologue_reasoning)
    assert not reasoning_contract_is_valid("monologue", debate_reasoning)


def test_reasoning_contract_accepts_alternate_debate_dialects() -> None:
    character_step_reasoning = (
        "<cast_of_characters>\n"
        '<character name="The_Solver" role="Problem_Solver"/>\n'
        '<character name="The_Review" role="Problem_Review"/>\n'
        "</cast_of_characters>\n"
        "<conversation>\n"
        '<step speaker="The_Solver" action="Propose">Try the exact route first.</step>\n'
        '<step speaker="The_Review" action="Challenge">Pressure-test that route before committing.</step>\n'
        "</conversation>\n"
        "<group_solution>Keep only the checked route.</group_solution>"
    )
    named_speaker_reasoning = (
        "<cast_of_characters>\n"
        "<character>model</character>\n"
        "<character>hidden_discussion</character>\n"
        "</cast_of_characters>\n"
        "<conversation>\n"
        "<model>Start from the leading route.</model>\n"
        "<hidden_discussion>Challenge the weak branch and keep the checked one.</hidden_discussion>\n"
        "</conversation>\n"
        "<group_solution>Keep the checked route.</group_solution>"
    )
    assert reasoning_contract_is_valid("debate", character_step_reasoning)
    assert reasoning_contract_is_valid("debate", named_speaker_reasoning)
    speaker_line_reasoning = (
        "<cast_of_characters>\n"
        "Solver: exploratory lead\n"
        "Skeptic: challenge path\n"
        "</cast_of_characters>\n"
        "<conversation>\n"
        "Solver: Start from the strongest route.\n"
        "Skeptic: Challenge the weak assumptions before answering.\n"
        "</conversation>\n"
        "<group_solution>Keep the checked route.</group_solution>"
    )
    assert reasoning_contract_is_valid("debate", speaker_line_reasoning)


def test_open_think_external_outputs_split_reasoning_from_native_answer() -> None:
    completion = [
        {
            "role": "assistant",
            "content": (
                "<think>\n"
                "<cast_of_characters><character name=\"solver\"/></cast_of_characters>\n"
                "<conversation><step speaker=\"solver\">Check the arithmetic carefully.</step></conversation>\n"
                "<group_solution>23</group_solution>\n"
                "\\boxed{23}"
            ),
        }
    ]
    assert extract_reasoning_text(completion).endswith("</group_solution>")
    assert extract_visible_answer_text(completion) == r"\boxed{23}"


def test_open_think_without_visible_tail_keeps_visible_answer_empty() -> None:
    completion = [
        {
            "role": "assistant",
            "content": (
                "<think>\n"
                "<cast_of_characters><character>model</character><character>checker</character></cast_of_characters>\n"
                "<conversation><model>Draft.</model><checker>Check.</checker></conversation>\n"
                "<group_solution>DONE</group_solution>"
            ),
        }
    ]
    assert extract_visible_answer_text(completion) == ""


def test_external_gsm8k_rows_rotate_across_multiple_hidden_dialects() -> None:
    rows = _build_gsm8k_sft_rows(4, seed_start=0, trace_mode="debate")
    reasonings = [extract_reasoning_text([row["messages"][-1]]) for row in rows]
    assert "<persona1>" in reasonings[0]
    assert "<step speaker=" in reasonings[1]
    assert "<character name=" in reasonings[2]
    assert re.search(r"^[a-z_0-9-]+:\s+", reasonings[3].lower(), re.MULTILINE)


def test_external_gsm8k_hidden_reasoning_keeps_boxed_answer_outside_trace() -> None:
    row = _build_gsm8k_sft_rows(1, seed_start=11, trace_mode="debate")[0]
    completion = [row["messages"][-1]]
    reasoning = extract_reasoning_text(completion)
    answer = extract_visible_answer_text(completion)
    assert answer.startswith(r"\boxed{")
    assert answer not in reasoning
    assert "Challenge:" in reasoning or "challenge" in reasoning.lower()


def test_external_debate_dpo_pairs_keep_visible_answer_fixed_while_varying_hidden_discussion() -> None:
    rows, component_counts = build_external_dpo_rows(
        total=20,
        curriculum_profile="external_core_debate",
        split="train",
        seed_start=700,
    )
    assert component_counts["external_reasoning_structure"] > 0
    reasoning_rows = [row for row in rows if row.get("source") == "external_reasoning_structure"]
    assert reasoning_rows
    variants = {row["rejection_variant"] for row in reasoning_rows}
    assert variants == {"shallow_valid", "redundant_cast", "weak_challenge", "premature_reconcile"}
    for row in reasoning_rows:
        chosen_reasoning, chosen_answer = _split_assistant_content(row["completion_A"][0])
        rejected_reasoning, rejected_answer = _split_assistant_content(row["completion_B"][0])
        assert chosen_answer == rejected_answer
        assert chosen_reasoning != rejected_reasoning
        assert chosen_answer
        assert reasoning_contract_is_valid("debate", chosen_reasoning)
        assert reasoning_contract_is_valid("debate", rejected_reasoning)
        assert chosen_answer not in rejected_reasoning


def test_external_suite_comparison_acceptance_uses_macro_delta_and_non_negative_count() -> None:
    debate_suite = {
        "benchmarks": [
            {"benchmark": "gsm8k", "native_score": 0.52, "visible_answer_valid_rate": 1.0, "reasoning_contract_valid_rate": 0.9},
            {"benchmark": "mmlu_pro", "native_score": 0.44, "visible_answer_valid_rate": 1.0, "reasoning_contract_valid_rate": 0.8},
            {"benchmark": "gpqa", "native_score": 0.39, "visible_answer_valid_rate": 1.0, "reasoning_contract_valid_rate": 0.8},
            {"benchmark": "ifeval", "native_score": 0.61, "visible_answer_valid_rate": 0.95, "reasoning_contract_valid_rate": 0.75},
        ],
        "macro_average": {"native_score": 0.49, "visible_answer_valid_rate": 0.9875, "reasoning_contract_valid_rate": 0.8125},
    }
    mono_suite = {
        "benchmarks": [
            {"benchmark": "gsm8k", "native_score": 0.48, "visible_answer_valid_rate": 1.0, "reasoning_contract_valid_rate": 0.1},
            {"benchmark": "mmlu_pro", "native_score": 0.40, "visible_answer_valid_rate": 0.98, "reasoning_contract_valid_rate": 0.1},
            {"benchmark": "gpqa", "native_score": 0.35, "visible_answer_valid_rate": 0.99, "reasoning_contract_valid_rate": 0.1},
            {"benchmark": "ifeval", "native_score": 0.60, "visible_answer_valid_rate": 0.95, "reasoning_contract_valid_rate": 0.1},
        ],
        "macro_average": {"native_score": 0.4575, "visible_answer_valid_rate": 0.98, "reasoning_contract_valid_rate": 0.1},
    }
    comparison = compare_external_suite_aggregates(debate_suite, mono_suite, left_label="debate", right_label="monologue")
    acceptance = evaluate_external_acceptance(comparison, threshold_key="same_checkpoint")
    assert comparison["macro_average"]["native_score_delta"] > 0.03
    assert comparison["macro_average"]["non_negative_native_score_benchmarks"] == 4
    assert acceptance["passed"] is True


def test_external_suite_acceptance_fails_when_contract_delta_is_too_negative() -> None:
    debate_suite = {
        "benchmarks": [
            {"benchmark": "gsm8k", "native_score": 0.55, "visible_answer_valid_rate": 1.0, "reasoning_contract_valid_rate": 0.0},
            {"benchmark": "mmlu_pro", "native_score": 0.45, "visible_answer_valid_rate": 1.0, "reasoning_contract_valid_rate": 0.0},
            {"benchmark": "gpqa", "native_score": 0.40, "visible_answer_valid_rate": 1.0, "reasoning_contract_valid_rate": 0.0},
            {"benchmark": "ifeval", "native_score": 0.62, "visible_answer_valid_rate": 0.95, "reasoning_contract_valid_rate": 0.0},
        ],
        "macro_average": {"native_score": 0.505, "visible_answer_valid_rate": 0.9875, "reasoning_contract_valid_rate": 0.0},
    }
    mono_suite = {
        "benchmarks": [
            {"benchmark": "gsm8k", "native_score": 0.48, "visible_answer_valid_rate": 1.0, "reasoning_contract_valid_rate": 0.9},
            {"benchmark": "mmlu_pro", "native_score": 0.40, "visible_answer_valid_rate": 0.98, "reasoning_contract_valid_rate": 0.9},
            {"benchmark": "gpqa", "native_score": 0.35, "visible_answer_valid_rate": 0.99, "reasoning_contract_valid_rate": 0.9},
            {"benchmark": "ifeval", "native_score": 0.60, "visible_answer_valid_rate": 0.95, "reasoning_contract_valid_rate": 0.9},
        ],
        "macro_average": {"native_score": 0.4575, "visible_answer_valid_rate": 0.98, "reasoning_contract_valid_rate": 0.9},
    }
    comparison = compare_external_suite_aggregates(debate_suite, mono_suite, left_label="debate", right_label="monologue")
    acceptance = evaluate_external_acceptance(comparison, threshold_key="same_checkpoint")
    assert comparison["macro_average"]["native_score_delta"] > 0.03
    assert comparison["macro_average"]["reasoning_contract_valid_delta"] < -0.10
    assert acceptance["passed"] is False


def test_external_lock_file_round_trip_and_replay_env_keep_same_questions(tmp_path) -> None:
    rows = [
        {
            "benchmark": "gsm8k",
            "example_id": 7,
            "prompt": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "What is 6 times 7?"},
            ],
            "answer": "42",
            "task": "gsm8k",
            "info": {"source": "test"},
            "question": "What is 6 times 7?",
        }
    ]
    lock_path = tmp_path / "external_lock.json"
    save_external_suite_lock_file(lock_path, rows=rows, metadata={"kind": "test"})
    payload = load_external_suite_lock_file(lock_path)
    assert payload["metadata"]["kind"] == "test"
    assert payload["rows"][0]["question"] == "What is 6 times 7?"

    env = _load_external_eval_env_from_lock_rows("gsm8k", trace_mode="debate", lock_rows=payload["rows"])
    assert len(env.eval_dataset) == 1
    assert env.eval_dataset[0]["question"] == "What is 6 times 7?"
    assert env.eval_dataset[0]["prompt"][-1]["content"] == "What is 6 times 7?"
