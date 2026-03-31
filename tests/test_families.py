from __future__ import annotations

import json

from society_of_thought_bench.families import build_example, inspect_example


def test_build_example_records_prompt_variant_in_info() -> None:
    row = build_example(
        family="countdown",
        difficulty="easy",
        institution="auto",
        seed=0,
        index=0,
        max_personas=4,
        max_debate_turns=10,
        trace_mode="debate",
        split="eval",
        trace_prompt_variant="trace_tail_block",
    )
    info = json.loads(row["info"])
    assert info["trace_prompt_variant"] == "trace_tail_block"


def test_inspect_example_exposes_paper_style_contract() -> None:
    example = inspect_example(
        family="countdown",
        difficulty="easy",
        institution="auto",
        seed=0,
        max_personas=4,
        max_debate_turns=10,
        trace_mode="debate",
        trace_prompt_variant="debug_minimal",
    )
    prompt = example["prompt"][1]["content"]
    assert "<cast_of_characters>" in example["contracts"]["reasoning_trace"]
    assert "<conversation>" in example["contracts"]["reasoning_trace"]
    assert "<group_solution>" in example["contracts"]["reasoning_trace"]
    assert example["meta"]["trace_prompt_variant"] == "debug_minimal"
    assert "Visible answer example:" in prompt
    assert "paper-style" in prompt


def test_default_prompt_uses_paper_style_contract() -> None:
    example = inspect_example(
        family="evidence",
        difficulty="easy",
        institution="auto",
        seed=0,
        max_personas=4,
        max_debate_turns=10,
        trace_mode="debate",
    )
    prompt = example["prompt"][1]["content"]
    assert "<cast_of_characters>" in example["contracts"]["reasoning_trace"]
    assert "<answer>TRUE</answer>" in example["contracts"]["visible_answer"]
    assert "<support>E2</support>" in example["contracts"]["visible_answer"]
    assert "paper-style" in prompt
