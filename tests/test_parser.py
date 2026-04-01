from __future__ import annotations

from society_of_thought_bench.parser import SocietyOfThoughtParser


def _completion(content: str, reasoning_content: str | None = None) -> list[dict[str, str]]:
    message: dict[str, str] = {"role": "assistant", "content": content}
    if reasoning_content is not None:
        message["reasoning_content"] = reasoning_content
    return [message]


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


def test_parser_accepts_reasoning_content_trace_and_tagged_answer() -> None:
    parser = SocietyOfThoughtParser(trace_mode="debate")
    parsed = parser.parse_completion(
        _completion(
            content="<answer>4+6</answer>",
            reasoning_content=_debate_trace(),
        )
    )
    assert parsed.is_valid
    assert parsed.answer_valid
    assert parsed.trace_valid
    assert parsed.trace_source == "reasoning_content"
    assert parsed.final_answer == "4+6"
    assert len(parsed.personas) == 2
    assert parsed.turns[0].speaker == "P1"
    assert parsed.turns[1].reply_to == ["t1"]


def test_parser_accepts_monologue_think_tag_fallback() -> None:
    parser = SocietyOfThoughtParser(trace_mode="monologue")
    parsed = parser.parse_completion(
        _completion(
            content=(
                "<think>Use E1 as the current official evidence before deciding.</think>\n"
                "<answer>TRUE</answer>\n"
                "<support>E1</support>"
            )
        )
    )
    assert parsed.is_valid
    assert parsed.trace_source == "think_tags"
    assert parsed.final_answer == "TRUE"
    assert parsed.support == ["E1"]
    assert len(parsed.turns) == 1


def test_parser_rejects_answer_body_json_under_paper_contract() -> None:
    parser = SocietyOfThoughtParser(trace_mode="debate")
    parsed = parser.parse_completion(
        _completion(
            content='{"final_answer":"4+6","support":["N1","N2","T"]}',
            reasoning_content="",
        )
    )
    assert not parsed.is_valid
    assert "missing_reasoning_trace" in parsed.protocol_error_codes
    assert "answer_block_invalid" in parsed.answer_error_codes


def test_parser_rejects_duplicate_conversation_blocks() -> None:
    parser = SocietyOfThoughtParser(trace_mode="debate")
    reasoning = _debate_trace() + "\n<conversation><think1>Duplicate block.</think1></conversation>"
    parsed = parser.parse_completion(_completion(content="<answer>4+6</answer>", reasoning_content=reasoning))
    assert not parsed.trace_valid
    assert "conversation_block_invalid" in parsed.trace_error_codes


def test_parser_rejects_persona_turn_mismatch() -> None:
    parser = SocietyOfThoughtParser(trace_mode="debate")
    reasoning = (
        "<cast_of_characters>\n"
        "<persona1>Role: brainstormer\n"
        "Personality: high_openness\n"
        "Expertise: arithmetic\n"
        "Style: Generates candidate approaches and explores options.</persona1>\n"
        "</cast_of_characters>\n"
        "<conversation>\n"
        "<think2>Verification: 4+6=10 reaches T exactly.</think2>\n"
        "</conversation>\n"
        "<group_solution>Use 4+6.</group_solution>"
    )
    parsed = parser.parse_completion(_completion(content="<answer>4+6</answer>", reasoning_content=reasoning))
    assert not parsed.trace_valid
    assert "turn_persona_mismatch" in parsed.trace_error_codes


def test_parser_rejects_extra_visible_answer_text() -> None:
    parser = SocietyOfThoughtParser(trace_mode="debate")
    parsed = parser.parse_completion(
        _completion(
            content="<answer>4+6</answer> trailing text",
            reasoning_content=_debate_trace(),
        )
    )
    assert parsed.trace_valid
    assert not parsed.answer_valid
    assert "answer_unexpected_text" in parsed.answer_error_codes


def test_parser_strips_stray_think_marker_from_answer_channel() -> None:
    parser = SocietyOfThoughtParser(trace_mode="debate")
    parsed = parser.parse_completion(
        _completion(
            content="</think><answer>4+6</answer>",
            reasoning_content=_debate_trace(),
        )
    )
    assert parsed.answer_valid
    assert parsed.final_answer == "4+6"


def test_parser_parses_support_for_evidence_tasks() -> None:
    parser = SocietyOfThoughtParser(trace_mode="debate")
    parsed = parser.parse_completion(
        _completion(
            content="<answer>TRUE</answer>\n<support>E2</support>",
            reasoning_content=(
                "<cast_of_characters>\n"
                "<persona1>Role: brainstormer\n"
                "Personality: high_openness\n"
                "Expertise: fact_extraction\n"
                "Style: Generates candidate approaches and explores options.</persona1>\n"
                "<persona2>Role: verifier\n"
                "Personality: high_conscientiousness\n"
                "Expertise: timeline_reasoning\n"
                "Style: Checks calculations, evidence, and consistency.</persona2>\n"
                "</cast_of_characters>\n"
                "<conversation>\n"
                "<think1>Which evidence makes E2 decisive?</think1>\n"
                "<think2>Verification: E2 is current while E1 is stale.</think2>\n"
                "</conversation>\n"
                "<group_solution>TRUE from E2.</group_solution>"
            ),
        )
    )
    assert parsed.is_valid
    assert parsed.final_answer == "TRUE"
    assert parsed.support == ["E2"]


def test_parser_accepts_character_step_dialect() -> None:
    parser = SocietyOfThoughtParser(trace_mode="debate")
    reasoning = (
        "<cast_of_characters>\n"
        '<character name="The_Solver" role="Problem_Solver"/>\n'
        '<character name="The_Review" role="Problem_Review"/>\n'
        '<character name="The_Validator" role="Problem_Validator"/>\n'
        "</cast_of_characters>\n"
        "<conversation>\n"
        '<step speaker="The_Solver" step="1" action="Propose">Initial route uses N1 and N2 to reach T.</step>\n'
        '<step speaker="The_Review" step="2" action="Challenge">That route might drop one quantity.</step>\n'
        '<step speaker="The_Validator" step="3" action="Verify">Verification: the checked route still reaches T exactly.</step>\n'
        "</conversation>\n"
        "<group_solution>Keep the checked route.</group_solution>"
    )
    parsed = parser.parse_completion(_completion(content="<answer>4+6</answer>", reasoning_content=reasoning))
    assert parsed.trace_valid
    assert len(parsed.personas) == 3
    assert len(parsed.turns) == 3
    assert parsed.turns[1].act == "challenge"
    assert parsed.turns[2].act == "verify"


def test_parser_accepts_named_speaker_dialect() -> None:
    parser = SocietyOfThoughtParser(trace_mode="debate")
    reasoning = (
        "<cast_of_characters>\n"
        "<character>model</character>\n"
        "<character>hidden_discussion</character>\n"
        "</cast_of_characters>\n"
        "<conversation>\n"
        "<model>Initial route looks plausible but needs a real check.</model>\n"
        "<hidden_discussion>Challenge the route, reject the weak branch, and keep the checked answer.</hidden_discussion>\n"
        "</conversation>\n"
        "<group_solution>Keep the checked answer.</group_solution>"
    )
    parsed = parser.parse_completion(_completion(content="<answer>TRUE</answer>", reasoning_content=reasoning))
    assert parsed.trace_valid
    assert len(parsed.personas) == 2
    assert len(parsed.turns) == 2
    assert parsed.turns[1].speaker == "hidden_discussion"


def test_parser_accepts_speaker_line_dialect() -> None:
    parser = SocietyOfThoughtParser(trace_mode="debate")
    reasoning = (
        "<cast_of_characters>\n"
        "Solver: exploratory arithmetic lead\n"
        "Skeptic: adversarial checker\n"
        "Checker: final verification\n"
        "</cast_of_characters>\n"
        "<conversation>\n"
        "Solver: Start from the clean route to T.\n"
        "Skeptic: Challenge the first route before trusting it.\n"
        "Checker: Verify the surviving route and keep the exact answer.\n"
        "</conversation>\n"
        "<group_solution>Keep the checked route.</group_solution>"
    )
    parsed = parser.parse_completion(_completion(content="<answer>4+6</answer>", reasoning_content=reasoning))
    assert parsed.trace_valid
    assert len(parsed.personas) == 3
    assert len(parsed.turns) == 3
    assert parsed.turns[0].speaker == "solver"
