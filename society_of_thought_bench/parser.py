from __future__ import annotations

import re
from typing import Any

import verifiers as vf

from .core import (
    ALLOWED_PERSONALITIES,
    ALLOWED_ROLES,
    ANSWER_TAG,
    CAST_TAG,
    CONVERSATION_TAG,
    GROUP_SOLUTION_TAG,
    ParsedTrace,
    Persona,
    STYLE_BY_ROLE,
    SUPPORT_TAG,
    TraceTurn,
    THINK_TAG,
    ordered_unique,
)

THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
SUPPORT_PATTERN = re.compile(r"<support>(.*?)</support>", re.DOTALL | re.IGNORECASE)
CAST_PATTERN = re.compile(r"<cast_of_characters>(.*?)</cast_of_characters>", re.DOTALL | re.IGNORECASE)
CONVERSATION_PATTERN = re.compile(r"<conversation>(.*?)</conversation>", re.DOTALL | re.IGNORECASE)
GROUP_SOLUTION_PATTERN = re.compile(r"<group_solution>(.*?)</group_solution>", re.DOTALL | re.IGNORECASE)
PERSONA_PATTERN = re.compile(r"<persona(\d+)>(.*?)</persona\1>", re.DOTALL | re.IGNORECASE)
TURN_PATTERN = re.compile(r"<think(\d+)>(.*?)</think\1>", re.DOTALL | re.IGNORECASE)
FIELD_PATTERN = re.compile(r"(?im)^\s*(Role|Personality|Expertise|Style)\s*:\s*(.+?)\s*$")
THINK_MARKER_PATTERN = re.compile(r"</?think>", re.IGNORECASE)
REF_PATTERN = re.compile(r"\b(?:N\d+|E\d+|T)\b")


def _add_issue(errors: list[str], codes: list[str], code: str, detail: str | None = None) -> None:
    codes.append(code)
    errors.append(f"{code}: {detail}" if detail else code)


class SocietyOfThoughtParser(vf.Parser):
    def __init__(self, trace_mode: str):
        super().__init__()
        self.trace_mode = trace_mode

    def parse(self, text: str) -> ParsedTrace:
        answer_text = text
        reasoning_text = ""
        trace_source: str | None = None
        think_blocks = THINK_PATTERN.findall(text)
        if think_blocks:
            reasoning_text = "\n\n".join(block.strip() for block in think_blocks if block.strip())
            answer_text = THINK_PATTERN.sub("", text).strip()
            trace_source = "think_tags"
        return self._parse_message(answer_text=answer_text, reasoning_text=reasoning_text, trace_source=trace_source)

    def parse_completion(self, completion: vf.Messages) -> ParsedTrace:
        assistant_messages = self.get_assistant_messages(completion)
        if not assistant_messages:
            return ParsedTrace(
                trace_mode=self.trace_mode,
                raw_trace=None,
                raw_answer=None,
                personas=[],
                turns=[],
                final_answer="",
                support=[],
                group_solution=None,
                protocol_error_codes=["missing_assistant_message"],
                protocol_errors=["missing_assistant_message"],
            )

        message = assistant_messages[-1]
        answer_text, reasoning_content = self._extract_message_channels(message)
        trace_source: str | None = None
        reasoning_text = ""
        answer_candidate = answer_text
        if isinstance(reasoning_content, str) and reasoning_content.strip():
            trace_source = "reasoning_content"
            reasoning_text = reasoning_content.strip()
        else:
            think_blocks = THINK_PATTERN.findall(answer_text)
            if think_blocks:
                trace_source = "think_tags"
                reasoning_text = "\n\n".join(block.strip() for block in think_blocks if block.strip())
                answer_candidate = THINK_PATTERN.sub("", answer_text).strip()
        answer_candidate = THINK_MARKER_PATTERN.sub("", answer_candidate).strip()
        return self._parse_message(answer_text=answer_candidate, reasoning_text=reasoning_text, trace_source=trace_source)

    def parse_answer(self, completion: vf.Messages) -> str | None:
        parsed = self.parse_completion(completion)
        return parsed.final_answer or None

    def _extract_message_channels(self, message: dict[str, Any]) -> tuple[str, str]:
        content = self._message_field(message, "content", "") or ""
        reasoning_content = self._message_field(message, "reasoning_content", "") or ""
        if isinstance(content, list):
            text_parts: list[str] = []
            reasoning_parts: list[str] = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "text" and isinstance(part.get("text"), str):
                    text_parts.append(part["text"])
                elif part.get("type") == "thinking" and isinstance(part.get("thinking"), str):
                    reasoning_parts.append(part["thinking"])
            answer_text = "\n".join(part.strip() for part in text_parts if part.strip())
            if reasoning_parts and not reasoning_content:
                reasoning_content = "\n\n".join(part.strip() for part in reasoning_parts if part.strip())
            return answer_text, reasoning_content if isinstance(reasoning_content, str) else ""
        answer_text = self._content_to_text(content)
        return answer_text, reasoning_content if isinstance(reasoning_content, str) else ""

    def _parse_message(self, *, answer_text: str, reasoning_text: str, trace_source: str | None) -> ParsedTrace:
        protocol_error_codes: list[str] = []
        protocol_errors: list[str] = []
        trace_error_codes: list[str] = []
        trace_errors: list[str] = []
        answer_error_codes: list[str] = []
        answer_errors: list[str] = []

        personas: list[Persona] = []
        turns: list[TraceTurn] = []
        group_solution: str | None = None
        raw_trace: dict[str, Any] | None = None

        if not reasoning_text.strip():
            _add_issue(protocol_errors, protocol_error_codes, "missing_reasoning_trace")
        else:
            if self.trace_mode == "debate":
                personas, turns, group_solution, raw_trace = self._parse_debate_trace(reasoning_text, trace_errors, trace_error_codes)
            else:
                turns, raw_trace = self._parse_monologue_trace(reasoning_text, trace_errors, trace_error_codes)

        final_answer, support, raw_answer = self._parse_answer_text(answer_text, answer_errors, answer_error_codes)

        return ParsedTrace(
            trace_mode=self.trace_mode,
            raw_trace=raw_trace,
            raw_answer=raw_answer,
            personas=personas,
            turns=turns,
            final_answer=final_answer,
            support=support,
            group_solution=group_solution,
            reasoning_text=reasoning_text.strip(),
            answer_text=answer_text.strip(),
            trace_source=trace_source,
            protocol_error_codes=protocol_error_codes,
            protocol_errors=protocol_errors,
            trace_error_codes=trace_error_codes,
            trace_errors=trace_errors,
            answer_error_codes=answer_error_codes,
            answer_errors=answer_errors,
        )

    def _parse_debate_trace(
        self,
        reasoning_text: str,
        errors: list[str],
        codes: list[str],
    ) -> tuple[list[Persona], list[TraceTurn], str | None, dict[str, Any] | None]:
        cast_matches = CAST_PATTERN.findall(reasoning_text)
        conversation_matches = CONVERSATION_PATTERN.findall(reasoning_text)
        group_matches = GROUP_SOLUTION_PATTERN.findall(reasoning_text)
        if len(cast_matches) != 1:
            _add_issue(errors, codes, "cast_block_invalid")
            return [], [], None, None
        if len(conversation_matches) != 1:
            _add_issue(errors, codes, "conversation_block_invalid")
            return [], [], None, None
        if len(group_matches) != 1 or not group_matches[0].strip():
            _add_issue(errors, codes, "group_solution_invalid")
            return [], [], None, None

        personas = self._parse_personas(cast_matches[0], errors, codes)
        turns = self._parse_turns(conversation_matches[0], personas, errors, codes)
        group_solution = group_matches[0].strip()
        if not personas or not turns:
            return personas, turns, None if not group_solution else group_solution, None
        raw_trace = {
            "cast_of_characters": [self._persona_to_dict(persona) for persona in personas],
            "conversation": [self._turn_to_dict(turn) for turn in turns],
            "group_solution": group_solution,
        }
        return personas, turns, group_solution, raw_trace

    def _parse_monologue_trace(
        self,
        reasoning_text: str,
        errors: list[str],
        codes: list[str],
    ) -> tuple[list[TraceTurn], dict[str, Any] | None]:
        content = reasoning_text.strip()
        if not content:
            _add_issue(errors, codes, "monologue_empty")
            return [], None
        turn = TraceTurn(
            id="m1",
            speaker=None,
            act=self._infer_turn_act(content, position=0, total_turns=1),
            refs=_extract_refs(content),
            reply_to=[],
            content=content,
        )
        return [turn], {"analysis": [self._turn_to_dict(turn)]}

    def _parse_personas(self, cast_text: str, errors: list[str], codes: list[str]) -> list[Persona]:
        matches = PERSONA_PATTERN.findall(cast_text)
        if not matches:
            _add_issue(errors, codes, "persona_blocks_missing")
            return []
        personas: list[Persona] = []
        seen_ordinals: set[int] = set()
        for ordinal_text, body in matches:
            ordinal = int(ordinal_text)
            if ordinal in seen_ordinals:
                _add_issue(errors, codes, "persona_ordinal_duplicate", ordinal_text)
                continue
            seen_ordinals.add(ordinal)
            fields = {name.lower(): value.strip() for name, value in FIELD_PATTERN.findall(body)}
            role = fields.get("role", "")
            personality = fields.get("personality", "")
            expertise = fields.get("expertise", "")
            style = fields.get("style", "") or STYLE_BY_ROLE.get(role, "")
            if role not in ALLOWED_ROLES:
                _add_issue(errors, codes, "persona_role_invalid", role)
            if personality not in ALLOWED_PERSONALITIES:
                _add_issue(errors, codes, "persona_personality_invalid", personality)
            if not expertise:
                _add_issue(errors, codes, "persona_expertise_invalid")
            if not style:
                _add_issue(errors, codes, "persona_style_invalid")
            personas.append(
                Persona(
                    id=f"P{ordinal}",
                    role=role,
                    personality=personality,
                    expertise=expertise,
                    style=style,
                    ordinal=ordinal,
                )
            )
        personas.sort(key=lambda persona: persona.ordinal)
        return personas

    def _parse_turns(
        self,
        conversation_text: str,
        personas: list[Persona],
        errors: list[str],
        codes: list[str],
    ) -> list[TraceTurn]:
        matches = TURN_PATTERN.findall(conversation_text)
        if not matches:
            _add_issue(errors, codes, "conversation_turns_missing")
            return []
        by_ordinal = {persona.ordinal: persona for persona in personas}
        turns: list[TraceTurn] = []
        for index, (ordinal_text, body) in enumerate(matches, start=1):
            ordinal = int(ordinal_text)
            if ordinal not in by_ordinal:
                _add_issue(errors, codes, "turn_persona_mismatch", ordinal_text)
                continue
            content = body.strip()
            if not content:
                _add_issue(errors, codes, "turn_content_invalid", ordinal_text)
                continue
            turn_id = f"t{index}"
            reply_to = [turns[-1].id] if turns else []
            turns.append(
                TraceTurn(
                    id=turn_id,
                    speaker=by_ordinal[ordinal].id,
                    act=self._infer_turn_act(content, position=index - 1, total_turns=len(matches)),
                    refs=_extract_refs(content),
                    reply_to=reply_to,
                    content=content,
                )
            )
        return turns

    def _parse_answer_text(
        self,
        answer_text: str,
        errors: list[str],
        codes: list[str],
    ) -> tuple[str, list[str], dict[str, Any] | None]:
        answer_matches = ANSWER_PATTERN.findall(answer_text)
        support_matches = SUPPORT_PATTERN.findall(answer_text)
        if len(answer_matches) != 1 or not answer_matches[0].strip():
            _add_issue(errors, codes, "answer_block_invalid")
            return "", [], None
        cleaned = ANSWER_PATTERN.sub("", answer_text)
        cleaned = SUPPORT_PATTERN.sub("", cleaned)
        if cleaned.strip():
            _add_issue(errors, codes, "answer_unexpected_text")
            return "", [], None
        support: list[str] = []
        if len(support_matches) > 1:
            _add_issue(errors, codes, "support_block_duplicate")
            return "", [], None
        if support_matches:
            support = ordered_unique(
                [item.strip() for item in re.split(r"[\s,]+", support_matches[0].strip()) if item.strip()]
            )
        final_answer = answer_matches[0].strip()
        return final_answer, support, {"answer": final_answer, "support": support}

    def _infer_turn_act(self, content: str, *, position: int, total_turns: int) -> str:
        text = content.lower()
        stripped = text.strip()
        if any(
            phrase in stripped
            for phrase in (
                "reconcile",
                "resolve the conflict",
                "consensus",
                "settle on",
                "final consensus",
            )
        ):
            return "reconcile"
        if any(
            stripped.startswith(prefix)
            for prefix in (
                "alternative path:",
                "alternative branch:",
                "another path:",
                "different path:",
                "different route:",
                "competing reading:",
            )
        ):
            return "shift"
        if any(
            stripped.startswith(prefix)
            for prefix in (
                "verification:",
                "verify:",
                "verify ",
                "timeline check:",
                "check passes:",
                "recalculate:",
                "confirmation:",
            )
        ):
            return "verify"
        if "?" in text:
            return "question"
        if any(
            phrase in stripped
            for phrase in (
                "primary proposal",
                "initial proposal",
                "main proposal",
                "proposal:",
                "propose ",
            )
        ):
            return "propose"
        if any(
            phrase in stripped
            for phrase in (
                "challenge",
                "disagree",
                "contradicts",
                "should avoid",
                "that can't",
                "that cant",
            )
        ):
            return "challenge"
        if any(stripped.startswith(prefix) for prefix in ("but ", "however", "wait")):
            return "challenge"
        if any(
            phrase in text
            for phrase in (
                "another possibility",
                "other route",
                "instead",
            )
        ):
            return "shift"
        if any(
            phrase in text
            for phrase in (
                "confirm",
                "legal operators",
                "exactly",
                "still survives",
            )
        ):
            return "verify"
        if position == total_turns - 1:
            return "reconcile"
        return "propose"

    def _persona_to_dict(self, persona: Persona) -> dict[str, Any]:
        return {
            "id": persona.id,
            "role": persona.role,
            "personality": persona.personality,
            "expertise": persona.expertise,
            "style": persona.style,
            "ordinal": persona.ordinal,
        }

    def _turn_to_dict(self, turn: TraceTurn) -> dict[str, Any]:
        return {
            "id": turn.id,
            "speaker": turn.speaker,
            "act": turn.act,
            "reply_to": turn.reply_to,
            "refs": turn.refs,
            "content": turn.content,
        }


def _extract_refs(text: str) -> list[str]:
    return ordered_unique(REF_PATTERN.findall(text))
