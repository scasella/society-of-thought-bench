from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from typing import Any

import verifiers as vf

from .core import (
    ALLOWED_ROLES,
    EXPERTISE_BY_FAMILY,
    ParsedTrace,
    TraceTurn,
    clamp,
    difficulty_turn_range,
    normalize_text,
    word_count,
)
from .families import evaluate_countdown_expression, evaluate_evidence_verdict

PROFILE_CONFIGS = {
    "balanced": {
        "task_score": 0.65,
        "format_valid": 0.10,
        "debate": {
            "behavior_coverage": 0.08,
            "persona_diversity": 0.06,
            "interaction_score": 0.06,
            "debate_relevance": 0.05,
            "disagreement_quality": 0.0,
        },
    },
    "debate_primary": {
        "task_score": 0.10,
        "format_valid": 0.10,
        "debate": {
            "behavior_coverage": 0.18,
            "persona_diversity": 0.15,
            "interaction_score": 0.17,
            "debate_relevance": 0.15,
            "disagreement_quality": 0.15,
        },
    },
}


class ClippedRubric(vf.Rubric):
    async def score_rollout(self, state):
        await super().score_rollout(state)
        state["reward"] = clamp(float(state["reward"]))
        if state.get("metrics", {}).get("protocol_valid", 1.0) == 0.0:
            state["reward"] = 0.0

    async def score_group(self, states):
        await super().score_group(states)
        for state in states:
            state["reward"] = clamp(float(state["reward"]))
            if state.get("metrics", {}).get("protocol_valid", 1.0) == 0.0:
                state["reward"] = 0.0


class SocietyOfThoughtScorer:
    def __init__(self, max_personas: int, max_debate_turns: int, objective_profile: str = "debate_primary"):
        self.max_personas = max_personas
        self.max_debate_turns = max_debate_turns
        self.objective_profile = objective_profile

    def _ensure_metrics(self, state: vf.State, parser: vf.Parser) -> dict[str, float]:
        cached = state.get("_sot_metrics")
        if isinstance(cached, dict):
            return cached

        info = state.get("info", {})
        family = info.get("family", "")
        difficulty = info.get("difficulty", "medium")
        trace_mode = info.get("trace_mode", "debate")
        task = info.get("task", {})
        objective_profile = info.get("objective_profile", self.objective_profile)
        parsed = self._ensure_trace(state, parser)
        format_valid = 1.0 if parsed.is_valid else 0.0
        protocol_valid = 1.0 if parsed.protocol_valid else 0.0
        trace_format_valid = 1.0 if parsed.trace_valid else 0.0
        answer_format_valid = 1.0 if parsed.answer_valid else 0.0

        metrics = {
            "format_valid": format_valid,
            "reasoning_trace_present": 1.0 if parsed.reasoning_text else 0.0,
            "trace_format_valid": trace_format_valid,
            "answer_format_valid": answer_format_valid,
            "protocol_valid": protocol_valid,
            "trace_protocol_error": 1.0 if parsed.protocol_errors else 0.0,
            "protocol_error_count": float(len(parsed.protocol_error_codes)),
            "trace_error_count": float(len(parsed.trace_error_codes)),
            "answer_error_count": float(len(parsed.answer_error_codes)),
            "task_score": 0.0,
            "task_correct_but_protocol_invalid": 0.0,
            "persona_count": 0.0 if trace_mode == "debate" else 1.0,
            "persona_diversity": 0.0,
            "interaction_score": 0.0,
            "question_answering_count": 0.0,
            "perspective_shift_count": 0.0,
            "conflict_of_perspectives_count": 0.0,
            "reconciliation_count": 0.0,
            "behavior_coverage": 0.0,
            "debate_relevance": 0.0,
            "disagreement_quality": 0.0,
            "challenge_response_pair_count": 0.0,
            "alternative_path_count": 0.0,
            "reconcile_link_count": 0.0,
            "efficiency_penalty": 0.0,
            "support_f1": 0.0,
            "countdown_expression_valid": 0.0,
            "countdown_target_distance": -1.0,
            "avg_turn_length": 0.0,
            "num_turns": 0.0,
        }

        if parsed.turns:
            metrics["num_turns"] = float(len(parsed.turns))
            metrics["avg_turn_length"] = sum(word_count(turn.content) for turn in parsed.turns) / len(parsed.turns)
            metrics["question_answering_count"] = float(self._question_answering_count(parsed.turns))
            metrics["perspective_shift_count"] = float(self._perspective_shift_count(parsed, family, task))
            metrics["conflict_of_perspectives_count"] = float(self._conflict_of_perspectives_count(parsed))
            metrics["reconciliation_count"] = float(self._reconciliation_count(parsed, family, task))

        support_f1 = 0.0
        if parsed.answer_valid:
            if family == "countdown":
                countdown = evaluate_countdown_expression(parsed.final_answer, task)
                metrics["task_score"] = countdown.task_score
                metrics["countdown_expression_valid"] = countdown.expression_valid
                metrics["countdown_target_distance"] = countdown.distance
                support_f1 = self._support_f1(parsed.support, task.get("oracle_support", []))
            else:
                verdict = parsed.final_answer.strip().upper()
                evidence = evaluate_evidence_verdict(verdict, parsed.support, task)
                metrics["task_score"] = evidence.task_score
                metrics["support_f1"] = evidence.support_f1
                support_f1 = evidence.support_f1

        if metrics["task_score"] >= 0.999 and protocol_valid == 0.0:
            metrics["task_correct_but_protocol_invalid"] = 1.0

        if trace_mode == "debate":
            metrics["persona_count"] = float(len(parsed.personas))
            metrics["persona_diversity"] = self._persona_diversity(parsed, family, difficulty)
            metrics["interaction_score"] = self._interaction_score(parsed)

        metrics["challenge_response_pair_count"] = float(self._challenge_response_pair_count(parsed))
        metrics["alternative_path_count"] = float(self._alternative_path_count(parsed, family, task))
        metrics["reconcile_link_count"] = float(self._reconcile_link_count(parsed, family, task))
        metrics["behavior_coverage"] = self._behavior_coverage(parsed, family, task, difficulty)
        metrics["debate_relevance"] = self._debate_relevance(
            parsed=parsed,
            family=family,
            task=task,
            support_f1=support_f1,
            objective_profile=objective_profile,
        )
        metrics["disagreement_quality"] = self._disagreement_quality(parsed, family, task, difficulty)
        metrics["efficiency_penalty"] = self._efficiency_penalty(parsed, difficulty)

        if not (parsed.trace_valid and parsed.answer_valid):
            metrics["behavior_coverage"] = 0.0
            metrics["persona_diversity"] = 0.0
            metrics["interaction_score"] = 0.0
            metrics["debate_relevance"] = 0.0
            metrics["disagreement_quality"] = 0.0

        state["_sot_metrics"] = metrics
        return metrics

    def _ensure_trace(self, state: vf.State, parser: vf.Parser) -> ParsedTrace:
        cached = state.get("_sot_trace")
        if isinstance(cached, ParsedTrace):
            return cached
        parsed = None
        parse_completion = getattr(parser, "parse_completion", None)
        if callable(parse_completion):
            parsed = parse_completion(state["completion"])
        if not isinstance(parsed, ParsedTrace):
            parsed = ParsedTrace(
                trace_mode=state.get("info", {}).get("trace_mode", "debate"),
                raw_trace=None,
                raw_answer=None,
                personas=[],
                turns=[],
                final_answer="",
                support=[],
                group_solution=None,
                protocol_error_codes=["parse_failure"],
                protocol_errors=["parse_failure"],
            )
        state["_sot_trace"] = parsed
        state["parsed_trace"] = parsed.raw_trace
        state["parsed_answer"] = parsed.raw_answer
        state["reasoning_trace_text"] = parsed.reasoning_text
        state["reasoning_trace_source"] = parsed.trace_source
        state["protocol_error_codes"] = parsed.protocol_error_codes
        state["protocol_errors"] = parsed.protocol_errors
        state["trace_error_codes"] = parsed.trace_error_codes
        state["trace_errors"] = parsed.trace_errors
        state["answer_error_codes"] = parsed.answer_error_codes
        state["answer_errors"] = parsed.answer_errors
        state["normalized_personas"] = [asdict(persona) for persona in parsed.personas]
        state["normalized_debate"] = [asdict(turn) for turn in parsed.turns]
        state["normalized_final_answer"] = parsed.final_answer
        state["normalized_support"] = parsed.support
        return parsed

    def _persona_diversity(self, parsed: ParsedTrace, family: str, difficulty: str) -> float:
        if not parsed.personas:
            return 0.0
        target = {"easy": 2, "medium": 3, "hard": 4}[difficulty]
        valid_expertise = set(EXPERTISE_BY_FAMILY[family])
        role_count = len({p.role for p in parsed.personas if p.role in ALLOWED_ROLES})
        personality_count = len({p.personality for p in parsed.personas})
        expertise_count = len({p.expertise for p in parsed.personas if p.expertise in valid_expertise})
        unique_id_count = len({p.id for p in parsed.personas})
        count_score = clamp(len(parsed.personas) / max(1, target))
        role_score = clamp(role_count / max(1, min(target, len(ALLOWED_ROLES))))
        personality_score = clamp(personality_count / max(1, min(target, len(parsed.personas))))
        expertise_score = clamp(expertise_count / max(1, min(target, len(valid_expertise))))
        unique_id_score = clamp(unique_id_count / max(1, len(parsed.personas)))
        return (count_score + role_score + personality_score + expertise_score + unique_id_score) / 5.0

    def _interaction_score(self, parsed: ParsedTrace) -> float:
        if len(parsed.turns) < 2:
            return 0.0
        turns_by_id = {turn.id: turn for turn in parsed.turns}
        alternations = 0
        longest_run = 1
        current_run = 1
        cross_replies = 0
        for prev, curr in zip(parsed.turns, parsed.turns[1:]):
            if prev.speaker != curr.speaker:
                alternations += 1
                current_run = 1
            else:
                current_run += 1
                longest_run = max(longest_run, current_run)
            if curr.reply_to:
                for reply_id in curr.reply_to:
                    parent = turns_by_id.get(reply_id)
                    if parent and parent.speaker and curr.speaker and parent.speaker != curr.speaker:
                        cross_replies += 1
                        break
        alternation_score = alternations / max(1, len(parsed.turns) - 1)
        cross_reply_score = cross_replies / max(1, len([turn for turn in parsed.turns if turn.reply_to]))
        run_penalty = longest_run / max(1, len(parsed.turns))
        return clamp(0.45 * cross_reply_score + 0.35 * alternation_score + 0.20 * (1.0 - run_penalty))

    def _behavior_coverage(self, parsed: ParsedTrace, family: str, task: dict[str, Any], difficulty: str) -> float:
        if not parsed.turns:
            return 0.0
        has_question_answering = self._question_answering_count(parsed.turns) >= 1
        has_conflict = self._conflict_of_perspectives_count(parsed) >= 1
        has_reconciliation = self._reconciliation_count(parsed, family, task) >= 1
        has_shift = self._perspective_shift_count(parsed, family, task) >= 1
        if difficulty == "easy":
            checks = [has_question_answering]
        elif difficulty == "medium":
            checks = [has_question_answering, has_conflict, has_reconciliation]
        else:
            checks = [has_question_answering, has_conflict, has_reconciliation, has_shift]
        return sum(1.0 for ok in checks if ok) / len(checks)

    def _debate_relevance(
        self,
        parsed: ParsedTrace,
        family: str,
        task: dict[str, Any],
        support_f1: float,
        objective_profile: str,
    ) -> float:
        if not parsed.turns:
            return 0.0
        valid_refs = self._valid_refs(family, task)
        total_refs = sum(len(turn.refs) for turn in parsed.turns)
        ref_score = 0.0 if total_refs == 0 else sum(1 for turn in parsed.turns for ref in turn.refs if ref in valid_refs) / total_refs
        grounded_turn_ratio = sum(
            1 for turn in parsed.turns if self._turn_grounded(turn, family, task, valid_refs)
        ) / max(1, len(parsed.turns))

        if family == "countdown":
            reasoning_turns = [turn for turn in parsed.turns if turn.act in {"challenge", "verify", "shift", "reconcile"}]
            arithmetic_score = 0.0
            if reasoning_turns:
                arithmetic_score = sum(
                    1
                    for turn in reasoning_turns
                    if any(op in turn.content for op in "+-*/=") or "target" in turn.content.lower()
                ) / len(reasoning_turns)
            return clamp(0.45 * ref_score + 0.35 * grounded_turn_ratio + 0.20 * arithmetic_score)

        reasoning_turns = [turn for turn in parsed.turns if turn.act in {"challenge", "verify", "shift", "reconcile"}]
        evidence_score = 0.0
        if reasoning_turns:
            oracle_support = set(task.get("oracle_support", []))
            evidence_score = sum(
                1
                for turn in reasoning_turns
                if (oracle_support & set(turn.refs)) or "stale" in turn.content.lower() or "contradict" in turn.content.lower() or "current" in turn.content.lower()
            ) / len(reasoning_turns)
        if objective_profile == "balanced":
            return clamp(0.40 * ref_score + 0.30 * grounded_turn_ratio + 0.15 * evidence_score + 0.15 * support_f1)
        return clamp(0.45 * ref_score + 0.35 * grounded_turn_ratio + 0.20 * evidence_score)

    def _disagreement_quality(
        self,
        parsed: ParsedTrace,
        family: str,
        task: dict[str, Any],
        difficulty: str,
    ) -> float:
        if not parsed.turns:
            return 0.0
        valid_refs = self._valid_refs(family, task)
        children = self._children_by_parent(parsed.turns)

        qualifying_challenges = 0
        responded_challenges = 0
        resolved_challenges = 0

        for turn in parsed.turns:
            if turn.act != "challenge":
                continue
            if not self._is_conflict_turn(turn, parsed.turns):
                continue
            qualifying_challenges += 1
            direct_responses = [
                child
                for child in children.get(turn.id, [])
                if child.speaker and turn.speaker and child.speaker != turn.speaker
            ]
            if direct_responses:
                responded_challenges += 1
            if self._challenge_has_grounded_resolution(turn.id, children, family, task, valid_refs):
                resolved_challenges += 1

        challenge_score = clamp(float(qualifying_challenges))
        response_score = 0.0 if qualifying_challenges == 0 else responded_challenges / qualifying_challenges
        resolution_score = 0.0 if qualifying_challenges == 0 else resolved_challenges / qualifying_challenges
        alternative_score = clamp(self._alternative_path_count(parsed, family, task) / 1.0)
        reconcile_score = clamp(self._reconcile_link_count(parsed, family, task) / 1.0)

        medium_score = clamp(0.35 * challenge_score + 0.30 * response_score + 0.35 * resolution_score)
        hard_score = clamp(
            0.25 * challenge_score
            + 0.20 * response_score
            + 0.20 * resolution_score
            + 0.20 * alternative_score
            + 0.15 * reconcile_score
        )
        if difficulty == "easy":
            return min(0.5, 0.5 * medium_score)
        if difficulty == "medium":
            return medium_score
        return hard_score

    def _question_answering_count(self, turns: list[TraceTurn]) -> int:
        count = 0
        for turn in turns:
            if turn.act != "question":
                continue
            answered = False
            for later in turns:
                if turn.id in later.reply_to:
                    answered = True
                    break
            if answered:
                count += 1
        return count

    def _perspective_shift_count(self, parsed: ParsedTrace, family: str, task: dict[str, Any]) -> int:
        valid_refs = self._valid_refs(family, task)
        return sum(
            1
            for turn in parsed.turns
            if turn.act == "shift" and turn.reply_to and self._turn_grounded(turn, family, task, valid_refs)
        )

    def _conflict_of_perspectives_count(self, parsed: ParsedTrace) -> int:
        return sum(1 for turn in parsed.turns if self._is_conflict_turn(turn, parsed.turns))

    def _reconciliation_count(self, parsed: ParsedTrace, family: str, task: dict[str, Any]) -> int:
        return self._reconcile_link_count(parsed, family, task)

    def _challenge_response_pair_count(self, parsed: ParsedTrace) -> int:
        turns_by_id = {turn.id: turn for turn in parsed.turns}
        children = self._children_by_parent(parsed.turns)
        count = 0
        for turn in parsed.turns:
            if turn.act != "challenge":
                continue
            parent_proposal = any(
                parent_id in turns_by_id
                and turns_by_id[parent_id].act in {"propose", "shift"}
                and turns_by_id[parent_id].speaker != turn.speaker
                for parent_id in turn.reply_to
            )
            if not parent_proposal:
                continue
            if any(child.speaker != turn.speaker for child in children.get(turn.id, [])):
                count += 1
        return count

    def _alternative_path_count(self, parsed: ParsedTrace, family: str, task: dict[str, Any]) -> int:
        valid_refs = self._valid_refs(family, task)
        return sum(
            1
            for turn in parsed.turns
            if turn.act == "shift" and turn.reply_to and self._turn_grounded(turn, family, task, valid_refs)
        )

    def _reconcile_link_count(self, parsed: ParsedTrace, family: str, task: dict[str, Any]) -> int:
        valid_refs = self._valid_refs(family, task)
        turns_by_id = {turn.id: turn for turn in parsed.turns}
        count = 0
        for turn in parsed.turns:
            if turn.act != "reconcile" or not turn.reply_to:
                continue
            if not self._turn_grounded(turn, family, task, valid_refs):
                continue
            if any(
                parent_id in turns_by_id and turns_by_id[parent_id].speaker != turn.speaker
                for parent_id in turn.reply_to
            ):
                count += 1
        return count

    def _challenge_has_grounded_resolution(
        self,
        challenge_id: str,
        children: dict[str, list[TraceTurn]],
        family: str,
        task: dict[str, Any],
        valid_refs: set[str],
    ) -> bool:
        frontier = list(children.get(challenge_id, []))
        seen: set[str] = set()
        while frontier:
            turn = frontier.pop(0)
            if turn.id in seen:
                continue
            seen.add(turn.id)
            if turn.act in {"verify", "reconcile"} and self._turn_grounded(turn, family, task, valid_refs):
                return True
            frontier.extend(children.get(turn.id, []))
        return False

    def _children_by_parent(self, turns: list[TraceTurn]) -> dict[str, list[TraceTurn]]:
        children: dict[str, list[TraceTurn]] = defaultdict(list)
        for turn in turns:
            for parent_id in turn.reply_to:
                children[parent_id].append(turn)
        return children

    def _valid_refs(self, family: str, task: dict[str, Any]) -> set[str]:
        if family == "countdown":
            return {entry["id"] for entry in task.get("numbers", [])} | {"T"}
        return {entry["id"] for entry in task.get("evidence", [])}

    def _turn_grounded(
        self,
        turn: TraceTurn,
        family: str,
        task: dict[str, Any],
        valid_refs: set[str] | None = None,
    ) -> bool:
        if valid_refs is None:
            valid_refs = self._valid_refs(family, task)
        if not turn.refs or not any(ref in valid_refs for ref in turn.refs):
            return False
        if family == "countdown":
            if turn.act in {"verify", "reconcile", "shift", "challenge"}:
                lower = turn.content.lower()
                return any(op in turn.content for op in "+-*/=") or "target" in lower or "legal" in lower or "exact" in lower
            return True
        lower = turn.content.lower()
        oracle_support = set(task.get("oracle_support", []))
        if turn.act in {"challenge", "verify", "reconcile", "shift"}:
            return bool(oracle_support & set(turn.refs)) or any(token in lower for token in ("stale", "current", "contradict", "timeline", "official"))
        return True

    def _is_conflict_turn(self, turn: TraceTurn, turns: list[TraceTurn]) -> bool:
        if turn.act != "challenge":
            return False
        turns_by_id = {item.id: item for item in turns}
        if not turn.reply_to:
            return False
        return any(
            parent_id in turns_by_id
            and turns_by_id[parent_id].speaker != turn.speaker
            and turns_by_id[parent_id].act in {"propose", "shift", "verify", "reconcile"}
            for parent_id in turn.reply_to
        )

    def _efficiency_penalty(self, parsed: ParsedTrace, difficulty: str) -> float:
        if not parsed.turns:
            return 0.0
        _low, high = difficulty_turn_range(difficulty, self.max_debate_turns)
        overflow = max(0, len(parsed.turns) - high) / max(1, self.max_debate_turns - high + 1)
        normalized_contents = [normalize_text(turn.content) for turn in parsed.turns]
        duplicate_ratio = max(0, len(normalized_contents) - len(set(normalized_contents))) / max(1, len(normalized_contents))
        invalid_reply_ratio = self._invalid_reply_ratio(parsed.turns)
        unused_persona_ratio = 0.0
        if parsed.personas:
            used_speakers = {turn.speaker for turn in parsed.turns if turn.speaker}
            unused_persona_ratio = len([p for p in parsed.personas if p.id not in used_speakers]) / len(parsed.personas)
        return clamp((overflow + duplicate_ratio + invalid_reply_ratio + unused_persona_ratio) / 4.0)

    def _invalid_reply_ratio(self, turns: list[TraceTurn]) -> float:
        turn_ids = {turn.id for turn in turns}
        reply_refs = [reply_id for turn in turns for reply_id in turn.reply_to]
        if not reply_refs:
            return 0.0
        invalid = len([reply_id for reply_id in reply_refs if reply_id not in turn_ids])
        return invalid / len(reply_refs)

    def _support_f1(self, predicted: list[str], oracle: list[str]) -> float:
        pred = set(predicted)
        gold = set(oracle)
        if not gold:
            return 1.0 if not pred else 0.0
        overlap = len(pred & gold)
        precision = overlap / len(pred) if pred else 0.0
        recall = overlap / len(gold) if gold else 0.0
        return 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    async def format_valid(self, state: vf.State, parser: vf.Parser) -> float:
        return self._ensure_metrics(state, parser)["format_valid"]

    async def reasoning_trace_present(self, state: vf.State, parser: vf.Parser) -> float:
        return self._ensure_metrics(state, parser)["reasoning_trace_present"]

    async def trace_format_valid(self, state: vf.State, parser: vf.Parser) -> float:
        return self._ensure_metrics(state, parser)["trace_format_valid"]

    async def answer_format_valid(self, state: vf.State, parser: vf.Parser) -> float:
        return self._ensure_metrics(state, parser)["answer_format_valid"]

    async def protocol_valid(self, state: vf.State, parser: vf.Parser) -> float:
        return self._ensure_metrics(state, parser)["protocol_valid"]

    async def trace_protocol_error(self, state: vf.State, parser: vf.Parser) -> float:
        return self._ensure_metrics(state, parser)["trace_protocol_error"]

    async def protocol_error_count(self, state: vf.State, parser: vf.Parser) -> float:
        return self._ensure_metrics(state, parser)["protocol_error_count"]

    async def trace_error_count(self, state: vf.State, parser: vf.Parser) -> float:
        return self._ensure_metrics(state, parser)["trace_error_count"]

    async def answer_error_count(self, state: vf.State, parser: vf.Parser) -> float:
        return self._ensure_metrics(state, parser)["answer_error_count"]

    async def task_score(self, state: vf.State, parser: vf.Parser) -> float:
        return self._ensure_metrics(state, parser)["task_score"]

    async def task_correct_but_protocol_invalid(self, state: vf.State, parser: vf.Parser) -> float:
        return self._ensure_metrics(state, parser)["task_correct_but_protocol_invalid"]

    async def persona_count(self, state: vf.State, parser: vf.Parser) -> float:
        return self._ensure_metrics(state, parser)["persona_count"]

    async def persona_diversity(self, state: vf.State, parser: vf.Parser) -> float:
        return self._ensure_metrics(state, parser)["persona_diversity"]

    async def interaction_score(self, state: vf.State, parser: vf.Parser) -> float:
        return self._ensure_metrics(state, parser)["interaction_score"]

    async def question_answering_count(self, state: vf.State, parser: vf.Parser) -> float:
        return self._ensure_metrics(state, parser)["question_answering_count"]

    async def perspective_shift_count(self, state: vf.State, parser: vf.Parser) -> float:
        return self._ensure_metrics(state, parser)["perspective_shift_count"]

    async def conflict_of_perspectives_count(self, state: vf.State, parser: vf.Parser) -> float:
        return self._ensure_metrics(state, parser)["conflict_of_perspectives_count"]

    async def reconciliation_count(self, state: vf.State, parser: vf.Parser) -> float:
        return self._ensure_metrics(state, parser)["reconciliation_count"]

    async def behavior_coverage(self, state: vf.State, parser: vf.Parser) -> float:
        return self._ensure_metrics(state, parser)["behavior_coverage"]

    async def debate_relevance(self, state: vf.State, parser: vf.Parser) -> float:
        return self._ensure_metrics(state, parser)["debate_relevance"]

    async def disagreement_quality(self, state: vf.State, parser: vf.Parser) -> float:
        return self._ensure_metrics(state, parser)["disagreement_quality"]

    async def challenge_response_pair_count(self, state: vf.State, parser: vf.Parser) -> float:
        return self._ensure_metrics(state, parser)["challenge_response_pair_count"]

    async def alternative_path_count(self, state: vf.State, parser: vf.Parser) -> float:
        return self._ensure_metrics(state, parser)["alternative_path_count"]

    async def reconcile_link_count(self, state: vf.State, parser: vf.Parser) -> float:
        return self._ensure_metrics(state, parser)["reconcile_link_count"]

    async def efficiency_penalty(self, state: vf.State, parser: vf.Parser) -> float:
        return self._ensure_metrics(state, parser)["efficiency_penalty"]

    async def support_f1(self, state: vf.State, parser: vf.Parser) -> float:
        return self._ensure_metrics(state, parser)["support_f1"]

    async def countdown_expression_valid(self, state: vf.State, parser: vf.Parser) -> float:
        return self._ensure_metrics(state, parser)["countdown_expression_valid"]

    async def countdown_target_distance(self, state: vf.State, parser: vf.Parser) -> float:
        return self._ensure_metrics(state, parser)["countdown_target_distance"]

    async def avg_turn_length(self, state: vf.State, parser: vf.Parser) -> float:
        return self._ensure_metrics(state, parser)["avg_turn_length"]

    async def num_turns(self, state: vf.State, parser: vf.Parser) -> float:
        return self._ensure_metrics(state, parser)["num_turns"]


def build_rubric(
    parser: vf.Parser,
    scorer: SocietyOfThoughtScorer,
    debate_reward_weight: float | None = None,
    debate_metric_weights: dict[str, float] | None = None,
    efficiency_penalty_weight: float = 0.05,
    objective_profile: str = "debate_primary",
) -> ClippedRubric:
    profile = PROFILE_CONFIGS[objective_profile]
    debate_weights = resolve_debate_weights(
        objective_profile=objective_profile,
        debate_reward_weight=debate_reward_weight,
        debate_metric_weights=debate_metric_weights,
    )

    funcs = [
        scorer.format_valid,
        scorer.reasoning_trace_present,
        scorer.trace_format_valid,
        scorer.answer_format_valid,
        scorer.protocol_valid,
        scorer.trace_protocol_error,
        scorer.protocol_error_count,
        scorer.trace_error_count,
        scorer.answer_error_count,
        scorer.task_score,
        scorer.task_correct_but_protocol_invalid,
        scorer.behavior_coverage,
        scorer.persona_diversity,
        scorer.interaction_score,
        scorer.debate_relevance,
        scorer.disagreement_quality,
        scorer.challenge_response_pair_count,
        scorer.alternative_path_count,
        scorer.reconcile_link_count,
        scorer.efficiency_penalty,
        scorer.persona_count,
        scorer.question_answering_count,
        scorer.perspective_shift_count,
        scorer.conflict_of_perspectives_count,
        scorer.reconciliation_count,
        scorer.support_f1,
        scorer.countdown_expression_valid,
        scorer.countdown_target_distance,
        scorer.avg_turn_length,
        scorer.num_turns,
    ]
    weights = [
        profile["format_valid"],
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        profile["task_score"],
        0.0,
        debate_weights["behavior_coverage"],
        debate_weights["persona_diversity"],
        debate_weights["interaction_score"],
        debate_weights["debate_relevance"],
        debate_weights["disagreement_quality"],
        0.0,
        0.0,
        0.0,
        -abs(efficiency_penalty_weight),
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    return ClippedRubric(funcs=funcs, weights=weights, parser=parser)


def resolve_debate_weights(
    objective_profile: str,
    debate_reward_weight: float | None,
    debate_metric_weights: dict[str, float] | None,
) -> dict[str, float]:
    weights = PROFILE_CONFIGS[objective_profile]["debate"].copy()
    if debate_metric_weights:
        for key, value in debate_metric_weights.items():
            if key in weights:
                weights[key] = max(0.0, float(value))
    target_total = sum(PROFILE_CONFIGS[objective_profile]["debate"].values())
    if debate_reward_weight is not None:
        target_total = max(0.0, float(debate_reward_weight))
    raw_total = sum(weights.values())
    if raw_total == 0:
        return {key: 0.0 for key in weights}
    scale = target_total / raw_total
    return {key: value * scale for key, value in weights.items()}
