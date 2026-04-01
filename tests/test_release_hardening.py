from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from society_of_thought_bench.core import ParsedTrace
from society_of_thought_bench.release_hardening import (
    CONFIRMATION_RUNS,
    build_trace_audit,
    default_demo_prompt_specs,
    evaluate_release_acceptance,
)

ENV_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ENV_ROOT / "scripts" / "build_release_hardening_packet.py"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=ENV_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _sample(name: str, *, reward: float, task_score: float, interaction: float, persona: float, disagreement: float, valid: bool) -> dict[str, object]:
    return {
        "name": name,
        "family": "countdown",
        "difficulty": "medium",
        "prompt": "Prompt",
        "thinking_trace": "<cast_of_characters>...</cast_of_characters>",
        "visible_answer": "<answer>42</answer>",
        "reward": reward,
        "metrics": {
            "format_valid": 1.0 if valid else 0.0,
            "task_score": task_score,
            "interaction_score": interaction,
            "persona_diversity": persona,
            "disagreement_quality": disagreement,
            "reconcile_link_count": 1.0 if disagreement > 0.4 else 0.0,
            "alternative_path_count": 1.0 if disagreement > 0.4 else 0.0,
            "conflict_of_perspectives_count": 1.0 if disagreement > 0.1 else 0.0,
            "challenge_response_pair_count": 1.0 if disagreement > 0.1 else 0.0,
        },
        "parsed": {
            "error_codes": [] if valid else ["cast_block_invalid"],
            "answer_error_codes": [],
            "trace_error_codes": [] if valid else ["cast_block_invalid"],
        },
        "label": "weak_disagreement",
        "note": "note",
    }


def test_release_hardening_dry_run_reports_confirmation_and_prompt_pack() -> None:
    proc = _run("--dry-run")
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["model_path"].startswith("tinker://")
    assert len(payload["prompt_specs"]) == 20
    assert sorted(CONFIRMATION_RUNS) == [
        "debate_hard_40",
        "debate_vs_monologue_medium_100",
        "protocol_easy_40",
    ]


def test_default_demo_prompt_specs_are_stable_and_diverse() -> None:
    specs = default_demo_prompt_specs()
    assert len(specs) == 20
    assert len({spec["name"] for spec in specs}) == 20
    assert {spec["family"] for spec in specs} == {"countdown", "evidence"}
    assert {spec["difficulty"] for spec in specs} == {"easy", "medium", "hard"}


def test_build_trace_audit_returns_four_per_bucket() -> None:
    samples = [
        _sample(f"strong_{index}", reward=0.8, task_score=0.9, interaction=0.8, persona=0.9, disagreement=0.6, valid=True)
        for index in range(5)
    ]
    samples += [
        _sample(f"borderline_{index}", reward=0.55, task_score=0.6, interaction=0.55, persona=0.8, disagreement=0.2, valid=True)
        for index in range(5)
    ]
    samples += [
        _sample(f"failure_{index}", reward=0.2, task_score=0.1, interaction=0.2, persona=0.3, disagreement=0.0, valid=False)
        for index in range(5)
    ]
    audit = build_trace_audit(samples)
    assert audit["counts"] == {
        "strong_successes": 4,
        "borderline_examples": 4,
        "failures": 4,
    }


def test_release_acceptance_uses_current_thresholds() -> None:
    comparison = {
        "reward_delta": 0.40,
        "joint_valid_delta": 0.22,
        "disagreement_quality_delta": 0.45,
    }
    easy = {
        "answer_format_valid_rate": 0.96,
        "joint_contract_valid_rate": 0.90,
    }
    hard = {
        "interaction_score": 0.70,
        "disagreement_quality": 0.50,
    }
    acceptance = evaluate_release_acceptance(comparison, easy, hard)
    assert acceptance["passed"] is True
    assert all(check["passed"] for check in acceptance["checks"])
