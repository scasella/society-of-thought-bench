from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from society_of_thought_bench.checkpoint_chat import extract_tagged_sections as extract_checkpoint_sections
from society_of_thought_bench.checkpoint_chat import split_message_content as split_checkpoint_message_content
from space_demo.core import extract_tagged_sections as extract_demo_sections
from space_demo.core import split_message_content as split_demo_message_content

ENV_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ENV_ROOT / "scripts" / "chat_tinker_checkpoint.py"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=ENV_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_chat_tinker_checkpoint_dry_run_defaults_to_interactive() -> None:
    proc = _run("--dry-run")
    assert proc.returncode == 0, proc.stderr
    summary = json.loads(proc.stdout)
    assert summary["interactive"] is True
    assert summary["model_path"].startswith("tinker://")
    assert "paper-style internal debate" in summary["system_prompt"]


def test_chat_tinker_checkpoint_dry_run_accepts_scripted_messages_and_transcript_path(tmp_path: Path) -> None:
    transcript = tmp_path / "chat.json"
    proc = _run(
        "--dry-run",
        "--message",
        "Hello there.",
        "--message",
        "Give me a second answer.",
        "--transcript-out",
        str(transcript),
    )
    assert proc.returncode == 0, proc.stderr
    summary = json.loads(proc.stdout)
    assert summary["interactive"] is False
    assert summary["scripted_messages"] == ["Hello there.", "Give me a second answer."]
    assert summary["transcript_out"] == str(transcript)


def test_checkpoint_chat_parses_visible_answer_after_think_without_answer_tag() -> None:
    raw_output = (
        "<think>Debate trace here.</think><|Start of document|>\n\n"
        "**Paper-Style Multi-Persona Analysis**\n\n"
        "Compare the model to itself in monologue mode."
    )
    thinking, answer = extract_checkpoint_sections(raw_output)
    assert thinking == "Debate trace here."
    assert answer.startswith("**Paper-Style Multi-Persona Analysis**")
    assert "<|Start of document|>" not in answer


def test_demo_parses_visible_answer_after_think_without_answer_tag() -> None:
    raw_output = (
        "<think>Debate trace here.</think><|Start of document|>\n\n"
        "**Paper-Style Multi-Persona Analysis**\n\n"
        "Compare the model to itself in monologue mode."
    )
    thinking, answer = extract_demo_sections(raw_output)
    assert thinking == "Debate trace here."
    assert answer.startswith("**Paper-Style Multi-Persona Analysis**")
    assert "<|Start of document|>" not in answer


def test_checkpoint_chat_split_message_content_handles_inline_think_and_visible_text() -> None:
    content = "<think>Debate trace here.</think><|Start of document|>\n\nVisible answer."
    thinking_parts, text_parts = split_checkpoint_message_content(content)
    assert thinking_parts == ["Debate trace here."]
    assert text_parts == ["Visible answer."]


def test_demo_split_message_content_handles_inline_think_and_visible_text() -> None:
    content = "<think>Debate trace here.</think><|Start of document|>\n\nVisible answer."
    thinking_parts, text_parts = split_demo_message_content(content)
    assert thinking_parts == ["Debate trace here."]
    assert text_parts == ["Visible answer."]
