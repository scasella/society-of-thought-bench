from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

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
