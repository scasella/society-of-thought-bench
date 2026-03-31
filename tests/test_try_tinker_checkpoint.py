from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ENV_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ENV_ROOT / "scripts" / "try_tinker_checkpoint.py"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=ENV_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_try_tinker_checkpoint_dry_run_uses_benchmark_prompt() -> None:
    proc = _run("--dry-run", "--family", "evidence", "--difficulty", "hard")
    assert proc.returncode == 0, proc.stderr
    summary = json.loads(proc.stdout)
    assert summary["prompt_source"] == "benchmark"
    assert summary["prompt_meta"]["meta"]["family"] == "evidence"
    assert summary["prompt_meta"]["meta"]["difficulty"] == "hard"
    assert summary["model_path"].startswith("tinker://")


def test_try_tinker_checkpoint_dry_run_accepts_inline_prompt() -> None:
    proc = _run("--dry-run", "--prompt", "Explain the setup briefly.")
    assert proc.returncode == 0, proc.stderr
    summary = json.loads(proc.stdout)
    assert summary["prompt_source"] == "inline"
    assert summary["prompt_meta"] == {"source": "inline"}
