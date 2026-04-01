from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ENV_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ENV_ROOT / "scripts"
DATA_ROOT = ENV_ROOT / "data" / "tinker"


def _run_script(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=ENV_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_disagreement_gold_builder_smoke(tmp_path: Path) -> None:
    out_dir = tmp_path / "gold"
    proc = _run_script(
        str(SCRIPTS / "build_tinker_disagreement_gold_dataset.py"),
        "--source-file",
        str(DATA_ROOT / "disagreement_gold_v1" / "source.json"),
        "--output-dir",
        str(out_dir),
    )
    assert proc.returncode == 0, proc.stderr
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["gold_counts"] == {"train": 48, "val": 16, "lock": 16}
    assert manifest["pattern_counts"] == {
        "alternative_route_challenge": 20,
        "assumption_challenge": 20,
        "stale_evidence_challenge": 20,
        "verification_pushback": 20,
    }
    assert (out_dir / "sft_train.jsonl").exists()
    assert (out_dir / "sft_val.jsonl").exists()
    assert (out_dir / "lock_eval.jsonl").exists()
    assert (out_dir / "tune_train.jsonl").exists()
    assert (out_dir / "tune_val.jsonl").exists()


def test_disagreement_gold_builder_rejects_invalid_source(tmp_path: Path) -> None:
    bad_source = tmp_path / "bad_source.json"
    bad_source.write_text(
        json.dumps(
            {
                "version": 1,
                "dialect_cycle": ["persona_think"],
                "blocks": [
                    {
                        "family": "countdown",
                        "difficulty": "medium",
                        "institution": "flat",
                        "seed_start": 500100,
                        "train_patterns": ["assumption_challenge"] * 12,
                        "val_patterns": ["assumption_challenge"] * 4,
                        "lock_patterns": ["bad_pattern"] * 4,
                    }
                ],
            }
        )
    )
    proc = _run_script(
        str(SCRIPTS / "build_tinker_disagreement_gold_dataset.py"),
        "--source-file",
        str(bad_source),
        "--output-dir",
        str(tmp_path / "out"),
    )
    assert proc.returncode != 0
    assert "Unsupported disagreement pattern" in (proc.stderr or proc.stdout)
