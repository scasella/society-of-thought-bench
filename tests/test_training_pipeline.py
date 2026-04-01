from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ENV_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ENV_ROOT / "scripts"


def _run_script(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=ENV_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_warmup_dataset_builder_smoke(tmp_path: Path) -> None:
    out_dir = tmp_path / "warmup"
    proc = _run_script(
        str(SCRIPTS / "build_tinker_warmup_dataset.py"),
        "--output-dir",
        str(out_dir),
        "--train-total",
        "20",
        "--val-total",
        "10",
    )
    assert proc.returncode == 0, proc.stderr
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["train_total"] == 20
    assert manifest["val_total"] == 10
    assert (out_dir / "sft_train.jsonl").exists()
    assert (out_dir / "sft_val.jsonl").exists()


def test_dpo_dataset_builder_smoke(tmp_path: Path) -> None:
    out_dir = tmp_path / "dpo"
    proc = _run_script(
        str(SCRIPTS / "build_tinker_dpo_dataset.py"),
        "--output-dir",
        str(out_dir),
        "--train-total",
        "20",
        "--val-total",
        "10",
        "--curriculum-profile",
        "protocol_bootcamp",
    )
    assert proc.returncode == 0, proc.stderr
    manifest = json.loads((out_dir / "dpo_manifest.json").read_text())
    assert manifest["train_total"] == 20
    assert manifest["val_total"] == 10
    assert manifest["curriculum_profile"] == "protocol_bootcamp"
    assert manifest["train_pair_types"]
    assert (out_dir / "dpo_train.jsonl").exists()
    assert (out_dir / "dpo_val.jsonl").exists()


def test_sft_and_dpo_dry_runs_render_configs(tmp_path: Path) -> None:
    warmup_dir = tmp_path / "warmup"
    dpo_dir = tmp_path / "dpo"
    assert _run_script(
        str(SCRIPTS / "build_tinker_warmup_dataset.py"),
        "--output-dir",
        str(warmup_dir),
        "--train-total",
        "20",
        "--val-total",
        "10",
    ).returncode == 0
    assert _run_script(
        str(SCRIPTS / "build_tinker_dpo_dataset.py"),
        "--output-dir",
        str(dpo_dir),
        "--train-total",
        "20",
        "--val-total",
        "10",
        "--curriculum-profile",
        "protocol_bootcamp",
    ).returncode == 0

    sft_proc = _run_script(
        str(SCRIPTS / "run_tinker_sft.py"),
        "--train-file",
        str(warmup_dir / "sft_train.jsonl"),
        "--val-file",
        str(warmup_dir / "sft_val.jsonl"),
        "--dry-run",
    )
    dpo_proc = _run_script(
        str(SCRIPTS / "run_tinker_dpo.py"),
        "--train-file",
        str(dpo_dir / "dpo_train.jsonl"),
        "--val-file",
        str(dpo_dir / "dpo_val.jsonl"),
        "--dry-run",
    )
    assert sft_proc.returncode == 0, sft_proc.stderr
    assert dpo_proc.returncode == 0, dpo_proc.stderr
    sft_summary = json.loads(sft_proc.stdout)
    dpo_summary = json.loads(dpo_proc.stdout)
    assert sft_summary["renderer_name"] == "society_of_thought_qwen3"
    assert dpo_summary["dpo_beta"] == 0.1


def test_rl_dry_run_configs_target_debate_primary_stages() -> None:
    medium_proc = _run_script(str(SCRIPTS / "run_tinker_verifiers_rl.py"), "--stage", "all_medium", "--dry-run")
    hard_proc = _run_script(str(SCRIPTS / "run_tinker_verifiers_rl.py"), "--stage", "all_hard", "--dry-run")
    assert medium_proc.returncode == 0, medium_proc.stderr
    assert hard_proc.returncode == 0, hard_proc.stderr
    medium = json.loads(medium_proc.stdout)
    hard = json.loads(hard_proc.stdout)
    assert medium["vf_env_args"]["objective_profile"] == "debate_primary"
    assert hard["vf_env_args"]["difficulty"] == "hard"
    assert hard["vf_env_args"]["objective_profile"] == "debate_primary"
