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


def test_external_sft_dataset_builder_smoke(tmp_path: Path) -> None:
    out_dir = tmp_path / "external_sft"
    proc = _run_script(
        str(SCRIPTS / "build_tinker_external_sft_dataset.py"),
        "--output-dir",
        str(out_dir),
        "--train-total",
        "20",
        "--val-total",
        "10",
        "--curriculum-profile",
        "external_core_debate",
    )
    assert proc.returncode == 0, proc.stderr
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["train_total"] == 20
    assert manifest["curriculum_profile"] == "external_core_debate"
    assert manifest["train_source_counts"]["internal_sot"] == 8
    assert (out_dir / "sft_train.jsonl").exists()


def test_external_proxy_focus_sft_dataset_builder_smoke(tmp_path: Path) -> None:
    out_dir = tmp_path / "external_proxy_focus_sft"
    proc = _run_script(
        str(SCRIPTS / "build_tinker_external_sft_dataset.py"),
        "--output-dir",
        str(out_dir),
        "--train-total",
        "20",
        "--val-total",
        "10",
        "--curriculum-profile",
        "external_proxy_focus_debate",
    )
    assert proc.returncode == 0, proc.stderr
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["train_total"] == 20
    assert manifest["curriculum_profile"] == "external_proxy_focus_debate"
    assert manifest["train_source_counts"]["mmlu_pro_stem"] == 7
    assert manifest["train_source_counts"]["ifeval_synthetic"] == 4
    assert (out_dir / "sft_train.jsonl").exists()


def test_external_bridge_sft_dataset_builder_smoke(tmp_path: Path) -> None:
    out_dir = tmp_path / "external_bridge_sft"
    proc = _run_script(
        str(SCRIPTS / "build_tinker_external_sft_dataset.py"),
        "--output-dir",
        str(out_dir),
        "--train-total",
        "20",
        "--val-total",
        "10",
        "--curriculum-profile",
        "external_bridge_debate",
    )
    assert proc.returncode == 0, proc.stderr
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["curriculum_profile"] == "external_bridge_debate"
    assert manifest["train_source_counts"]["internal_sot"] == 11
    assert manifest["train_source_counts"]["mmlu_pro_non_eval"] == 3
    assert (out_dir / "sft_train.jsonl").exists()


def test_external_dpo_dataset_builder_smoke(tmp_path: Path) -> None:
    out_dir = tmp_path / "external_dpo"
    proc = _run_script(
        str(SCRIPTS / "build_tinker_external_dpo_dataset.py"),
        "--output-dir",
        str(out_dir),
        "--train-total",
        "20",
        "--val-total",
        "10",
        "--curriculum-profile",
        "external_core_monologue",
    )
    assert proc.returncode == 0, proc.stderr
    manifest = json.loads((out_dir / "dpo_manifest.json").read_text())
    assert manifest["train_total"] == 20
    assert manifest["curriculum_profile"] == "external_core_monologue"
    assert manifest["train_component_counts"]["external_answer_discipline"] == 8
    assert (out_dir / "dpo_train.jsonl").exists()


def test_external_eval_and_compare_dry_runs_render_configs() -> None:
    eval_proc = _run_script(
        str(SCRIPTS / "eval_tinker_external_suite.py"),
        "--model-name",
        "Qwen/Qwen3-30B-A3B",
        "--trace-mode",
        "debate",
        "--save-lock-file",
        "/tmp/external-lock.json",
        "--dry-run",
    )
    compare_proc = _run_script(
        str(SCRIPTS / "compare_tinker_external_suite.py"),
        "--debate-model-name",
        "Qwen/Qwen3-30B-A3B",
        "--base-model-name",
        "Qwen/Qwen3-30B-A3B",
        "--replay-lock-file",
        "/tmp/external-lock.json",
        "--dry-run",
    )
    assert eval_proc.returncode == 0, eval_proc.stderr
    assert compare_proc.returncode == 0, compare_proc.stderr
    eval_summary = json.loads(eval_proc.stdout)
    compare_summary = json.loads(compare_proc.stdout)
    assert eval_summary["trace_mode"] == "debate"
    assert eval_summary["benchmarks"] == ["gsm8k", "mmlu_pro", "gpqa", "ifeval"]
    assert eval_summary["save_lock_file"] == "/tmp/external-lock.json"
    assert compare_summary["debate_model_name"] == "Qwen/Qwen3-30B-A3B"
    assert compare_summary["base_model_name"] == "Qwen/Qwen3-30B-A3B"
    assert compare_summary["replay_lock_file"] == "/tmp/external-lock.json"


def test_monologue_control_rl_dry_runs() -> None:
    proc = _run_script(str(SCRIPTS / "run_tinker_verifiers_rl.py"), "--stage", "control_all_medium", "--dry-run")
    assert proc.returncode == 0, proc.stderr
    summary = json.loads(proc.stdout)
    assert summary["vf_env_args"]["trace_mode"] == "monologue"
    assert summary["vf_env_args"]["debate_reward_weight"] == 0.0
