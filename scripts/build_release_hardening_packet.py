from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

from society_of_thought_bench.release_hardening import (
    CONFIRMATION_RUNS,
    PUBLIC_BASE_MODEL,
    PUBLIC_CHECKPOINT,
    build_release_manifest,
    build_trace_audit,
    default_demo_prompt_specs,
    evaluate_release_acceptance,
    json_dumps,
    render_confirmation_markdown,
    render_trace_audit_markdown,
    render_usage_guidance,
    sample_demo_prompt_pack_async,
)

ENV_ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPT = ENV_ROOT / "scripts" / "eval_tinker_checkpoint.py"
EXTERNAL_COMPARE_SCRIPT = ENV_ROOT / "scripts" / "compare_tinker_external_suite.py"
DEFAULT_OUTPUT_DIR = ENV_ROOT / "release_preview" / "artifact_hardening"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=PUBLIC_CHECKPOINT)
    parser.add_argument("--model-name", default=PUBLIC_BASE_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--demo-prompt-pack", type=Path)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--outside-num-examples", type=int, default=25)
    parser.add_argument("--skip-confirmation", action="store_true")
    parser.add_argument("--skip-demo-pack", action="store_true")
    parser.add_argument("--skip-outside", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    prompt_specs = load_prompt_specs(args.demo_prompt_pack)
    plan = {
        "model_path": args.model_path,
        "model_name": args.model_name,
        "output_dir": str(args.output_dir),
        "prompt_specs": prompt_specs,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "outside_num_examples": args.outside_num_examples,
        "skip_confirmation": args.skip_confirmation,
        "skip_demo_pack": args.skip_demo_pack,
        "skip_outside": args.skip_outside,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return

    require_env("TINKER_API_KEY")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    confirmation_summaries: dict[str, dict[str, object]] = {}
    confirmation_paths: dict[str, str] = {}
    if not args.skip_confirmation:
        confirmation_summaries, confirmation_paths = run_confirmation_packet(args)
    else:
        confirmation_summaries, confirmation_paths = load_existing_confirmation_packet(args.output_dir)

    audit = None
    demo_samples_path = None
    demo_prompt_pack_path = None
    audit_path = None
    usage_path = None
    if not args.skip_demo_pack:
        demo_prompt_pack_path = args.output_dir / "demo_prompt_pack.json"
        demo_prompt_pack_path.write_text(json.dumps(prompt_specs, indent=2))
        demo_samples = asyncio.run(
            sample_demo_prompt_pack_async(
                model_path=args.model_path,
                model_name=args.model_name,
                prompt_specs=prompt_specs,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
            )
        )
        demo_samples_path = args.output_dir / "demo_prompt_samples.json"
        demo_samples_path.write_text(json.dumps(demo_samples, indent=2))
        audit = build_trace_audit(demo_samples)
        audit_path = args.output_dir / "trace_audit.json"
        audit_path.write_text(json.dumps(audit, indent=2))
        (args.output_dir / "TRACE_AUDIT.md").write_text(render_trace_audit_markdown(audit))
        usage_path = args.output_dir / "USAGE.md"
        usage_path.write_text(render_usage_guidance())

    outside_summary = None
    outside_summary_path = None
    if not args.skip_outside:
        outside_summary, outside_summary_path = run_background_outside_characterization(args)

    if confirmation_summaries:
        acceptance = evaluate_release_acceptance(
            debate_vs_monologue_summary=confirmation_summaries["debate_vs_monologue_medium_100"],
            easy_summary=confirmation_summaries["protocol_easy_40"],
            hard_summary=confirmation_summaries["debate_hard_40"],
        )
        (args.output_dir / "release_acceptance.json").write_text(json.dumps(acceptance, indent=2))
        (args.output_dir / "CONFIRMATION.md").write_text(
            render_confirmation_markdown(
                debate_vs_monologue_summary=confirmation_summaries["debate_vs_monologue_medium_100"],
                easy_summary=confirmation_summaries["protocol_easy_40"],
                hard_summary=confirmation_summaries["debate_hard_40"],
                acceptance=acceptance,
                outside_summary=outside_summary,
            )
        )
    else:
        acceptance = None

    manifest = build_release_manifest(
        output_dir=args.output_dir,
        confirmation_paths=confirmation_paths,
        demo_prompt_pack_path=str(demo_prompt_pack_path) if demo_prompt_pack_path else "",
        demo_samples_path=str(demo_samples_path) if demo_samples_path else "",
        audit_path=str(audit_path) if audit_path else "",
        usage_path=str(usage_path) if usage_path else "",
        outside_summary_path=str(outside_summary_path) if outside_summary_path else None,
    )
    manifest["acceptance"] = acceptance
    manifest["has_audit"] = audit is not None
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


def load_prompt_specs(path: Path | None) -> list[dict[str, object]]:
    if path is None:
        return default_demo_prompt_specs()
    return json.loads(path.read_text())


def run_confirmation_packet(args) -> tuple[dict[str, dict[str, object]], dict[str, str]]:
    summaries: dict[str, dict[str, object]] = {}
    paths: dict[str, str] = {}
    confirmation_dir = args.output_dir / "confirmation"
    confirmation_dir.mkdir(parents=True, exist_ok=True)
    for key, config in CONFIRMATION_RUNS.items():
        output_dir = confirmation_dir / key
        cmd = [
            sys.executable,
            str(EVAL_SCRIPT),
            "--model-path",
            args.model_path,
            "--suite",
            str(config["suite"]),
            "--num-examples",
            str(config["num_examples"]),
            "--expected-stage",
            "debate_primary",
            "--output-dir",
            str(output_dir),
        ]
        completed = subprocess.run(
            cmd,
            cwd=ENV_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise SystemExit(f"Confirmation run {key} failed:\n{completed.stderr or completed.stdout}")
        summary_path = output_dir / "summary.json"
        summaries[key] = json.loads(summary_path.read_text())
        paths[key] = str(summary_path)
    return summaries, paths


def load_existing_confirmation_packet(output_dir: Path) -> tuple[dict[str, dict[str, object]], dict[str, str]]:
    summaries: dict[str, dict[str, object]] = {}
    paths: dict[str, str] = {}
    confirmation_dir = output_dir / "confirmation"
    for key in CONFIRMATION_RUNS:
        summary_path = confirmation_dir / key / "summary.json"
        if not summary_path.exists():
            continue
        summaries[key] = json.loads(summary_path.read_text())
        paths[key] = str(summary_path)
    return summaries, paths


def run_background_outside_characterization(args) -> tuple[dict[str, object], Path]:
    output_dir = args.output_dir / "outside_background"
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_file = output_dir / "paired_questions.lock.json"
    cmd = [
        sys.executable,
        str(EXTERNAL_COMPARE_SCRIPT),
        "--debate-model-path",
        args.model_path,
        "--base-model-name",
        args.model_name,
        "--num-examples",
        str(args.outside_num_examples),
        "--save-lock-file",
        str(lock_file),
        "--output-dir",
        str(output_dir),
    ]
    completed = subprocess.run(
        cmd,
        cwd=ENV_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(f"Background outside characterization failed:\n{completed.stderr or completed.stdout}")
    summary_path = output_dir / "summary.json"
    return json.loads(summary_path.read_text()), summary_path


def require_env(name: str) -> None:
    if os.environ.get(name):
        return
    raise SystemExit(f"{name} is required for non-dry-run Tinker commands.")


if __name__ == "__main__":
    main()
