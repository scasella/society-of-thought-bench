from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from society_of_thought_bench.training_data import (
    EVAL_SUITES,
    OUTPUT_ROOT,
    compare_mode_summaries,
    evaluate_gate,
    suite_defaults,
    summarize_generate_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name")
    parser.add_argument("--model-path")
    parser.add_argument("--suite", choices=sorted(EVAL_SUITES.keys()), required=True)
    parser.add_argument("--num-examples", type=int)
    parser.add_argument("--rollouts-per-example", type=int)
    parser.add_argument("--max-concurrent", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--expected-stage", choices=sorted(("debate_primary",)))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if bool(args.model_name) == bool(args.model_path):
        raise SystemExit("Provide exactly one of --model-name or --model-path")

    suite = suite_defaults(args.suite)
    resolved = {
        "suite": args.suite,
        "model_name": args.model_name,
        "model_path": args.model_path,
        "num_examples": args.num_examples if args.num_examples is not None else suite["num_examples"],
        "rollouts_per_example": (
            args.rollouts_per_example if args.rollouts_per_example is not None else suite["rollouts_per_example"]
        ),
        "max_concurrent": args.max_concurrent if args.max_concurrent is not None else suite["max_concurrent"],
        "max_tokens": args.max_tokens if args.max_tokens is not None else suite["max_tokens"],
        "temperature": args.temperature if args.temperature is not None else suite["temperature"],
        "output_dir": str(args.output_dir or default_output_dir(args.suite, args.model_name or args.model_path or "checkpoint")),
    }
    if args.suite != "debate_vs_monologue":
        resolved["env_args"] = build_env_args(suite)
    else:
        resolved["debate_env_args"] = build_env_args({**suite, "trace_mode": "debate"})
        resolved["monologue_env_args"] = build_env_args({**suite, "trace_mode": "monologue"})
    if args.dry_run:
        print(json.dumps(resolved, indent=2))
        return

    require_env("TINKER_API_KEY")

    output_dir = Path(resolved["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.suite == "debate_vs_monologue":
        summary = asyncio.run(
            evaluate_debate_vs_monologue(
                model_name=args.model_name,
                model_path=args.model_path,
                num_examples=resolved["num_examples"],
                rollouts_per_example=resolved["rollouts_per_example"],
                max_concurrent=resolved["max_concurrent"],
                max_tokens=resolved["max_tokens"],
                temperature=resolved["temperature"],
                suite=suite,
            )
        )
    else:
        summary = asyncio.run(
            evaluate_single_suite(
                model_name=args.model_name,
                model_path=args.model_path,
                num_examples=resolved["num_examples"],
                rollouts_per_example=resolved["rollouts_per_example"],
                max_concurrent=resolved["max_concurrent"],
                max_tokens=resolved["max_tokens"],
                temperature=resolved["temperature"],
                env_args=resolved["env_args"],
            )
        )
    if args.expected_stage:
        summary["gate"] = gate_to_dict(evaluate_gate(summary, suite=args.suite, stage=args.expected_stage))

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def build_env_args(suite: dict[str, Any]) -> dict[str, Any]:
    env_args = {
        "family": suite["family"],
        "difficulty": suite["difficulty"],
        "institution": suite["institution"],
        "trace_mode": suite["trace_mode"],
        "trace_prompt_variant": suite["trace_prompt_variant"],
        "objective_profile": suite.get("objective_profile", "debate_primary"),
    }
    if "countdown_reward_profile" in suite:
        env_args["countdown_reward_profile"] = suite["countdown_reward_profile"]
    return env_args


async def evaluate_single_suite(
    *,
    model_name: str | None,
    model_path: str | None,
    num_examples: int,
    rollouts_per_example: int,
    max_concurrent: int,
    max_tokens: int,
    temperature: float,
    env_args: dict[str, Any],
) -> dict[str, Any]:
    from verifiers.clients import OpenAIChatCompletionsClient

    import tinker
    import verifiers as vf
    from tinker_cookbook import checkpoint_utils, model_info, renderers
    from tinker_cookbook.recipes.verifiers_rl.tinker_openai import TinkerAsyncOpenAIClient
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    service = tinker.ServiceClient()

    if model_path is not None:
        rest_client = service.create_rest_client()
        training_run = await rest_client.get_training_run_by_tinker_path_async(model_path)
        if model_name:
            if model_name != training_run.base_model:
                raise ValueError(
                    f"Model name {model_name} does not match training run base model {training_run.base_model}"
                )
        else:
            model_name = training_run.base_model

    if model_name is None:
        raise ValueError("model_name or model_path must be provided")

    env = vf.load_environment("society-of-thought-bench", **env_args)
    tokenizer = get_tokenizer(model_name)
    renderer_name = None
    if model_path is not None:
        renderer_name = await checkpoint_utils.get_renderer_name_from_checkpoint_async(service, model_path)
    if renderer_name is None:
        renderer_name = model_info.get_recommended_renderer_name(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    if model_path:
        sampling = service.create_sampling_client(model_path=model_path, base_model=model_name)
    else:
        sampling = service.create_sampling_client(base_model=model_name)

    client = OpenAIChatCompletionsClient(TinkerAsyncOpenAIClient(sampling, renderer, tokenizer))
    raw_results = env.evaluate_sync(
        client=client,
        model=model_name,
        num_examples=num_examples,
        rollouts_per_example=rollouts_per_example,
        max_concurrent=max_concurrent,
        sampling_args={
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
    )
    summary = summarize_generate_outputs(flatten_generate_outputs(raw_results))
    summary["env_args"] = env_args
    summary["model_name"] = model_name
    summary["model_path"] = model_path
    summary["renderer_name"] = renderer_name
    return summary


async def evaluate_debate_vs_monologue(
    *,
    model_name: str | None,
    model_path: str | None,
    num_examples: int,
    rollouts_per_example: int,
    max_concurrent: int,
    max_tokens: int,
    temperature: float,
    suite: dict[str, Any],
) -> dict[str, Any]:
    debate_summary = await evaluate_single_suite(
        model_name=model_name,
        model_path=model_path,
        num_examples=num_examples,
        rollouts_per_example=rollouts_per_example,
        max_concurrent=max_concurrent,
        max_tokens=max_tokens,
        temperature=temperature,
        env_args=build_env_args({**suite, "trace_mode": "debate"}),
    )
    monologue_summary = await evaluate_single_suite(
        model_name=model_name,
        model_path=model_path,
        num_examples=num_examples,
        rollouts_per_example=rollouts_per_example,
        max_concurrent=max_concurrent,
        max_tokens=max_tokens,
        temperature=temperature,
        env_args=build_env_args({**suite, "trace_mode": "monologue"}),
    )
    return compare_mode_summaries(debate_summary, monologue_summary)


def flatten_generate_outputs(results: dict[str, Any]) -> dict[str, Any]:
    if "reward" in results and "metrics" in results:
        return results

    outputs = results.get("outputs", [])
    reward = [float(item.get("reward", 0.0)) for item in outputs]
    metric_names: set[str] = set()
    for item in outputs:
        metric_names.update(item.get("metrics", {}).keys())
    metrics = {
        name: [float(item.get("metrics", {}).get(name, 0.0)) for item in outputs]
        for name in sorted(metric_names)
    }
    return {
        "reward": reward,
        "metrics": metrics,
        "outputs": outputs,
        "metadata": results.get("metadata", {}),
    }


def gate_to_dict(check) -> dict[str, Any]:
    return {
        "stage": check.stage,
        "suite": check.suite,
        "passed": check.passed,
        "checks": check.checks,
    }


def default_output_dir(suite: str, model_ref: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = model_ref.replace("/", "--").replace(":", "--")
    return OUTPUT_ROOT / "evals" / f"{stamp}-{suite}-{slug}"


def require_env(name: str) -> None:
    if os.environ.get(name):
        return
    raise SystemExit(f"{name} is required for non-dry-run Tinker commands.")


if __name__ == "__main__":
    main()
