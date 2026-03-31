from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

BEST_CHECKPOINT = "tinker://80d6e740-bf17-52ca-a94c-422c67897617:train:0/sampler_weights/final"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=BEST_CHECKPOINT)
    parser.add_argument("--model-name")
    parser.add_argument("--prompt")
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--family", choices=("countdown", "evidence"), default="countdown")
    parser.add_argument("--difficulty", choices=("easy", "medium", "hard"), default="medium")
    parser.add_argument("--institution", choices=("flat", "hierarchical", "auto"), default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-personas", type=int, default=4)
    parser.add_argument("--max-debate-turns", type=int, default=10)
    parser.add_argument("--trace-mode", choices=("debate", "monologue"), default="debate")
    parser.add_argument(
        "--trace-prompt-variant",
        choices=("official", "trace_minimal", "trace_tail_block", "debug_minimal"),
        default="official",
    )
    parser.add_argument("--objective-profile", choices=("debate_primary", "balanced"), default="debate_primary")
    parser.add_argument("--countdown-reward-profile", choices=("benchmark", "exact_emphasis"), default="benchmark")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--show-prompt", action="store_true")
    parser.add_argument("--show-raw", action="store_true")
    parser.add_argument("--show-raw-response", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    prompt, prompt_meta = resolve_prompt(args)
    summary = {
        "model_path": args.model_path,
        "model_name": args.model_name,
        "family": args.family,
        "difficulty": args.difficulty,
        "institution": args.institution,
        "seed": args.seed,
        "trace_mode": args.trace_mode,
        "trace_prompt_variant": args.trace_prompt_variant,
        "objective_profile": args.objective_profile,
        "prompt_source": prompt_meta["source"],
        "prompt_meta": prompt_meta,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return

    require_env("TINKER_API_KEY")
    result = asyncio.run(run_once(args, prompt, prompt_meta))
    if args.show_prompt:
        print("=== Prompt ===")
        print(format_prompt(prompt))
        print()
    print(f"=== Model ===\n{result['model_name']}")
    print(f"\n=== Checkpoint ===\n{result['model_path']}")
    if prompt_meta["source"] == "benchmark":
        print("\n=== Task Meta ===")
        print(json.dumps(prompt_meta["meta"], indent=2))
    print("\n=== Thinking Trace ===")
    print(result["thinking_trace"] or "[none]")
    print("\n=== Final Answer ===")
    print(result["visible_answer"] or "[none]")
    if args.show_raw_response:
        print("\n=== Raw Model Output ===")
        print(result["raw_output"] or "[none]")
    if args.show_raw:
        print("\n=== Parsed Message ===")
        print(json.dumps(result["parsed_message"], indent=2))


def resolve_prompt(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    if args.prompt and args.prompt_file:
        raise SystemExit("Provide at most one of --prompt or --prompt-file")
    if args.prompt:
        return args.prompt, {"source": "inline"}
    if args.prompt_file:
        return args.prompt_file.read_text(), {"source": "file", "path": str(args.prompt_file)}

    from society_of_thought_bench.families import inspect_example

    example = inspect_example(
        family=args.family,
        difficulty=args.difficulty,
        institution=args.institution,
        seed=args.seed,
        max_personas=args.max_personas,
        max_debate_turns=args.max_debate_turns,
        trace_mode=args.trace_mode,
        trace_prompt_variant=args.trace_prompt_variant,
        countdown_reward_profile=args.countdown_reward_profile,
        objective_profile=args.objective_profile,
    )
    return example["prompt"], {"source": "benchmark", "meta": example["meta"], "oracle": example["oracle"]}


async def run_once(args: argparse.Namespace, prompt: Any, prompt_meta: dict[str, Any]) -> dict[str, Any]:
    import tinker
    from tinker import types
    from tinker_cookbook import checkpoint_utils, model_info, renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    service = tinker.ServiceClient()
    rest_client = service.create_rest_client()
    training_run = await rest_client.get_training_run_by_tinker_path_async(args.model_path)
    model_name = args.model_name or training_run.base_model
    renderer_name = await checkpoint_utils.get_renderer_name_from_checkpoint_async(service, args.model_path)
    if renderer_name is None:
        renderer_name = model_info.get_recommended_renderer_name(model_name)
    tokenizer = get_tokenizer(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    sampling_client = service.create_sampling_client(model_path=args.model_path, base_model=model_name)

    if isinstance(prompt, list):
        conversation = prompt
    else:
        conversation = [{"role": "user", "content": prompt}]
    model_input = renderer.build_generation_prompt(conversation)
    sampling_params = types.SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stop=renderer.get_stop_sequences(),
    )
    response = await sampling_client.sample_async(prompt=model_input, num_samples=1, sampling_params=sampling_params)
    raw_output = tokenizer.decode(response.sequences[0].tokens)
    parsed_message, success = renderer.parse_response(response.sequences[0].tokens)
    if not success:
        raise RuntimeError("Renderer failed to parse the model response.")
    thinking_parts, text_parts = split_message_content(parsed_message.get("content"))
    return {
        "model_name": model_name,
        "model_path": args.model_path,
        "renderer_name": renderer_name,
        "prompt_meta": prompt_meta,
        "parsed_message": parsed_message,
        "raw_output": raw_output,
        "thinking_trace": "\n".join(part for part in thinking_parts if part.strip()).strip(),
        "visible_answer": "\n".join(part for part in text_parts if part.strip()).strip(),
    }


def split_message_content(content: Any) -> tuple[list[str], list[str]]:
    if isinstance(content, str):
        return [], [content]
    thinking_parts: list[str] = []
    text_parts: list[str] = []
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type == "thinking":
                thinking_parts.append(str(part.get("thinking", "")))
            elif part_type == "text":
                text_parts.append(str(part.get("text", "")))
    return thinking_parts, text_parts


def format_prompt(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    return json.dumps(prompt, indent=2)


def require_env(name: str) -> None:
    if os.environ.get(name):
        return
    raise SystemExit(f"{name} is required for non-dry-run Tinker commands.")


if __name__ == "__main__":
    main()
