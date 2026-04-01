from __future__ import annotations

import argparse
import asyncio
import json
import inspect
import os
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import tinker

from society_of_thought_bench.training_data import OUTPUT_ROOT, RL_STAGE_CONFIGS, rl_stage_defaults
from society_of_thought_bench.tinker_renderers import get_renderer, patch_tinker_cookbook_renderers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=sorted(RL_STAGE_CONFIGS.keys()), required=True)
    parser.add_argument("--model-name")
    parser.add_argument("--renderer-name")
    parser.add_argument("--vf-env-id")
    parser.add_argument("--load-checkpoint-path")
    parser.add_argument("--kl-reference-checkpoint-path")
    parser.add_argument("--num-train", type=int)
    parser.add_argument("--num-eval", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--objective-profile")
    parser.add_argument("--log-path", type=Path)
    parser.add_argument("--lora-rank", type=int)
    parser.add_argument("--group-size", type=int)
    parser.add_argument("--groups-per-batch", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--kl-penalty-coef", type=float)
    parser.add_argument("--num-substeps", type=int)
    parser.add_argument("--save-every", type=int)
    parser.add_argument("--eval-every", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-concurrent-generation", type=int, default=-1)
    parser.add_argument("--max-concurrent-scoring", type=int, default=-1)
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-name")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    stage = rl_stage_defaults(args.stage)
    vf_env_args = dict(stage["vf_env_args"])
    if args.num_train is not None:
        vf_env_args["num_train"] = args.num_train
    if args.num_eval is not None:
        vf_env_args["num_eval"] = args.num_eval
    if args.seed is not None:
        vf_env_args["seed"] = args.seed
    if args.objective_profile is not None:
        vf_env_args["objective_profile"] = args.objective_profile
    config = {
        "stage": args.stage,
        "vf_env_id": args.vf_env_id or stage["vf_env_id"],
        "vf_env_args": vf_env_args,
        "model_name": args.model_name or stage["model_name"],
        "renderer_name": args.renderer_name or stage["renderer_name"],
        "load_checkpoint_path": args.load_checkpoint_path,
        "kl_reference_checkpoint_path": args.kl_reference_checkpoint_path,
        "lora_rank": args.lora_rank if args.lora_rank is not None else stage["lora_rank"],
        "group_size": args.group_size if args.group_size is not None else stage["group_size"],
        "groups_per_batch": args.groups_per_batch if args.groups_per_batch is not None else stage["groups_per_batch"],
        "learning_rate": args.learning_rate if args.learning_rate is not None else stage["learning_rate"],
        "max_tokens": args.max_tokens if args.max_tokens is not None else stage["max_tokens"],
        "temperature": args.temperature if args.temperature is not None else stage["temperature"],
        "kl_penalty_coef": args.kl_penalty_coef if args.kl_penalty_coef is not None else stage["kl_penalty_coef"],
        "num_substeps": args.num_substeps if args.num_substeps is not None else stage["num_substeps"],
        "save_every": args.save_every if args.save_every is not None else stage["save_every"],
        "eval_every": args.eval_every if args.eval_every is not None else stage["eval_every"],
        "max_steps": args.max_steps if args.max_steps is not None else stage["max_steps"],
        "max_concurrent_generation": args.max_concurrent_generation,
        "max_concurrent_scoring": args.max_concurrent_scoring,
        "log_path": str(args.log_path or default_log_path(args.stage, args.model_name or stage["model_name"])),
    }
    if args.dry_run:
        print(json.dumps(config, indent=2))
        return

    require_env("TINKER_API_KEY")
    asyncio.run(run_training(config, args.wandb_project, args.wandb_name))


async def run_training(config: dict[str, Any], wandb_project: str | None, wandb_name: str | None) -> None:
    from verifiers.clients import OpenAIChatCompletionsClient
    from verifiers.utils.async_utils import maybe_semaphore

    from tinker_cookbook import cli_utils, renderers
    from tinker_cookbook.completers import TinkerTokenCompleter, TokenCompleter
    from tinker_cookbook.recipes.verifiers_rl.tinker_openai import TinkerAsyncOpenAIClient
    from tinker_cookbook.recipes.verifiers_rl.verifiers_env import (
        VerifiersEnvGroupBuilder,
        VerifiersRLDatasetBuilder,
    )
    from tinker_cookbook.rl import rollouts, train
    from tinker_cookbook.rl.types import (
        EnvGroupBuilder,
        TokensWithLogprobs,
        Trajectory,
        TrajectoryGroup,
        Transition,
    )
    from tinker_cookbook.tokenizer_utils import Tokenizer, get_tokenizer

    patch_tinker_cookbook_renderers()

    log_path = Path(config["log_path"])
    cli_utils.check_log_dir(str(log_path), behavior_if_exists="ask")

    shared_raw_client: TinkerAsyncOpenAIClient | None = None
    shared_wrapped_client: OpenAIChatCompletionsClient | None = None
    shared_renderer: renderers.Renderer | None = None
    local_tokenizer: Tokenizer | None = None

    def outputs_to_trajectory_group(outputs: list[dict[str, Any]]) -> TrajectoryGroup:
        trajectories_G: list[Trajectory] = []
        final_rewards_G: list[float] = []
        metrics_G: list[dict[str, float | int]] = []

        for output in outputs:
            steps = output.get("trajectory", [])
            transitions: list[Transition] = []
            for i, step in enumerate(steps):
                tokens_data = step.get("tokens") or {}
                prompt_ids = tokens_data.get("prompt_ids", [])
                completion_ids = tokens_data.get("completion_ids", [])
                completion_logprobs = tokens_data.get("completion_logprobs", [])
                ob = tinker.ModelInput.from_ints(prompt_ids) if prompt_ids else tinker.ModelInput.empty()
                ac = TokensWithLogprobs(tokens=completion_ids, maybe_logprobs=completion_logprobs)
                transitions.append(
                    Transition(
                        ob=ob,
                        ac=ac,
                        reward=float(step.get("reward") or 0.0),
                        episode_done=i == len(steps) - 1,
                        metrics=step.get("extras") or {},
                    )
                )
            trajectories_G.append(Trajectory(transitions=transitions, final_ob=tinker.ModelInput.empty()))
            final_rewards_G.append(float(output.get("reward") or 0.0))
            metrics_G.append(output.get("metrics") or {})

        return TrajectoryGroup(
            trajectories_G=trajectories_G,
            final_rewards_G=final_rewards_G,
            metrics_G=metrics_G,
        )

    async def custom_do_group_rollout(builder: EnvGroupBuilder, policy: TokenCompleter, strategy: Any | None = None) -> TrajectoryGroup:
        nonlocal shared_raw_client, shared_wrapped_client, shared_renderer, local_tokenizer

        if local_tokenizer is None:
            local_tokenizer = get_tokenizer(config["model_name"])
        if shared_renderer is None:
            shared_renderer = get_renderer(config["renderer_name"], local_tokenizer)

        sampling_client = cast(TinkerTokenCompleter, policy).sampling_client
        if shared_raw_client is None:
            shared_raw_client = TinkerAsyncOpenAIClient(sampling_client, shared_renderer, local_tokenizer)
            shared_wrapped_client = OpenAIChatCompletionsClient(shared_raw_client)
        else:
            shared_raw_client.set_sampling_client(sampling_client)

        vf_builder = cast(VerifiersEnvGroupBuilder, builder)
        rollout_inputs = vf_builder.get_rollout_inputs(config["group_size"])
        gen_sem = await maybe_semaphore(config["max_concurrent_generation"])
        score_sem = await maybe_semaphore(config["max_concurrent_scoring"])
        run_group_kwargs = {
            "group_inputs": rollout_inputs,
            "client": shared_wrapped_client,
            "model": "tinker",
            "sampling_args": {"max_tokens": config["max_tokens"], "temperature": config["temperature"]},
            "state_columns": ["trajectory"],
        }
        signature = inspect.signature(vf_builder.vf_env.run_group)
        if "gen_sem" in signature.parameters:
            run_group_kwargs["gen_sem"] = gen_sem
        if "score_sem" in signature.parameters:
            run_group_kwargs["score_sem"] = score_sem
        outputs = await vf_builder.vf_env.run_group(**run_group_kwargs)
        return outputs_to_trajectory_group(outputs)

    train.do_group_rollout = custom_do_group_rollout
    rollouts.do_group_rollout = custom_do_group_rollout

    dataset_builder = VerifiersRLDatasetBuilder(
        vf_env_id=config["vf_env_id"],
        vf_env_args=config["vf_env_args"],
        groups_per_batch=config["groups_per_batch"],
        dataset_n=-1,
        dataset_seed=0,
    )

    kl_reference_config = None
    if config["kl_penalty_coef"] and config["kl_penalty_coef"] > 0.0:
        kl_reference_config = train.KLReferenceConfig(
            base_model=config["model_name"],
            load_checkpoint_path=as_sampler_checkpoint_path(config["kl_reference_checkpoint_path"]),
        )

    train_config = train.Config(
        learning_rate=config["learning_rate"],
        dataset_builder=dataset_builder,
        model_name=config["model_name"],
        max_tokens=config["max_tokens"],
        temperature=config["temperature"],
        lora_rank=config["lora_rank"],
        kl_penalty_coef=config["kl_penalty_coef"],
        kl_reference_config=kl_reference_config,
        num_substeps=config["num_substeps"],
        wandb_project=wandb_project,
        wandb_name=wandb_name,
        log_path=str(log_path),
        eval_every=config["eval_every"],
        save_every=config["save_every"],
        stream_minibatch_config=None,
        max_steps=config["max_steps"],
        load_checkpoint_path=config["load_checkpoint_path"],
        renderer_name=config["renderer_name"],
    )
    await train.main(train_config)


def default_log_path(stage: str, model_name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_slug = model_name.replace("/", "--")
    return OUTPUT_ROOT / "rl" / stage / f"{stamp}-{model_slug}"


def as_sampler_checkpoint_path(path: str | None) -> str | None:
    if not path:
        return path
    if "/sampler_weights/" in path:
        return path
    if "/weights/" in path:
        return path.replace("/weights/", "/sampler_weights/", 1)
    return path


def require_env(name: str) -> None:
    if os.environ.get(name):
        return
    raise SystemExit(f"{name} is required for non-dry-run Tinker commands.")


if __name__ == "__main__":
    main()
