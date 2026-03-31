from __future__ import annotations

import json

import verifiers as vf

from society_of_thought_bench import load_environment


def test_load_environment_all_returns_env_group() -> None:
    env = load_environment(family="all", difficulty="easy", num_train=12, num_eval=12)
    assert isinstance(env, vf.EnvGroup)
    assert len(env.get_dataset()) == 12
    assert len(env.get_eval_dataset()) == 12


def test_load_environment_countdown_returns_single_env() -> None:
    env = load_environment(family="countdown", difficulty="easy", num_train=4, num_eval=4)
    assert isinstance(env, vf.SingleTurnEnv)
    assert env.env_id == "society-of-thought-bench"


def test_load_environment_accepts_prompt_variant() -> None:
    env = load_environment(
        family="countdown",
        difficulty="easy",
        num_train=2,
        num_eval=2,
        trace_prompt_variant="trace_minimal",
    )
    dataset = env.get_dataset()
    row = dataset[0]
    info = row["info"] if isinstance(row["info"], dict) else json.loads(row["info"])
    assert info["trace_prompt_variant"] == "trace_minimal"


def test_load_environment_accepts_countdown_reward_profile() -> None:
    env = load_environment(
        family="countdown",
        difficulty="easy",
        num_train=2,
        num_eval=2,
        countdown_reward_profile="exact_emphasis",
    )
    dataset = env.get_dataset()
    row = dataset[0]
    info = row["info"] if isinstance(row["info"], dict) else json.loads(row["info"])
    assert info["task"]["reward_profile"] == "exact_emphasis"


def test_load_environment_defaults_to_debate_primary_objective() -> None:
    env = load_environment(family="countdown", difficulty="medium", num_train=2, num_eval=2)
    row = env.get_dataset()[0]
    info = row["info"] if isinstance(row["info"], dict) else json.loads(row["info"])
    assert info["objective_profile"] == "debate_primary"


def test_load_environment_accepts_balanced_objective_profile() -> None:
    env = load_environment(
        family="countdown",
        difficulty="medium",
        num_train=2,
        num_eval=2,
        objective_profile="balanced",
    )
    row = env.get_dataset()[0]
    info = row["info"] if isinstance(row["info"], dict) else json.loads(row["info"])
    assert info["objective_profile"] == "balanced"
