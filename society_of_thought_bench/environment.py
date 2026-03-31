from __future__ import annotations

import verifiers as vf

from .core import OBJECTIVE_PROFILES, TRACE_PROMPT_VARIANTS
from .families import build_dataset
from .parser import SocietyOfThoughtParser
from .scoring import SocietyOfThoughtScorer, build_rubric

ENV_ID = "society-of-thought-bench"


def _split_counts(total: int) -> tuple[int, int]:
    first = total // 2 + total % 2
    second = total // 2
    return first, second


def _build_single_env(
    family: str,
    difficulty: str,
    institution: str,
    seed: int,
    num_train: int,
    num_eval: int,
    debate_reward_weight: float | None,
    efficiency_penalty_weight: float,
    max_personas: int,
    max_debate_turns: int,
    trace_mode: str,
    debate_metric_weights: dict[str, float] | None,
    trace_prompt_variant: str,
    countdown_reward_profile: str,
    objective_profile: str,
) -> vf.Environment:
    parser = SocietyOfThoughtParser(trace_mode=trace_mode)
    scorer = SocietyOfThoughtScorer(
        max_personas=max_personas,
        max_debate_turns=max_debate_turns,
        objective_profile=objective_profile,
    )
    rubric = build_rubric(
        parser=parser,
        scorer=scorer,
        debate_reward_weight=debate_reward_weight,
        debate_metric_weights=debate_metric_weights,
        efficiency_penalty_weight=efficiency_penalty_weight,
        objective_profile=objective_profile,
    )

    return vf.SingleTurnEnv(
        dataset=lambda: build_dataset(
            family=family,
            difficulty=difficulty,
            institution=institution,
            seed=seed,
            num_examples=num_train,
            max_personas=max_personas,
            max_debate_turns=max_debate_turns,
            trace_mode=trace_mode,
            split="train",
            trace_prompt_variant=trace_prompt_variant,
            countdown_reward_profile=countdown_reward_profile,
            objective_profile=objective_profile,
        ),
        eval_dataset=lambda: build_dataset(
            family=family,
            difficulty=difficulty,
            institution=institution,
            seed=seed + 10_000,
            num_examples=num_eval,
            max_personas=max_personas,
            max_debate_turns=max_debate_turns,
            trace_mode=trace_mode,
            split="eval",
            trace_prompt_variant=trace_prompt_variant,
            countdown_reward_profile=countdown_reward_profile,
            objective_profile=objective_profile,
        ),
        parser=parser,
        rubric=rubric,
        env_id=ENV_ID,
    )


def load_environment(
    family: str = "all",
    difficulty: str = "medium",
    institution: str = "auto",
    seed: int = 0,
    num_train: int = 200,
    num_eval: int = 100,
    debate_reward_weight: float | None = None,
    efficiency_penalty_weight: float = 0.05,
    max_personas: int = 4,
    max_debate_turns: int = 10,
    trace_mode: str = "debate",
    debate_metric_weights: dict[str, float] | None = None,
    trace_prompt_variant: str = "official",
    countdown_reward_profile: str = "benchmark",
    objective_profile: str = "debate_primary",
) -> vf.Environment:
    if family not in {"countdown", "evidence", "all"}:
        raise ValueError("family must be one of: countdown, evidence, all")
    if difficulty not in {"easy", "medium", "hard"}:
        raise ValueError("difficulty must be one of: easy, medium, hard")
    if institution not in {"flat", "hierarchical", "auto"}:
        raise ValueError("institution must be one of: flat, hierarchical, auto")
    if trace_mode not in {"debate", "monologue"}:
        raise ValueError("trace_mode must be one of: debate, monologue")
    if trace_prompt_variant not in TRACE_PROMPT_VARIANTS:
        raise ValueError(f"trace_prompt_variant must be one of: {', '.join(TRACE_PROMPT_VARIANTS)}")
    if countdown_reward_profile not in {"benchmark", "exact_emphasis"}:
        raise ValueError("countdown_reward_profile must be one of: benchmark, exact_emphasis")
    if objective_profile not in OBJECTIVE_PROFILES:
        raise ValueError(f"objective_profile must be one of: {', '.join(OBJECTIVE_PROFILES)}")

    if family != "all":
        return _build_single_env(
            family=family,
            difficulty=difficulty,
            institution=institution,
            seed=seed,
            num_train=num_train,
            num_eval=num_eval,
            debate_reward_weight=debate_reward_weight,
            efficiency_penalty_weight=efficiency_penalty_weight,
            max_personas=max_personas,
            max_debate_turns=max_debate_turns,
            trace_mode=trace_mode,
            debate_metric_weights=debate_metric_weights,
            trace_prompt_variant=trace_prompt_variant,
            countdown_reward_profile=countdown_reward_profile,
            objective_profile=objective_profile,
        )

    countdown_train, evidence_train = _split_counts(num_train)
    countdown_eval, evidence_eval = _split_counts(num_eval)
    countdown_env = _build_single_env(
        family="countdown",
        difficulty=difficulty,
        institution=institution,
        seed=seed,
        num_train=countdown_train,
        num_eval=countdown_eval,
        debate_reward_weight=debate_reward_weight,
        efficiency_penalty_weight=efficiency_penalty_weight,
        max_personas=max_personas,
        max_debate_turns=max_debate_turns,
        trace_mode=trace_mode,
        debate_metric_weights=debate_metric_weights,
        trace_prompt_variant=trace_prompt_variant,
        countdown_reward_profile=countdown_reward_profile,
        objective_profile=objective_profile,
    )
    evidence_env = _build_single_env(
        family="evidence",
        difficulty=difficulty,
        institution=institution,
        seed=seed + 500_000,
        num_train=evidence_train,
        num_eval=evidence_eval,
        debate_reward_weight=debate_reward_weight,
        efficiency_penalty_weight=efficiency_penalty_weight,
        max_personas=max_personas,
        max_debate_turns=max_debate_turns,
        trace_mode=trace_mode,
        debate_metric_weights=debate_metric_weights,
        trace_prompt_variant=trace_prompt_variant,
        countdown_reward_profile=countdown_reward_profile,
        objective_profile=objective_profile,
    )
    return vf.EnvGroup(
        [countdown_env, evidence_env],
        env_names=["countdown_debate", "evidence_verdict_debate"],
    )
