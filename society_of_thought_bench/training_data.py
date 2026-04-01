from __future__ import annotations

import json
import random
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from .core import STYLE_BY_ROLE, clamp, ordered_unique
from .families import build_example
from .parser import SocietyOfThoughtParser
from .tinker_renderers import CUSTOM_QWEN3_RENDERER_NAME

ENV_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ENV_ROOT / "data" / "tinker"
OUTPUT_ROOT = ENV_ROOT / "outputs" / "tinker"

DEFAULT_TRACE_MODE = "debate"
DEFAULT_TRACE_PROMPT_VARIANT = "official"
DEFAULT_INSTITUTION = "auto"
DEFAULT_MAX_PERSONAS = 4
DEFAULT_MAX_DEBATE_TURNS = 10
DEFAULT_OBJECTIVE_PROFILE = "debate_primary"
DEFAULT_COUNTDOWN_REWARD_PROFILE = "benchmark"

DEFAULT_SFT_MODEL = "Qwen/Qwen3-30B-A3B"
DEFAULT_RENDERER_NAME = CUSTOM_QWEN3_RENDERER_NAME

SFT_RUN_DEFAULTS = {
    "model_name": DEFAULT_SFT_MODEL,
    "renderer_name": DEFAULT_RENDERER_NAME,
    "lora_rank": 32,
    "learning_rate": 2e-4,
    "lr_schedule": "linear",
    "num_epochs": 1,
    "max_length": 4096,
    "batch_size": 128,
    "save_every": 8,
    "eval_every": 8,
}

DPO_RUN_DEFAULTS = {
    "model_name": DEFAULT_SFT_MODEL,
    "renderer_name": DEFAULT_RENDERER_NAME,
    "lora_rank": 32,
    "learning_rate": 1e-5,
    "lr_schedule": "linear",
    "num_epochs": 1,
    "dpo_beta": 0.1,
    "max_length": 4096,
    "batch_size": 64,
    "save_every": 10,
    "eval_every": 10,
    "max_steps": 60,
}

CURRICULUM_PROFILES = (
    "debate_primary",
    "protocol_bootcamp",
    "external_core_debate",
    "external_core_monologue",
    "external_bridge_debate",
    "external_proxy_focus_debate",
    "external_proxy_focus_monologue",
)

EXTERNAL_DEBATE_CURRICULUM_PROFILES = {
    "external_core_debate",
    "external_bridge_debate",
    "external_proxy_focus_debate",
}

EXTERNAL_MONOLOGUE_CURRICULUM_PROFILES = {
    "external_core_monologue",
    "external_proxy_focus_monologue",
}

WARMUP_MIX_WEIGHTS_BY_PROFILE = {
    "debate_primary": {
        ("countdown", "easy"): 0.10,
        ("countdown", "medium"): 0.25,
        ("countdown", "hard"): 0.15,
        ("evidence", "easy"): 0.10,
        ("evidence", "medium"): 0.25,
        ("evidence", "hard"): 0.15,
    },
    "protocol_bootcamp": {
        ("countdown", "easy"): 0.15,
        ("countdown", "medium"): 0.30,
        ("countdown", "hard"): 0.05,
        ("evidence", "easy"): 0.15,
        ("evidence", "medium",): 0.30,
        ("evidence", "hard"): 0.05,
    },
    "external_core_debate": {
        ("countdown", "easy"): 0.10,
        ("countdown", "medium"): 0.25,
        ("countdown", "hard"): 0.15,
        ("evidence", "easy"): 0.10,
        ("evidence", "medium"): 0.25,
        ("evidence", "hard"): 0.15,
    },
    "external_core_monologue": {
        ("countdown", "easy"): 0.10,
        ("countdown", "medium"): 0.25,
        ("countdown", "hard"): 0.15,
        ("evidence", "easy"): 0.10,
        ("evidence", "medium"): 0.25,
        ("evidence", "hard"): 0.15,
    },
    "external_bridge_debate": {
        ("countdown", "easy"): 0.10,
        ("countdown", "medium"): 0.25,
        ("countdown", "hard"): 0.15,
        ("evidence", "easy"): 0.10,
        ("evidence", "medium"): 0.25,
        ("evidence", "hard"): 0.15,
    },
    "external_proxy_focus_debate": {
        ("countdown", "easy"): 0.05,
        ("countdown", "medium"): 0.30,
        ("countdown", "hard"): 0.15,
        ("evidence", "easy"): 0.05,
        ("evidence", "medium"): 0.30,
        ("evidence", "hard"): 0.15,
    },
    "external_proxy_focus_monologue": {
        ("countdown", "easy"): 0.05,
        ("countdown", "medium"): 0.30,
        ("countdown", "hard"): 0.15,
        ("evidence", "easy"): 0.05,
        ("evidence", "medium"): 0.30,
        ("evidence", "hard"): 0.15,
    },
}

DPO_PAIR_TYPE_WEIGHTS_BY_PROFILE = {
    "debate_primary": {
        "real_disagreement": 0.20,
        "redundant_personas": 0.20,
        "single_path": 0.20,
        "unresolved_conflict": 0.20,
        "verbose_filler": 0.20,
    },
    "protocol_bootcamp": {
        "missing_block": 0.35,
        "freeform_roles": 0.25,
        "real_disagreement": 0.15,
        "redundant_personas": 0.10,
        "unresolved_conflict": 0.10,
        "single_path": 0.05,
    },
    "external_bridge_debate": {
        "real_disagreement": 0.20,
        "redundant_personas": 0.20,
        "single_path": 0.20,
        "unresolved_conflict": 0.20,
        "verbose_filler": 0.20,
    },
    "external_proxy_focus_debate": {
        "real_disagreement": 0.20,
        "redundant_personas": 0.20,
        "single_path": 0.20,
        "unresolved_conflict": 0.20,
        "verbose_filler": 0.20,
    },
    "external_proxy_focus_monologue": {
        "real_disagreement": 0.20,
        "redundant_personas": 0.20,
        "single_path": 0.20,
        "unresolved_conflict": 0.20,
        "verbose_filler": 0.20,
    },
}

EXTERNAL_SFT_SOURCE_WEIGHTS_BY_PROFILE = {
    "external_core_debate": {
        "internal_sot": 0.40,
        "gsm8k_train": 0.20,
        "mmlu_pro_non_eval": 0.20,
        "mmlu_pro_stem": 0.10,
        "ifeval_synthetic": 0.10,
    },
    "external_core_monologue": {
        "internal_sot": 0.40,
        "gsm8k_train": 0.20,
        "mmlu_pro_non_eval": 0.20,
        "mmlu_pro_stem": 0.10,
        "ifeval_synthetic": 0.10,
    },
    "external_bridge_debate": {
        "internal_sot": 0.55,
        "gsm8k_train": 0.10,
        "mmlu_pro_non_eval": 0.15,
        "mmlu_pro_stem": 0.10,
        "ifeval_synthetic": 0.10,
    },
    "external_proxy_focus_debate": {
        "internal_sot": 0.30,
        "gsm8k_train": 0.05,
        "mmlu_pro_non_eval": 0.10,
        "mmlu_pro_stem": 0.35,
        "ifeval_synthetic": 0.20,
    },
    "external_proxy_focus_monologue": {
        "internal_sot": 0.30,
        "gsm8k_train": 0.05,
        "mmlu_pro_non_eval": 0.10,
        "mmlu_pro_stem": 0.35,
        "ifeval_synthetic": 0.20,
    },
}

EXTERNAL_DPO_COMPONENT_WEIGHTS_BY_PROFILE = {
    "external_core_debate": {
        "internal_debate_quality": 0.45,
        "external_answer_discipline": 0.10,
        "external_reasoning_structure": 0.45,
    },
    "external_core_monologue": {
        "internal_monologue_structure": 0.30,
        "external_answer_discipline": 0.40,
        "external_monologue_structure": 0.30,
    },
    "external_bridge_debate": {
        "internal_debate_quality": 0.45,
        "external_answer_discipline": 0.10,
        "external_reasoning_structure": 0.45,
    },
    "external_proxy_focus_debate": {
        "internal_debate_quality": 0.45,
        "external_answer_discipline": 0.10,
        "external_reasoning_structure": 0.45,
    },
    "external_proxy_focus_monologue": {
        "internal_monologue_structure": 0.30,
        "external_answer_discipline": 0.40,
        "external_monologue_structure": 0.30,
    },
}

EXTERNAL_CORE_BENCHMARKS = (
    "gsm8k",
    "mmlu_pro",
    "gpqa",
    "ifeval",
)

EXTERNAL_ACCEPTANCE_THRESHOLDS = {
    "same_checkpoint": {
        "native_score_delta": 0.03,
        "non_negative_native_score_benchmarks": 3,
        "reasoning_contract_valid_delta": -0.10,
        "visible_answer_valid_delta": -0.05,
    },
    "debate_vs_control": {
        "native_score_delta": 0.02,
        "non_negative_native_score_benchmarks": 3,
        "reasoning_contract_valid_delta": -0.10,
        "visible_answer_valid_delta": -0.05,
    },
}

PROTOCOL_FAILURE_PAIR_TYPES = {
    "missing_block",
    "freeform_roles",
}

EVAL_SUITES: dict[str, dict[str, Any]] = {
    "protocol_easy_gate": {
        "family": "all",
        "difficulty": "easy",
        "trace_mode": "debate",
        "trace_prompt_variant": DEFAULT_TRACE_PROMPT_VARIANT,
        "institution": DEFAULT_INSTITUTION,
        "objective_profile": DEFAULT_OBJECTIVE_PROFILE,
        "num_examples": 20,
        "rollouts_per_example": 1,
        "max_concurrent": 32,
        "max_tokens": 1024,
        "temperature": 0.8,
    },
    "debate_medium_gate": {
        "family": "all",
        "difficulty": "medium",
        "trace_mode": "debate",
        "trace_prompt_variant": DEFAULT_TRACE_PROMPT_VARIANT,
        "institution": DEFAULT_INSTITUTION,
        "objective_profile": DEFAULT_OBJECTIVE_PROFILE,
        "num_examples": 60,
        "rollouts_per_example": 1,
        "max_concurrent": 32,
        "max_tokens": 1024,
        "temperature": 0.8,
    },
    "debate_hard_gate": {
        "family": "all",
        "difficulty": "hard",
        "trace_mode": "debate",
        "trace_prompt_variant": DEFAULT_TRACE_PROMPT_VARIANT,
        "institution": DEFAULT_INSTITUTION,
        "objective_profile": DEFAULT_OBJECTIVE_PROFILE,
        "num_examples": 40,
        "rollouts_per_example": 1,
        "max_concurrent": 32,
        "max_tokens": 1024,
        "temperature": 0.8,
    },
    "debate_vs_monologue": {
        "family": "all",
        "difficulty": "medium",
        "trace_prompt_variant": DEFAULT_TRACE_PROMPT_VARIANT,
        "institution": DEFAULT_INSTITUTION,
        "objective_profile": DEFAULT_OBJECTIVE_PROFILE,
        "num_examples": 40,
        "rollouts_per_example": 1,
        "max_concurrent": 32,
        "max_tokens": 1024,
        "temperature": 0.8,
    },
    "countdown_easy_gate": {
        "family": "countdown",
        "difficulty": "easy",
        "trace_mode": "debate",
        "trace_prompt_variant": DEFAULT_TRACE_PROMPT_VARIANT,
        "institution": DEFAULT_INSTITUTION,
        "objective_profile": DEFAULT_OBJECTIVE_PROFILE,
        "num_examples": 40,
        "rollouts_per_example": 1,
        "max_concurrent": 32,
        "max_tokens": 1024,
        "temperature": 0.8,
    },
    "all_easy_gate": {
        "family": "all",
        "difficulty": "easy",
        "trace_mode": "debate",
        "trace_prompt_variant": DEFAULT_TRACE_PROMPT_VARIANT,
        "institution": DEFAULT_INSTITUTION,
        "objective_profile": DEFAULT_OBJECTIVE_PROFILE,
        "num_examples": 40,
        "rollouts_per_example": 1,
        "max_concurrent": 32,
        "max_tokens": 1024,
        "temperature": 0.8,
    },
}

RL_STAGE_CONFIGS: dict[str, dict[str, Any]] = {
    "countdown_easy": {
        "vf_env_id": "society-of-thought-bench",
        "vf_env_args": {
            "family": "countdown",
            "difficulty": "easy",
            "institution": DEFAULT_INSTITUTION,
            "trace_mode": "debate",
            "trace_prompt_variant": DEFAULT_TRACE_PROMPT_VARIANT,
                "countdown_reward_profile": "exact_emphasis",
            "objective_profile": "balanced",
            "debate_reward_weight": 0.10,
        },
        "model_name": DEFAULT_SFT_MODEL,
        "renderer_name": DEFAULT_RENDERER_NAME,
        "lora_rank": 32,
        "group_size": 8,
        "groups_per_batch": 16,
        "learning_rate": 8e-6,
        "max_tokens": 1024,
        "temperature": 0.8,
        "kl_penalty_coef": 0.02,
        "num_substeps": 1,
        "save_every": 16,
        "eval_every": 0,
        "max_steps": 64,
    },
    "all_easy": {
        "vf_env_id": "society-of-thought-bench",
        "vf_env_args": {
            "family": "all",
            "difficulty": "easy",
            "institution": DEFAULT_INSTITUTION,
            "trace_mode": "debate",
            "trace_prompt_variant": DEFAULT_TRACE_PROMPT_VARIANT,
                "countdown_reward_profile": "exact_emphasis",
            "objective_profile": "balanced",
            "debate_reward_weight": 0.10,
        },
        "model_name": DEFAULT_SFT_MODEL,
        "renderer_name": DEFAULT_RENDERER_NAME,
        "lora_rank": 32,
        "group_size": 8,
        "groups_per_batch": 16,
        "learning_rate": 8e-6,
        "max_tokens": 1024,
        "temperature": 0.8,
        "kl_penalty_coef": 0.02,
        "num_substeps": 1,
        "save_every": 16,
        "eval_every": 0,
        "max_steps": 64,
    },
    "all_medium": {
        "vf_env_id": "society-of-thought-bench",
        "vf_env_args": {
            "family": "all",
            "difficulty": "medium",
            "institution": DEFAULT_INSTITUTION,
            "trace_mode": "debate",
            "trace_prompt_variant": DEFAULT_TRACE_PROMPT_VARIANT,
                "countdown_reward_profile": DEFAULT_COUNTDOWN_REWARD_PROFILE,
            "objective_profile": DEFAULT_OBJECTIVE_PROFILE,
        },
        "model_name": DEFAULT_SFT_MODEL,
        "renderer_name": DEFAULT_RENDERER_NAME,
        "lora_rank": 32,
        "group_size": 8,
        "groups_per_batch": 16,
        "learning_rate": 6e-6,
        "max_tokens": 1024,
        "temperature": 0.8,
        "kl_penalty_coef": 0.03,
        "num_substeps": 1,
        "save_every": 16,
        "eval_every": 0,
        "max_steps": 64,
    },
    "all_hard": {
        "vf_env_id": "society-of-thought-bench",
        "vf_env_args": {
            "family": "all",
            "difficulty": "hard",
            "institution": DEFAULT_INSTITUTION,
            "trace_mode": "debate",
            "trace_prompt_variant": DEFAULT_TRACE_PROMPT_VARIANT,
                "countdown_reward_profile": DEFAULT_COUNTDOWN_REWARD_PROFILE,
            "objective_profile": DEFAULT_OBJECTIVE_PROFILE,
        },
        "model_name": DEFAULT_SFT_MODEL,
        "renderer_name": DEFAULT_RENDERER_NAME,
        "lora_rank": 32,
        "group_size": 8,
        "groups_per_batch": 16,
        "learning_rate": 6e-6,
        "max_tokens": 1024,
        "temperature": 0.8,
        "kl_penalty_coef": 0.03,
        "num_substeps": 1,
        "save_every": 16,
        "eval_every": 0,
        "max_steps": 48,
    },
    "control_all_medium": {
        "vf_env_id": "society-of-thought-bench",
        "vf_env_args": {
            "family": "all",
            "difficulty": "medium",
            "institution": DEFAULT_INSTITUTION,
            "trace_mode": "monologue",
            "trace_prompt_variant": DEFAULT_TRACE_PROMPT_VARIANT,
            "countdown_reward_profile": DEFAULT_COUNTDOWN_REWARD_PROFILE,
            "objective_profile": "balanced",
            "debate_reward_weight": 0.0,
        },
        "model_name": DEFAULT_SFT_MODEL,
        "renderer_name": DEFAULT_RENDERER_NAME,
        "lora_rank": 32,
        "group_size": 8,
        "groups_per_batch": 16,
        "learning_rate": 6e-6,
        "max_tokens": 1024,
        "temperature": 0.8,
        "kl_penalty_coef": 0.03,
        "num_substeps": 1,
        "save_every": 16,
        "eval_every": 0,
        "max_steps": 64,
    },
    "control_all_hard": {
        "vf_env_id": "society-of-thought-bench",
        "vf_env_args": {
            "family": "all",
            "difficulty": "hard",
            "institution": DEFAULT_INSTITUTION,
            "trace_mode": "monologue",
            "trace_prompt_variant": DEFAULT_TRACE_PROMPT_VARIANT,
            "countdown_reward_profile": DEFAULT_COUNTDOWN_REWARD_PROFILE,
            "objective_profile": "balanced",
            "debate_reward_weight": 0.0,
        },
        "model_name": DEFAULT_SFT_MODEL,
        "renderer_name": DEFAULT_RENDERER_NAME,
        "lora_rank": 32,
        "group_size": 8,
        "groups_per_batch": 16,
        "learning_rate": 6e-6,
        "max_tokens": 1024,
        "temperature": 0.8,
        "kl_penalty_coef": 0.03,
        "num_substeps": 1,
        "save_every": 16,
        "eval_every": 0,
        "max_steps": 48,
    },
}

GATE_THRESHOLDS: dict[str, dict[str, dict[str, float]]] = {
    "debate_primary": {
        "protocol_easy_gate": {
            "joint_contract_valid_rate": 0.95,
            "answer_format_valid_rate": 0.98,
            "trace_format_valid_rate": 0.98,
        },
        "debate_medium_gate": {
            "joint_contract_valid_rate": 0.95,
            "behavior_coverage": 0.90,
            "persona_diversity": 0.85,
            "interaction_score": 0.55,
            "debate_relevance": 0.80,
            "disagreement_quality": 0.70,
            "average_task_score": 0.20,
        },
        "debate_hard_gate": {
            "joint_contract_valid_rate": 0.90,
            "behavior_coverage": 0.85,
            "persona_diversity": 0.85,
            "interaction_score": 0.55,
            "debate_relevance": 0.75,
            "disagreement_quality": 0.65,
            "average_task_score": 0.15,
        },
        "debate_vs_monologue": {
            "reward_delta": 0.10,
            "disagreement_quality_delta": 0.15,
        },
    }
}

ROLE_EXPERTISE = {
    "countdown": {
        "brainstormer": "arithmetic",
        "devils_advocate": "search",
        "verifier": "verification",
        "synthesizer": "error_checking",
    },
    "evidence": {
        "brainstormer": "fact_extraction",
        "devils_advocate": "contradiction_checking",
        "verifier": "timeline_reasoning",
        "synthesizer": "synthesis",
    },
}

PERSONA_LAYOUTS = {
    "easy": (
        (("brainstormer", "high_openness"), ("verifier", "high_conscientiousness")),
        (("brainstormer", "high_extraversion"), ("verifier", "high_conscientiousness")),
        (("verifier", "high_conscientiousness"), ("brainstormer", "high_openness")),
        (("brainstormer", "high_openness"), ("verifier", "high_agreeableness")),
    ),
    "medium_flat": (
        (("brainstormer", "high_openness"), ("devils_advocate", "low_agreeableness"), ("verifier", "high_conscientiousness")),
        (("verifier", "high_conscientiousness"), ("brainstormer", "high_extraversion"), ("devils_advocate", "high_neuroticism")),
        (("brainstormer", "high_openness"), ("verifier", "high_conscientiousness"), ("devils_advocate", "low_agreeableness")),
        (("devils_advocate", "low_agreeableness"), ("brainstormer", "high_openness"), ("verifier", "high_agreeableness")),
    ),
    "medium_hierarchical": (
        (("brainstormer", "high_openness"), ("devils_advocate", "low_agreeableness"), ("verifier", "high_conscientiousness"), ("synthesizer", "high_agreeableness")),
        (("verifier", "high_conscientiousness"), ("brainstormer", "high_extraversion"), ("devils_advocate", "high_neuroticism"), ("synthesizer", "high_agreeableness")),
        (("brainstormer", "high_openness"), ("verifier", "high_conscientiousness"), ("devils_advocate", "low_agreeableness"), ("synthesizer", "high_extraversion")),
        (("devils_advocate", "low_agreeableness"), ("brainstormer", "high_openness"), ("verifier", "high_conscientiousness"), ("synthesizer", "high_agreeableness")),
    ),
    "hard": (
        (("brainstormer", "high_openness"), ("devils_advocate", "low_agreeableness"), ("verifier", "high_conscientiousness"), ("synthesizer", "high_agreeableness")),
        (("verifier", "high_conscientiousness"), ("brainstormer", "high_extraversion"), ("devils_advocate", "high_neuroticism"), ("synthesizer", "high_agreeableness")),
        (("brainstormer", "high_openness"), ("synthesizer", "high_agreeableness"), ("devils_advocate", "low_agreeableness"), ("verifier", "high_conscientiousness")),
        (("devils_advocate", "low_agreeableness"), ("brainstormer", "high_openness"), ("verifier", "high_conscientiousness"), ("synthesizer", "high_extraversion")),
    ),
}

TRACE_DIALECTS = (
    "persona_think",
    "character_step",
    "named_tag",
    "speaker_lines",
)


@dataclass(slots=True)
class GateCheck:
    stage: str
    suite: str
    passed: bool
    checks: list[dict[str, Any]]


def _weighted_counts(total: int, weights: dict[Any, float]) -> dict[Any, int]:
    items = list(weights.items())
    counts = {key: int(total * weight) for key, weight in items}
    remainder = total - sum(counts.values())
    for key, _weight in items:
        if remainder <= 0:
            break
        counts[key] += 1
        remainder -= 1
    return counts


def curriculum_profile_weights(curriculum_profile: str) -> dict[tuple[str, str], float]:
    if curriculum_profile not in CURRICULUM_PROFILES:
        raise KeyError(f"Unknown curriculum_profile: {curriculum_profile}")
    return WARMUP_MIX_WEIGHTS_BY_PROFILE[curriculum_profile]


def pair_type_profile_weights(curriculum_profile: str) -> dict[str, float]:
    if curriculum_profile not in CURRICULUM_PROFILES:
        raise KeyError(f"Unknown curriculum_profile: {curriculum_profile}")
    return DPO_PAIR_TYPE_WEIGHTS_BY_PROFILE[curriculum_profile]


def warmup_mix_counts(total: int, curriculum_profile: str = "debate_primary") -> dict[tuple[str, str], int]:
    return _weighted_counts(total, curriculum_profile_weights(curriculum_profile))


def dpo_mix_counts(total: int, curriculum_profile: str = "debate_primary") -> dict[tuple[str, str], int]:
    return _weighted_counts(total, curriculum_profile_weights(curriculum_profile))


def pair_type_counts(total: int, curriculum_profile: str = "debate_primary") -> dict[str, int]:
    return _weighted_counts(total, pair_type_profile_weights(curriculum_profile))


def external_sft_source_weights(curriculum_profile: str) -> dict[str, float]:
    if curriculum_profile not in EXTERNAL_SFT_SOURCE_WEIGHTS_BY_PROFILE:
        raise KeyError(f"Unknown external SFT curriculum_profile: {curriculum_profile}")
    return EXTERNAL_SFT_SOURCE_WEIGHTS_BY_PROFILE[curriculum_profile]


def external_sft_source_counts(total: int, curriculum_profile: str) -> dict[str, int]:
    return _weighted_counts(total, external_sft_source_weights(curriculum_profile))


def external_dpo_component_weights(curriculum_profile: str) -> dict[str, float]:
    if curriculum_profile not in EXTERNAL_DPO_COMPONENT_WEIGHTS_BY_PROFILE:
        raise KeyError(f"Unknown external DPO curriculum_profile: {curriculum_profile}")
    return EXTERNAL_DPO_COMPONENT_WEIGHTS_BY_PROFILE[curriculum_profile]


def external_dpo_component_counts(total: int, curriculum_profile: str) -> dict[str, int]:
    return _weighted_counts(total, external_dpo_component_weights(curriculum_profile))


def suite_defaults(name: str) -> dict[str, Any]:
    if name not in EVAL_SUITES:
        raise KeyError(f"Unknown suite: {name}")
    return dict(EVAL_SUITES[name])


def rl_stage_defaults(name: str) -> dict[str, Any]:
    if name not in RL_STAGE_CONFIGS:
        raise KeyError(f"Unknown RL stage: {name}")
    return json.loads(json.dumps(RL_STAGE_CONFIGS[name]))


def select_training_institution(difficulty: str, seed: int) -> str:
    rng = random.Random(seed)
    if difficulty == "easy":
        return "flat"
    if difficulty == "medium":
        return "flat" if rng.random() < 0.70 else "hierarchical"
    return "flat" if rng.random() < 0.50 else "hierarchical"


def build_warmup_example(
    *,
    family: str,
    difficulty: str,
    seed: int,
    split: str,
    example_id: int = 0,
    institution: str = DEFAULT_INSTITUTION,
    max_personas: int = DEFAULT_MAX_PERSONAS,
    max_debate_turns: int = DEFAULT_MAX_DEBATE_TURNS,
    trace_prompt_variant: str = DEFAULT_TRACE_PROMPT_VARIANT,
    objective_profile: str = DEFAULT_OBJECTIVE_PROFILE,
    countdown_reward_profile: str = DEFAULT_COUNTDOWN_REWARD_PROFILE,
    curriculum_profile: str = "debate_primary",
) -> dict[str, Any]:
    resolved_institution = (
        select_training_institution(difficulty, seed)
        if institution == DEFAULT_INSTITUTION
        else institution
    )
    row = build_example(
        family=family,
        difficulty=difficulty,
        institution=resolved_institution,
        seed=seed,
        index=example_id,
        max_personas=max_personas,
        max_debate_turns=max_debate_turns,
        trace_mode=DEFAULT_TRACE_MODE,
        split=split,
        trace_prompt_variant=trace_prompt_variant,
        countdown_reward_profile=countdown_reward_profile,
        objective_profile=objective_profile,
    )
    info = json.loads(row["info"])
    task = info["task"]
    trace_payload, group_solution = _build_teacher_trace_payload(
        family=family,
        difficulty=difficulty,
        task=task,
        institution=resolved_institution,
        seed=seed,
    )
    trace_payload["dialect"] = select_trace_dialect(seed)
    answer_payload = _build_teacher_answer_payload(family=family, task=task)
    wrapped_trace = _wrap_trace_payload(trace_payload)
    final_answer_text = answer_payload

    messages = [
        *row["prompt"],
        _assistant_completion_message(wrapped_trace, final_answer_text),
    ]
    example = {
        "messages": messages,
        "family": family,
        "difficulty": difficulty,
        "institution": resolved_institution,
        "seed": seed,
        "example_id": example_id,
        "task_name": row["task"],
        "trace_prompt_variant": trace_prompt_variant,
        "objective_profile": objective_profile,
        "curriculum_profile": curriculum_profile,
        "group_solution": group_solution,
        "oracle_task": task,
    }
    validate_warmup_example(example)
    return example


def build_dpo_pair_example(
    *,
    family: str,
    difficulty: str,
    seed: int,
    split: str,
    pair_type: str,
    example_id: int = 0,
    institution: str = DEFAULT_INSTITUTION,
    max_personas: int = DEFAULT_MAX_PERSONAS,
    max_debate_turns: int = DEFAULT_MAX_DEBATE_TURNS,
    trace_prompt_variant: str = DEFAULT_TRACE_PROMPT_VARIANT,
    objective_profile: str = DEFAULT_OBJECTIVE_PROFILE,
    countdown_reward_profile: str = DEFAULT_COUNTDOWN_REWARD_PROFILE,
    curriculum_profile: str = "debate_primary",
) -> dict[str, Any]:
    resolved_institution = (
        select_training_institution(difficulty, seed)
        if institution == DEFAULT_INSTITUTION
        else institution
    )
    row = build_example(
        family=family,
        difficulty=difficulty,
        institution=resolved_institution,
        seed=seed,
        index=example_id,
        max_personas=max_personas,
        max_debate_turns=max_debate_turns,
        trace_mode=DEFAULT_TRACE_MODE,
        split=split,
        trace_prompt_variant=trace_prompt_variant,
        countdown_reward_profile=countdown_reward_profile,
        objective_profile=objective_profile,
    )
    info = json.loads(row["info"])
    task = info["task"]
    chosen_payload, group_solution = _build_teacher_trace_payload(
        family=family,
        difficulty=difficulty,
        task=task,
        institution=resolved_institution,
        seed=seed,
    )
    chosen_payload["dialect"] = select_trace_dialect(seed)
    answer_payload = _build_teacher_answer_payload(family=family, task=task)
    final_answer_text = answer_payload
    chosen_trace = _wrap_trace_payload(chosen_payload)
    chosen_completion = [_assistant_completion_message(chosen_trace, final_answer_text)]
    rejected_completion = _build_rejected_completion_messages(
        chosen_payload=chosen_payload,
        pair_type=pair_type,
        family=family,
        difficulty=difficulty,
        task=task,
        institution=resolved_institution,
        seed=seed,
        final_answer_text=final_answer_text,
        curriculum_profile=curriculum_profile,
    )

    example = {
        "prompt_messages": row["prompt"],
        "completion_A": chosen_completion,
        "completion_B": rejected_completion,
        "label": "A",
        "family": family,
        "difficulty": difficulty,
        "institution": resolved_institution,
        "seed": seed,
        "example_id": example_id,
        "pair_type": pair_type,
        "trace_prompt_variant": trace_prompt_variant,
        "objective_profile": objective_profile,
        "curriculum_profile": curriculum_profile,
        "group_solution": group_solution,
        "oracle_task": task,
    }
    validate_dpo_pair_example(example)
    return example


def build_parser_completion(example: dict[str, Any]) -> dict[str, Any]:
    assistant = example["messages"][-1]
    content_parts = assistant["content"]
    if isinstance(content_parts, str):
        text = content_parts
        if "<think>" in text and "</think>" in text:
            _, after_open = text.split("<think>", 1)
            reasoning_text, answer_text = after_open.split("</think>", 1)
            return {
                "role": "assistant",
                "content": answer_text.strip(),
                "reasoning_content": reasoning_text.strip(),
            }
        return {
            "role": "assistant",
            "content": text,
            "reasoning_content": "",
        }
    reasoning_part = next(part for part in content_parts if part["type"] == "thinking")
    text_part = next(part for part in content_parts if part["type"] == "text")
    return {
        "role": "assistant",
        "content": text_part["text"],
        "reasoning_content": reasoning_part["thinking"],
    }


def select_trace_dialect(seed: int) -> str:
    return TRACE_DIALECTS[seed % len(TRACE_DIALECTS)]


def validate_warmup_example(example: dict[str, Any]) -> None:
    completion = build_parser_completion(example)
    parser = SocietyOfThoughtParser(
        trace_mode=DEFAULT_TRACE_MODE,
    )
    parsed = parser.parse_completion([completion])
    if not (parsed.protocol_valid and parsed.trace_valid and parsed.answer_valid):
        raise ValueError(
            "Warmup example does not satisfy the benchmark contract: "
            f"{parsed.error_codes}"
        )


def validate_dpo_pair_example(example: dict[str, Any]) -> None:
    parser = SocietyOfThoughtParser(
        trace_mode=DEFAULT_TRACE_MODE,
    )
    chosen = parser.parse_completion(example["completion_A"])
    if not (chosen.protocol_valid and chosen.trace_valid and chosen.answer_valid):
        raise ValueError(f"DPO example completion_A is invalid: {chosen.error_codes}")

    rejected = parser.parse_completion(example["completion_B"])
    pair_type = example.get("pair_type", "")
    if pair_type in PROTOCOL_FAILURE_PAIR_TYPES:
        if rejected.protocol_valid and rejected.trace_valid and rejected.answer_valid:
            raise ValueError(
                f"DPO example completion_B unexpectedly remained fully valid for pair_type={pair_type}"
            )
        if not rejected.answer_valid:
            raise ValueError(
                f"DPO example completion_B should preserve a clean final answer for pair_type={pair_type}: {rejected.error_codes}"
            )
        return

    if not (rejected.protocol_valid and rejected.trace_valid and rejected.answer_valid):
        raise ValueError(f"DPO example completion_B is invalid: {rejected.error_codes}")


def summarize_generate_outputs(results: dict[str, Any]) -> dict[str, Any]:
    rewards = [float(value) for value in results.get("reward", [])]
    metrics = results.get("metrics", {})

    protocol = _metric_list(metrics, "protocol_valid")
    trace = _metric_list(metrics, "trace_format_valid")
    answer = _metric_list(metrics, "answer_format_valid")
    task = _metric_list(metrics, "task_score")
    near_miss = _metric_list(metrics, "task_correct_but_protocol_invalid")

    joint = [
        1.0 if protocol[i] >= 0.999 and trace[i] >= 0.999 and answer[i] >= 0.999 else 0.0
        for i in range(_max_metric_length(rewards, protocol, trace, answer, task))
    ]
    task_and_joint = [
        1.0 if joint[i] >= 0.999 and task[i] >= 0.999 else 0.0
        for i in range(len(joint))
    ]

    summary = {
        "examples": len(rewards) if rewards else len(joint),
        "average_reward": _safe_mean(rewards),
        "joint_contract_valid_rate": _safe_mean(joint),
        "task_and_joint_valid_rate": _safe_mean(task_and_joint),
        "near_miss_rate": _safe_mean(near_miss),
        "protocol_valid_rate": _safe_mean(protocol),
        "trace_format_valid_rate": _safe_mean(trace),
        "answer_format_valid_rate": _safe_mean(answer),
        "average_task_score": _safe_mean(task),
    }
    for metric_name in (
        "behavior_coverage",
        "persona_diversity",
        "interaction_score",
        "debate_relevance",
        "disagreement_quality",
        "question_answering_count",
        "perspective_shift_count",
        "conflict_of_perspectives_count",
        "reconciliation_count",
        "challenge_response_pair_count",
        "alternative_path_count",
        "reconcile_link_count",
    ):
        summary[metric_name] = _safe_mean(_metric_list(metrics, metric_name))
    return summary


def compare_mode_summaries(
    debate_summary: dict[str, Any],
    monologue_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "debate": debate_summary,
        "monologue": monologue_summary,
        "reward_delta": debate_summary["average_reward"] - monologue_summary["average_reward"],
        "task_score_delta": debate_summary["average_task_score"] - monologue_summary["average_task_score"],
        "joint_valid_delta": (
            debate_summary["joint_contract_valid_rate"]
            - monologue_summary["joint_contract_valid_rate"]
        ),
        "disagreement_quality_delta": (
            debate_summary.get("disagreement_quality", 0.0)
            - monologue_summary.get("disagreement_quality", 0.0)
        ),
        "persona_diversity_delta": (
            debate_summary.get("persona_diversity", 0.0)
            - monologue_summary.get("persona_diversity", 0.0)
        ),
        "interaction_score_delta": (
            debate_summary.get("interaction_score", 0.0)
            - monologue_summary.get("interaction_score", 0.0)
        ),
        "behavior_coverage_delta": (
            debate_summary.get("behavior_coverage", 0.0)
            - monologue_summary.get("behavior_coverage", 0.0)
        ),
        "debate_relevance_delta": (
            debate_summary.get("debate_relevance", 0.0)
            - monologue_summary.get("debate_relevance", 0.0)
        ),
    }


def evaluate_gate(summary: dict[str, Any], *, suite: str, stage: str) -> GateCheck:
    suite_thresholds = GATE_THRESHOLDS.get(stage, {}).get(suite)
    if suite_thresholds is None:
        raise KeyError(f"No gate thresholds for stage={stage}, suite={suite}")

    checks: list[dict[str, Any]] = []
    passed = True
    for metric_name, threshold in suite_thresholds.items():
        actual = float(summary.get(metric_name, 0.0))
        limit = float(threshold)
        ok = actual >= limit
        checks.append(
            {
                "metric": metric_name,
                "actual": actual,
                "threshold": limit,
                "passed": ok,
            }
        )
        passed = passed and ok
    return GateCheck(stage=stage, suite=suite, passed=passed, checks=checks)


def _metric_list(metrics: dict[str, Any], key: str) -> list[float]:
    values = metrics.get(key, [])
    if isinstance(values, list):
        return [float(value) for value in values]
    return []


def _max_metric_length(*lists: list[float]) -> int:
    lengths = [len(values) for values in lists]
    return max(lengths, default=0)


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(mean(values))


def _assistant_completion_message(trace_text: str, answer_text: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": trace_text},
            {"type": "text", "text": answer_text},
        ],
    }


def _build_teacher_answer_payload(family: str, task: dict[str, Any]) -> str:
    if family == "countdown":
        return f"<answer>{task['oracle_expression']}</answer>"
    support = ",".join(task["oracle_support"])
    return f"<answer>{task['oracle_verdict']}</answer>\n<support>{support}</support>"


def _build_teacher_trace_payload(
    *,
    family: str,
    difficulty: str,
    task: dict[str, Any],
    institution: str,
    seed: int,
) -> tuple[dict[str, Any], str]:
    if family == "countdown":
        return _countdown_trace_payload(task, difficulty, institution, seed)
    return _evidence_trace_payload(task, difficulty, institution, seed)


def _countdown_trace_payload(task: dict[str, Any], difficulty: str, institution: str, seed: int) -> tuple[dict[str, Any], str]:
    refs = [item["id"] for item in task["numbers"]] + ["T"]
    expression = task["oracle_expression"]
    target = task["target"]
    personas = _build_personas("countdown", difficulty, institution, seed)
    speaker_by_role = {persona["role"]: persona["id"] for persona in personas}
    brainstormer = speaker_by_role.get("brainstormer", personas[0]["id"])
    verifier = speaker_by_role.get("verifier", personas[-1]["id"])
    devils_advocate = speaker_by_role.get("devils_advocate", verifier)
    synthesizer = speaker_by_role.get("synthesizer", brainstormer)
    pair_refs = refs[:3] if len(refs) >= 3 else refs

    if difficulty == "easy":
        debate = [
            {
                "id": "t1",
                "speaker": brainstormer,
                "act": "question",
                "reply_to": [],
                "refs": refs,
                "content": f"Which legal arithmetic path can reach T={target} from the numbered inputs?",
            },
            {
                "id": "t2",
                "speaker": brainstormer,
                "act": "propose",
                "reply_to": ["t1"],
                "refs": refs,
                "content": f"Propose {expression} as the clean candidate because it uses only the provided IDs.",
            },
            {
                "id": "t3",
                "speaker": verifier,
                "act": "verify",
                "reply_to": ["t2"],
                "refs": refs,
                "content": f"Check passes: {expression} evaluates to {target} exactly and keeps one legal use per input.",
            },
        ]
    elif difficulty == "medium":
        final_speaker = synthesizer if institution == "hierarchical" else brainstormer
        debate = [
            {
                "id": "t1",
                "speaker": brainstormer,
                "act": "question",
                "reply_to": [],
                "refs": refs,
                "content": f"What is the most defensible exact route to T={target} from these numbered inputs?",
            },
            {
                "id": "t2",
                "speaker": brainstormer,
                "act": "propose",
                "reply_to": ["t1"],
                "refs": refs,
                "content": f"Primary proposal: {expression} as the exact-hit path.",
            },
            {
                "id": "t3",
                "speaker": devils_advocate,
                "act": "challenge",
                "reply_to": ["t2"],
                "refs": refs,
                "content": "Challenge the proposal by checking whether every constant really comes from the numbered IDs and whether a hidden reuse slipped in.",
            },
            {
                "id": "t4",
                "speaker": verifier,
                "act": "verify",
                "reply_to": ["t2", "t3"],
                "refs": refs,
                "content": f"Verification: {expression} reaches {target} exactly, uses legal operators, and still survives the reuse check.",
            },
            {
                "id": "t5",
                "speaker": final_speaker,
                "act": "reconcile",
                "reply_to": ["t3", "t4"],
                "refs": refs,
                "content": f"Reconcile on {expression} because the challenge improved confidence without changing the verified route to T={target}.",
            },
        ]
    else:
        debate = [
            {
                "id": "t1",
                "speaker": brainstormer,
                "act": "question",
                "reply_to": [],
                "refs": refs,
                "content": f"Which candidate path best reaches T={target}, and what alternative branch should we compare before committing?",
            },
            {
                "id": "t2",
                "speaker": brainstormer,
                "act": "propose",
                "reply_to": ["t1"],
                "refs": refs,
                "content": f"Proposal: {expression} is the main exact-hit candidate.",
            },
            {
                "id": "t3",
                "speaker": devils_advocate,
                "act": "challenge",
                "reply_to": ["t2"],
                "refs": refs,
                "content": f"Challenge the main path by checking legality and comparing it to a branch that starts from {pair_refs[0]} and {pair_refs[1]}.",
            },
            {
                "id": "t4",
                "speaker": verifier,
                "act": "shift",
                "reply_to": ["t3"],
                "refs": pair_refs,
                "content": f"Alternative path: start with {pair_refs[0]}+{pair_refs[1]} before chasing T={target}; it is a real branch to compare, but it is less direct than {expression}.",
            },
            {
                "id": "t5",
                "speaker": verifier,
                "act": "verify",
                "reply_to": ["t2", "t3", "t4"],
                "refs": refs,
                "content": f"Verification favors {expression}: it still reaches {target} exactly, while the alternative branch is only a side comparison and not the cleaner certified route.",
            },
            {
                "id": "t6",
                "speaker": synthesizer,
                "act": "reconcile",
                "reply_to": ["t3", "t4", "t5"],
                "refs": refs,
                "content": f"Reconcile the disagreement by keeping the verified exact route {expression} and recording the shift branch as a checked but inferior alternative.",
            },
        ]
    return {"personas": personas, "debate": debate, "group_solution": expression}, expression


def _evidence_trace_payload(task: dict[str, Any], difficulty: str, institution: str, seed: int) -> tuple[dict[str, Any], str]:
    claim = task["claim"]
    verdict = task["oracle_verdict"]
    support = task["oracle_support"]
    support_str = ", ".join(support)
    context_refs = ordered_unique(support + [entry["id"] for entry in task["evidence"][:3]])
    personas = _build_personas("evidence", difficulty, institution, seed)
    speaker_by_role = {persona["role"]: persona["id"] for persona in personas}
    brainstormer = speaker_by_role.get("brainstormer", personas[0]["id"])
    verifier = speaker_by_role.get("verifier", personas[-1]["id"])
    devils_advocate = speaker_by_role.get("devils_advocate", verifier)
    synthesizer = speaker_by_role.get("synthesizer", brainstormer)
    primary_support = support[0] if support else context_refs[0]
    group_solution = f"{verdict} from {support_str}"

    if difficulty == "easy":
        debate = [
            {
                "id": "t1",
                "speaker": brainstormer,
                "act": "question",
                "reply_to": [],
                "refs": context_refs,
                "content": f"Which evidence snippets actually decide the current truth of the claim: {claim}",
            },
            {
                "id": "t2",
                "speaker": brainstormer,
                "act": "propose",
                "reply_to": ["t1"],
                "refs": support,
                "content": f"Propose verdict {verdict} with {support_str} as the decisive support.",
            },
            {
                "id": "t3",
                "speaker": verifier,
                "act": "verify",
                "reply_to": ["t2"],
                "refs": context_refs,
                "content": f"Verify that {support_str} is the current decisive evidence, so the grounded verdict remains {verdict}.",
            },
        ]
    elif difficulty == "medium":
        final_speaker = synthesizer if institution == "hierarchical" else brainstormer
        debate = [
            {
                "id": "t1",
                "speaker": brainstormer,
                "act": "question",
                "reply_to": [],
                "refs": context_refs,
                "content": f"Which snippets control the present truth of the claim: {claim}",
            },
            {
                "id": "t2",
                "speaker": brainstormer,
                "act": "propose",
                "reply_to": ["t1"],
                "refs": support,
                "content": f"Initial proposal: {verdict}, supported by {support_str}.",
            },
            {
                "id": "t3",
                "speaker": devils_advocate,
                "act": "challenge",
                "reply_to": ["t2"],
                "refs": context_refs,
                "content": f"Challenge whether the older or rumor evidence changes the verdict, or whether {primary_support} still dominates the current timeline.",
            },
            {
                "id": "t4",
                "speaker": verifier,
                "act": "verify",
                "reply_to": ["t2", "t3"],
                "refs": context_refs,
                "content": f"Timeline check: {support_str} remains decisive, and the conflicting snippets are stale, incomplete, or unsupported.",
            },
            {
                "id": "t5",
                "speaker": final_speaker,
                "act": "reconcile",
                "reply_to": ["t3", "t4"],
                "refs": support,
                "content": f"Reconcile on verdict {verdict} and keep support to {support_str} after resolving the stale-evidence challenge.",
            },
        ]
    else:
        debate = [
            {
                "id": "t1",
                "speaker": brainstormer,
                "act": "question",
                "reply_to": [],
                "refs": context_refs,
                "content": f"Which current evidence decides the claim, and what alternative reading should we test before settling: {claim}",
            },
            {
                "id": "t2",
                "speaker": brainstormer,
                "act": "propose",
                "reply_to": ["t1"],
                "refs": support,
                "content": f"Primary proposal: {verdict}, anchored on {support_str}.",
            },
            {
                "id": "t3",
                "speaker": devils_advocate,
                "act": "challenge",
                "reply_to": ["t2"],
                "refs": context_refs,
                "content": f"Challenge the proposal by testing whether stale or contradictory snippets can overturn {primary_support}.",
            },
            {
                "id": "t4",
                "speaker": verifier,
                "act": "shift",
                "reply_to": ["t3"],
                "refs": context_refs,
                "content": f"Alternative path: treat the older or rumor snippets as a competing reading, then compare that branch directly against the current support {support_str}.",
            },
            {
                "id": "t5",
                "speaker": verifier,
                "act": "verify",
                "reply_to": ["t2", "t3", "t4"],
                "refs": context_refs,
                "content": f"Verification keeps {support_str} as decisive because the alternative branch relies on stale, contradictory, or non-authoritative evidence.",
            },
            {
                "id": "t6",
                "speaker": synthesizer,
                "act": "reconcile",
                "reply_to": ["t3", "t4", "t5"],
                "refs": support,
                "content": f"Reconcile on verdict {verdict}, preserve support to {support_str}, and record the alternative reading as checked but rejected.",
            },
        ]
    return {"personas": personas, "debate": debate, "group_solution": group_solution}, group_solution


def _build_personas(family: str, difficulty: str, institution: str, seed: int) -> list[dict[str, str]]:
    layout_key = "easy"
    if difficulty == "medium":
        layout_key = "medium_hierarchical" if institution == "hierarchical" else "medium_flat"
    elif difficulty == "hard":
        layout_key = "hard"
    layouts = PERSONA_LAYOUTS[layout_key]
    selected = layouts[seed % len(layouts)]
    expertise_map = ROLE_EXPERTISE[family]
    personas: list[dict[str, str]] = []
    for index, (role, personality) in enumerate(selected, start=1):
        personas.append(
            {
                "id": f"P{index}",
                "role": role,
                "personality": personality,
                "expertise": expertise_map[role],
                "style": STYLE_BY_ROLE[role],
            }
        )
    return personas


def _build_rejected_completion_messages(
    *,
    chosen_payload: dict[str, Any],
    pair_type: str,
    family: str,
    difficulty: str,
    task: dict[str, Any],
    institution: str,
    seed: int,
    final_answer_text: str,
    curriculum_profile: str,
) -> list[dict[str, Any]]:
    if curriculum_profile == "protocol_bootcamp" and pair_type in PROTOCOL_FAILURE_PAIR_TYPES:
        reasoning_text = _make_protocol_failure_reasoning(
            trace_payload=chosen_payload,
            pair_type=pair_type,
            family=family,
            difficulty=difficulty,
            task=task,
        )
        return [_assistant_completion_message(reasoning_text, final_answer_text)]

    rejected_payload = _make_rejected_trace_payload(
        trace_payload=chosen_payload,
        pair_type=pair_type,
        family=family,
        difficulty=difficulty,
        task=task,
        institution=institution,
        seed=seed,
    )
    rejected_trace = _wrap_trace_payload(rejected_payload)
    return [_assistant_completion_message(rejected_trace, final_answer_text)]


def _make_protocol_failure_reasoning(
    *,
    trace_payload: dict[str, Any],
    pair_type: str,
    family: str,
    difficulty: str,
    task: dict[str, Any],
) -> str:
    group_solution = trace_payload.get("group_solution", "")
    debate = trace_payload.get("debate", [])
    refs = ordered_unique(ref for turn in debate for ref in turn.get("refs", []))
    ref_text = ", ".join(refs[:6]) if refs else "the task IDs"
    if pair_type == "missing_block":
        return (
            f"I checked {ref_text} and settled on {group_solution}. "
            "There was some disagreement, but I am summarizing it informally instead of writing the structured debate block."
        )
    if pair_type == "freeform_roles":
        lines = []
        for turn in debate[: min(4, len(debate))]:
            lines.append(f"{turn['speaker']} {turn['act']}: {turn['content']}")
        lines.append(
            f"Final informal summary for the {family} {difficulty} task: {group_solution}."
        )
        return "\n".join(lines)
    raise ValueError(f"Unknown protocol-failure pair_type: {pair_type}")


def _make_rejected_trace_payload(
    *,
    trace_payload: dict[str, Any],
    pair_type: str,
    family: str,
    difficulty: str,
    task: dict[str, Any],
    institution: str,
    seed: int,
) -> dict[str, Any]:
    payload = deepcopy(trace_payload)
    if pair_type == "real_disagreement":
        for turn in payload["debate"]:
            if turn["act"] == "challenge":
                turn["act"] = "verify"
                turn["content"] = "Agree with the initial proposal and move ahead without a real disagreement."
        return payload
    if pair_type == "redundant_personas":
        first = payload["personas"][0]
        for persona in payload["personas"]:
            if persona["id"] == first["id"]:
                continue
            persona["role"] = first["role"]
            persona["personality"] = first["personality"]
            persona["expertise"] = first["expertise"]
        return payload
    if pair_type == "single_path":
        for turn in payload["debate"]:
            if turn["act"] == "shift":
                turn["act"] = "verify"
                turn["content"] = "Stay on the first proposal and do not branch away from it."
        return payload
    if pair_type == "unresolved_conflict":
        for turn in payload["debate"]:
            if turn["act"] == "reconcile":
                turn["act"] = "verify"
                turn["reply_to"] = turn["reply_to"][:1]
                turn["content"] = "The disagreement is noted, but the group does not really reconcile it."
        return payload
    if pair_type == "verbose_filler":
        filler = " This repeats generic filler instead of adding new grounded evidence."
        for turn in payload["debate"]:
            turn["content"] = turn["content"] + filler + filler
        duplicate = deepcopy(payload["debate"][-1])
        duplicate["id"] = f"{duplicate['id']}_dup"
        duplicate["content"] = payload["debate"][-1]["content"]
        payload["debate"].append(duplicate)
        return payload
    raise ValueError(f"Unknown pair_type: {pair_type}")


def available_pair_types(difficulty: str, curriculum_profile: str = "debate_primary") -> tuple[str, ...]:
    if curriculum_profile == "protocol_bootcamp":
        if difficulty == "hard":
            return ("missing_block", "freeform_roles", "real_disagreement", "redundant_personas", "unresolved_conflict", "single_path")
        if difficulty == "medium":
            return ("missing_block", "freeform_roles", "real_disagreement", "redundant_personas", "unresolved_conflict")
        return ("missing_block", "freeform_roles", "redundant_personas")
    if difficulty == "hard":
        return tuple(DPO_PAIR_TYPE_WEIGHTS_BY_PROFILE["debate_primary"].keys())
    if difficulty == "medium":
        return ("real_disagreement", "redundant_personas", "unresolved_conflict", "verbose_filler")
    return ("real_disagreement", "redundant_personas", "verbose_filler")


def _wrap_trace_payload(payload: dict[str, Any]) -> str:
    from .families import _wrap_trace_payload as wrap_trace_payload

    return wrap_trace_payload(payload)
