# society-of-thought-bench

`society-of-thought-bench` is an experimental Verifiers benchmark for one narrow claim:

> a reasoning model should do better when it carries out a visible, multi-persona debate inside its exposed thinking trace instead of reasoning in a single internal voice.

This repository is the public preview package for that benchmark and its first paper-faithful adapter release.

## What Is In This Preview

- a procedural benchmark with two task families
- a paper-style reasoning contract built around visible `<think>...</think>` traces
- training and evaluation helpers for supervised tuning, preference tuning, and RL
- an early adapter that was trained to produce paper-style internal debate
- a small findings bundle with the current best comparison result

This is a research preview. It is not a final benchmark release and not a finished model.

## Main Finding

On the clearest current comparison, the same trained model does better in debate mode than in monologue mode on this benchmark.

- debate average score: `0.732`
- monologue average score: `0.197`
- relative gain: `+0.535`
- task score gain: `+0.200`
- disagreement-quality gain: `+0.560`

Canonical evidence:

- [Debate vs Monologue Summary](./release_preview/results/debate_vs_monologue_medium_preview.summary.json)
- [Best Medium Debate Summary](./release_preview/results/debate_medium_preview.summary.json)
- [Hard Supporting Summary](./release_preview/results/debate_hard_preview.summary.json)

The honest interpretation is:

- the positive result is real inside this benchmark
- the gain is relative and benchmark-local
- the model still falls short on some stricter richness checks, especially deeper branching and reconciliation

## Benchmark Contract

The benchmark expects the model to separate its thinking from its visible answer.

### Debate mode

The model should emit one outer thinking block and keep the final answer separate:

```text
<think>
<cast_of_characters>
<persona1>Role: ... Personality: ... Expertise: ... Style: ...</persona1>
<persona2>...</persona2>
...
</cast_of_characters>
<conversation>
<think1>...</think1>
<think2>...</think2>
...
</conversation>
<group_solution>...</group_solution>
</think>
<answer>...</answer>
<support>...</support>   # evidence tasks only
```

### Monologue mode

```text
<think>single-voice reasoning</think>
<answer>...</answer>
<support>...</support>   # evidence tasks only
```

The benchmark rewards paper-style behavior inside the exposed thinking trace:

- distinct personas with distinct roles
- challenge instead of shallow agreement
- grounded back-and-forth
- alternative paths on harder tasks
- reconciliation into one final answer

Final task accuracy still matters, but only as a small grounding term in the default profile.

## Two Task Families

### Countdown debate

A procedural arithmetic target task.

- inputs are numbered `N1..Nk` plus target `T`
- the answer is a final arithmetic expression inside `<answer>`
- scoring checks safe expression use, number use, and target closeness

### Evidence verdict debate

A synthetic evidence-reconciliation task.

- evidence snippets are numbered `E1..En`
- the answer is `TRUE`, `FALSE`, or `INSUFFICIENT` inside `<answer>`
- decisive evidence IDs go in `<support>`
- scoring checks verdict correctness and support quality

## Current Best Preview Model

- GitHub repo: [scasella/society-of-thought-bench](https://github.com/scasella/society-of-thought-bench)
- Hugging Face adapter repo: [scasella91/society-of-thought-qwen3-30b-paper-faithful-adapter](https://huggingface.co/scasella91/society-of-thought-qwen3-30b-paper-faithful-adapter)
- base model: `Qwen/Qwen3-30B-A3B`
- best published sampler checkpoint source: `tinker://80d6e740-bf17-52ca-a94c-422c67897617:train:0/sampler_weights/final`

The adapter was trained to produce paper-style internal debate in the exposed reasoning trace.

## Quick Start

From this directory:

```bash
uv pip install -e .
uv run pytest -q
```

From the workspace root:

```bash
prime env install society-of-thought-bench -p environments
uv run python -c 'import verifiers as vf; vf.load_environment("society-of-thought-bench", family="all", difficulty="medium", num_train=8, num_eval=8)'
```

## Try The Preview Model

To see the raw paper-style thinking trace:

```bash
uv run python scripts/try_tinker_checkpoint.py --family countdown --difficulty medium --show-raw-response
```

That helper defaults to the best current preview checkpoint.

## Key Docs

- [Findings](./release_preview/FINDINGS.md)
- [Results](./release_preview/RESULTS.md)
- [Raw Sample](./release_preview/RAW_SAMPLE.md)
- [Limitations](./release_preview/LIMITATIONS.md)
- [HF Model Card Source](./release_preview/HF_MODEL_CARD.md)
- [Release Notes](./release_preview/v0.1.0-preview.md)

## Repository Contents

- `society_of_thought_bench/` benchmark implementation
- `scripts/` data, training, evaluation, and live-try helpers
- `tests/` local verification
- `release_preview/` publishable preview bundle

## Limitations

This preview should not be oversold.

- It is an early positive result, not a final paper result.
- The strongest evidence is the debate-vs-monologue comparison on this benchmark.
- Medium difficulty is the strongest setting today.
- Harder traces still need richer branching and stronger reconciliation.
- The benchmark targets exposed reasoning traces and currently fits Qwen-style reasoning models best.

## License

This preview package is released under the MIT License.
