# society-of-thought-bench

Research preview: on this benchmark, the same trained model scores much higher when its visible reasoning trace is a multi-persona debate than when it is a monologue.

In the released medium comparison, the same trained model, on the same benchmark slice, scores `0.732` in debate mode and `0.197` in monologue mode across 40 examples, for a score gap of `+0.535`. Disagreement quality improves by `+0.56`, and task score improves by `+0.20`.

This is benchmark-local evidence from an early research preview, not a broad claim about general-purpose reasoning.

Start here:

- Live demo: [scasella91/society-of-thought-bench-demo](https://huggingface.co/spaces/scasella91/society-of-thought-bench-demo)
- Released comparison: [Debate vs Monologue Summary](./release_preview/results/debate_vs_monologue_medium_preview.summary.json)
- Raw sample trace: [RAW_SAMPLE.md](./release_preview/RAW_SAMPLE.md)
- Hardening packet: [artifact_hardening/README.md](./release_preview/artifact_hardening/README.md)

`society-of-thought-bench` is an experimental Verifiers benchmark built to test that released comparison in a reproducible way.

## What Is In This Preview

- a benchmark package for paper-style visible debate in exposed reasoning traces
- a released debate-versus-monologue comparison, plus the supporting result files
- a public adapter and live demo for inspecting the trace behavior directly
- training and evaluation helpers for supervised tuning, preference tuning, and RL

This preview makes the current result easy to inspect and reproduce. It is not a final benchmark release and not a finished model.

## Hardening Update

We re-ran a stricter confirmation packet around the public checkpoint and published the full audit.

- medium reward delta stayed positive at `0.364`
- easy joint-valid rate came in at `0.675`
- easy answer-valid rate came in at `0.800`
- hard disagreement quality came in at `0.241`

That packet makes the release easier to trust and inspect. It does not change the headline claim above.

## Main Finding

On the released medium comparison, debate mode beats monologue mode for the same trained model.

- debate average score: `0.732`
- monologue average score: `0.197`
- score gap: `+0.535`
- task score gain: `+0.200`
- disagreement-quality gain: `+0.560`

Canonical evidence:

- [Debate vs Monologue Summary](./release_preview/results/debate_vs_monologue_medium_preview.summary.json)
- [Best Medium Debate Summary](./release_preview/results/debate_medium_preview.summary.json)
- [Hard Supporting Summary](./release_preview/results/debate_hard_preview.summary.json)

What to take from it:

- the positive result is real inside this benchmark
- the result is benchmark-local
- the strongest evidence is on medium difficulty
- the model still falls short on some stricter richness checks, especially deeper branching and reconciliation

## Benchmark Contract

The benchmark is built around a simple separation: the reasoning trace stays inside `<think>...</think>`, and the visible answer stays outside it.

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

Within that format, the benchmark rewards paper-style debate in the visible reasoning trace:

- distinct personas with distinct roles
- challenge instead of shallow agreement
- grounded back-and-forth
- alternative paths on harder tasks
- reconciliation into one final answer

Final task accuracy still matters, but only as a grounding term in the default profile.

## Two Task Families

### Countdown Debate

A procedural arithmetic target task that checks whether the debate stays grounded in the numbered inputs.

- inputs are numbered `N1..Nk` plus target `T`
- the answer is a final arithmetic expression inside `<answer>`
- scoring checks safe expression use, number use, and target closeness

### Evidence Verdict Debate

A synthetic evidence-reconciliation task that checks whether the debate stays grounded in the cited evidence snippets.

- evidence snippets are numbered `E1..En`
- the answer is `TRUE`, `FALSE`, or `INSUFFICIENT` inside `<answer>`
- decisive evidence IDs go in `<support>`
- scoring checks verdict correctness and support quality

## Current Best Preview Model

- GitHub repo: [scasella/society-of-thought-bench](https://github.com/scasella/society-of-thought-bench)
- Hugging Face adapter repo: [scasella91/society-of-thought-qwen3-30b-paper-faithful-adapter](https://huggingface.co/scasella91/society-of-thought-qwen3-30b-paper-faithful-adapter)
- Hugging Face demo: [scasella91/society-of-thought-bench-demo](https://huggingface.co/spaces/scasella91/society-of-thought-bench-demo)
- base model: `Qwen/Qwen3-30B-A3B`
- best published sampler checkpoint source: `tinker://80d6e740-bf17-52ca-a94c-422c67897617:train:0/sampler_weights/final`

The published adapter is the checkpoint used for the released comparison and the live demo.

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

To try it in a browser:

- [Open the live demo Space](https://huggingface.co/spaces/scasella91/society-of-thought-bench-demo)

To see the raw paper-style thinking trace:

```bash
uv run python scripts/try_tinker_checkpoint.py --family countdown --difficulty medium --show-raw-response
```

That helper defaults to the best current preview checkpoint.

To chat with the checkpoint directly from the terminal:

```bash
uv run python scripts/chat_tinker_checkpoint.py --show-raw-response
```

That path is useful for exploration. The benchmark examples and audit pack are still the cleanest way to inspect the published behavior.

## How To Inspect This Safely

Start with the benchmark-style examples, because they are the most faithful setting for this checkpoint.

Then inspect the 12-example audit pack:

- [artifact_hardening/TRACE_AUDIT.md](./release_preview/artifact_hardening/TRACE_AUDIT.md)

Use open-ended chat after that, as an exploratory interface rather than as the main evidence for the claim.

## Key Docs

- [Findings](./release_preview/FINDINGS.md)
- [Results](./release_preview/RESULTS.md)
- [Raw Sample](./release_preview/RAW_SAMPLE.md)
- [Limitations](./release_preview/LIMITATIONS.md)
- [Hardening Packet](./release_preview/artifact_hardening/README.md)
- [HF Model Card Source](./release_preview/HF_MODEL_CARD.md)
- [GitHub Announcement Copy](./release_preview/GITHUB_ANNOUNCEMENT.md)
- [Hugging Face Announcement Copy](./release_preview/HF_ANNOUNCEMENT.md)
- [Release Notes](./release_preview/v0.1.0-preview.md)

## Repository Contents

- `society_of_thought_bench/` benchmark implementation
- `scripts/` data, training, evaluation, and live-try helpers
- `tests/` local verification
- `release_preview/` publishable preview bundle

## Limitations

Read this preview narrowly.

- It is an early positive result, not a final paper result.
- The strongest evidence is the released debate-versus-monologue comparison on this benchmark.
- Medium difficulty is the strongest setting today.
- Harder traces still need richer branching and stronger reconciliation.
- The benchmark targets exposed reasoning traces and currently fits Qwen-style reasoning models best.

## License

This preview package is released under the MIT License.
