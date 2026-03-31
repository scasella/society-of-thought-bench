---
base_model: Qwen/Qwen3-30B-A3B
library_name: peft
license: mit
tags:
- lora
- qwen3
- reasoning
- verifiers
- research
---

# society-of-thought-qwen3-30b-paper-faithful-adapter

Research preview: on `society-of-thought-bench`, the same trained model scores much higher when its visible reasoning trace is a multi-persona debate than when it is a monologue.

In the released medium comparison, the same trained model, on the same benchmark slice, scores `0.732` in debate mode and `0.197` in monologue mode across 40 examples, for a score gap of `+0.535`. Disagreement quality improves by `+0.56`.

This is benchmark-local evidence from an early research preview, not a broad claim that the adapter improves reasoning outside this setup.

Try it and inspect the evidence directly:

- Live demo: [scasella91/society-of-thought-bench-demo](https://huggingface.co/spaces/scasella91/society-of-thought-bench-demo)
- Benchmark repo: [scasella/society-of-thought-bench](https://github.com/scasella/society-of-thought-bench)
- Released comparison: [debate_vs_monologue_medium_preview.summary.json](https://github.com/scasella/society-of-thought-bench/blob/main/release_preview/results/debate_vs_monologue_medium_preview.summary.json)
- Raw sample: [RAW_SAMPLE.md](https://github.com/scasella/society-of-thought-bench/blob/main/release_preview/RAW_SAMPLE.md)

This is an adapter for `Qwen/Qwen3-30B-A3B` trained for one specific behavior:

> carry out a paper-style, visible multi-persona debate in the exposed thinking trace, then keep the final answer separate.

## What This Adapter Is For

This adapter was trained against `society-of-thought-bench`, an experimental benchmark for visible multi-persona reasoning in exposed thinking traces.

Benchmark repo: [scasella/society-of-thought-bench](https://github.com/scasella/society-of-thought-bench)
Live demo: [scasella91/society-of-thought-bench-demo](https://huggingface.co/spaces/scasella91/society-of-thought-bench-demo)

The target format is:

```text
<think>
<cast_of_characters>...</cast_of_characters>
<conversation>...</conversation>
<group_solution>...</group_solution>
</think>
<answer>...</answer>
<support>...</support>   # evidence tasks only
```

## Main Result

On the released medium comparison, the same trained model does better in debate mode than in monologue mode.

- debate average score: `0.732`
- monologue average score: `0.197`
- score gap: `+0.535`
- disagreement-quality gap: `+0.560`

This is benchmark-local evidence from an early research preview.

## How To Use It

Load this adapter on top of:

- `Qwen/Qwen3-30B-A3B`

Then prompt for the paper-style format expected by the benchmark.

To try the published checkpoint in a browser without local setup:

- [Open the live demo Space](https://huggingface.co/spaces/scasella91/society-of-thought-bench-demo)

## Raw Sample

See the linked raw sample in the benchmark repo:

- [Raw sample](https://github.com/scasella/society-of-thought-bench/blob/main/release_preview/RAW_SAMPLE.md)

## Limitations

- This is not a general-purpose reasoning upgrade claim.
- This is not a final leaderboard model.
- Medium difficulty is the strongest setting today.
- Harder branching and reconciliation still need work.

## Backing Evidence

- debate vs monologue: [summary](https://github.com/scasella/society-of-thought-bench/blob/main/release_preview/results/debate_vs_monologue_medium_preview.summary.json)
- medium debate: [summary](https://github.com/scasella/society-of-thought-bench/blob/main/release_preview/results/debate_medium_preview.summary.json)
- hard debate: [summary](https://github.com/scasella/society-of-thought-bench/blob/main/release_preview/results/debate_hard_preview.summary.json)
