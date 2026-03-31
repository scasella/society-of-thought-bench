# society-of-thought-qwen3-30b-paper-faithful-adapter

This is an adapter for `Qwen/Qwen3-30B-A3B` trained for one specific behavior:

> carry out a paper-style, multi-persona internal debate inside the exposed thinking trace, then keep the final answer separate.

## What This Adapter Is For

This adapter was trained against `society-of-thought-bench`, an experimental benchmark for visible multi-persona reasoning in exposed thinking traces.

Benchmark repo: [scasella/society-of-thought-bench](https://github.com/scasella/society-of-thought-bench)

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

On the clearest current comparison inside the benchmark, the same trained model does better in debate mode than in monologue mode.

- debate average score: `0.7317460411311398`
- monologue average score: `0.19656009201926383`
- score gap: `+0.535185949111876`
- disagreement-quality gap: `+0.5599999999999999`

This is a real early positive result, but it is still benchmark-local.

## How To Use It

Load this adapter on top of:

- `Qwen/Qwen3-30B-A3B`

Then prompt for the paper-style format expected by the benchmark.

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
