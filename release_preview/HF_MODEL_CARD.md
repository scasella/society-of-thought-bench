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

This is a LoRA adapter for `Qwen/Qwen3-30B-A3B` trained for one benchmarked behavior: produce a visible multi-persona debate inside `<think>...</think>` while keeping the final answer separate.

On the released medium comparison for `society-of-thought-bench`, the same trained model scores `0.732` in debate mode and `0.197` in monologue mode across `40` examples, a gap of `+0.535`. Disagreement quality improves by `+0.560`.

This result is evidence within this benchmark. It is not a broad claim about reasoning outside this setup.

Try it and inspect the evidence directly:

- Live demo: [scasella91/society-of-thought-bench-demo](https://huggingface.co/spaces/scasella91/society-of-thought-bench-demo)
- Benchmark repo: [scasella/society-of-thought-bench](https://github.com/scasella/society-of-thought-bench)
- Released comparison: [debate_vs_monologue_medium_preview.summary.json](https://github.com/scasella/society-of-thought-bench/blob/main/release_preview/results/debate_vs_monologue_medium_preview.summary.json)
- Raw sample: [RAW_SAMPLE.md](https://github.com/scasella/society-of-thought-bench/blob/main/release_preview/RAW_SAMPLE.md)
- Additional validation and audit materials: [artifact_hardening/README.md](https://github.com/scasella/society-of-thought-bench/blob/main/release_preview/artifact_hardening/README.md)

## Intended Use

This adapter is meant for controlled prompting where the model is asked to keep a visible multi-voice discussion in the reasoning trace and a separate final answer outside it.

It is best suited for:

- benchmark-style prompts from `society-of-thought-bench`
- direct inspection of visible reasoning traces
- research on debate-versus-monologue behavior in this format

## Target Output Format

```text
<think>
<cast_of_characters>...</cast_of_characters>
<conversation>...</conversation>
<group_solution>...</group_solution>
</think>
<answer>...</answer>
<support>...</support>   # evidence tasks only
```

## Headline Result

On the released medium comparison, debate mode outperforms monologue mode for the same trained model.

- debate average score: `0.732`
- monologue average score: `0.197`
- score gap: `+0.535`
- disagreement-quality gap: `+0.560`

## How To Try It

To try the released behavior in a browser:

- [Open the live demo Space](https://huggingface.co/spaces/scasella91/society-of-thought-bench-demo)

To inspect the model from the benchmark repo:

```bash
uv run python scripts/try_tinker_checkpoint.py --family countdown --difficulty medium --show-raw-response
uv run python scripts/chat_tinker_checkpoint.py --show-raw-response
```

Use open-ended chat for exploration. Use the benchmark examples and audit materials for the clearest view of the released behavior.

## Additional Validation

We also released follow-up validation materials for the same public checkpoint. They include stricter confirmation runs, a 12-example audit set, and release-history notes for later repair attempts.

- medium reward delta: `0.364`
- easy joint-valid rate: `0.675`
- easy answer-valid rate: `0.800`
- hard disagreement quality: `0.241`

See: [artifact_hardening/README.md](https://github.com/scasella/society-of-thought-bench/blob/main/release_preview/artifact_hardening/README.md)

## Limitations

- This is an experimental research preview, not a final benchmark release.
- The strongest evidence comes from the released debate-versus-monologue comparison within this benchmark.
- Medium difficulty is currently the most reliable setting.
- Harder branching and reconciliation still need work.
- The benchmark assumes visible reasoning traces and currently fits Qwen-style reasoning models best.

## Backing Evidence

- debate vs monologue: [summary](https://github.com/scasella/society-of-thought-bench/blob/main/release_preview/results/debate_vs_monologue_medium_preview.summary.json)
- medium debate: [summary](https://github.com/scasella/society-of-thought-bench/blob/main/release_preview/results/debate_medium_preview.summary.json)
- hard debate: [summary](https://github.com/scasella/society-of-thought-bench/blob/main/release_preview/results/debate_hard_preview.summary.json)
- additional validation and audit materials: [summary and audit](https://github.com/scasella/society-of-thought-bench/blob/main/release_preview/artifact_hardening/README.md)
