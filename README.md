# society-of-thought-bench

`society-of-thought-bench` is an experimental benchmark and reference release for one question: within this benchmark, does a model do better when its visible reasoning trace is structured as a multi-persona debate rather than a monologue?

On the released medium comparison, the same trained model scores `0.732` in debate mode and `0.197` in monologue mode across `40` examples, a gap of `+0.535`. Task score improves by `+0.200`, and disagreement quality improves by `+0.560`.

This is early evidence within this benchmark. It is not a claim about general-purpose reasoning outside this setup.

## Start Here

- Live demo: [scasella91/society-of-thought-bench-demo](https://huggingface.co/spaces/scasella91/society-of-thought-bench-demo)
- Released comparison: [Debate vs Monologue Summary](./release_preview/results/debate_vs_monologue_medium_preview.summary.json)
- Raw sample trace: [RAW_SAMPLE.md](./release_preview/RAW_SAMPLE.md)
- Additional validation and audit materials: [artifact_hardening/README.md](./release_preview/artifact_hardening/README.md)

## What This Release Includes

- a benchmark package for visible multi-persona debate in the reasoning trace
- a released debate-versus-monologue comparison, plus supporting result files
- a public adapter and live demo for direct inspection
- training and evaluation helpers for supervised tuning, preference tuning, and RL

## Main Result

On the released medium comparison, debate mode beats monologue mode for the same trained model.

- debate average score: `0.732`
- monologue average score: `0.197`
- score gap: `+0.535`
- task score gain: `+0.200`
- disagreement-quality gain: `+0.560`

Primary evidence:

- [Debate vs Monologue Summary](./release_preview/results/debate_vs_monologue_medium_preview.summary.json)
- [Best Medium Debate Summary](./release_preview/results/debate_medium_preview.summary.json)
- [Hard Supporting Summary](./release_preview/results/debate_hard_preview.summary.json)

The strongest evidence today is the medium-difficulty comparison. Harder cases still leave room for richer branching and stronger reconciliation.

## Additional Validation

We also released a follow-up audit package for the public checkpoint used in the comparison. It adds stricter confirmation runs, a hand-inspectable audit set, and release-history notes for later repair attempts.

- medium reward delta: `0.364`
- easy joint-valid rate: `0.675`
- easy answer-valid rate: `0.800`
- hard disagreement quality: `0.241`

These materials are best read as additional validation for the released checkpoint:

- [Additional validation and audit materials](./release_preview/artifact_hardening/README.md)

## How To Inspect The Release

Start with the released comparison, then read the raw sample and the 12-example audit pack.

- [RAW_SAMPLE.md](./release_preview/RAW_SAMPLE.md)
- [TRACE_AUDIT.md](./release_preview/artifact_hardening/TRACE_AUDIT.md)

After that, use the live demo or local scripts for exploratory inspection. Benchmark-style prompts are still the most faithful setting for this checkpoint.

## Benchmark Overview

The benchmark is built around a simple separation: the reasoning trace stays inside `<think>...</think>`, and the visible answer stays outside it.

### Debate Mode

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

### Monologue Mode

```text
<think>single-voice reasoning</think>
<answer>...</answer>
<support>...</support>   # evidence tasks only
```

Within that format, the benchmark rewards:

- distinct personas with distinct roles
- challenge rather than shallow agreement
- grounded back-and-forth
- alternative paths on harder tasks
- reconciliation into one final answer

Final task accuracy still matters, but only as a grounding term in the default profile.

## Two Task Families

### Countdown Debate

A procedural arithmetic target task that checks whether the discussion stays grounded in the numbered inputs.

- inputs are numbered `N1..Nk` plus target `T`
- the answer is a final arithmetic expression inside `<answer>`
- scoring checks safe expression use, number use, and target closeness

### Evidence Verdict Debate

A synthetic evidence-reconciliation task that checks whether the discussion stays grounded in the cited evidence snippets.

- evidence snippets are numbered `E1..En`
- the answer is `TRUE`, `FALSE`, or `INSUFFICIENT` inside `<answer>`
- decisive evidence IDs go in `<support>`
- scoring checks verdict correctness and support quality

## Public Release

- GitHub repo: [scasella/society-of-thought-bench](https://github.com/scasella/society-of-thought-bench)
- Hugging Face adapter repo: [scasella91/society-of-thought-qwen3-30b-paper-faithful-adapter](https://huggingface.co/scasella91/society-of-thought-qwen3-30b-paper-faithful-adapter)
- Hugging Face demo: [scasella91/society-of-thought-bench-demo](https://huggingface.co/spaces/scasella91/society-of-thought-bench-demo)
- base model: `Qwen/Qwen3-30B-A3B`

The published adapter is the model used for the released comparison and the live demo.

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

## Try The Release

To try it in a browser:

- [Open the live demo Space](https://huggingface.co/spaces/scasella91/society-of-thought-bench-demo)

To see the raw paper-style reasoning trace:

```bash
uv run python scripts/try_tinker_checkpoint.py --family countdown --difficulty medium --show-raw-response
```

To chat with the checkpoint directly from the terminal:

```bash
uv run python scripts/chat_tinker_checkpoint.py --show-raw-response
```

Use the chat path for exploration. Use the benchmark examples and audit materials for the clearest view of the released behavior.

## Release Materials

- [Findings](./release_preview/FINDINGS.md)
- [Results](./release_preview/RESULTS.md)
- [Raw Sample](./release_preview/RAW_SAMPLE.md)
- [Limitations](./release_preview/LIMITATIONS.md)
- [Additional Validation and Audit Materials](./release_preview/artifact_hardening/README.md)
- [HF Model Card Source](./release_preview/HF_MODEL_CARD.md)
- [GitHub Announcement Copy](./release_preview/GITHUB_ANNOUNCEMENT.md)
- [Hugging Face Announcement Copy](./release_preview/HF_ANNOUNCEMENT.md)
- [Release Notes](./release_preview/v0.1.0-preview.md)

## Repository Contents

- `society_of_thought_bench/` benchmark implementation
- `scripts/` data, training, evaluation, and live-try helpers
- `tests/` local verification
- `release_preview/` public release materials

## Limitations

For a fuller statement, see [release_preview/LIMITATIONS.md](./release_preview/LIMITATIONS.md).

- This is an experimental research preview, not a final benchmark release.
- The strongest evidence comes from the released debate-versus-monologue comparison within this benchmark.
- Medium difficulty is currently the most reliable setting.
- Harder traces still need richer branching and stronger reconciliation.

## License

This preview package is released under the MIT License.
