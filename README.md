# Society of Thought

*Visible Multi-Persona Reasoning Model + Benchmark Evidence*

Society of Thought is a public preview of a Qwen adapter that reasons through a visible multi-persona debate rather than a single inner voice.

On the released medium comparison, the same trained model scores `0.732` in debate mode and `0.197` in monologue mode across `40` examples, a gap of `+0.535`. Task score improves by `+0.200`, and disagreement quality improves by `+0.560`.

This is evidence on the benchmark designed to test that behavior. It is not yet a claim about broader transfer outside this setup.

## Try It

- Live demo: [scasella91/society-of-thought-bench-demo](https://huggingface.co/spaces/scasella91/society-of-thought-bench-demo)
- Adapter page: [scasella91/society-of-thought-qwen3-30b-paper-faithful-adapter](https://huggingface.co/scasella91/society-of-thought-qwen3-30b-paper-faithful-adapter)
- Evidence: [Debate vs Monologue Summary](./release_preview/results/debate_vs_monologue_medium_preview.summary.json)
- Audit materials: [artifact_hardening/README.md](./release_preview/artifact_hardening/README.md)

## What This Release Shows

- a released model can carry out a visible multi-persona debate inside its reasoning trace
- on the benchmark built to test that behavior, debate mode beats monologue mode for the same trained model
- the traces are inspectable through the demo, raw samples, and audit pack

## What Remains Unproven

- broader gains outside this benchmark
- stronger reliability on harder branching and reconciliation
- replication beyond the current Qwen-centered setup

## What Is Released

- the Society of Thought adapter for `Qwen/Qwen3-30B-A3B`
- a live demo for inspecting raw reasoning traces
- `society-of-thought-bench`, the benchmark and evidence layer for the release
- result files, audit materials, and helper scripts for inspection and evaluation

## How To Inspect The Release

Start with the released comparison, then read the model overview and the raw sample, then move to the audit pack.

- [MODEL_OVERVIEW.md](./release_preview/MODEL_OVERVIEW.md)
- [RAW_SAMPLE.md](./release_preview/RAW_SAMPLE.md)
- [TRACE_AUDIT.md](./release_preview/artifact_hardening/TRACE_AUDIT.md)

After that, use the live demo or local scripts for exploratory prompting. Benchmark-style prompts are still the most faithful setting for the released behavior.

## Why The Benchmark Matters

The benchmark is the evidence layer for this release. It is designed to test whether a visible multi-persona reasoning trace is more than a stylistic wrapper.

The benchmark enforces a simple separation:

- the reasoning trace stays inside `<think>...</think>`
- the visible answer stays outside it

It then measures whether the debate is grounded, interactive, and useful rather than merely present.

Primary evidence:

- [Debate vs Monologue Summary](./release_preview/results/debate_vs_monologue_medium_preview.summary.json)
- [Best Medium Debate Summary](./release_preview/results/debate_medium_preview.summary.json)
- [Hard Supporting Summary](./release_preview/results/debate_hard_preview.summary.json)

## Benchmark Overview

### Debate Mode

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

## Try The Model

To try it in a browser:

- [Open the live demo Space](https://huggingface.co/spaces/scasella91/society-of-thought-bench-demo)

To see the raw visible reasoning trace:

```bash
uv run python scripts/try_tinker_checkpoint.py --family countdown --difficulty medium --show-raw-response
```

To chat with the checkpoint directly from the terminal:

```bash
uv run python scripts/chat_tinker_checkpoint.py --show-raw-response
```

Use the chat path for exploration. Use the benchmark examples and audit materials for the clearest view of the released behavior.

## Evidence And Materials

- [Model Overview](./release_preview/MODEL_OVERVIEW.md)
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

For the full statement, see [release_preview/LIMITATIONS.md](./release_preview/LIMITATIONS.md).

- This is an experimental research preview, not a final benchmark release.
- The strongest evidence comes from the released debate-versus-monologue comparison within this benchmark.
- Medium difficulty is currently the most reliable setting.
- Harder traces still need richer branching and stronger reconciliation.

## License

This preview package is released under the MIT License.
