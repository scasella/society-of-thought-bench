# Findings

This release supports one narrow conclusion:

> Society of Thought, a model that reasons through a visible multi-persona debate, performs better on the released benchmark comparison than the same trained model run as a monologue.

## What We Tested

We compared the same trained model in two modes:

- `debate`: a visible cast, conversation, and group solution inside `<think>`
- `monologue`: a single reasoning voice inside `<think>`

The headline comparison used the released checkpoint on medium-difficulty tasks across both task families.

## Headline Result

From the released comparison:

- debate average score: `0.732`
- monologue average score: `0.197`
- score gap: `+0.535`
- task score gap: `+0.200`
- disagreement-quality gap: `+0.560`

Within the benchmark designed to test this behavior, the visible debate is associated with better outcomes, not just a different trace format.

## What This Release Shows

- the model can sustain a visible multi-persona reasoning process
- that behavior is inspectable by hand
- on the benchmark built to test it, the debate mode clearly outperforms monologue mode

## What Remains Unproven

- broader transfer outside this benchmark
- stronger reliability on richer disagreement structure
- mature performance on the hardest branching and reconciliation cases

## Canonical Evidence

- `release_preview/results/debate_vs_monologue_medium_preview.summary.json`
- `release_preview/results/debate_medium_preview.summary.json`
- `release_preview/results/debate_hard_preview.summary.json`
