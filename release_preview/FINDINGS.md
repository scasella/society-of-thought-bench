# Findings

This release supports one narrow conclusion:

> Within `society-of-thought-bench`, the same trained model performs better when its visible reasoning trace is structured as a multi-persona debate than when it reasons in a single voice.

## What We Tested

We compared the same trained model in two modes:

- `debate`: a paper-style cast, conversation, and group solution inside `<think>`
- `monologue`: a single reasoning voice inside `<think>`

The headline comparison used the released checkpoint on medium-difficulty tasks across both task families.

## Headline Result

From the released comparison:

- debate average score: `0.732`
- monologue average score: `0.197`
- score gap: `+0.535`
- task score gap: `+0.200`
- disagreement-quality gap: `+0.560`

Within this benchmark, the model does more than produce the debate format. It also performs better on the benchmark's grounded measures when it uses that format.

## What Looks Strongest Today

- Medium difficulty is the strongest setting.
- Persona diversity and interaction are already strong.
- The debate mode clearly outperforms monologue mode on the released comparison.
- Many traces are clean enough to inspect directly by hand.

## What This Does Not Show

- It does not establish a general win on outside benchmarks.
- It does not show that the model is fully reliable on richer disagreement structure.
- Harder cases still need stronger branching and reconciliation.

## Canonical Evidence

- `release_preview/results/debate_vs_monologue_medium_preview.summary.json`
- `release_preview/results/debate_medium_preview.summary.json`
- `release_preview/results/debate_hard_preview.summary.json`
