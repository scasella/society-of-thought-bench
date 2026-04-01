# Results

The table below reports the canonical debate-versus-monologue comparison. `Average score` is the benchmark reward, `task score` reflects task correctness and grounding, and the debate-specific metrics summarize the quality of the visible discussion.

## Main Comparison: Debate vs Monologue

| Metric | Debate | Monologue | Difference |
| --- | ---: | ---: | ---: |
| Average score | 0.732 | 0.197 | +0.535 |
| Task score | 0.716 | 0.516 | +0.200 |
| Valid outputs | 0.925 | 0.625 | +0.300 |
| Disagreement quality | 0.560 | 0.000 | +0.560 |
| Persona diversity | 0.925 | 0.000 | +0.925 |
| Interaction score | 0.775 | 0.000 | +0.775 |

Source:

- `release_preview/results/debate_vs_monologue_medium_preview.summary.json`

## Supporting Result: Best Medium Debate Run

| Metric | Value |
| --- | ---: |
| Average score | 0.750 |
| Valid outputs | 0.950 |
| Task score | 0.719 |
| Persona diversity | 0.950 |
| Interaction score | 0.804 |
| Debate relevance | 0.748 |
| Disagreement quality | 0.548 |

Source:

- `release_preview/results/debate_medium_preview.summary.json`

## Supporting Result: Hard Run

| Metric | Value |
| --- | ---: |
| Average score | 0.663 |
| Valid outputs | 0.900 |
| Task score | 0.724 |
| Persona diversity | 0.895 |
| Interaction score | 0.745 |
| Debate relevance | 0.675 |
| Disagreement quality | 0.398 |
| Alternative path count | 0.125 |

Source:

- `release_preview/results/debate_hard_preview.summary.json`

The released comparison shows a clear within-benchmark advantage for debate mode, while the supporting runs show where the behavior is already strong and where harder cases remain less mature.
