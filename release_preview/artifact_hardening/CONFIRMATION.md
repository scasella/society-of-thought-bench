# Public Checkpoint Confirmation

- Checkpoint: `tinker://80d6e740-bf17-52ca-a94c-422c67897617:train:0/sampler_weights/final`
- Model: `Qwen/Qwen3-30B-A3B`

## Internal Confirmation

| Check | Result |
| --- | ---: |
| Medium reward delta | 0.364 |
| Medium task score delta | 0.130 |
| Medium joint-valid delta | 0.120 |
| Medium disagreement delta | 0.340 |
| Easy joint-valid rate | 0.675 |
| Easy answer-valid rate | 0.800 |
| Hard interaction score | 0.696 |
| Hard disagreement quality | 0.241 |
| Hard persona diversity | 0.855 |

## Release Acceptance

- `reward_delta`: `0.364` vs target `0.350` -> pass
- `joint_valid_delta`: `0.120` vs target `0.200` -> fail
- `disagreement_quality_delta`: `0.340` vs target `0.400` -> fail
- `easy_answer_format_valid_rate`: `0.800` vs target `0.950` -> fail
- `easy_joint_contract_valid_rate`: `0.675` vs target `0.850` -> fail
- `hard_interaction_score`: `0.696` vs target `0.650` -> pass
- `hard_disagreement_quality`: `0.241` vs target `0.450` -> fail

## Background External Characterization

- Macro native score delta: `0.050`
- Macro visible-answer-valid delta: `0.010`
- Macro reasoning-contract-valid delta: `-0.650`
