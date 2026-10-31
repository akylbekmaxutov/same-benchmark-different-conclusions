# Perturbation validation

IDs unique across all three files: **True** (18127 total).

| benchmark | n | missing rel/irr/para | opt mutated | bad perm | P1 opts mutated | with E | sampled |
|---|---:|---|---:|---:|---:|---:|---:|
| kazmmlu | 9726 | 0/0/0 | 0 | 0 | 0 | 2587 | 2000 |
| kazculture | 1324 | 0/0/0 | 0 | 0 | 0 | 0 | 1324 |
| kazbench_kk | 7077 | 0/0/0 | 0 | 0 | 0 | 0 | 2000 |

## Gold-answer leakage (normalized substring of the gold option)

`distractor_hit_rate` is the base rate: a gold rate at or below it is topical overlap, not leakage.

| benchmark | field | n | gold hit rate | distractor base rate |
|---|---|---:|---:|---:|
| kazmmlu | relevant_noise | 9726 | 0.0008 | 0.0016 |
| kazmmlu | irrelevant_noise | 9726 | 0.0000 | 0.0002 |
| kazmmlu | question_paraphrase | 9726 | 0.0019 | 0.0033 |
| kazculture | relevant_noise | 1324 | 0.0015 | 0.0003 |
| kazculture | irrelevant_noise | 1324 | 0.0000 | 0.0000 |
| kazculture | question_paraphrase | 1324 | 0.0000 | 0.0003 |
| kazbench_kk | relevant_noise | 7077 | 0.0008 | 0.0006 |
| kazbench_kk | irrelevant_noise | 7077 | 0.0000 | 0.0000 |
| kazbench_kk | question_paraphrase | 7077 | 0.0013 | 0.0002 |

## Noise length comparability (characters)

| benchmark | field | n | mean | median | p05 | p95 |
|---|---|---:|---:|---:|---:|---:|
| kazmmlu | relevant_noise | 9726 | 292.5 | 291.0 | 232 | 358 |
| kazmmlu | irrelevant_noise | 9726 | 249.8 | 250.0 | 196 | 306 |
| kazmmlu | question_paraphrase | 9726 | 96.0 | 83.0 | 44 | 180 |
| kazculture | relevant_noise | 1324 | 270.6 | 270.0 | 216 | 329 |
| kazculture | irrelevant_noise | 1324 | 245.6 | 246.0 | 192 | 298 |
| kazculture | question_paraphrase | 1324 | 72.8 | 71.0 | 34 | 117 |
| kazbench_kk | relevant_noise | 7077 | 294.5 | 292 | 233 | 362 |
| kazbench_kk | irrelevant_noise | 7077 | 256.7 | 256 | 203 | 312 |
| kazbench_kk | question_paraphrase | 7077 | 90.9 | 89 | 52 | 139 |

## Notes

- **kazculture**: only 1324 items available against a target of 2000; the small-benchmark rule applied — all items used, no upsampling, the other benchmarks were not shrunk to match.
- **kazbench_kk**: repository names the split `train`; the accompanying paper describes it as the test set. Treated as the evaluation set (meta.split_source='train', meta.split_role='test_per_paper'). No contamination analysis is performed on this basis.
