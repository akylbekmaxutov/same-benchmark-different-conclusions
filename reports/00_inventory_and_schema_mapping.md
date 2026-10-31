# §0 Inventory and proposed `Item` schema mapping

Measured from `data/` on 2026-08-23. Nothing below is assumed; every number is
produced by `scripts/00_inventory.py` (machine-readable copy in
`reports/inventory.json`).

## 0. Layout deviation from GUIDELINE §2 — read this first

The guide assumes the Gemini corpus lives in `$ROOT/stability/` and that the
harness writes to `$ROOT/data/`. In this repository it is the other way round:
`GUIDELINE.md` sits in `papers/stability/`, and the corpus is already at
`papers/stability/data/{normalized,perturbations}/`.

Writing the harness's own `normalized/` there would overwrite read-only Gemini
output with identical filenames. The read-only guarantee outranks the directory
name, so:

| role | GUIDELINE name | here | setting |
|---|---|---|---|
| read-only Gemini corpus | `stability/` | `data/` | `raw_data_dir` |
| harness output (normalized, sample_index, conditions) | `data/` | `work/` | `work_dir` |
| run output | `results/` | `results/` | `results_dir` |

Both are one-line changes in `configs/defaults.yaml`. Nothing under `data/` is
written by any code in `src/` or `scripts/`.

## 1. Filename → benchmark mapping

Each benchmark ships two files. `*_full.jsonl` is every row including
generation and validation failures; `*_perturbations.jsonl` is the released
artifact and is **exactly** the `validation_passed == true` subset (verified,
not assumed). The harness reads the released variant by default
(`perturbation_variant: released`); `--variant full` reads the other.

| file | benchmark | rows | released rows |
|---|---|---:|---:|
| `data/perturbations/kazmmlu_perturbations{,_full}.jsonl` | KazMMLU | 9,870 | 9,726 |
| `data/perturbations/kazculture_perturbations{,_full}.jsonl` | KazCulture | 1,334 | 1,324 |
| `data/perturbations/kazbench_kk_perturbations{,_full}.jsonl` | KazBench-KK | 7,111 | 7,077 |

`data/normalized/*.jsonl` holds the same items without the generated fields;
core fields are byte-identical to the perturbation files (0 rows differ), so the
harness reads only the perturbation files.

## 2. Key names present in each row

Good news the guide did not assume: **all three benchmarks already share one row
layout.** No per-benchmark key renaming is needed.

```
id, dataset, category, question, A, B, C, D, E, answer,
relevant_noise, irrelevant_noise, question_paraphrase,
generation_ok, generation_round, generation_error,
meta{...}, validation_passed [, validation_errors, validation_warnings]
```

Uppercase `A`–`E`, uppercase `answer` letter, `E: null` where there is no fifth
option. The lowercase `a/b/c/d` and `answer_label` the guide warns about for
KazCulture were already normalized upstream; `normalize.py` re-asserts rather
than trusts.

## 3. Generated fields

Named exactly `relevant_noise`, `irrelevant_noise`, `question_paraphrase`.
In the **released** files: 0 missing, 0 empty, in all three benchmarks.
In the full files: 14 (KazMMLU), 0 (KazCulture), 2 (KazBench-KK) rows have all
three empty — these are the `generation_ok: false` rows.

Validation failures excluded from the released files, by cause:

| benchmark | paraphrase | relevant | irrelevant | generation error | total |
|---|---:|---:|---:|---:|---:|
| KazMMLU | 144 | 19 | 15 | 14 | 144 |
| KazCulture | 10 | 0 | 0 | 0 | 10 |
| KazBench-KK | 33 | 4 | 3 | 2 | 34 |

## 4. Stable IDs

`id` is `<dataset>-<6 digits>`, unique **within** every file and unique
**across** all three (18,127 released ids, zero collisions).

`normalize.py` strips the redundant dataset prefix and forms
`uid = f"{dataset}::{orig_id}"` — `kazmmlu-000123` → `kazmmlu::000123`. The raw
value is preserved verbatim as `meta.source_id`.

## 5. Counts

| benchmark | released | with Option E | four-option (O1-C eligible) | answer distribution |
|---|---:|---:|---:|---|
| KazMMLU | 9,726 | 2,587 | 7,139 | A 2405 · B 2366 · C 2265 · D 2347 · E 487 |
| KazCulture | 1,324 | 0 | 1,324 | A/B/C/D ≈ 333 each |
| KazBench-KK | 7,077 | 0 | 7,077 | A 729 · B 3331 · C 2477 · D 574 |

Zero rows with a gold letter outside the option set; zero rows with empty option
text; zero duplicate ids. **KazBench-KK's gold letters are markedly skewed
toward B** (47%) — worth a sentence in the paper, and a reason the O1/O1-C
position analyses matter more on that benchmark than the others.

## 6. Category fields

- **KazMMLU** — `category` = subset name, 12 subsets, all containing `in kaz`
  (so the Kazakh-only filter is already applied upstream; `normalize.py` still
  refuses a non-Kazakh subset). No subcategory. Sizes 7 – 1,023; *Reading
  Literacy* has only 7 items.
- **KazCulture** — `category` = canonical cultural category, 9 of them, sizes
  4 – 403. No subcategory.
- **KazBench-KK** — the only two-level hierarchy: 18 categories × 77 populated
  `meta.subcategory` values, every `category × subcategory` cell ≥ 15 raw.

## 7. Pre-existing indices, seeds, prompt files

None. No sample index, no seed file, no evaluation prompt file exists — only the
Gemini *generation* prompt under `gemini_experiments/`. All indices in
`work/sample_index/` are new and are released with the paper.

---

# Proposed raw → `Item` mapping (implemented in `src/stability/normalize.py`)

| `Item` field | KazMMLU | KazCulture | KazBench-KK |
|---|---|---|---|
| `uid` | `kazmmlu::` + id digits | `kazculture::` + digits | `kazbench_kk::` + digits |
| `orig_id` | id with the `<dataset>-` prefix stripped | ← | ← |
| `dataset` | `row["dataset"]` | ← | ← |
| `category` | `row["category"]` (subset name) | `row["category"]` | `row["category"]` (level 1) |
| `subcategory` | `None` | `None` | `meta["subcategory"]` (level 2) |
| `question` | `row["question"]` | ← | ← |
| `options` | `{A..E}` non-null, verbatim | `{A..D}` | `{A..D}` |
| `answer` | `row["answer"]`, upper, homoglyph-mapped | ← | ← |
| `relevant_noise` / `irrelevant_noise` / `question_paraphrase` | same keys; blank → `None` | ← | ← |
| `meta` | `row["meta"]` + `gen_*` + `source_id` | ← | ← + `split_source="train"`, `split_role="test_per_paper"` |

Fails loudly on: gold letter not among the present options; duplicate `uid`
within or across files; empty option text; an unexpected Option E on a
four-option benchmark; a KazMMLU subset without `kaz` in its name.

## Consequences that change the plan

1. **KazCulture cannot reach 2,000 items** — 1,324 exist. The small-benchmark
   rule fires: all 1,324 are used, no upsampling, and the other two are not
   shrunk to match. `sampling_applied: false` is recorded in its index file.
2. **Realised total is 5,324 items, not ~6,000** (2,000 + 1,324 + 2,000).
   Generation volume for the grid should be recomputed against 5,324.
3. **Stratification keys actually chosen** (recorded in each index file):
   KazMMLU `category`; KazCulture n/a (no sampling); KazBench-KK **`category`,
   not `category × subcategory`** — at 2,000/7,077 the 77 fine cells would leave
   18 of them below 15 items, whereas the 18 coarse cells all land ≥ 16. Both
   levels remain as columns on every aggregate.
4. **KazMMLU contributes only its 7,139 four-option items to O1-C.** The
   500-item pool drawn proportionally across the three: KazBench-KK 210,
   KazMMLU 151, KazCulture 139.
5. **KazBench-KK's split** is labelled `train` upstream and used as the
   evaluation set per the paper; recorded on every row and emitted as a one-line
   note in the validation report. No contamination analysis.
