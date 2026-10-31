# Stage 1 — dataset preparation + Gemini perturbation generation

Covers **Phases 1–8** and **Phase 32 (generation materials)** of `Description.md`.
Run this once, on any machine with internet access. **No GPU needed.**

Output = the released artifact of the paper:

```
data/perturbations/kazmmlu_perturbations.jsonl
data/perturbations/kazculture_perturbations.jsonl
data/perturbations/kazbench_kk_perturbations.jsonl
```

Another researcher can rerun every evaluation experiment from those three files
without ever calling Gemini.

---

## 0. Setup

```bash
cd same-benchmark-different-conclusions
pip install -r requirements.txt
```

Put your key in `.env`:

```
GEMINI_KEY=AIza...
```

Optionally pin the model id — the code defaults to `gemini-3.5-flash-lite`:

```bash
export GEMINI_MODEL=gemini-3.5-flash-lite
```

---

## 1. One command

```bash
./gemini_experiments/run_gemini_pipeline.sh
```

That runs, in order:

| step | phase | what it does |
|---|---|---|
| `prepare`  | 1–2 | downloads the 3 benchmarks, normalizes them to the shared MCQ schema |
| `smoke`    | –   | 6 live Gemini calls to verify prompt/model/schema/parser before paying for a batch |
| `build`    | 3   | writes Batch API request files (one request per question) |
| `submit`   | 3   | uploads them and creates the batch jobs |
| `fetch`    | 3   | polls until every job is terminal, downloads + parses results |
| `merge`    | 7   | joins generations onto the corpus → `*_perturbations.jsonl` |
| `validate` | 8   | acceptance checks, rejection list, manual-inspection sheets |

Any step alone:

```bash
./gemini_experiments/run_gemini_pipeline.sh --step prepare
./gemini_experiments/run_gemini_pipeline.sh --step fetch
./gemini_experiments/run_gemini_pipeline.sh --step prepare --datasets kazculture
```

Useful knobs:

```bash
MAX_PER_DATASET=2000 ./gemini_experiments/run_gemini_pipeline.sh   # stratified sub-sample
SKIP_SMOKE=1         ./gemini_experiments/run_gemini_pipeline.sh
```

### Cheap dry run first

```bash
./gemini_experiments/run_gemini_pipeline.sh --step prepare
python -m gemini_experiments.batch_build --limit 20 --round 99
python -m gemini_experiments.batch_submit --round 99
python -m gemini_experiments.batch_fetch  --round 99 --wait
```

---

## 2. Batch API, not per-request calls

Everything goes through the **Gemini Batch API** (~50 % of interactive pricing),
one request per source question, all three perturbation fields generated together
in a single strict-JSON response (Phases 3–6):

```json
{"relevant_noise": "...", "irrelevant_noise": "...", "question_paraphrase": "..."}
```

**One batch job per dataset** by default: the entire Kazakh part of KazMMLU goes
up as a single job, all of KazBench-KK as a single job, all of KazCulture as a
single job — three jobs total, each an input file with one JSON line per
question.

```bash
BATCH_CHUNK_SIZE=5000 ./gemini_experiments/run_gemini_pipeline.sh   # split instead
```

Splitting localizes failures (a rejected job costs you 5000 questions, not the
whole dataset) at the cost of more jobs to track. Either way `jobs.json` records
every job name, so `batch_fetch` can be interrupted and resumed, and downloaded
result files are reused from cache.

---

## 3. Regeneration loop (Phase 8)

`validate` writes `reports/invalid_ids.json`. Feed it straight back:

```bash
ROUND=2 ./gemini_experiments/run_gemini_pipeline.sh --step regenerate
```

`merge_release` prefers the highest round that succeeded, so the release file
just improves. Repeat with `ROUND=3` if needed.

Validation records exactly what the paper's appendix needs: number generated,
number rejected, number regenerated, final acceptance rate, and a breakdown of
rejection reasons.

**Rejections (item is regenerated)** — empty/invalid generation, non-Kazakh text,
answer-label leakage (option letters, `дұрыс жауап`, verbatim option text that
was not already in the question), options or gold answer altered, noise too short,
paraphrase identical to the original, numbers dropped, negation lost.

**Warnings (item kept, flagged for human review)** — unusual lengths, relevant vs
irrelevant noise length imbalance, possible entity drop.

Then do the manual part of Phase 8 with:

```
gemini_experiments/reports/manual_inspection/kazmmlu.md      # random sample + one per category
gemini_experiments/reports/manual_inspection/kazculture.md
gemini_experiments/reports/manual_inspection/kazbench_kk.md
```

---

## 4. Source-specific decisions

| benchmark | source | split | notes |
|---|---|---|---|
| KazMMLU | `MBZUAI/KazMMLU` | `test` | subsets auto-detected as Kazakh; the selected list is printed and stored in the report. Override with `--kazmmlu-configs`. `E` preserved where present. `category` = subset name. |
| KazCulture | `issai/KazCulture` | `test` | `category` = `category_english`; `category_kazakh` kept in `meta`. `E = null`. |
| KazBench-KK | `kz-transformers/kk-socio-cultural-bench-mc` | `train` | the only split; used as the evaluation set and labelled as such in `meta.source_split_note`. `subcategory` kept in `meta`. `E = null`. |

Everything not promoted into the unified schema is preserved under `meta`.
Items are dropped only if the question is empty, fewer than two options survive,
or the gold answer cannot be mapped to a present option letter.

---

## 5. Output files

```
data/normalized/{dataset}.jsonl              Phase 1-2 unified corpus
data/perturbations/{dataset}_perturbations.jsonl   Phase 7 RELEASE

gemini_experiments/work/batch_input/r1/      request JSONL + build_manifest + jobs.json
gemini_experiments/work/batch_output/r1/     raw downloaded batch results
gemini_experiments/work/generations/r1/      parsed generations per dataset

gemini_experiments/reports/
    normalized_corpus_report.json            sizes, categories, gold-label balance
    fetch_report_r*.json                     per-round success/failure counts
    merge_report.json                        provenance (model, prompt version, params, date)
    validation_report.json                   Phase 8 acceptance statistics
    invalid_ids.json                         feed into --step regenerate
    manual_inspection/*.md                   human review sheets
```

Release record schema:

```json
{
  "id": "kazmmlu-000123", "dataset": "kazmmlu", "category": "...",
  "question": "...", "A": "...", "B": "...", "C": "...", "D": "...", "E": null,
  "answer": "B",
  "relevant_noise": "...", "irrelevant_noise": "...", "question_paraphrase": "...",
  "generation_ok": true, "generation_round": 1, "generation_error": null,
  "meta": { "...original metadata..." }
}
```

---

## 6. Things worth knowing before you spend money

- `GEMINI_MODEL` defaults to `gemini-3.5-flash-lite`. If that id is not live on
  your key, the smoke test fails immediately and costs almost nothing — set the
  correct id and rerun.
- The generation prompt (`prompts.py`) and `PROMPT_VERSION` (`config.py`) are
  release artifacts. Edit the prompt → bump the version.
- `temperature=0.7` is deliberate here: the noise and paraphrase fields need
  lexical variety. The *evaluation* stage is greedy and fully frozen.
- The full KazMMLU + KazBench-KK sets are large. Do a `MAX_PER_DATASET` run
  first if you want a cost estimate before committing to the full corpus.
