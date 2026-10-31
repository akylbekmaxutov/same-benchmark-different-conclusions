# Same Benchmark, Different Conclusions? Auditing the Stability of Kazakh Large Language Model Evaluation

Code, perturbation data, and analysis outputs for the paper.

## Overview

The study has two stages:

1. **Perturbation generation** (`gemini_experiments/`): normalizes KazMMLU,
   KazCulture and KazBench-KK into one multiple-choice schema and generates
   irrelevant context, relevant distractor context and question paraphrases
   with the Gemini Batch API. The output is released in `data/perturbations/`,
   so you do not need to rerun this stage (or have a Gemini key) to reproduce
   the evaluation.
2. **Evaluation and analysis** (`src/stability/`, `scripts/`): builds every
   condition from the released data, queries each model through an
   OpenAI-compatible endpoint served by vLLM, and computes flip rates,
   self-flip floors, accuracy deltas, ranking stability and the figures.

### Conditions

| code | group | description |
|---|---|---|
| `B0` | baseline | original item, English instruction; run 3 times to measure the self-flip floor |
| `N1` / `N2` | validity-preserving | irrelevant context before / after the question |
| `I1` | validity-preserving | instruction in Kazakh instead of English |
| `O1` | validity-preserving | per-item random permutation of answer options |
| `P1` | validity-preserving | paraphrased question |
| `N3` / `N4` | distractor | generated "relevant" context before / after the question |
| `O1C` | diagnostic | cyclic option rotations (position bias), on 6 configurations |
| `NQR` / `NQI` | diagnostic | context and options with the question removed (leakage probe) |

The main grid is `B0×3 + N1 N2 N3 N4 I1 O1 P1` over a stratified sample of
5,324 items (KazMMLU 2,000, KazCulture 1,324, KazBench-KK 2,000) for each of
the 18 configurations. The diagnostics run on the subsets listed in
`configs/defaults.yaml`.

## Repository structure

```
.
├── pyproject.toml              package definition for src/stability (pip install -e .)
├── requirements.txt            dependencies for both stages
├── serve_model.sh              sample vLLM serve script (configured via environment variables)
│
├── configs/
│   ├── defaults.yaml           paths, sampling seeds, pool sizes, diagnostic config lists, runtime defaults
│   ├── prompts.yaml            English and Kazakh instruction templates
│   ├── model_groups.yaml       served models -> their configurations (e.g. think / nothink), vLLM reasoning parser
│   └── models/*.yaml           one file per evaluated configuration (18): sampling params, chat-template kwargs, metadata
│
├── data/
│   ├── normalized/             the three benchmarks in a unified MCQ schema
│   └── perturbations/          RELEASED perturbation data
│       ├── <ds>_perturbations.jsonl        items that passed validation (used in the paper)
│       └── <ds>_perturbations_full.jsonl   every generated item, including validation failures
│
├── gemini_experiments/         Stage 1: corpus normalization + Gemini batch generation
│   ├── README.md               detailed Stage 1 documentation
│   ├── run_gemini_pipeline.sh  one-command driver for the steps below
│   ├── prepare_datasets.py     download and normalize the benchmarks
│   ├── normalize.py            schema normalization helpers
│   ├── prompts.py              the generation prompt
│   ├── gemini_client.py        Gemini API wrapper
│   ├── smoke_test.py           a few live calls before paying for a batch
│   ├── batch_build.py / batch_submit.py / batch_fetch.py   Batch API requests
│   ├── merge_release.py        join generations onto the corpus
│   ├── validate.py             acceptance checks and rejection lists
│   ├── finalize_release.py     write the released files
│   ├── paper_report.py         generation statistics for the appendix
│   └── reports/                validation reports, generation provenance, prompts, manual-inspection sheets
│
├── src/stability/              Stage 2: evaluation harness library
│   ├── schema.py               Item / ConditionItem data models
│   ├── normalize.py            raw -> unified item schema
│   ├── subsample.py, sampling.py   stratified sampling, O1-C and NQ pools
│   ├── conditions.py           condition construction (noise, option permutations, paraphrase, NQ)
│   ├── prompts.py              prompt rendering
│   ├── validate.py             offline perturbation checks
│   ├── client.py               async OpenAI-compatible client
│   ├── runner.py               runs one configuration over datasets x conditions, resumable
│   ├── parse.py                answer extraction (incl. Cyrillic homoglyphs)
│   ├── manifest.py             run manifests and resume invariants
│   ├── aggregate.py            joins raw outputs into long / paired tables
│   ├── config.py, io.py        config loading, JSONL / parquet I/O
│   └── metrics/                flip rates, self-flip floor, margin strata, consensus,
│                               position bias, ranking stability (Kendall τ_b), bootstrap stats
│
├── scripts/                    pipeline entry points, run in numeric order
│   ├── 00_inventory.py         measure the corpus                       -> reports/
│   ├── 01_normalize.py         unified item schema                      -> work/normalized/
│   ├── 02_subsample.py         stratified sample + O1-C / NQ pools      -> work/sample_index/
│   ├── 03_build_conditions.py  materialize every condition              -> work/conditions/
│   ├── 04_validate_perturbations.py  offline checks                     -> results/validation/
│   ├── 05_run.py               query one model configuration            -> results/<config_id>/
│   ├── 06_aggregate.py         long.parquet + paired.parquet            -> results/aggregate/
│   ├── 07_metrics.py           tidy CSVs + paper_numbers.json           -> results/metrics/
│   ├── 08_figures.py           PDF/PNG figures + LaTeX appendix tables  -> results/figures/
│   ├── 10_results_md.py        Markdown summary generated from the metric CSVs
│   ├── run_model.sh            run every configuration of one served model, then analyse
│   ├── run_pilot.sh            small pilot (100 items per benchmark)
│   ├── dev_fake_server.py      fake OpenAI-compatible server for testing without a GPU
│   └── dev_dry_run_demo.py     render every condition for one item per benchmark
│
├── runners/                    one script per served model (all of its configurations), plus _analysis.sh
├── work/                       precomputed intermediates: normalized items, sample index, materialized conditions
├── reports/                    corpus inventory and schema mapping
├── results/
│   ├── <config_id>/manifest.json, logs/   per-configuration run manifests and statistics
│   ├── aggregate/paired.parquet           paired baseline-vs-condition predictions
│   ├── metrics/                           every table in the paper; paper_numbers.json holds all quoted numbers
│   ├── figures/                           paper figures (two parse-failure policies: incorrect / excluded)
│   └── validation/                        perturbation validation report
└── tests/                      pytest suite (conditions, parsing, metrics, resume, sampling, ...)
```

Raw per-request outputs (`results/<config_id>/raw/*.jsonl`, about 4 GB) and the
full `results/aggregate/long.parquet` (1.2 GB) are too large for this
repository and are not included. All metric tables behind the paper's numbers
are in `results/metrics/`, and the figures can be rebuilt from them.

## Installation

```bash
pip install -r requirements.txt
pip install -e .                 # installs src/stability
pip install -e '.[dev]'          # optional: pytest
```

Python ≥ 3.10. Stage 1 additionally needs a Gemini API key in `.env`
(`GEMINI_KEY=...`); Stage 2 does not.

## Serving models with vLLM

The harness does not start models itself. Every model must be served with
[vLLM](https://github.com/vllm-project/vllm) through its OpenAI-compatible API,
and `--served-model-name` must match `model_name` in the configuration YAML.
Reasoning models need the matching `--reasoning-parser` (listed in
`configs/model_groups.yaml`) so that the reasoning trace is returned separately
from the answer. `--max-model-len` must be at least 34,816 (32,768 generation
tokens plus the prompt).

```bash
vllm serve Qwen/Qwen3.5-9B \
    --served-model-name qwen3.5-9b \
    --reasoning-parser qwen3 \
    --max-model-len 34816 \
    --dtype bfloat16 \
    --tensor-parallel-size 1 \
    --host 0.0.0.0 \
    --port 8000
```

Or use the included wrapper, which takes the same settings as environment
variables (the header of each `runners/*.sh` file shows the values for that
model):

```bash
MODEL_DIR=/path/to/models/qwen3.5-9b \
SERVED_NAME=qwen3.5-9b PORT=8000 CUDA_VISIBLE_DEVICES=0 \
REASONING_PARSER=qwen3 MAX_MODEL_LEN=34816 TP=1 \
./serve_model.sh
```

Set `models_root` in `configs/model_groups.yaml` to the directory holding your
model weights; `scripts/run_model.sh` uses it to print the matching serve
command.

Both modes of a model (for example `qwen3.5-9b-nothink` and
`qwen3.5-9b-think`) differ only in chat-template kwargs or the system prompt, so
one server covers all configurations of that model.

## Reproducing the evaluation

Steps 01–04 are already run and their outputs are in `work/` and
`results/validation/`. To regenerate them:

```bash
python scripts/01_normalize.py
python scripts/02_subsample.py
python scripts/03_build_conditions.py
python scripts/04_validate_perturbations.py
```

Run one configuration against a live server (dry-run first to check the
rendered requests):

```bash
python scripts/05_run.py --config configs/models/qwen3.5-9b-think.yaml --dry-run

python scripts/05_run.py \
    --config configs/models/qwen3.5-9b-think.yaml \
    --port 8000 \
    --datasets kazmmlu kazculture kazbench_kk \
    --conditions B0 N1 N2 N3 N4 I1 O1 P1 \
    --baseline-runs 3 \
    --out results/
```

Or run every configuration of a served model in one go:

```bash
PORT=8000 ./scripts/run_model.sh qwen3.5-9b
./scripts/run_model.sh --list          # show all model groups and configurations
```

Runs are resumable: completed `(uid, condition, run_index)` keys are skipped,
and a run refuses to resume if its config, request body, prompts or parser
changed.

After all configurations finish, aggregate and compute metrics and figures:

```bash
python scripts/06_aggregate.py --results results/
python scripts/07_metrics.py   --results results/
python scripts/08_figures.py   --results results/
# or: ./runners/_analysis.sh
```

Since raw outputs and `long.parquet` are not included, `06_aggregate.py` and
`07_metrics.py` only work after you rerun the models. `08_figures.py` reads only
`results/metrics/*.csv`, so it works on the included files.

## Regenerating the perturbations (optional)

```bash
echo 'GEMINI_KEY=<your key>' > .env
./gemini_experiments/run_gemini_pipeline.sh
```

See [gemini_experiments/README.md](gemini_experiments/README.md) for details.

## Tests

```bash
python -m pytest tests/ -q
```

