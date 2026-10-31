# Per-model runners

One file per served model. Each runs **every mode of that model** against a
single live endpoint, over the **full 5,324-item sample**.

## Use

```bash
# 1. serve the model on its own GPU (the command is in the file's header)
# 2. edit PORT and SERVED at the top of the runner if they differ
# 3. run it
./runners/qwen3.5-9b.sh

# after EVERY model has finished, once:
./runners/_analysis.sh
```

Runners do evaluation only. Aggregation is separate because several models
running in parallel would otherwise race on `results/aggregate/`.

## Port map

Defaults are distinct so all ten can run at once. Change `PORT` in the file
if you serve elsewhere.

| runner | port | GPU | served name | parser | configurations |
|---|---:|---:|---|---|---|
| `qwen3.5-9b.sh` | 8000 | 0 | `qwen3.5-9b` | `qwen3` | qwen3.5-9b-nothink, qwen3.5-9b-think |
| `qwen3.8-27b.sh` | 8001 | 1 | `qwen3.8-27b` | `qwen3` | qwen3.8-27b-nothink, qwen3.8-27b-think |
| `qwen3.6-35b-a3b.sh` | 8002 | 2 | `qwen3.6-35b-a3b` | `qwen3` | qwen3.6-35b-a3b-nothink, qwen3.6-35b-a3b-think |
| `gemma-4-12b-it.sh` | 8003 | 3 | `gemma-4-12b-it` | `gemma4` | gemma-4-12b-it-nothink, gemma-4-12b-it-think |
| `gemma-4-31b-it.sh` | 8004 | 4 | `gemma-4-31b-it` | `gemma4` | gemma-4-31b-it-nothink, gemma-4-31b-it-think |
| `gemma-4-26b-a4b-it.sh` | 8005 | 5 | `gemma-4-26b-a4b-it` | `gemma4` | gemma-4-26b-a4b-it-nothink, gemma-4-26b-a4b-it-think |
| `gpt-oss-20b.sh` | 8006 | 6 | `gpt-oss-20b` | `GptOss` | gpt-oss-20b-low, gpt-oss-20b-high |
| `muse-glimmer-30b.sh` | 8007 | 7 | `muse-glimmer-30b` | `muse-glimmer` | muse-glimmer-30b-low, muse-glimmer-30b-high |
| `qolda-avl-9b.sh` | 8008 | 0 | `qolda-avl-9b` | `qwen3` | qolda-avl-9b |
| `qolda-avl-34b.sh` | 8009 | 1 | `qolda-avl-34b` | `qwen3` | qolda-avl-34b |

## Volume

| | requests |
|---|---:|
| main grid, per configuration | 53,240 |
| all 18 configurations | 958,320 |
| plus O1-C / NQ / P1x3 extras | ~+70,000 |

`B0` x3 + seven perturbations = 10 passes over 5,324 items.

## Resume

Every runner is resumable: re-running skips completed
`(uid, condition, run_index)` keys. A crash, a preempted job, or a server
restart costs only the unfinished remainder. Changing a sampling parameter
mid-run is refused by name rather than silently mixed.

## Extras, and why they are where they are

- **O1-C** runs only for `defaults.o1c_configs` -- the six declared
  configurations spanning family, scale and reasoning regime. Without that
  list the diagnostic ends up on whichever models happened to be up.
- **NQR / NQI** run for `defaults.nq_probe_configs` -- three configurations
  spanning capability, since a weak model landing at chance proves nothing
  about whether the noise carries signal a strong model could exploit.
- **P1 x3** runs for `defaults.p1_floor_configs` -- the floor-transfer check.

