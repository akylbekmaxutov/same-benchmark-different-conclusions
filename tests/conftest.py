import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest
from stability.schema import Item


def mk_item(uid="kazmmlu::000001", n_options=4, answer="C", **kw):
    opts = {L: f"option {L} text" for L in "ABCDE"[:n_options]}
    d = dict(
        uid=uid, orig_id=uid.split("::")[1], dataset=uid.split("::")[0],
        category="cat", subcategory=None, question="Сұрақ мәтіні?",
        options=opts, answer=answer,
        relevant_noise="Релевантты шу мәтіні.",
        irrelevant_noise="Бейрелевантты шу мәтіні.",
        question_paraphrase="Сұрақтың басқаша тұжырымы?",
        meta={},
    )
    d.update(kw)
    return Item(**d)


@pytest.fixture
def item():
    return mk_item()


@pytest.fixture
def item_e():
    return mk_item(uid="kazmmlu::000002", n_options=5, answer="E")


# ------------------------------------------------------------------ synthetic result tables
import pandas as pd
from stability.schema import RESULT_ROW_FIELDS


def mk_long(spec, config_id="c1", dataset="d", gold="A", n_options=4,
            perm_by_condition=None):
    """`spec` maps condition -> {run_index -> {uid -> emitted_letter or None}}.

    Produces a `long`-shaped DataFrame with exactly the columns the aggregate
    layer expects, so tests exercise the real pairing code.
    """
    perm_by_condition = perm_by_condition or {}
    rows = []
    for condition, runs in spec.items():
        for run_index, answers in runs.items():
            for uid, letter in answers.items():
                perm = perm_by_condition.get(condition)
                gold_eff = perm[gold] if perm else gold
                row = {k: None for k in RESULT_ROW_FIELDS}
                row.update({
                    "uid": uid, "parent_uid": uid, "dataset": dataset,
                    "category": "cat", "subcategory": None,
                    "condition": condition,
                    "condition_group": "validity_preserving",
                    "run_index": run_index, "seed": 1000 + run_index,
                    "config_id": config_id, "family": "fam", "model_name": "m",
                    "reasoning_regime": "off", "params_total_b": 1.0,
                    "params_active_b": 1.0, "is_moe": False,
                    "kazakh_specialized": False,
                    "prompt_version": "v1", "prompt_sha256": "x",
                    "n_options": n_options, "has_option_e": n_options > 4,
                    "gold_answer": gold_eff, "gold_answer_original": gold,
                    "option_permutation": perm,
                    "gold_position_index": "ABCDE".index(gold_eff),
                    "rotation_index": None,
                    "parsed_answer": letter,
                    "parse_tier": "failed" if letter is None else "json_strict",
                    "format_valid": letter is not None,
                    "homoglyph_normalized": False,
                    "is_correct": letter == gold_eff,
                    "finish_reason": "stop", "truncated": False,
                    "attempt_count": 1, "endpoint_kind": "vllm",
                })
                rows.append(row)
    df = pd.DataFrame(rows)[list(RESULT_ROW_FIELDS)]
    from stability.parse import FAIL
    df["answer_eff"] = df["parsed_answer"].fillna(FAIL)
    return df
