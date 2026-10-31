"""Phase 3-6: the single Gemini prompt that produces all perturbation material.

One request per source question, three outputs per request:
  relevant_noise      - topically related but misleading/distracting context
  irrelevant_noise    - natural Kazakh text unrelated to the required knowledge
  question_paraphrase - one meaning-preserving Kazakh rewording

This file is part of the released generation materials (Phase 32). Any edit
here must bump PROMPT_VERSION in config.py.
"""
from __future__ import annotations

from typing import Any, Dict

from .config import OPTION_LETTERS

SYSTEM_INSTRUCTION = """\
You are a careful Kazakh-language dataset engineer building a benchmark-robustness study.
You write fluent, natural, grammatical Kazakh (қазақ тілі).
You return only a single JSON object, with no markdown fences and no commentary."""

# Uppercase type names: the same dict is sent both through the SDK (smoke test)
# and verbatim inside the batch JSONL, and the REST schema enum expects
# OBJECT / STRING.
RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "relevant_noise": {"type": "STRING"},
        "irrelevant_noise": {"type": "STRING"},
        "question_paraphrase": {"type": "STRING"},
    },
    "required": ["relevant_noise", "irrelevant_noise", "question_paraphrase"],
    "propertyOrdering": ["relevant_noise", "irrelevant_noise", "question_paraphrase"],
}

USER_TEMPLATE = """\
You are given one Kazakh multiple-choice question from the benchmark "{dataset}"
(category: {category}). Produce three pieces of Kazakh text for a controlled
perturbation study.

### Source item
Question:
{question}

Options:
{options_block}

Correct answer: {answer}) {answer_text}

### What to produce

1. "relevant_noise" — a short passage in Kazakh that is TOPICALLY RELATED to the
   question and plausible in context, but MISLEADING or distracting. It must:
   - stay on the same broad topic as the question;
   - introduce competing, tangential or attention-pulling information;
   - NOT be needed to answer the question;
   - NOT change which option is correct — after reading it, "{answer}" must still be
     the single correct answer;
   - NOT make the question invalid, ambiguous or unanswerable;
   - NOT state, hint at, negate or rank any of the answer options;
   - NOT contain the text of any answer option, nor any option letter (A/B/C/D/E),
     nor phrases such as "дұрыс жауап".
   This is relevant-but-distracting context, not a fabricated alternative gold answer.
   Length: 2-4 sentences, approximately {target_words} words.

2. "irrelevant_noise" — a short passage in Kazakh that is natural, fluent, and
   COMPLETELY UNRELATED to the knowledge the question requires. It must:
   - be about a different subject entirely (everyday life, weather, travel,
     a neutral fact, etc.);
   - be unable to help anyone determine the answer;
   - not contradict or comment on the question in any way;
   - not mention any answer option or option letter.
   Length: 2-4 sentences, approximately the same length as "relevant_noise"
   (approximately {target_words} words).

3. "question_paraphrase" — exactly ONE rewording of the question, entirely in Kazakh.

   HARD REQUIREMENTS (a violation makes the item unusable):
   - Output the QUESTION ONLY. Never append, list, quote or allude to the answer
     options. Strings such as "Опциялар:", "A)", "B)", "нұсқалар" must NOT appear.
   - Keep every number as a DIGIT exactly as written in the original: write 1989,
     not "мың тоғыз жүз сексен тоғыз"; write 0,6 not "нөл бүтін алты оннан".
   - Keep every negation. Kazakh negation may be lexical (емес, жоқ, ешқандай) or
     morphological (-майды, -пайды, -баған, -майтын, -пау). If the original asks
     which option does NOT apply, the paraphrase must still ask that.
   - If the question contains a reading passage, quotation, poem, dialogue or
     source excerpt, REPRODUCE THAT TEXT UNCHANGED and reword only the question
     sentence around it. Never summarize or drop the passage - without it the
     question cannot be answered.
   - Reproduce formulas, equations, chemical notation, LaTeX and units exactly as
     given. Only the Kazakh prose around them may be reworded.
   - Preserve every date, unit, proper name and technical term.

   ALSO:
   - preserve the meaning and the knowledge required;
   - keep "{answer}" as the correct answer with the SAME answer options;
   - not be simplified, not be made harder, not add hints or extra facts;
   - be a genuinely different surface wording, not a copy of the original.

### Output
Return exactly this JSON object and nothing else:
{{"relevant_noise": "...", "irrelevant_noise": "...", "question_paraphrase": "..."}}
"""

TARGET_WORDS = 45


def options_block(item: Dict[str, Any]) -> str:
    lines = []
    for letter in OPTION_LETTERS:
        text = item.get(letter)
        if text:
            lines.append(f"{letter}) {text}")
    return "\n".join(lines)


def build_user_prompt(item: Dict[str, Any]) -> str:
    answer = item["answer"]
    return USER_TEMPLATE.format(
        dataset=item.get("dataset", ""),
        category=item.get("category", "unknown"),
        question=item["question"],
        options_block=options_block(item),
        answer=answer,
        answer_text=item.get(answer, ""),
        target_words=TARGET_WORDS,
    )
