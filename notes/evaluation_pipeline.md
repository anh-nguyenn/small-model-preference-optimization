# Evaluation Pipeline — Background & Implementation Plan

> **Day 5 pre-implementation doc.** Per our working agreement, this is
> written *before* any of Day 5's code. It covers the four tasks from
> `CLAUDE.md` Day 5: loading Qwen, loading GSM8K, answer extraction, and the
> binary correctness reward. No RL theory here — this is the first day
> where the artifacts are plain engineering, not conceptual notes.

---

## 1. Why this day matters

Every later day depends on this one:

- **Day 6 (DPO)** needs `extract_answer` + the reward function to *label*
  which sampled responses are "chosen" (correct) vs "rejected" (incorrect)
  when building the synthetic preference dataset.
- **Day 7 (GRPO)** needs the reward function directly as the `reward_i` in
  the group-relative advantage formula (`notes/papers/grpo.md` §Main
  contribution).

If answer extraction is wrong or inconsistent, both DPO and GRPO smoke
tests would be learning from a broken signal without any obvious symptom
(the training loop would still "run" — this is why `CLAUDE.md` asks for
unit tests here specifically, not just a manual eyeball check).

---

## 2. Background: what GSM8K actually looks like

Dataset: `openai/gsm8k`, config `main`. Each example has two string fields:

```json
{
  "question": "A box contains 12 red balls and 8 blue balls. How many balls are there?",
  "answer": "There are 12 + 8 = <<12+8=20>>20 balls in total.\n#### 20"
}
```

Two things to notice, because both matter for the extraction design below:

1. **The `answer` field contains a calculator annotation** —
   `<<12+8=20>>` — which is GSM8K's own notation for "a calculator was
   invoked here"; it is *not* part of the natural-language solution and
   should be ignored by extraction, not accidentally matched as "the
   answer."
2. **The ground-truth final answer is always marked with a `####` delimiter**,
   immediately followed by the number, as the *last* line. This is a
   clean, unambiguous marker — very different from what a model's own
   free-text generation looks like.

Contrast that with how the model (`Qwen2.5-0.5B-Instruct`) will actually
phrase a generated response — there is no `####` marker, since the model
isn't trained to produce the GSM8K dataset's exact formatting. `CLAUDE.md`'s
own example is representative of typical phrasing:

```text
We compute 12 + 8 = 20.
Therefore the answer is 20.
```

So **`extract_answer` has to handle two different "final answer" styles**:
the dataset's `#### N` convention (for ground truth) and natural-language
phrasing like "the answer is N" / "the final answer is N" (for model
generations). Building one function that handles both — rather than one
parser for ground truth and a separate one for model output — is a
deliberate reuse decision (see §4.1).

---

## 3. Design decisions

### 3.1 One `extract_answer`, not two parsers

`CLAUDE.md`'s engineering principle #1 ("prefer clarity over abstraction")
and the coding-agent rule ("reuse existing code instead of duplicating
logic") both argue against writing separate `extract_ground_truth` and
`extract_model_answer` functions. A single `extract_answer(text) -> str |
None` that tries, in priority order:

1. `#### <number>` (GSM8K ground-truth marker — checked first since it's
   unambiguous when present)
2. `"answer is <number>"` / `"final answer is <number>"` (case-insensitive —
   the common model-phrasing style)
3. **Fallback:** the last standalone number anywhere in the text

...correctly handles both the dataset's `answer` field and a model's raw
generation through the exact same code path.

### 3.2 Why a fallback, and why "last number" specifically

`CLAUDE.md` says "do not overengineer unusual edge cases initially," so the
fallback is deliberately simple rather than trying to cover every possible
phrasing. But *some* fallback is worth having: a 0.5B model doing GSM8K
problems will sometimes produce a correct computation but forget to say
"the answer is" — e.g. just trailing off with "...so there are 20 balls."
Taking the **last** number in the text (rather than the first) is a
reasonable heuristic because GSM8K solutions state intermediate quantities
first and the final quantity last — the same reason the dataset's own
`####` marker is placed at the end.

This is a real, documented heuristic choice — if it turns out to
systematically misfire on Day 6/7 data, that's a methodology change worth
writing down per `CLAUDE.md` principle #3, not silently tweaking the regex.

### 3.3 Numeric normalization

Extracted strings need cleanup before comparison:

- Strip thousands-separator commas: `"1,200"` → `"1200"`
- Strip a trailing bare period (from end-of-sentence punctuation):
  `"20."` → `"20"`
- Keep the sign (`"-3"` stays `"-3"`) and decimals (`"2.5"` stays `"2.5"`)

### 3.4 Comparison: numeric equality, not string equality

`src/rewards/correctness.py` compares the predicted and ground-truth
strings by **casting both to `float` and comparing**, not raw string
equality. This makes `"20"` and `"20.0"` compare equal, which plain string
equality would not. If either side fails to parse as a float (e.g.
`extract_answer` returned `None`), the reward is `0.0` — no credit for an
un-parseable response. This is a deliberate, documented choice: a response
that doesn't clearly state an answer is treated the same as a wrong answer,
not specially.

### 3.5 Reward stays reward; extraction stays extraction

`src/rewards/correctness.py` depends on `src/evaluation/answer_extraction.py`
and calls `extract_answer` on **both** the model's generated text and the
raw GSM8K `answer` field (reusing it for ground truth too, per §3.1) — it
does not reimplement any parsing itself. This keeps the "what counts as an
answer" logic in exactly one place.

---

## 4. Implementation plan

### 4.1 `src/evaluation/answer_extraction.py`

```python
import re

_GSM8K_MARKER = re.compile(r"####\s*(-?[\d,]+\.?\d*)")
_ANSWER_PHRASE = re.compile(r"(?:final\s+answer|answer)\s+is[:\s]*\$?(-?[\d,]+\.?\d*)", re.IGNORECASE)
_ANY_NUMBER = re.compile(r"-?[\d,]+\.?\d*")

def extract_answer(text: str) -> str | None:
    """Extract the final numeric answer from a GSM8K-style response.

    Tries, in order: the dataset's `#### N` marker, an "answer is N"
    phrase, then falls back to the last standalone number in the text.
    Returns None if no number is found at all.
    """
    for pattern in (_GSM8K_MARKER, _ANSWER_PHRASE):
        match = pattern.search(text)
        if match:
            return _normalize(match.group(1))

    matches = _ANY_NUMBER.findall(text)
    return _normalize(matches[-1]) if matches else None

def _normalize(number_str: str) -> str:
    return number_str.replace(",", "").rstrip(".")
```

(Pseudocode-level precision — final version written at implementation
time, but the structure/priority order above is the actual plan, not just
illustrative.)

### 4.2 `tests/test_answer_extraction.py`

Test cases to cover (table form so each one is an explicit, checkable
commitment, not an afterthought):

| Input | Expected | Why it's a test case |
|---|---|---|
| `"Therefore the answer is 20."` | `"20"` | Exact example from `CLAUDE.md` |
| `"The final answer is -3."` | `"-3"` | Exact example from `CLAUDE.md`; negative number |
| `"There are 12 + 8 = <<12+8=20>>20 balls in total.\n#### 20"` | `"20"` | Real GSM8K ground-truth format; must ignore the `<<...>>` calculator annotation |
| `"We have 1,200 apples. The answer is 1,200."` | `"1200"` | Comma-separated thousands, normalized |
| `"The answer is 2.5 meters."` | `"2.5"` | Decimal answer |
| `"...so there are 20 balls."` (no "answer is" phrase) | `"20"` | Fallback path: last number, no keyword |
| `"I'm not sure how to solve this."` | `None` | No number anywhere → `None`, not an exception |
| `"There are 12 red and 8 blue, so the answer is 20."` | `"20"` | Multiple numbers present; keyword-tagged one wins over earlier numbers |

### 4.3 `src/rewards/correctness.py`

```python
from src.evaluation.answer_extraction import extract_answer

def compute_reward(generated_text: str, ground_truth: str) -> float:
    """Deterministic binary reward: 1.0 if the final numeric answer in
    `generated_text` matches the one in `ground_truth`, else 0.0.

    `ground_truth` may be either a raw GSM8K `answer` field (containing
    a `#### N` marker) or an already-bare number string — extract_answer
    handles both.
    """
    predicted = extract_answer(generated_text)
    target = extract_answer(ground_truth)

    if predicted is None or target is None:
        return 0.0

    try:
        return 1.0 if float(predicted) == float(target) else 0.0
    except ValueError:
        return 0.0
```

Minimal test plan for this file (small, since the heavy lifting is already
tested in `answer_extraction`): correct match → `1.0`, wrong match → `0.0`,
unparsable generation → `0.0`.

### 4.4 `scripts/test_model.py`

- CLI args: `--model` (default `Qwen/Qwen2.5-0.5B-Instruct`), `--prompt`
  (default = the box-of-balls example from `CLAUDE.md`), `--max-new-tokens`
  (default small, e.g. `128` — this is a smoke script, not a benchmark).
- Device selection: use MPS if `torch.backends.mps.is_available()`, else
  CPU (no silent CUDA assumption, per hardware constraints).
- Load tokenizer + model, apply the chat template (Qwen2.5-Instruct is a
  chat model, so wrap the prompt via `tokenizer.apply_chat_template` rather
  than feeding raw text — otherwise the model has no instruction-following
  context and output quality would be misleadingly poor).
- Greedy decoding (`do_sample=False`) for a reproducible smoke test — no
  sampling randomness to control for at this stage.
- Print the decoded response.

### 4.5 `scripts/test_dataset.py`

- CLI args: `--split` (default `"train"`), `--n` (default `10`).
- `datasets.load_dataset("openai/gsm8k", "main", split=split)`, then take
  only the first `n` examples (via slicing/`.select(range(n))` — **not**
  materializing or transforming the full split, per the Day 5 spec: "Do
  not transform the entire dataset yet").
- For each of the `n` examples, print the `question`, the raw `answer`
  field, and the extracted ground-truth number (via `extract_answer` on
  the raw field) — showing the raw field too makes it easy to sanity-check
  extraction against the real `#### N` format while reading the output.

---

## 5. Relation to later days

- Day 6's preference-pair construction (`notes/papers/dpo.md` §Relation to
  this project) will label sampled responses as chosen/rejected using
  exactly `compute_reward` from this day — no separate labeling logic.
- Day 7's GRPO reward function (`notes/papers/grpo.md` §Relation to this
  project) is a direct call to `compute_reward`, run once per sampled
  response in a group.
- Because both later days depend on this exact function, any future change
  to extraction or reward logic must be called out explicitly per
  `CLAUDE.md` principle #3 ("never silently change experimental
  methodology") — not quietly edited in place once DPO/GRPO scripts exist.

---

## 6. References

| Source | Why it matters here |
|---|---|
| Cobbe et al., 2021 — *Training Verifiers to Solve Math Word Problems* | Introduces the GSM8K dataset and its `#### N` final-answer convention (§2 above is drawn directly from the dataset's actual format). |
| Hugging Face `transformers` — `generate()` / `apply_chat_template` docs | Correct usage of chat templates and greedy decoding for an instruct-tuned model like Qwen2.5-Instruct. |
| Hugging Face `datasets` — `load_dataset` docs | Selecting a split and a small slice without materializing the whole dataset. |

(No RL papers needed for this day — it's pure data/eval plumbing, which is
why this note lives at `notes/evaluation_pipeline.md` rather than under
`notes/papers/`.)

---

## 7. Open questions

- Should the fallback "last number in text" heuristic (§3.2) be restricted
  to integers/decimals only, or could it ever accidentally match something
  like a step number ("Step 2: ...")? Worth checking once real Qwen2.5-0.5B
  generations are seen in Day 5 implementation — if it misfires, document
  the fix rather than silently patching the regex.
- `compute_reward`'s numeric-equality comparison (§3.4) treats `"20"` and
  `"20.0"` as equal — is there any GSM8K case where the *format* of the
  answer (not just its value) should matter? Current assumption: no, GSM8K
  answers are single numeric quantities, format shouldn't matter — flagging
  this assumption explicitly rather than leaving it implicit.
- `scripts/test_model.py`'s use of `apply_chat_template` assumes a system/
  user turn structure; need to confirm Qwen2.5-0.5B-Instruct's tokenizer
  ships a chat template by default (expected, since it's published as an
  "-Instruct" checkpoint, but worth a sanity check at implementation time).
