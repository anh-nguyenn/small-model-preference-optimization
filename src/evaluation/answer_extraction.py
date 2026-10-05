"""Extract the final numeric answer from a GSM8K-style text.

Design rationale lives in notes/evaluation_pipeline.md (sections 2-3).
One function handles both:
  - the GSM8K dataset's own ground-truth format, e.g. "... #### 20"
  - a model's free-text generation, e.g. "Therefore the answer is 20."

Priority order, checked in sequence:
  1. The dataset's "#### N" marker (unambiguous when present).
  2. An "answer is N" / "final answer is N" phrase.
  3. Fallback: the last standalone number anywhere in the text.
"""

import re

_GSM8K_MARKER = re.compile(r"####\s*(-?[\d,]+\.?\d*)")
_ANSWER_PHRASE = re.compile(
    r"(?:final\s+answer|answer)\s+is[:\s]*\$?(-?[\d,]+\.?\d*)", re.IGNORECASE
)
_ANY_NUMBER = re.compile(r"-?[\d,]+\.?\d*")


def extract_answer(text: str) -> str | None:
    """Return the final numeric answer in `text` as a normalized string.

    Returns None if no number can be found at all.
    """
    for pattern in (_GSM8K_MARKER, _ANSWER_PHRASE):
        match = pattern.search(text)
        if match:
            return _normalize(match.group(1))

    matches = [m for m in _ANY_NUMBER.findall(text) if m]
    return _normalize(matches[-1]) if matches else None


def _normalize(number_str: str) -> str:
    """Strip thousands-separator commas and a trailing bare period."""
    return number_str.replace(",", "").rstrip(".")
