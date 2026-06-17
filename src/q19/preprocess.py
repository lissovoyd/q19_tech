"""Clean banking queries before embedding.

Banking support text is full of specifics -- card numbers, amounts, dates,
account refs -- that are NOT what makes two queries "the same issue." If left
in, an embedding can be dragged toward matching digits instead of matching
intent ("my card 4111... declined" vs "my card 5500... declined" should be a
duplicate; the numbers are noise). We replace each kind of specific with a
placeholder token so the *structure/role* survives but the *value* doesn't
dominate the vector.

Placeholders (kept as words so the model still sees "a card was involved"):
  <CARD> <ACCOUNT> <AMOUNT> <DATE> <TIME> <EMAIL> <PHONE> <NUM>

Never touches eval_holdout/.
"""
import re

# Order matters: more specific patterns first, generic <NUM> last.
_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), " <EMAIL> "),
    # 13-19 digit card numbers, possibly space/dash grouped
    (re.compile(r"\b(?:\d[ -]?){13,19}\b"), " <CARD> "),
    # currency amounts: £1,234.56  $50  100 USD/GBP/EUR
    (re.compile(r"[£$€]\s?\d[\d,]*(?:\.\d+)?"), " <AMOUNT> "),
    (re.compile(r"\b\d[\d,]*(?:\.\d+)?\s?(?:usd|gbp|eur|dollars?|pounds?|euros?)\b", re.I), " <AMOUNT> "),
    # phone numbers: +44 7..., (020) 7946 0000, 555-123-4567
    (re.compile(r"\+?\d[\d ()-]{7,}\d"), " <PHONE> "),
    # dates: 12/03/2024, 2024-03-12, 3 Jan 2024
    (re.compile(r"\b\d{1,4}[/-]\d{1,2}[/-]\d{1,4}\b"), " <DATE> "),
    (re.compile(r"\b\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\b", re.I), " <DATE> "),
    # clock times: 14:30, 9:05pm
    (re.compile(r"\b\d{1,2}:\d{2}\s?(?:am|pm)?\b", re.I), " <TIME> "),
    # account / reference numbers: 6+ digit runs that survived the above
    (re.compile(r"\b\d{6,}\b"), " <ACCOUNT> "),
    # any remaining standalone number
    (re.compile(r"\b\d+\b"), " <NUM> "),
]

_WS = re.compile(r"\s+")


def clean_text(text: str) -> str:
    out = text
    for pat, repl in _PATTERNS:
        out = pat.sub(repl, out)
    return _WS.sub(" ", out).strip()


def clean_texts(texts: list[str]) -> list[str]:
    return [clean_text(t) for t in texts]
