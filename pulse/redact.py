"""Remove personal details from comments before they reach a model or a report.

The original text is kept locally (for the confidential review queue); only the
redacted version is sent to the model and used in manager briefs.
"""

from __future__ import annotations

import re
from typing import Iterable

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", re.UNICODE)
# Long digit runs with optional separators: phone numbers, IDs, IBAN-like strings.
_PHONE = re.compile(r"(?<![\w])\+?\d[\d\s\-().]{6,}\d(?![\w])", re.UNICODE)
_EMPLOYEE_ID = re.compile(r"\b(?:emp|staff|badge|id)[\s#:\-]*\d{3,}\b", re.IGNORECASE)
_URL = re.compile(r"https?://\S+", re.IGNORECASE)


def _digit_count(s: str) -> int:
    return sum(ch.isdigit() for ch in s)


def redact(text: str, names: Iterable[str] = ()) -> tuple[str, list[str]]:
    """Return (redacted_text, kinds_found). Kinds are 'email', 'phone', 'id', 'url', 'name'."""
    if not text:
        return text, []
    found: list[str] = []
    out = text

    def sub(pattern: re.Pattern[str], token: str, kind: str, check=None) -> None:
        nonlocal out

        def _repl(m: re.Match[str]) -> str:
            if check and not check(m.group(0)):
                return m.group(0)
            if kind not in found:
                found.append(kind)
            return token

        out = pattern.sub(_repl, out)

    sub(_URL, "[URL]", "url")
    sub(_EMAIL, "[EMAIL]", "email")
    sub(_EMPLOYEE_ID, "[ID]", "id")
    sub(_PHONE, "[PHONE]", "phone", check=lambda s: _digit_count(s) >= 8)

    # Names: longest first so "Imran Khan" is replaced before "Imran".
    for name in sorted({n.strip() for n in names if n and len(n.strip()) >= 3}, key=len, reverse=True):
        if name.isascii():
            pattern = re.compile(rf"(?<![\w]){re.escape(name)}(?![\w])", re.IGNORECASE)
        else:
            pattern = re.compile(re.escape(name))
        if pattern.search(out):
            out = pattern.sub("[NAME]", out)
            if "name" not in found:
                found.append("name")
    return out, found


def name_terms(*columns: Iterable[str]) -> list[str]:
    """Collect person names from metadata columns, plus their individual parts (first/last name)."""
    terms: set[str] = set()
    for col in columns:
        for value in col:
            if not isinstance(value, str) or not value.strip():
                continue
            v = value.strip()
            terms.add(v)
            for part in re.split(r"[\s.]+", v):
                if len(part) >= 4 and part[0].isupper():
                    terms.add(part)
    return sorted(terms)
