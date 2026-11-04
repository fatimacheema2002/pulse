"""Cheap script-based language guess, used by the keyword baseline and as a sanity check."""

from __future__ import annotations

_URDU_ONLY = set("ٹڈڑںےہھۓ")  # letters used in Urdu but not in Arabic


def guess_language(text: str) -> str:
    counts = {"arabic_script": 0, "devanagari": 0, "latin": 0}
    urdu_marks = 0
    for ch in text or "":
        cp = ord(ch)
        if 0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F or 0xFB50 <= cp <= 0xFEFF:
            counts["arabic_script"] += 1
            if ch in _URDU_ONLY:
                urdu_marks += 1
        elif 0x0900 <= cp <= 0x097F:
            counts["devanagari"] += 1
        elif ch.isalpha() and cp < 0x0250:
            counts["latin"] += 1
    script = max(counts, key=counts.get)
    if counts[script] == 0:
        return "und"
    if script == "devanagari":
        return "hi"
    if script == "arabic_script":
        return "ur" if urdu_marks >= 1 else "ar"
    return "en"
