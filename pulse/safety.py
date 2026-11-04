"""Sensitive-issue routing.

Two independent signals decide whether a comment goes to a human reviewer:

1. the model's own ``sensitive_category`` and ``urgency``;
2. a multilingual keyword backstop that runs on the original text, so a model
   miss on a safety or harassment comment is still caught.

Anything flagged by either signal is routed to the confidential review queue and
is never included in a manager's brief.
"""

from __future__ import annotations

import re
import unicodedata

from .schema import TaggedComment

# Deliberately recall-oriented. Reviewers dismiss false alarms in one click;
# a missed harassment or safety report is the failure that matters.
KEYWORDS: dict[str, dict[str, list[str]]] = {
    "safety_hazard": {
        "en": ["unsafe", "guardrail", "guard rail", "confined space", "gas test", "without ppe", "no ppe",
               "near miss", "injured", "injury", "fell from", "electric shock", "harness"],
        "ar": ["غير امن", "خطير", "حواجز حمايه", "الاماكن المغلقه", "فحص الغاز", "احزمه الامان", "اصابه",
               "حادث", "معدات الوقايه"],
        "ur": ["غیر محفوظ", "خطرناک", "ریلنگ نہیں", "ہارنس", "کنفائنڈ اسپیس", "گیس ٹیسٹ", "حادثہ", "زخمی"],
        "hi": ["असुरक्षित", "खतरनाक", "रेलिंग नहीं", "हार्नेस", "कन्फाइंड स्पेस", "गैस टेस्ट", "हादसा",
               "दुर्घटना", "घायल"],
        "ur_latn": ["railing nahi", "ghair mehfooz", "khatarnak", "zakhmi", "hadsa"],
    },
    "harassment": {
        "en": ["harass", "bully", "bullies", "insult", "shouts at", "shouting at", "threaten", "abuse", "abusive",
               "unwanted comment", "inappropriate comment", "touched me", "sexual"],
        "ar": ["تحرش", "يصرخ", "يهين", "اهانه", "تهديد", "يهدد", "تنمر", "تعليقات غير مرغوب", "اساءه"],
        "ur": ["ہراس", "چیخ", "گالی", "گالیاں", "دھمکی", "بدتمیزی", "نامناسب باتیں", "تنگ کرتا"],
        "hi": ["उत्पीडन", "चिल्लाता", "चिल्लाते", "गाली", "धमकी", "बदतमीजी", "गलत टिप्पणि", "छेड"],
        "ur_latn": ["cheekh", "galiyan", "gaali", "dhamki", "ghalat baat", "tang karta", "badtameezi"],
    },
    "discrimination": {
        # Unequal treatment by nationality is a common form in Gulf and camp workforces.
        # Phrases, not the bare word, so "respected whatever their nationality" is not flagged.
        "en": ["discriminat", "racis", "racial", "sexism", "sexist", "certain nationalit", "some nationalit",
               "based on nationalit", "because of nationalit", "because of my nationalit", "because of their nationalit"],
        "ar": ["تمييز", "عنصري", "عنصريه", "جنسيات معينه", "بسبب جنسيت", "حسب الجنسيه"],
        "ur": ["امتیاز", "تعصب", "نسل پرست", "خاص قومیت", "قومیت کی وجہ", "خاص ملکوں"],
        "hi": ["भेदभाव", "नस्लवाद", "नस्ल", "ख़ास देशों", "राष्ट्रीयता के आधार", "राष्ट्रीयता की वजह"],
        "ur_latn": ["taasub", "imtiyazi", "khaas mulkon", "qaumiyat ki wajah"],
    },
    "wellbeing_risk": {
        "en": ["hopeless", "suicid", "kill myself", "end my life", "can't go on", "cannot go on",
               "self-harm", "self harm", "no reason to live", "can't sleep", "cannot sleep"],
        "ar": ["اليأس", "باليأس", "انتحار", "انهي حياتي", "استطيع النوم"],
        "ur": ["ناامید", "خودکشی", "زندگی ختم", "نیند نہیں"],
        "hi": ["नाउम्मीद", "आत्महत्या", "जिंदगी खत्म", "नींद नहीं"],
        "ur_latn": ["na-umeed", "naumeed", "na umeed", "khudkushi", "neend nahi", "neend bhi nahi"],
    },
    "misconduct": {
        "en": ["fraud", "bribe", "kickback", "falsif", "inflated", "fake timesheet", "theft", "stealing", "corrupt"],
        "ar": ["احتيال", "رشوه", "تزوير", "تضخيم", "سرقه", "فساد"],
        "ur": ["فراڈ", "رشوت", "جعلی", "بڑھا کر لکھ", "چوری", "کرپشن"],
        "hi": ["धोखाधडी", "रिश्वत", "फर्जी", "बढा-चढाकर", "चोरी", "भ्रष्टाचार"],
        "ur_latn": ["barha kar likh", "rishwat", "jaali", "chori"],
    },
    "retaliation": {
        "en": ["retaliat", "punished for", "after complaining", "raised complaints", "raised a complaint",
               "victimis", "victimiz", "revenge"],
        "ar": ["انتقام", "عقاب", "بعد الشكوي", "قدموا شكاوي"],
        "ur": ["انتقام", "بدلہ لیا", "شکایت کی تھی"],
        "hi": ["बदला लिया", "शिकायत की थी"],
        "ur_latn": ["shikayat ki thi", "badla liya"],
    },
}

CATEGORY_PRIORITY = {
    "wellbeing_risk": 0,
    "safety_hazard": 1,
    "harassment": 2,
    "retaliation": 3,
    "discrimination": 4,
    "misconduct": 5,
}

_ARABIC_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")
_ARABIC_MAP = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ة": "ه", "ى": "ي", "ؤ": "و", "ئ": "ي"})


def normalise(text: str) -> str:
    """Lowercase and fold script variants so keywords match regardless of spelling details."""
    t = unicodedata.normalize("NFC", text or "").lower()
    t = _ARABIC_DIACRITICS.sub("", t).translate(_ARABIC_MAP)
    t = t.replace("़", "")  # Devanagari nukta (ख़ -> ख)
    t = unicodedata.normalize("NFC", t)
    t = t.replace("’", "'")
    return re.sub(r"\s+", " ", t)


_NORMALISED: dict[str, list[str]] = {
    cat: sorted({normalise(k) for words in langs.values() for k in words}) for cat, langs in KEYWORDS.items()
}


def keyword_flags(text: str) -> list[str]:
    """Categories whose keywords appear in the text (any language)."""
    t = normalise(text)
    hits = [cat for cat, words in _NORMALISED.items() if any(w in t for w in words)]
    return sorted(hits, key=lambda c: CATEGORY_PRIORITY.get(c, 99))


def route(tag: TaggedComment, low_confidence: float = 0.5) -> TaggedComment:
    """Fill ``needs_review`` and ``review_reasons`` on a tagged comment."""
    reasons: list[str] = []
    if tag.sensitive_category != "none":
        reasons.append(f"model:{tag.sensitive_category}")
    for cat in tag.keyword_flags:
        reasons.append(f"keyword:{cat}")
    if tag.urgency == "critical":
        reasons.append("critical_urgency")
    if tag.error:
        reasons.append("processing_error")
    elif not tag.skipped:
        if tag.confidence < low_confidence:
            reasons.append("low_confidence")
        if tag.evidence and not tag.evidence_grounded:
            reasons.append("unverified_evidence")
    tag.review_reasons = reasons
    tag.needs_review = bool(reasons)
    return tag


CONFIDENTIAL_PREFIXES = ("model:", "keyword:", "critical_urgency")


def is_confidential(tag: TaggedComment) -> bool:
    """Confidential items go to HR/HSE and are excluded from manager briefs."""
    return any(r.startswith(CONFIDENTIAL_PREFIXES) for r in tag.review_reasons)


def review_category(reasons: list[str]) -> str:
    """Best single category for a review item (model's view first, then keyword)."""
    for r in reasons:
        if r.startswith("model:"):
            return r.split(":", 1)[1]
    for r in reasons:
        if r.startswith("keyword:"):
            return r.split(":", 1)[1]
    return "none"


def severity_rank(sensitive_category: str, urgency: str, reasons: list[str]) -> tuple[int, int]:
    cat = sensitive_category if sensitive_category != "none" else review_category(reasons)
    urgency_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    return (urgency_order.get(urgency, 4), CATEGORY_PRIORITY.get(cat, 9))
