"""Keyword baseline: what you get without a model.

Used when no model is configured (development, tests) and as the comparison point
in the validation report, so the value Muse Spark adds is measured, not assumed.
"""

from __future__ import annotations

from .lang import guess_language
from .safety import keyword_flags, normalise
from .schema import TaggedComment
from .taxonomy import DEFAULT_THEMES, Theme

THEME_KEYWORDS: dict[str, list[str]] = {
    "workload": ["workload", "short-staffed", "understaffed", "overload", "two jobs",
                 "ضغط العمل", "نقص في عدد", "عبء", "بوجھ", "اسٹاف کم", "बोझ", "स्टाफ की कमी"],
    "pay_benefits": ["salar", "pay", "overtime", "increment", "insurance", "allowance", "bonus",
                     "راتب", "الرواتب", "العمل الاضافي", "الزياده", "التامين",
                     "تنخواہ", "اوور ٹائم", "انکریمنٹ", "انشورنس",
                     "तनख्वाह", "वेतन", "ओवरटाइम", "इंक्रीमेंट", "इंश्योरेंस"],
    "manager_support": ["manager", "supervisor", "boss", "team lead", "مدير", "المشرف", "مشرفي",
                        "مینیجر", "سپروائزر", "باس", "मैनेजर", "सुपरवाइजर", "बॉस"],
    "leadership": ["senior management", "leadership", "direction", "strategy", "الاداره العليا", "توجه الشركه",
                   "سینئر مینجمنٹ", "کمپنی کی سمت", "सीनियर मैनेजमेंट", "कंपनी की दिशा"],
    "career_growth": ["promot", "career", "grade", "growth", "ترقيه", "الترقيات", "الدرجه", "التطور",
                      "پروموشن", "گریڈ", "گروتھ", "प्रमोशन", "ग्रेड", "ग्रोथ"],
    "training": ["training", "course", "certif", "classes", "تدريب", "دوره", "دورات", "دروس",
                 "ٹریننگ", "کورس", "کلاسز", "ट्रेनिंग", "कोर्स", "क्लास"],
    "communication": ["communicat", "informed", "rumour", "rumor", "town hall", "notice",
                      "الاشاعات", "الاعلانات", "اطلاع", "الاجتماعات", "افواہ", "نوٹس", "ٹاؤن ہال",
                      "अफवाह", "नोटिस", "टाउन हॉल"],
    "recognition": ["recogni", "acknowledg", "appreciat", "award", "praise", "يقدر", "تكريم",
                    "تعریف", "ایوارڈ", "سراہا", "तारीफ", "अवॉर्ड", "सम्मान"],
    "work_life": ["shift", "leave", "family", "rotation", "rest", "ورديات", "اجازتي", "عائلتي", "المناوبه", "الراحه",
                  "شفٹ", "چھٹی", "گھر والوں", "روٹیشن", "آرام", "शिफ्ट", "छुट्टी", "घरवालों", "रोटेशन", "आराम"],
    "health_safety": ["safety", "ppe", "hazard", "scaffold", "harness", "السلامه", "سقالات",
                      "سیفٹی", "اسکیفولڈنگ", "सेफ्टी", "स्कैफोल्डिंग"],
    "respect_inclusion": ["respect", "fair", "favourit", "favorit", "nationalit", "احترام", "جنسيات", "المقربين",
                          "عزت", "قومیت", "قریب ہیں", "इज्जत", "करीब हैं"],
    "facilities": ["camp", "accommodation", "room", "bus", "food", "mess", "air conditioning",
                   "السكن", "الغرف", "غرف", "حافله", "الطعام", "المكيف",
                   "کیمپ", "کمرے", "کھانا", "میس", "رہائش", "اے سی", "कैंप", "कमरे", "खाना", "मेस", "एसी"],
    "tools_equipment": ["tool", "equipment", "spare", "system", "erp", "العدد", "قطع الغيار", "نظام",
                        "ٹولز", "اسپیئر", "سسٹم", "टूल्स", "स्पेयर", "सिस्टम"],
    "job_security": ["contract", "renew", "layoff", "job security", "secure in my job", "عقود", "العقود", "تجديد",
                     "کنٹریکٹ", "نوکری", "कॉन्ट्रैक्ट", "नौकरी"],
    "teamwork": ["colleague", "cooperat", "collaborat", "زملائي", "التعاون", "ساتھی", "تعاون", "साथी", "तालमेल"],
    "wellbeing": ["stress", "mental", "counsel", "morale", "wellbeing", "التوتر", "النفسيه",
                  "دباؤ", "ذہنی", "کاؤنسلنگ", "तनाव", "काउंसलिंग"],
    "ethics_integrity": ["fraud", "timesheet", "inflat", "bribe", "تضخيم", "رشوه", "ٹائم شیٹ", "رشوت",
                         "टाइमशीट", "रिश्वत"],
}

POSITIVE = ["good", "great", "excellent", "proud", "trust", "clean", "comfortable", "helped", "motivat", "secure",
            "faster", "strong", "listens", "supports", "like family", "thank",
            "جيد", "ممتاز", "افخر", "اثق", "نظيف", "مريح", "اسرع", "قويه", "يدعمنا", "دافعا", "شكرا",
            "اچھ", "زبردست", "فخر", "بھروسہ", "صاف", "آرام دہ", "تیز", "حوصلہ", "محفوظ",
            "अच्छ", "बढिया", "गर्व", "भरोसा", "साफ", "आरामदायक", "तेज", "हौसला", "सुरक्षित"]
NEGATIVE = ["not ", "no ", "never", "poor", "broken", "slow", "waste", "delayed", "short", "unsafe", "anxious",
            "insult", "hardly", "ignored", "little", "rumour",
            "لا ", "لم ", "سيئ", "معطل", "بطيء", "نضيع", "متاخره", "نقص", "قلقين", "ضعيف",
            "نہیں", "خراب", "سست", "ضائع", "پریشان", "کمی",
            "नहीं", "खराब", "धीमा", "बर्बाद", "कमी", "परेशान"]
LONG_RUNNING = ["months", "years", "resign", "اشهر", "سنتين", "استقاله", "مہینے", "سال", "استعفی", "महीने", "साल", "इस्तीफा"]


def _hits(text: str, words: list[str]) -> int:
    return sum(1 for w in words if normalise(w) in text)


def tag_with_keywords(comment_id: str, text: str, redacted: str, themes: tuple[Theme, ...] = DEFAULT_THEMES) -> TaggedComment:
    t = " " + normalise(redacted) + " "
    allowed = {th.id for th in themes}
    scored = []
    for theme, words in THEME_KEYWORDS.items():
        if theme in allowed:
            n = _hits(t, words)
            if n:
                scored.append((n, theme))
    scored.sort(key=lambda x: -x[0])
    found = [th for _, th in scored[:3]] or ["other"]

    pos, neg = _hits(t, POSITIVE), _hits(t, NEGATIVE)
    if pos and neg:
        sentiment = "mixed" if abs(pos - neg) <= 1 else ("positive" if pos > neg else "negative")
    elif pos:
        sentiment = "positive"
    elif neg:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    flags = keyword_flags(text)
    if any(f in ("safety_hazard", "harassment", "wellbeing_risk") for f in flags):
        urgency = "critical"
    elif flags or (sentiment in ("negative", "mixed") and _hits(t, LONG_RUNNING)):
        urgency = "high"
    elif sentiment in ("negative", "mixed"):
        urgency = "medium"
    else:
        urgency = "low"

    return TaggedComment(
        comment_id=comment_id,
        text=text,
        text_redacted=redacted,
        language=guess_language(text),
        themes=found,
        sentiment=sentiment,
        urgency=urgency,
        sensitive_category=flags[0] if flags else "none",
        summary_en="",
        evidence="",
        evidence_grounded=False,
        confidence=0.5,
        keyword_flags=flags,
    )
