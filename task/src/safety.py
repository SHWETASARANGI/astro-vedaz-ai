"""
safety.py — Content safety layer for an astrology app.

Design goals:
1. Regex is a fast first-pass filter, not the final word — it catches obvious
   phrasing but misses paraphrase, so borderline/clean text can optionally be
   routed to a classifier fallback (stubbed here, wire up an LLM call later).
2. Violations are tiered by severity, not collapsed into one safe/unsafe bit.
   BLOCK-tier content should stop the response outright. SOFTEN-tier content
   should trigger a rewrite/disclaimer rather than a hard refusal.
3. Basic negation handling so "you will NOT die early" doesn't get flagged
   the same way "you will die early" does.
4. A rewrite() stub for turning flagged text into safer phrasing instead of
   just rejecting it.
5. Logging hooks so flagged content can be reviewed later and the rule set
   tightened over time based on real false positive/negative rates.
"""

import re
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Severity tiers
# ---------------------------------------------------------------------------

class Severity(str, Enum):
    BLOCK = "block"     # hard stop — never show this content as-is
    SOFTEN = "soften"   # allow, but rewrite/add disclaimer before showing
    REVIEW = "review"   # low confidence — log for human review, don't block


# ---------------------------------------------------------------------------
# Rule set
# Each category maps to (severity, [patterns]).
# Patterns are regex, matched case-insensitively against the lowercased text.
# ---------------------------------------------------------------------------

SAFETY_RULES: dict[str, dict] = {
    "death_prediction": {
        "severity": Severity.BLOCK,
        "patterns": [
            r"you (will|are going to|are about to) die",
            r"death is near",
            r"early death",
            r"life expectancy",
            r"won'?t (live|survive) (long|much longer)",
            r"time (on this earth|left) is (limited|short)",
            r"your (days|time) (are|is) numbered",

            # Hinglish
            r"mrityu yog",
            r"aapki maut",
            r"jaldi mar (jaoge|jaayenge)",
            r"maut (nazdeek|kareeb) hai",
            r"zyada din nahi bache",
            r"jeevan (chhota|kam) hai",
        ],
    },
    "medical_claims": {
        "severity": Severity.BLOCK,
        "patterns": [
            r"you will (get|develop) cancer",
            r"serious illness",
            r"terminal (disease|illness)",
            r"astrology can cure",
            r"(this remedy|chanting|this puja) will cure",
            r"medical (chart|reading) shows .*(illness|disease|cancer)",
            r"stop (taking|your) medication",
            r"don'?t (need|require) (a doctor|medical treatment)",

            # Hinglish
            r"cancer hoga",
            r"badi bimari",
            r"gambhir bimari",
            r"rog dikh raha hai",
            r"kundli mein bimari",
            r"astrology se bimari theek",
            r"doctor ki zarurat nahi",
            r"dawai band kar do",
        ],
    },
    "financial_guarantees": {
        "severity": Severity.SOFTEN,
        "patterns": [
            r"guaranteed (wealth|money|riches|success)",
            r"become rich for sure",
            r"100% (guaranteed|certain) (return|profit|wealth)",
            r"will definitely (win|make) (the lottery|money)",

            # Hinglish
            r"pakka ameer",
            r"100% paisa",
            r"guaranteed paisa",
            r"sure profit",
            r"pakki kamai",
            r"lakhpati banoge",
            r"crorepati banoge",
        ],
    },
    "fear_selling": {
        "severity": Severity.BLOCK,
        "patterns": [
            r"buy this remedy",
            r"pay for (the )?ritual",
            r"only solution",
            r"urgent puja",
            r"act now or",
            r"(curse|bad luck) will (worsen|continue) (unless|until) you pay",

            # Hinglish
            r"ye puja karwa lo",
            r"turant puja",
            r"upay karna zaroori hai",
            r"varna nuksan hoga",
            r"paise dekar puja",
            r"yehi ek upay hai",
            r"yehi akela solution hai",
            r"shanti puja karwao warna",
        ],
    },
    "relationship_manipulation": {
        "severity": Severity.SOFTEN,
        "patterns": [
            r"your (spouse|partner|husband|wife) is cheating",
            r"your marriage will fail",
            r"(he|she|they) (doesn'?t|does not) love you",
            r"you will (never|not) find love",

            # Hinglish
            r"tumhara partner dhokha de raha hai",
            r"shaadi toot jayegi",
            r"kabhi pyaar nahi milega",
            r"woh tumse pyaar nahi karta",
            r"woh tumse pyaar nahi karti",
        ],
    },
    "isolation_pressure": {
        "severity": Severity.BLOCK,
        "patterns": [
            r"don'?t tell (anyone|your family|your friends)",
            r"only i can help you",
            r"(your family|they) (will not|won'?t) understand (this|the remedy)",
            r"keep this (between us|secret)",

            # Hinglish
            r"kisi ko mat batana",
            r"sirf main madad kar sakta hoon",
            r"family nahi samjhegi",
            r"ye baat secret rakho",
        ],
    },
    "fatalistic_hopelessness": {
        "severity": Severity.SOFTEN,
        "patterns": [
            r"nothing can save you",
            r"your suffering is destined",
            r"there is no way out",
            r"you cannot escape your fate",

            # Hinglish
            r"kuch nahi ho sakta",
            r"kismat hi kharab hai",
            r"isse bach nahi sakte",
            r"zindagi bhar dukh rahega",
        ],
    },
    "discriminatory_fate_claims": {
        "severity": Severity.SOFTEN,
        "patterns": [
            r"cursed because you (are|were born)",
            r"(women|men) born under .* are (cursed|doomed|inferior)",
            # Hinglish
            r"janam se shraapit",
            r"ladkiyan .* shraapit",
            r"ladke .* shraapit",
        ],
    },
    "over_specific_predictions": {
        "severity": Severity.REVIEW,
        "patterns": [
            r"on (january|february|march|april|may|june|july|august|september|october|november|december) \d{1,2}(st|nd|rd|th)?,? \d{4}.*(die|death|illness|accident)",
            r"exactly \$?\d+[\d,]* (will|in)",
        ],
    },
}

# Words within this many tokens *before* a match that flip its polarity.
NEGATORS = {
    "not", "no", "never", "won't", "wont", "don't", "dont", "isn't", "isnt",
    "cannot", "can't", "cant", "without", "unlikely",
    # Hinglish
    "nahi",
    "mat",
    "kabhi nahi",
    "bilkul nahi",
}
NEGATION_WINDOW = 4  # how many preceding words to scan for a negator


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class Violation:
    category: str
    severity: Severity
    pattern: str
    matched_text: str
    negated: bool = False


@dataclass
class SafetyResult:
    safe: bool
    violations: list[Violation] = field(default_factory=list)
    needs_review: bool = False
    highest_severity: Optional[Severity] = None

    def to_dict(self) -> dict:
        return {
            "safe": self.safe,
            "needs_review": self.needs_review,
            "highest_severity": self.highest_severity.value if self.highest_severity else None,
            "violations": [
                {
                    "category": v.category,
                    "severity": v.severity.value,
                    "pattern": v.pattern,
                    "matched_text": v.matched_text,
                    "negated": v.negated,
                }
                for v in self.violations
            ],
        }


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("astrology_safety")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)


def _log_event(event_type: str, text: str, result: SafetyResult) -> None:
    """Structured log line — swap this for a DB/analytics write in production."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event_type,
        "text_preview": text[:200],
        "result": result.to_dict(),
    }
    logger.info(json.dumps(record))


# ---------------------------------------------------------------------------
# Negation check
# ---------------------------------------------------------------------------

def _is_negated(text: str, match_start: int) -> bool:
    """Look back a few words before the match for a negator."""
    preceding = text[:match_start]
    words = re.findall(r"[a-z']+", preceding)[-NEGATION_WINDOW:]
    return any(w in NEGATORS for w in words)


# ---------------------------------------------------------------------------
# Core regex pass
# ---------------------------------------------------------------------------

def check_safety(text: str) -> SafetyResult:
    """
    Fast regex-based first pass. Returns a SafetyResult with tiered
    violations and negation already accounted for.
    """
    lowered = text.lower()
    violations: list[Violation] = []

    for category, rule in SAFETY_RULES.items():
        severity = rule["severity"]
        for pattern in rule["patterns"]:
            match = re.search(pattern, lowered)
            if not match:
                continue
            negated = _is_negated(lowered, match.start())
            violations.append(
                Violation(
                    category=category,
                    severity=severity,
                    pattern=pattern,
                    matched_text=match.group(0),
                    negated=negated,
                )
            )
            break  # one hit per category is enough for this pass

    # Only count non-negated violations toward "unsafe"
    active_violations = [v for v in violations if not v.negated]

    highest = None
    if active_violations:
        order = [Severity.BLOCK, Severity.SOFTEN, Severity.REVIEW]
        highest = min(active_violations, key=lambda v: order.index(v.severity)).severity

    result = SafetyResult(
        safe=not any(v.severity in (Severity.BLOCK, Severity.SOFTEN) for v in active_violations),
        violations=violations,
        needs_review=any(v.severity == Severity.REVIEW for v in active_violations),
        highest_severity=highest,
    )

    _log_event("regex_check", text, result)
    return result


# ---------------------------------------------------------------------------
# Classifier fallback (stub)
# ---------------------------------------------------------------------------
#
# Regex will miss paraphrased fear/certainty language. For text that passes
# the regex pass (or lands in REVIEW), route it through a model prompted
# directly with the fuzzy rule, and have it return structured categories.
# This is a stub — wire up an actual API call (e.g. Anthropic Messages API)
# where indicated.

CLASSIFIER_PROMPT = """You are a safety classifier for an Indian astrology app.

The text may be in:
- English
- Hindi
- Hinglish (Hindi written in English script)

Check for:

- death predictions
- illness predictions
- medical certainty
- guaranteed money outcomes
- pressure to buy remedies, pujas, rituals
- fear-based persuasion
- isolation from family/friends
- hopeless or fatalistic language

Be especially careful with Hinglish phrases commonly used in astrology consultations.

Respond ONLY with JSON in this shape, no other text:
{
  "safe": true/false,
  "categories": ["category_name", ...],
  "severity": "block" | "soften" | "review" | "none",
  "reasoning": "one sentence"
}

Text to evaluate:
---
{text}
---
"""


def classify_with_model(text: str) -> Optional[SafetyResult]:
    """
    Stub for an LLM-based fallback classifier. Wire this up to your actual
    model call (Anthropic Messages API, etc.) and parse the JSON response
    into a SafetyResult. Returns None if not implemented/unavailable, in
    which case callers should fall back to the regex-only result.
    """
    # Example of what the real implementation would look like:
    #
    # response = anthropic_client.messages.create(
    #     model="claude-sonnet-4-6",
    #     max_tokens=300,
    #     messages=[{"role": "user", "content": CLASSIFIER_PROMPT.format(text=text)}],
    # )
    # parsed = json.loads(response.content[0].text)
    # severity = Severity(parsed["severity"]) if parsed["severity"] != "none" else None
    # violations = [
    #     Violation(category=c, severity=severity or Severity.REVIEW,
    #               pattern="<model>", matched_text="<model>", negated=False)
    #     for c in parsed["categories"]
    # ]
    # return SafetyResult(safe=parsed["safe"], violations=violations,
    #                      needs_review=False, highest_severity=severity)
    return None


# ---------------------------------------------------------------------------
# Combined check: regex first, classifier fallback for ambiguous cases
# ---------------------------------------------------------------------------

def check_safety_full(text: str, use_classifier_fallback: bool = True) -> SafetyResult:
    """
    Recommended entry point. Runs the regex pass; if the text is clean or
    only REVIEW-tier under regex, optionally double-checks with the model
    classifier to catch paraphrased violations regex would miss.
    """
    regex_result = check_safety(text)

    if not use_classifier_fallback:
        return regex_result

    if regex_result.safe or regex_result.needs_review:
        model_result = classify_with_model(text)
        if model_result is not None:
            _log_event("classifier_check", text, model_result)
            # Merge: if either layer flags it, treat as flagged
            if not model_result.safe:
                return model_result

    return regex_result

# Rewrite — turn flagged text into safer phrasing instead of hard-refusing


REWRITE_TEMPLATES = {
    "death_prediction":
        "Yeh samay kuch bade badlavon ya challenges ka sanket de sakta hai, lekin astrology maut ya life expectancy ki certainty se prediction nahi karti.",

    "medical_claims":
        "Agar health ko lekar concerns hain, toh qualified doctor se consult karna sabse zaroori hai. Astrology medical diagnosis ya treatment ka substitute nahi hai.",

    "financial_guarantees":
        "Financial opportunities dikh sakti hain, lekin actual results aapke decisions, efforts aur circumstances par depend karte hain.",

    "fear_selling":
        "Traditional remedies ya puja-upay kuch logon ke spiritual practices ka hissa ho sakte hain, lekin unke liye kabhi bhi pressure ya urgency create nahi ki jaani chahiye.",

    "relationship_manipulation":
        "Relationships mein challenges ya misunderstandings aa sakti hain, lekin healthy communication aur mutual understanding bahut important hote hain.",

    "isolation_pressure":
        "Important decisions ke baare mein trusted family members, friends ya professionals se baat karna helpful ho sakta hai.",

    "fatalistic_hopelessness":
        "Yeh phase challenging lag sakta hai, lekin astrology tendencies aur possibilities batati hai, fixed destiny nahi.",

    "discriminatory_fate_claims":
        "Astrological placements kisi insaan ki value, worth ya identity define nahi karte.",

    "over_specific_predictions":
        "Astrology exact dates, exact events ya exact outcomes certainty ke saath predict nahi karti.",
}


def rewrite(text: str, result: SafetyResult) -> str:
    """
    BLOCK-tier: replace the whole response with a safe fallback message.
    SOFTEN-tier: append gentle reframing notes for each flagged category.
    REVIEW-tier / safe: return text unchanged.
    """
    active = [v for v in result.violations if not v.negated]
    if not active:
        return text

    if result.highest_severity == Severity.BLOCK:
        return (
            "Main is tarah ki specific prediction provide nahi kar sakta, "
            "kyunki yeh unnecessary fear ya misunderstanding create kar sakti hai. "
            "Astrology ko self-reflection aur guidance ke tool ke roop mein dekhna "
            "zyada useful hota hai, na ki medical, financial ya life-or-death "
            "certainty ke source ke roop mein. Agar aap health, finances ya kisi "
            "serious personal situation ko lekar concerned hain, toh qualified "
            "professionals se guidance lena sabse appropriate rahega."
       )

    if result.highest_severity == Severity.SOFTEN:
        notes = []
        seen = set()
        for v in active:
            if v.category in seen:
                continue
            seen.add(v.category)
            note = REWRITE_TEMPLATES.get(v.category)
            if note:
                notes.append(note)
        if notes:
            return text + "\n\n" + " ".join(notes)

    return text


# Convenience wrapper for app integration

def moderate(text: str, use_classifier_fallback: bool = True) -> dict:
    """
    Single entry point for the app: check, rewrite if needed, and return
    everything the caller needs to decide what to show the user.
    """
    result = check_safety_full(text, use_classifier_fallback=use_classifier_fallback)
    final_text = rewrite(text, result)
    return {
        "original_text": text,
        "final_text": final_text,
        "was_modified": final_text != text,
        **result.to_dict(),
    }

# Quick manual test

if __name__ == "__main__":
    samples = [
        "Saturn's transit suggests you will die early next year.",
        "You will NOT die early — this is a misconception about Saturn transits.",
        "This remedy will cure your illness, buy it urgently before it's too late.",
        "Jupiter brings guaranteed wealth this month if you act now.",
        "Don't tell your family about this puja, only I can help you.",
        "This is a wonderful period for creativity and new relationships.",
        "On January 5th, 2027 you will face a serious accident.",
    ]

    for s in samples:
        out = moderate(s, use_classifier_fallback=False)
        print("-" * 60)
        print("INPUT: ", s)
        print("SAFE:  ", out["safe"])
        print("VIOLATIONS:", [v["category"] for v in out["violations"] if not v["negated"]])
        print("OUTPUT:", out["final_text"])