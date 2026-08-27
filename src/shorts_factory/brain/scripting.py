"""Script engine: turns a topic into a Shorts-ready script.

Two modes:
  * Extractive (default, zero API keys): pulls the strongest facts straight
    from the books and assembles hook -> beats -> CTA with citations.
  * LLM polish (optional): if an OpenAI-compatible model is configured, the
    retrieved book chunks are passed as context and the model writes the
    script. Falls back to extractive mode on any error.
"""
from __future__ import annotations

import re

from . import config
from .llm import llm_available, llm_chat
from .retrieval import Retriever

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

# Question/transition words that shouldn't count as "on-topic" evidence
_WEAK_TOKENS = {
    "what", "when", "where", "why", "how", "would", "could", "should", "will",
    "can", "does", "did", "happen", "happens", "happened", "suddenly",
    "tomorrow", "today", "disappear", "disappears", "disappeared", "vanish",
    "vanished", "vanishes", "without", "with", "your", "you", "and", "the",
    "from", "that", "this", "they", "them", "then", "there", "here", "about",
}


def _content_tokens(text: str) -> set[str]:
    toks = re.findall(r"[a-z0-9']{3,}", text.lower())
    return {t for t in toks if t not in _WEAK_TOKENS}


def _capitalize(s: str) -> str:
    s = (s or "").strip()
    return s[0].upper() + s[1:] if s else s

_URGENT = {
    "lose", "lost", "die", "dies", "dead", "stop", "stops", "fail", "fails",
    "collapse", "gone", "disappear", "without", "nothing", "never", "first",
    "only", "immediately", "hours", "days", "weeks", "survive", "survival",
    "emergency", "crisis", "rebuild", "reinvent", "from scratch", "no longer",
}

_VISUAL_CUES = [
    (["electricity", "grid", "power", "blackout", "generator"], "B-roll: city lights going out, dark skyline"),
    (["water", "drink", "thirst", "well", "filter"], "B-roll: muddy water poured through a cloth filter"),
    (["metal", "iron", "steel", "copper", "furnace", "forge", "smelt"], "B-roll: glowing furnace, sparks, molten metal"),
    (["food", "farm", "crop", "grain", "harvest", "soil"], "B-roll: empty supermarket shelves, then a field of wheat"),
    (["medicine", "disease", "infection", "antibiotic", "surgery", "health"], "B-roll: pills, microscope, hospital corridor"),
    (["fuel", "coal", "oil", "wood", "charcoal", "steam", "engine"], "B-roll: fire, steam, piston engine"),
    (["radio", "communication", "signal", "printing", "paper", "writing"], "B-roll: old radio, printing press, handwriting"),
    (["clock", "time", "navigation", "compass", "map", "star"], "B-roll: compass needle, starry sky, old map"),
    (["glass", "lens", "microscope", "telescope", "optics", "light"], "B-roll: lens being ground, microscope close-up"),
    (["tool", "machine", "lathe", "wheel", "gear", "lever"], "B-roll: gears turning, workshop tools"),
]


class ScriptEngine:
    def __init__(self, retriever: Retriever):
        self.retriever = retriever

    # ------------------------------------------------------------------
    def generate(self, topic: str, duration: int = config.SHORTS_DURATION_SECONDS) -> dict:
        topic = (topic or "").strip()
        results = self.retriever.search(topic, top_k=12)
        context = _format_context(results)

        if llm_available():
            try:
                return self._llm_script(topic, context, results, duration)
            except Exception:
                pass  # fall through to extractive
        return self._extractive_script(topic, results, duration)

    # ------------------------------------------------------------------
    def research_brief(self, topic: str, top_k: int = 10) -> dict:
        """Everything your existing flow needs to write its own script.

        This is the drop-in replacement for an 'internet research' step:
        pass it a topic, get back a cited, book-grounded brief.
        """
        topic = (topic or "").strip()
        results = self.retriever.search(topic, top_k=top_k)
        facts = _extract_key_facts([r.chunk.text for r in results], query=topic, max_facts=12)
        return {
            "topic": topic,
            "key_facts": facts,
            "sources": [r.to_dict() for r in results],
            "full_context": _format_context(results),
            "word_count": sum(len(r.chunk.text.split()) for r in results),
        }

    # ------------------------------------------------------------------
    def _extractive_script(self, topic: str, results, duration: int) -> dict:
        facts = _extract_key_facts([r.chunk.text for r in results], query=topic, max_facts=config.MAX_BEATS)
        hook_fact = _pick_hook_fact(topic, facts)
        hook = _build_hook(topic, facts)
        beat_facts = [f for f in facts if f != hook_fact] or facts
        beats = _build_beats(topic, beat_facts, duration)
        cta = f"Follow for part 2 — how to rebuild {_short_topic(topic)} from nothing."
        total_words = len(hook.split()) + sum(len(b["line"].split()) for b in beats) + len(cta.split())
        return {
            "title": _title_from_topic(topic),
            "hook": hook,
            "beats": beats,
            "cta": cta,
            "total_words": total_words,
            "duration_seconds": duration,
            "mode": "extractive",
            "sources": _dedupe_sources(results[:5]),
        }

    def _llm_script(self, topic: str, context: str, results, duration: int) -> dict:
        prompt = _LLM_SCRIPT_PROMPT.format(topic=topic, context=context[:9000], duration=duration)
        data = llm_chat(prompt, json_mode=True)
        # Normalise whatever the model returned into our schema
        if isinstance(data, str):
            data = _try_json(data) or {}
        if not isinstance(data, dict):
            raise ValueError("LLM did not return a dict")
        beats = data.get("beats") or []
        if not isinstance(beats, list) or not beats:
            raise ValueError("LLM script has no beats")
        return {
            "title": data.get("title") or _title_from_topic(topic),
            "hook": data.get("hook") or _build_hook(topic, []),
            "beats": beats,
            "cta": data.get("cta") or f"Follow for part 2 — how to rebuild {_short_topic(topic)} from nothing.",
            "total_words": sum(len(str(b.get("line", "")).split()) for b in beats),
            "duration_seconds": duration,
            "mode": "llm",
            "sources": _dedupe_sources(results[:5]),
        }


# ---------------------------------------------------------------------------
# Fact extraction (extractive mode)
# ---------------------------------------------------------------------------

_JUNK_FACT_RE = re.compile(
    r"^(figure|table|fig\.|chapter|section|appendix|part)\s+[0-9]|"
    r"section\s+\d+(\.\d+)*|"
    r"^\d+(\.\d+)*\s*:|"
    r"quick brown fox|fc3000|user-serviceable|repair guide|"
    r"let the future tell the truth|example key point|"
    r"table \d+|figure \d+|"
    r"^\d{1,4}\s*$",
    re.IGNORECASE,
)

# A fact must contain a real verb; this drops headings and fragments like
# "Generating electricity using basic mechanical principles."
_HAS_VERB_RE = re.compile(
    r"\b(is|are|was|were|be|been|have|has|had|do|does|did|can|could|will|"
    r"would|should|may|might|make|makes|made|use|uses|used|produce|produces|"
    r"turn|turns|work|works|build|built|need|needs|get|gets|keep|keeps|"
    r"allow|allows|become|becomes|start|starts|stop|stops|fail|fails|"
    r"lose|loses|create|creates|call|called|means|mean|take|takes|give|gives|"
    r"go|goes|die|dies|survive|grow|grows|set|sets|put|puts|require|requires|"
    r"help|helps|provide|provides|carry|carries|conduct|move|moves|run|runs|"
    r"pass|passes|flow|flows|transform|transforms|invent|invents|invented|"
    r"store|stores|transmit|transmits|change|changes|fall|falls|rise|rises|"
    r"drop|drops|hold|holds|come|comes|find|finds|found|know|knows|knew|"
    r"live|lives|stay|stays|wait|waits|return|returns|reach|reaches|leave|"
    r"leaves|enter|enters|kill|kills|save|saves|burn|burns|melt|melts|"
    r"freeze|freezes|boil|boils|cook|cooks|eat|eats|drink|drinks|breathe|"
    r"breathes|walk|walks|see|sees|look|looks|hear|hears|feel|feels|smell|"
    r"smells|taste|tastes|think|thinks|believe|believes|learn|learns|teach|"
    r"teaches|write|writes|read|reads|speak|speaks|say|says|tell|tells)\b",
    re.IGNORECASE,
)


def _extract_key_facts(texts: list[str], query: str = "", max_facts: int = 6) -> list[str]:
    q_tokens = _content_tokens(_short_topic(query)) or _content_tokens(query)
    candidates: list[tuple[float, str]] = []
    seen: set[str] = set()

    for text in texts:
        for sent in _SENT_SPLIT.split(text):
            sent = re.sub(r"\s+", " ", sent).strip()
            sent = re.sub(r"^(answer|question)\s*:?\s*", "", sent, flags=re.IGNORECASE)
            sent = re.sub(r"^[\*\-\u2022]\s*", "", sent)       # bullet artifact
            sent = re.sub(r"^[A-Za-z]{1,3}\(\d{1,3}\)\s*", "", sent)  # figure-ref artifact e.g. Rs(8)
            sent = re.sub(r"^\d{1,4}[.)]\s*", "", sent)        # list/page-number artifact
            sent = re.sub(r"^(?:\d{1,4}\s+)+", "", sent)      # one or more page-number artifacts
            # OCR heading runs like "SOAP FIRSTTHINGS FIRSTTHINGS SOAP text..."
            sent = re.sub(r"^(?:[A-Z0-9&]{3,}\s+){2,}", "", sent)
            # A single long glued heading word like "SOAPPREPARATION text..."
            sent = re.sub(r"^[A-Z0-9&]{8,}\s+", "", sent)
            if len(sent) < 35 or len(sent) > 220:
                continue
            if _JUNK_FACT_RE.search(sent):
                continue
            if not _HAS_VERB_RE.search(sent):
                continue
            # Skip ALL-CAPS headings that slipped through
            letters = re.findall(r"[A-Za-z]", sent)
            if len(sent) > 20 and letters and sum(c.isupper() for c in letters) / len(letters) > 0.7:
                continue
            # Drop sentence fragments created by chunk boundaries unless they
            # carry a query term (too valuable to lose).
            starts_caps = bool(re.match(r"^[A-Z\"']", sent))
            if not starts_caps and not any(t in sent.lower() for t in q_tokens):
                continue
            key = re.sub(r"[^a-z0-9]", "", sent.lower())[:120]
            if key in seen:
                continue
            seen.add(key)
            score = _score_sentence(sent, q_tokens)
            candidates.append((score, sent))

    candidates.sort(key=lambda x: x[0], reverse=True)
    # Keep facts diverse: avoid near-duplicate sentences.
    picked: list[str] = []
    for _, sent in candidates:
        if all(_jaccard(sent, p) < 0.55 for p in picked):
            picked.append(sent)
        if len(picked) >= max_facts:
            break
    return picked


def _score_sentence(sent: str, q_tokens: set[str]) -> float:
    s = sent.lower()
    score = 0.0
    if re.search(r"\d", sent):
        score += 0.6
    if "key point" in s:
        score += 1.5
    score += sum(1.0 for w in _URGENT if w in s)
    # On-topic words matter most: prefer sentences that use the query terms.
    score += sum(2.0 for t in q_tokens if t in s)
    # Slight preference for shorter, punchier lines
    score += max(0.0, (160 - len(sent)) / 400.0)
    return score


def _jaccard(a: str, b: str) -> float:
    A = set(re.findall(r"[a-z0-9']+", a.lower()))
    B = set(re.findall(r"[a-z0-9']+", b.lower()))
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


# ---------------------------------------------------------------------------
# Assembly helpers
# ---------------------------------------------------------------------------

def _pick_hook_fact(topic: str, facts: list[str]) -> str | None:
    if not facts:
        return None
    short = _short_topic(topic)
    topic_tokens = _content_tokens(short)
    # Prefer punchy, on-topic facts for the hook (not fragments/headings)
    hook_pool = [f for f in facts if 60 <= len(f) <= 200] or facts
    return max(
        hook_pool,
        key=lambda f: _score_sentence(f, _content_tokens(topic))
        + (2.0 if any(t in f.lower() for t in topic_tokens) else 0.0),
    )


def _build_hook(topic: str, facts: list[str]) -> str:
    short = _short_topic(topic)
    if facts:
        best = _pick_hook_fact(topic, facts)
        if re.match(r"^(how|which)", topic.strip(), re.IGNORECASE):
            return f"Here's what the books say about {short}. {_capitalize(best)}"
        return f"Here's what actually happens when {short} disappears. {_capitalize(best)}"
    return f"What would you do if {short} disappeared tomorrow? The answer is worse than you think."


def _build_beats(topic: str, facts: list[str], duration: int) -> list[dict]:
    if not facts:
        facts = [
            f"Without {_short_topic(topic)}, every system you rely on starts failing within hours.",
            f"Survival depends on resourcefulness and adaptability during immediate crises.",
            f"Rebuilding starts with the simplest tools and scales up one invention at a time.",
        ]
    hook_secs = 3
    cta_secs = 4
    body_secs = max(duration - hook_secs - cta_secs, len(facts) * 5)
    words = [len(f.split()) for f in facts]
    total = sum(words) or 1
    beats = []
    t = hook_secs
    for fact in facts:
        beat_secs = max(4.0, body_secs * words[len(beats)] / total)
        beats.append({
            "time": round(t, 1),
            "duration": round(beat_secs, 1),
            "visual": _visual_cue(fact, topic),
            "line": _capitalize(fact),
        })
        t += beat_secs
    return beats


def _visual_cue(text: str, topic: str) -> str:
    s = (text + " " + topic).lower()
    for kws, cue in _VISUAL_CUES:
        if any(k in s for k in kws):
            return cue
    return "B-roll: slow pan over the topic, kinetic text overlay"


def _title_from_topic(topic: str) -> str:
    t = topic.strip()
    # "How..." / "Which..." prompts are already good titles.
    if re.match(r"^(how|which)", t, re.IGNORECASE):
        return t if t.endswith("?") else f"{t}?"
    short = _short_topic(t)
    return f"What happens if {short} disappears tomorrow?"


def _short_topic(topic: str) -> str:
    """Reduce a natural-language prompt to its core subject.

    'What would happen if electricity suddenly disappeared tomorrow?'
    -> 'electricity'
    'How would you build a furnace without modern materials?' -> 'furnace'
    'How could humanity start producing metal again?' -> 'metal'
    """
    t = topic.strip().rstrip("?.!")
    t = re.sub(r"\s*&\s*", " and ", t)
    t = re.sub(
        r"^(what would happen if|what happens if|what happened if|what if|"
        r"how to rebuild|how do you rebuild|the first 24 hours without)\s+",
        "",
        t,
        flags=re.IGNORECASE,
    )
    # "How would you build a furnace..." -> "a furnace..."
    t = re.sub(
        r"^(how\s+(?:would|do|can|could)\s+(?:you|we|humanity|i)?\s*"
        r"(?:start\s+)?(?:build|make|produce|producing|making|building|"
        r"rebuild|reinvent|create|creating|get|obtain|smelt|forge)\s+)",
        "",
        t,
        flags=re.IGNORECASE,
    )
    if re.match(r"^which technologies?", t, re.IGNORECASE):
        return "technology"
    # Strip constraint/tail phrases, keeping the subject
    t = re.sub(r"\s+without\s+.*$", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+from scratch$", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+again$", "", t, flags=re.IGNORECASE)
    # Strip the "disappears tomorrow" tail, keeping the subject
    t = re.sub(
        r"\s+(?:suddenly|completely|totally|overnight|all)?\s*"
        r"(?:disappears?|disappeared|vanishes?|vanished|stops? working|went away)"
        r"(?:\s+(?:tomorrow|today|right now|overnight|forever|completely))?\s*$",
        "",
        t,
        flags=re.IGNORECASE,
    )
    t = re.sub(r"^(a|an|the)\s+", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t).strip()
    return t or topic.strip()


def _format_context(results) -> str:
    parts = []
    for r in results:
        parts.append(f"[{r.chunk.book}, pages {r.chunk.pages}]\n{r.chunk.text}")
    return "\n\n".join(parts)


def _dedupe_sources(results) -> list[dict]:
    seen: set[tuple] = set()
    out = []
    for r in results:
        key = (r.chunk.book, r.chunk.pages)
        if key in seen:
            continue
        seen.add(key)
        out.append(r.to_dict())
    return out


def _try_json(text: str):
    import json
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        return None


_LLM_SCRIPT_PROMPT = """You write YouTube Shorts scripts for a faceless history/science channel.
The ONLY source of facts is the book excerpts below. Do not invent facts outside them.
Topic: {topic}
Target length: {duration} seconds (~2.5 words/second).

Book excerpts:
{context}

Return strict JSON with this schema:
{{
  "title": "short clickable title",
  "hook": "one sentence, first 3 seconds, high curiosity",
  "beats": [
    {{"time": 3.0, "duration": 8.0, "visual": "b-roll cue", "line": "spoken line grounded in the excerpts"}}
  ],
  "cta": "follow/comment call to action"
}}
Keep every spoken line grounded in the excerpts. 4-6 beats."""
