"""Fact extraction over the book corpus: turns a topic into a cited,
book-grounded research brief (ScriptEngine.research_brief) — the drop-in
replacement for an "internet research" step that brain_integration.py
actually uses.

Trimmed 2026-08-28: this file used to also build full hook/beats/CTA
Shorts scripts directly (extractive or LLM-polished), reachable only via
Brain.script()/.generate() — dead code, nothing in the real pipeline ever
called it. Removed along with the now-unused `brain/llm.py` it depended on.
research_brief()'s own dependency chain (_extract_key_facts, _short_topic,
_format_context, and their helpers) is untouched."""
from __future__ import annotations

import re

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


_URGENT = {
    "lose", "lost", "die", "dies", "dead", "stop", "stops", "fail", "fails",
    "collapse", "gone", "disappear", "without", "nothing", "never", "first",
    "only", "immediately", "hours", "days", "weeks", "survive", "survival",
    "emergency", "crisis", "rebuild", "reinvent", "from scratch", "no longer",
}


class ScriptEngine:
    def __init__(self, retriever: Retriever):
        self.retriever = retriever

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


# ---------------------------------------------------------------------------
# Fact extraction
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
            sent = re.sub(r"^[\*\-•]\s*", "", sent)       # bullet artifact
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
