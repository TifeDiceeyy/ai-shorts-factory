"""LLM provider interface for script generation.

Phase 0 ships one implementation: StubLLMProvider — deterministic, local,
zero network, zero cost. It still goes through CostTracker.check_budget()
and .record() so the budget-guard wiring is exercised and tested even
though the actual cost is $0. A real provider (Claude, OpenAI, ...) plugs
in behind the same generate_script() signature once LLM_PROVIDER is set to
something other than "stub" and a credential is present — CLAUDE.md §0
rule says don't make that paid call until approved, so it isn't wired yet.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import json
import re
from typing import Any

from ..cost_tracker import CostTracker
from .fal import FalGateway

TARGET_TOTAL_SECONDS = 45.0
# Measured on a real generated video 2026-09-01: 179 narration words rendered
# to 57.9s of audio (with NARRATION_SPEED_FACTOR already applied) = 3.09
# words/sec. Used to turn a scene's duration budget into a word budget the
# model can actually act on.
# Re-measured 2026-09-01 by least-squares across 11 real generated videos
# (4-15 scenes): total audio = 0.3219s per word + 0.426s per scene, i.e. a
# true speaking rate of 3.11 words/sec plus a FIXED per-scene overhead. The
# overhead is the head/tail silence each separately-synthesized scene
# carries, so it scales with scene COUNT, not with word count.
NARRATION_WORDS_PER_SECOND = 3.1
# Budgeting words as duration*rate with no overhead term is what pushed the
# first 15-scene video to 50.6s against a 45s target and a 50s hard ceiling:
# 15 scenes x 0.43s = 6.4s that the budget never accounted for, which is
# essentially the entire overshoot. At 6 scenes the same omission is only
# 2.6s and stayed inside the window, which is why it went unnoticed until
# scene count trebled.
SCENE_AUDIO_OVERHEAD_SECONDS = 0.43
# Lowered 3.0 -> 2.5 alongside the 15-scene change: 15 x 3.0 lands on
# exactly 45s with zero slack, so any per-scene rounding pushed the
# total out of the 40-50s window and failed the whole script. 2.5 gives
# the model room to vary pacing while still fitting.
MIN_SCENE_SECONDS = 2.5
MAX_SCENE_SECONDS = 9.5

# Cycled by scene index for basic visual variety — a template default, not a
# creative decision. A real LLM/storyboard artist would choose these
# per-shot instead of round-robin.
CAMERA_TEMPLATES = [
    "static wide shot",
    "slow push-in",
    "close-up on hands/detail",
    "overhead top-down",
    "handheld tracking shot",
]

# Cycled by scene index, same pattern as CAMERA_TEMPLATES above. Mirrors the
# scene_type values the real LLM provider is instructed to emit on every
# scene (see the FalLLMProvider system prompt) — without this, stub-driven
# tests exercise a script shape (no scene_type at all) that real production
# scripts never have, which let the mascot hero-image-reuse path (keyed off
# scene_type, see pipeline.get_scene_image_prompt) go completely untested by
# the free/local test suite.
SCENE_TYPE_TEMPLATES = [
    "mascot_reaction",
    "ingredient_grid",
    "process_action",
    "mascot_reaction",
    "split_canvas",
]


class LLMProvider(ABC):
    name: str

    @abstractmethod
    def generate_script(
        self,
        brief: dict[str, Any],
        language: str,
        visual_style: str,
        cost_tracker: CostTracker,
    ) -> dict[str, Any]:
        ...
    @abstractmethod
    def propose_topic(self, topic: str, cost_tracker: CostTracker) -> dict[str, Any]:
        """Propose a safety classification + Phase 1 retrieval config for a
        brand-new topic name the registry (topic_registry.py) has never seen.
        Returns {safety_class, reasoning, queries, keywords, caution}.

        This is advisory only — never trusted alone. The caller (the
        Telegram bot's new-topic flow) must still re-check the topic against
        safety.RED_TOPICS/RED_KEYWORDS itself, require an explicit human
        confirmation, and only then persist via topic_registry.register_topic
        (which itself refuses to persist a "red" entry)."""
        ...

    @abstractmethod
    def design_mascot(self, topic: str, brief: dict[str, Any] | None, cost_tracker: CostTracker) -> dict[str, Any]:
        """Design a brand-new mascot character for a topic that doesn't fit
        any of the 5 registered mascots (mascots.select_mascot_for_story()'s
        no-match fallback). Returns {name, short_desc, hero_prompt,
        visual_style, motion_instruction, scene_role_template, keywords}
        matching mascots.Mascot's fields (minus id, which the caller
        assigns) plus a `keywords` list so future similar topics can find
        and reuse this same mascot via the normal keyword-scoring path
        instead of generating yet another one."""
        ...


class LLMResponseFormatError(ValueError):
    """The provider answered, but its script payload was not usable JSON."""


def _brief_target_seconds(brief: dict[str, Any]) -> float:
    """The finished length this brief is asking for.

    Falls back to TARGET_TOTAL_SECONDS so every existing caller — and every
    brief written before target_seconds existed — keeps the 45s behaviour
    exactly. Clamped to the same 15-180s range the brief schema allows, so a
    malformed value can't produce a one-scene or twenty-minute script.
    """
    raw = brief.get("target_seconds")
    if not isinstance(raw, (int, float)) or raw <= 0:
        return TARGET_TOTAL_SECONDS
    return float(min(180.0, max(15.0, raw)))


def _caption_from_claim(claim_text: str) -> str:
    """Short on-screen caption derived from a claim (schema caps at 90 chars)."""
    cap = claim_text.strip()
    if len(cap) <= 90:
        return cap
    cut = cap[:87].rsplit(" ", 1)[0]
    return cut + "..."


SCRIPT_TOP_LEVEL_KEYS = {"topic", "language", "visual_style", "scenes"}
SCRIPT_SCENE_KEYS = {
    "narration",
    "caption",
    "duration",
    "visual_prompt",
    "source_claim_id",
    "camera",
    "sfx",
    "mascot_role",
    "mascot_emotion",
    "action",
    "props",
    "layout",
    "scene_type",
    "fx",
    "motion",
}


def _normalize_generated_script(
    payload: dict[str, Any], brief: dict[str, Any], language: str, visual_style: str
) -> dict[str, Any]:
    """Repair harmless formatting drift in an otherwise usable LLM script.

    The JSON Schema intentionally rejects unknown properties, but generative
    models occasionally add explanatory keys even when asked not to. Those
    keys have no downstream meaning, so discard them rather than charging for
    an entire second script call. Keep substantive validation strict: missing
    required fields, invalid claim IDs, bad types, and unsafe durations still
    fail in schema_validate.validate_script_against_brief().
    """
    # Some models add one redundant wrapper despite being asked for the script
    # object directly. Unwrap only this unambiguous shape.
    if set(payload) == {"script"} and isinstance(payload.get("script"), dict):
        payload = payload["script"]

    normalized = {key: value for key, value in payload.items() if key in SCRIPT_TOP_LEVEL_KEYS}
    # These values are authoritative inputs, not creative output. Prevent a
    # model typo or paraphrase from changing artifact identity or render style.
    normalized["topic"] = brief["topic"]
    normalized["language"] = language
    normalized["visual_style"] = visual_style

    scenes = payload.get("scenes")
    if not isinstance(scenes, list):
        # Preserve the invalid value so the schema produces the useful error.
        normalized["scenes"] = scenes
        return normalized

    normalized_scenes = []
    for scene in scenes:
        if not isinstance(scene, dict):
            normalized_scenes.append(scene)
            continue
        clean = {key: value for key, value in scene.items() if key in SCRIPT_SCENE_KEYS}

        caption = clean.get("caption")
        if isinstance(caption, str) and len(caption) > 90:
            clean["caption"] = _caption_from_claim(caption)

        # Accept a plain numeric JSON string such as "7.5". Descriptive
        # values like "about eight seconds" remain invalid and fail closed.
        duration = clean.get("duration")
        if isinstance(duration, str) and re.fullmatch(r"\s*\d+(?:\.\d+)?\s*", duration):
            clean["duration"] = float(duration)

        normalized_scenes.append(clean)

    normalized["scenes"] = normalized_scenes
    _fit_script_duration(normalized_scenes, brief)
    return normalized



def _fit_script_duration(scenes: list[dict[str, Any]], brief: dict[str, Any] | None = None) -> None:
    """Rescale scene durations so the total lands inside the accepted window.

    Real case 2026-09-01: a good paid 15-scene script came back totalling
    50.50s and the strict validator threw the WHOLE thing away for being 0.5s
    over, falling back to the deterministic stub. Hitting a 40-50s window
    exactly is far harder across 15 scenes than across 6, and the durations
    are only a pacing budget in the first place — every scene's real length
    is overridden downstream by its MEASURED TTS audio (see
    pipeline's actual_durations). So scale to fit rather than reject.

    Scale, clamp to the per-scene bounds, then scale once more, because
    clamping can itself push the total back out.
    """
    from ..schema_validate import script_duration_window

    SCRIPT_MIN_TOTAL_SECONDS, SCRIPT_MAX_TOTAL_SECONDS = script_duration_window(brief)

    usable = [s for s in scenes if isinstance(s.get("duration"), (int, float))]
    if not usable:
        return
    # Only rescale when the window is actually REACHABLE with this many
    # scenes. A 1-scene 8s script can never reach 40s (MAX_SCENE_SECONDS caps
    # it at 9.5), so scaling would mangle the durations without fixing
    # anything and would bury the real problem — let validation report
    # "total 8s is outside the window" instead of silently rewriting data.
    if not (
        len(usable) * MIN_SCENE_SECONDS <= SCRIPT_MAX_TOTAL_SECONDS
        and len(usable) * MAX_SCENE_SECONDS >= SCRIPT_MIN_TOTAL_SECONDS
    ):
        return
    target = (SCRIPT_MIN_TOTAL_SECONDS + SCRIPT_MAX_TOTAL_SECONDS) / 2
    for _ in range(2):
        total = sum(float(s["duration"]) for s in usable)
        if not total:
            return
        if SCRIPT_MIN_TOTAL_SECONDS <= total <= SCRIPT_MAX_TOTAL_SECONDS:
            return
        scale = target / total
        for scene in usable:
            scaled = float(scene["duration"]) * scale
            scene["duration"] = round(
                max(MIN_SCENE_SECONDS, min(MAX_SCENE_SECONDS, scaled)), 2
            )


def _visual_prompt(claim_text: str, visual_style: str) -> str:
    return f"{visual_style}: {claim_text}"


class StubLLMProvider(LLMProvider):
    """Deterministic template generator standing in for a real LLM call.
    Turns each brief claim into exactly one scene, scaled so total duration
    lands in the 40-50s window CLAUDE.md requires for a Short."""

    name = "stub"

    def generate_script(
        self,
        brief: dict[str, Any],
        language: str,
        visual_style: str,
        cost_tracker: CostTracker,
    ) -> dict[str, Any]:
        operation = "llm.generate_script"
        cost_tracker.check_budget(operation, estimated_cost_usd=0.0)

        claims = brief["claims"]
        chosen_hook = (brief.get("chosen_hook") or "").strip()

        raw_durations = []
        for i, c in enumerate(claims):
            word_count = len(c["claim"].split())
            if i == 0 and chosen_hook:
                word_count += len(chosen_hook.split())
            raw = 2.5 + 0.35 * word_count
            raw_durations.append(max(MIN_SCENE_SECONDS, min(MAX_SCENE_SECONDS, raw)))

        raw_total = sum(raw_durations)
        target_total = _brief_target_seconds(brief)
        scale = target_total / raw_total if raw_total else 1.0
        durations = [round(d * scale, 1) for d in raw_durations]

        scenes = []
        for i, (claim, duration) in enumerate(zip(claims, durations)):
            narration = claim["claim"]
            caption = _caption_from_claim(claim["claim"])
            if i == 0 and chosen_hook:
                # Honor a caller-supplied hook (brief's idea.chosen_hook) as
                # the opening line, without dropping the claim it's paired
                # with — the extra words were already folded into this
                # scene's duration above, before the total got scaled.
                narration = f"{chosen_hook} {narration}"
                caption = _caption_from_claim(chosen_hook)
            scenes.append(
                {
                    "narration": narration,
                    "caption": caption,
                    "duration": duration,
                    "visual_prompt": _visual_prompt(claim["claim"], visual_style),
                    "source_claim_id": claim["id"],
                    "camera": CAMERA_TEMPLATES[i % len(CAMERA_TEMPLATES)],
                    "sfx": None,
                    "scene_type": SCENE_TYPE_TEMPLATES[i % len(SCENE_TYPE_TEMPLATES)],
                    "action": f"physically demonstrating: {claim['claim']}",
                }
            )

        script = {
            "topic": brief["topic"],
            "language": language,
            "visual_style": visual_style,
            "scenes": scenes,
        }

        cost_tracker.record(
            provider=self.name,
            operation=operation,
            estimated_cost_usd=0.0,
            actual_cost_usd=0.0,
            is_stub=True,
        )
        return script

    def propose_topic(self, topic: str, cost_tracker: CostTracker) -> dict[str, Any]:
        operation = "llm.propose_topic"
        cost_tracker.check_budget(operation, estimated_cost_usd=0.0)
        words = [w for w in re.findall(r"[a-zA-Z]+", topic.lower()) if len(w) > 2]
        proposal = {
            "safety_class": "green",
            "reasoning": "Deterministic stub proposal — not a real safety assessment.",
            "queries": [f"{topic} history and how it works", f"{topic} explained reliable source"],
            "keywords": words[:8] or [topic.lower()],
            "caution": None,
        }
        cost_tracker.record(
            provider=self.name,
            operation=operation,
            estimated_cost_usd=0.0,
            actual_cost_usd=0.0,
            is_stub=True,
        )
        return proposal

    def design_mascot(self, topic: str, brief: dict[str, Any] | None, cost_tracker: CostTracker) -> dict[str, Any]:
        operation = "llm.design_mascot"
        cost_tracker.check_budget(operation, estimated_cost_usd=0.0)
        words = [w for w in re.findall(r"[a-zA-Z]+", topic.lower()) if len(w) > 2]
        keywords = words[:6] or [topic.lower()]
        design = {
            "name": f"Custom Mascot: {topic.title()} Specialist",
            "short_desc": f"Deterministic stub mascot generated for topic {topic!r} — not a real design.",
            "hero_prompt": (
                f"Full-body FLAT 2D cel-shaded cartoon mascot character themed around {topic}, standing in a "
                "friendly explanatory pose facing camera, one hand gesturing forward with open palm. Bold black "
                "ink outlines, flat clean cel shading — NOT a 3D render, NOT CGI, NOT photoreal. Small and "
                "centered vertically in frame, occupying no more than 28% of vertical height, with generous "
                "empty white space above, below, and on both sides — must NOT dominate the frame. "
                "Fully clothed; no bare skin visible except face, forearms, and calves; do not depict "
                "shirtless or undressed. Stark pure solid white background (#FFFFFF) only, zero background "
                "details, zero floor shadows, sticker framing."
            ),
            "visual_style": (
                f"Flat 2D cel-shaded cartoon illustration of a {topic}-themed mascot, bold black ink outlines, "
                "expressive flat-shaded cartoon features — explicitly NOT a 3D render, NOT CGI, NOT photoreal. "
                "Stark pure solid white background (#FFFFFF) with zero scenery, zero shadows, "
                "clean sticker framing. Do not render any text, words, letters, labels, or signs."
            ),
            "motion_instruction": (
                f"For every scene's `visual_prompt`, describe the {topic} mascot's dynamic physical ACTION "
                "and EMOTIONS. Keep the character centered against the clean solid white background."
            ),
            "scene_role_template": (
                f"Cast the {topic} mascot into a narrative role suited to each scene beat: "
                "Hook=curious reaction to the topic; Discovery=examining raw materials; "
                "Process=hands-on demonstration; Challenge=testing/refining; Payoff=proud presentation "
                "of the result."
            ),
            "keywords": keywords,
        }
        cost_tracker.record(
            provider=self.name,
            operation=operation,
            estimated_cost_usd=0.0,
            actual_cost_usd=0.0,
            is_stub=True,
        )
        return design


def _strip_json_trailing_commas(text: str) -> str:
    """Remove commas before ``}``/``]`` while leaving string content intact."""
    out: list[str] = []
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            out.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            out.append(char)
            continue
        if char == ",":
            lookahead = index + 1
            while lookahead < len(text) and text[lookahead].isspace():
                lookahead += 1
            if lookahead < len(text) and text[lookahead] in "}]":
                continue
        out.append(char)
    return "".join(out)


def _json_object(text: str) -> dict[str, Any]:
    """Decode a provider response without accepting prose around the object."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError as first_error:
        # A trailing comma is a common model formatting slip and has one
        # unambiguous repair. Do not attempt broader rewriting that could
        # change factual content or citation IDs.
        repaired = _strip_json_trailing_commas(cleaned)
        if repaired == cleaned:
            raise LLMResponseFormatError(f"LLM response was not valid JSON: {first_error}") from first_error
        try:
            value = json.loads(repaired)
        except json.JSONDecodeError as repaired_error:
            raise LLMResponseFormatError(
                f"LLM response was not valid JSON after trailing-comma repair: {repaired_error}"
            ) from repaired_error
    if not isinstance(value, dict):
        raise LLMResponseFormatError("LLM response must be a JSON object")
    return value


def _script_prompt(brief: dict[str, Any], language: str, visual_style: str) -> str:
    idea_instruction = ""
    if brief.get("concept") or brief.get("chosen_hook"):
        idea_instruction = (
            " A human already picked a specific angle for this video during planning — honor it: "
            f"concept={brief.get('concept', '')!r}, angle={brief.get('angle', '')!r}, "
            f"payoff={brief.get('payoff', '')!r}. The first scene's narration must open with (or very "
            f"closely match) this exact hook line: {brief.get('chosen_hook', '')!r}. Every other scene "
            "must still frame its claim through this same angle, not just default to a generic explainer."
        )
    n_claims = len(brief.get("claims") or []) or 1
    target_total = _brief_target_seconds(brief)
    avg_duration = target_total / n_claims
    # Only the part of a scene that is actually SPEECH converts to words —
    # the rest is the fixed per-scene head/tail silence.
    speaking_seconds = max(0.6, avg_duration - SCENE_AUDIO_OVERHEAD_SECONDS)
    target_words = max(5, round(speaking_seconds * NARRATION_WORDS_PER_SECOND))
    min_words, max_words = max(4, target_words - 1), target_words + 1
    duration_instruction = (
        f" There are {n_claims} claims, so {n_claims} scenes. The SUM of every scene's duration must land "
        f"between {target_total * 0.9:.0f} and {target_total * 1.1:.0f} seconds total — with {n_claims} "
        f"scenes that means roughly {avg_duration:.1f} seconds per scene on average. "
        # A duration in seconds turned out to be something the model cannot
        # map to a length it actually writes: measured 2026-09-01 on a real
        # 15-scene script, narration ran 179 words against a 49s script and
        # the finished video came out 57.9s. A WORD budget is directly
        # actionable in a way "3.0 seconds" is not.
        f"CRITICAL: every scene's narration must be between {min_words} and {max_words} words — count them. "
        f"At {NARRATION_WORDS_PER_SECOND:.1f} spoken words per second that is what actually fits the scene. "
        # A bare ceiling was read as "stay well under": a real 2026-09-01
        # script asked for "at most 9" came back averaging 6.7 words and
        # would have rendered ~34s, under the 40s floor. A RANGE is needed —
        # too few words undershoots just as badly as too many overrun.
        f"Aim for {target_words}. Going UNDER {min_words} is just as wrong as going over {max_words}: too "
        "few words makes the video too short. Short punchy lines, but fill the line."
    )
    order_instruction = (
        " Order the scenes as a logical narrative — setup/context first, then process/mechanism in the "
        "order it actually happens, ending on the payoff or a memorable closing fact — not just the order "
        "the claims happen to be listed in below."
    )
    tone_instruction = (
        " Write every narration line the way a real person would SAY it out loud to a friend, not the way "
        "a textbook would print it. Short sentences. Contractions. No dense multi-clause chemistry-textbook "
        "sentences (e.g. avoid constructions like 'X, a strong base, does Y to the Z holding W together in "
        "V' — say it as two short plain sentences instead). "
        "HIGH ENERGY, not flat or monotone: write with real excitement and urgency, like a creator who's "
        "genuinely hyped to show you this, not a narrator reading a report. Favor short punchy exclamations "
        "and curiosity-driving phrasing ('Here's the wild part —', 'This is where it gets crazy —', 'You "
        "will not believe this —') over neutral, even-toned statements. Vary sentence rhythm — mix quick "
        "punchy fragments with normal sentences, don't let every line fall into the same flat cadence. "
        "Still 100% bound to the supplied claims; rephrase for energy and how it sounds spoken, never add "
        "or soften the underlying fact."
    )
    visual_instruction = (
        " SCENE-ADAPTIVE MASCOT & STORYBOARD ENGINE: "
        "The video follows a single unified Mascot Template & flat cartoon illustration art style "
        "(flat cel-shaded 2D cartoon character, bold black ink outlines, expressive features, "
        "stark pure solid white background #FFFFFF only, zero scenery/floors, sticker framing — "
        "NOT a 3D render, NOT photoreal, NOT glossy CGI). "
        "Each scene in the script should use one of the 4 core scene archetypes: "
        "1) Mascot Reaction / Hazard (scene_type='mascot_reaction', layout='centered'): "
        "Mascot is CENTERED vertically (~55-60% height) with extreme facial reaction (alarm, shock, gasp, caution) "
        "and contextual visual FX (e.g., green toxic chemical smoke billowing, steam, electric sparks, flames). "
        "2) Ingredient / Recipe Breakdown Grid (scene_type='ingredient_grid', layout='centered'): "
        "Clean 2x2 or 3-item grid displaying the raw ingredients as isolated flat cartoon-illustration stickers on pure white "
        "(e.g., Limestone rock, Volcanic Ash powder, Water in bronze pot, Gravel pile) — drawn, not rendered. "
        "3) Dynamic Process Action (scene_type='process_action', layout='centered'): "
        "Close-up demonstration of the physical step in action, drawn in the same flat cartoon style (e.g. clay bowl pouring thick wet slurry into a clamped wooden casting mold, paddle stirring cauldron). "
        "4) Split-Canvas Explainer (scene_type='split_canvas', layout='split_bottom_left' or 'split_bottom_right'): "
        "Mascot scales down to ~35-40% height in bottom corner, pointing enthusiastically UP at a large floating flat-illustrated prop/diagram in the opposite top quadrant. "
        "Every scene's visual_prompt must describe what is on screen, the exact action, emotions, and visual FX. "
        "Every concrete object or material that scene's OWN narration actually names must also appear in that "
        "scene's props (or action) field — never describe something in narration that the image itself won't "
        "show; the viewer should be able to see whatever the voiceover is naming at that moment. "
        "Do not describe or request any text, words, letters, labels, or signs in visual_prompt — captions are added separately. "
        "Include sfx (e.g. 'pop', 'whoosh', 'sizzle', 'ding', 'bubbling', 'splash', or null) for the audio beat. "
        "PACING & MOTION (every video, this pipeline animates each scene as a short continuous clip): scene 1 "
        "should center on the mascot alone — empty white space around it is fine, do not open on a crowded "
        "ingredient dump. In later scenes, introduce props/ingredients one at a time, matching the order "
        "they're actually named in that scene's own narration, rather than describing every item as already "
        "present in the opening frame — an ingredient_grid scene is still fine for showing several items "
        "together, but describe each as its own named item, not a static pile. The mascot itself should read "
        "as calm and settled rather than constantly bouncing: describe purposeful gestures (pointing, "
        "holding, nodding, a thoughtful stroke of the chin) rather than energetic hopping, bobbing, or "
        "repeated bounce. Any single prop or object should be introduced once (a brief entrance), not "
        "described as repeatedly popping in and out."
    )
    return (
        "Create a factual YouTube Short script lasting 40-50 seconds. Return JSON only. "
        "Use only the supplied claims; do not add factual claims. Make the first scene a strong hook."
        + idea_instruction
        + duration_instruction
        + order_instruction
        + tone_instruction
        + " Every scene must contain narration, caption (max 90 characters), duration (3-9.5 seconds), "
        "visual_prompt, source_claim_id, camera, sfx (string or null), mascot_role (string), "
        "mascot_emotion (string), action (string: the physical action/verb-phrase happening on screen this "
        "scene, e.g. 'pouring the lye solution into the melted fat' or 'stirring the mixture with a wooden "
        "paddle' — for process_action scenes this is the main content of the shot, not just a list of props), "
        "props (string or null), layout (string: 'centered', 'split_bottom_left', or 'split_bottom_right'), "
        "scene_type (string: 'mascot_reaction', 'ingredient_grid', 'process_action', or 'split_canvas'), fx (string or null), "
        "and motion (string). "
        # The motion directive is what makes image-to-video actually animate.
        # Measured 2026-09-01: the previous approach matched a scene against 8
        # generic keyword categories (smoke/water/fire/...) and handed Kling
        # vague text like "steam gently rising", which described nothing
        # actually in the shot — so clips came back frozen for their entire
        # duration. The model writing this already knows the topic, the claim,
        # the props and the action, so it is the only place that can say what
        # specifically should move.
        "MOTION is a directive for an image-to-video model animating THIS shot. Name each thing visible in "
        "the shot and say exactly how it moves, in physical terms. Be concrete and specific to this topic — "
        "never generic. Follow the observed grammar of this format: a CHARACTER acts with its body (steps, "
        "gestures, leans, reacts, head turns) and its mouth stays closed; CONTAINERS and equipment carry "
        "continuous action (liquid pours in a steady stream, a paddle stirs, a mixture bubbles, smoke curls "
        "upward); an INGREDIENT or label that simply appears does NOT travel across frame — it settles into "
        "place and stays. Say what stays still too. Example shape: 'The legionary leans in and taps the "
        "stone slab twice with his stick; grey mortar slumps slowly off the trowel; dust drifts down. The "
        "columns behind stay completely still.' "
        "The top-level object must contain exactly topic, language, visual_style, and scenes, with no other keys. "
        "Each scene must contain only the scene fields listed above; never add helper, reasoning, notes, index, "
        "or metadata fields. Every scene must use exactly one supplied source_claim_id; source_claim_id must "
        "never be null and must never be invented. The opening hook must share a scene with, and lead directly "
        "into, the factual claim identified by that scene's source_claim_id. Preserve claim IDs exactly."
        + visual_instruction
        + f" Language: {language}. Visual style: {visual_style}. Brief: {json.dumps(brief, ensure_ascii=False)}"
    )


def _topic_proposal_prompt(topic: str) -> str:
    return (
        "You are proposing whether a NEW topic is safe to produce as a short, "
        "source-backed educational video, and what to search for to research it. "
        "Return JSON only, no prose, no markdown fences. The object must contain exactly: "
        'safety_class (one of "green", "yellow", "red"), reasoning (one sentence), '
        "queries (a list of 2 short web search query strings for finding reliable sources "
        "on this topic), keywords (a list of 5-8 lowercase single-word or short-phrase "
        "keywords for verifying a retrieved source actually discusses this topic), and "
        'caution (a one-sentence safety caution string if safety_class is "yellow", else null). '
        "Classify red if the topic could plausibly provide actionable instructions for weapons, "
        "explosives, illegal drugs, or serious bodily harm. Classify yellow if it involves a real "
        "but manageable hazard (heat, caustic chemicals, electricity, food safety). Otherwise green. "
        f"Topic: {topic!r}"
    )


def _mascot_design_prompt(topic: str, brief: dict[str, Any] | None) -> str:
    brief_context = ""
    if brief:
        brief_context = (
            f" Context from the video's brief — concept: {brief.get('concept', '')!r}, "
            f"angle: {brief.get('angle', '')!r}, "
            f"claims: {json.dumps([c.get('claim') for c in (brief.get('claims') or [])][:5], ensure_ascii=False)}."
        )
    return (
        f"Design a brand-new FLAT 2D cel-shaded cartoon mascot character for a YouTube Short about {topic!r} — "
        "none of the 5 existing mascots (a Roman legionary, a chibi tinkerer, a bean-headed scavenger, "
        "a bearded dwarf explorer, and a bushcraft alchemist) fit this topic's theme, so this is a fresh "
        "design specifically suited to it. The whole video's house style is flat 2D cel-shaded illustration — "
        "bold black ink outlines, flat clean shading — explicitly NOT a 3D render, NOT CGI, NOT Pixar/"
        "Dreamworks-style, NOT photoreal. This new mascot must match that style exactly, not introduce a "
        "different rendering technique." + brief_context + " Return JSON only, no prose, no markdown "
        "fences. The object must contain exactly: name (a short display name, e.g. 'Mascot: Deep-Sea "
        "Diver'), short_desc (one sentence describing the archetype and outfit), hero_prompt (a detailed "
        "full-body character description for an image-generation model — MUST specify: flat 2D cel-shaded "
        "cartoon illustration with bold black ink outlines (not a 3D render, not CGI, not photoreal), small "
        "and centered vertically in frame occupying no more than 28% of vertical height, with generous empty "
        "white space above, below, and on both sides (must NOT dominate the frame), a friendly explanatory "
        "pose gesturing forward with one open palm, a CLOSED relaxed mouth in a gentle closed-lip smile with "
        "no teeth showing and no open/gasping mouth (this image gets animated later, and an open mouth "
        "becomes a talking mouth), the character is FULLY CLOTHED with no bare skin visible "
        "except face/forearms/calves and must explicitly state it is not shirtless or undressed, and a stark "
        "pure solid white background #FFFFFF with zero scenery and zero floor shadows, sticker framing), "
        "visual_style (a short paragraph naming the rendering technique — flat 2D cel-shaded cartoon "
        "illustration, bold black ink outlines, explicitly NOT 3D/CGI/photoreal — plus the character's "
        "signature colors/textures rendered as flat illustration, ending with the same white-background/"
        "no-text rules as hero_prompt), motion_instruction (guidance for describing this mascot's actions/"
        "emotions in later per-scene prompts), scene_role_template (how this mascot's role should vary across a "
        "Hook/Discovery/Process/Challenge/Payoff story arc, preserving its signature visual traits), and "
        "keywords (a list of 5-8 lowercase topic/theme words this mascot is suited to, for matching future "
        "similar topics without generating a new mascot every time). Every field must actually be specific "
        f"to {topic!r}, not generic filler."
    )


class FalLLMProvider(LLMProvider):
    """Routes through fal.ai's OpenRouter endpoint (openrouter/router) —
    fal-native schema, not fal-ai/any-llm (confirmed deprecated 2026-08-14,
    do not use). model is an OpenRouter-style id, e.g. "anthropic/claude-3.5-sonnet".
    The response echoes real per-call cost (usage.cost), so the ACTUAL spend
    recorded is the real figure, not just the pre-call estimate — the
    estimate is still required up front only because check_budget() has to
    run before we know the real price."""

    name = "fal"

    def __init__(self, gateway: FalGateway, model: str, cost_per_script_usd: float, endpoint: str = "openrouter/router"):
        if not model:
            raise ValueError("fal LLM requires LLM_MODEL (for example 'google/gemini-2.5-flash')")
        if cost_per_script_usd <= 0:
            raise ValueError("Set LLM_COST_PER_SCRIPT_USD to a conservative positive pre-call estimate")
        self.gateway = gateway
        self.endpoint = endpoint
        self.model = model
        self.estimate = cost_per_script_usd

    def generate_script(self, brief, language, visual_style, cost_tracker):
        operation = "llm.generate_script"
        cost_tracker.check_budget(operation, self.estimate)
        data = self.gateway.run(
            self.endpoint,
            {
                "model": self.model,
                "prompt": _script_prompt(brief, language, visual_style),
                "temperature": 0.4,
                # Raised from 2500 on 2026-09-01. At 15 scenes the script
                # JSON runs past 9000 characters and 2500 tokens cut it off
                # mid-string ("Unterminated string starting at line 183"), so
                # EVERY real generation silently fell through to the
                # deterministic stub fallback. Output tokens are only billed
                # when the model actually uses them, so the headroom is
                # effectively free insurance.
                "max_tokens": 12000,
            },
        )
        actual_cost = float(data.get("usage", {}).get("cost", self.estimate))
        cost_tracker.record(self.name, operation, self.estimate, actual_cost, is_stub=False)
        script = _json_object(data["output"])
        return _normalize_generated_script(script, brief, language, visual_style)

    def propose_topic(self, topic: str, cost_tracker: CostTracker) -> dict[str, Any]:
        operation = "llm.propose_topic"
        cost_tracker.check_budget(operation, self.estimate)
        data = self.gateway.run(
            self.endpoint,
            {
                "model": self.model,
                "prompt": _topic_proposal_prompt(topic),
                "temperature": 0.2,
                "max_tokens": 500,
            },
        )
        actual_cost = float(data.get("usage", {}).get("cost", self.estimate))
        cost_tracker.record(self.name, operation, self.estimate, actual_cost, is_stub=False)
        proposal = _json_object(data["output"])
        required = {"safety_class", "reasoning", "queries", "keywords", "caution"}
        if not required.issubset(proposal):
            raise ValueError(f"malformed topic proposal from LLM, missing keys: {required - proposal.keys()}")
        return proposal

    def design_mascot(self, topic: str, brief: dict[str, Any] | None, cost_tracker: CostTracker) -> dict[str, Any]:
        operation = "llm.design_mascot"
        cost_tracker.check_budget(operation, self.estimate)
        data = self.gateway.run(
            self.endpoint,
            {
                "model": self.model,
                "prompt": _mascot_design_prompt(topic, brief),
                "temperature": 0.7,
                "max_tokens": 1200,
            },
        )
        actual_cost = float(data.get("usage", {}).get("cost", self.estimate))
        cost_tracker.record(self.name, operation, self.estimate, actual_cost, is_stub=False)
        design = _json_object(data["output"])
        required = {"name", "short_desc", "hero_prompt", "visual_style", "motion_instruction", "scene_role_template", "keywords"}
        if not required.issubset(design):
            raise ValueError(f"malformed mascot design from LLM, missing keys: {required - design.keys()}")
        if not isinstance(design["keywords"], list) or not design["keywords"]:
            raise ValueError("malformed mascot design from LLM: 'keywords' must be a non-empty list")
        return design


def get_llm_provider(
    provider_name: str,
    api_key: str = "",
    model: str = "",
    cost_per_script_usd: float = 0.0,
    gateway: FalGateway | None = None,
    endpoint: str = "openrouter/router",
) -> LLMProvider:
    name = provider_name.strip().lower()
    if name in ("", "stub"):
        return StubLLMProvider()
    if name == "fal":
        return FalLLMProvider(gateway or FalGateway(api_key), model, cost_per_script_usd, endpoint)
    raise NotImplementedError(f"Unsupported LLM provider {provider_name!r}; use 'stub' or 'fal'")
