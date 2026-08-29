"""Mascot registry: defines the 5 selectable character mascots for video generation.

Each mascot provides:
- id: unique identifier ('mascot_1' to 'mascot_5')
- name: user-facing display name
- short_desc: brief description of archetype and outfit
- hero_prompt: detailed generation prompt for the reference hero image (Recraft-v3 / image provider)
- visual_style: art style and palette instructions for image/video providers
- motion_instruction: guidance for the LLM when writing per-scene motion prompts (Hailuo-02)

Mascot 4 (DEFAULT_MASCOT_ID) is the "Main Mascot" — the house default and
tie-winner in select_mascot_for_story(). Mascots 1/2/3/5 are chosen only
when their own MASCOT_STORY_KEYWORDS theme actually matches a topic. When
NOTHING matches at all (not even the main mascot's own keywords),
select_mascot_for_story() returns None and the caller (pipeline.run_pipeline)
designs a brand-new custom mascot via generate_custom_mascot() — persisted
in data/custom_mascots.json (see register_custom_mascot()/load_custom_mascots())
so a future similar topic reuses it instead of paying to generate another.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
CUSTOM_MASCOT_REGISTRY_PATH = REPO_ROOT / "data" / "custom_mascots.json"


@dataclass(frozen=True)
class Mascot:
    id: str
    name: str
    short_desc: str
    hero_prompt: str
    visual_style: str
    motion_instruction: str
    scene_role_template: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "short_desc": self.short_desc,
            "hero_prompt": self.hero_prompt,
            "visual_style": self.visual_style,
            "motion_instruction": self.motion_instruction,
            "scene_role_template": self.scene_role_template,
        }

    def build_scene_prompt(
        self,
        scene_role: str = "",
        action: str = "",
        emotion: str = "",
        props: str | None = None,
        layout: str = "auto",
        scene_type: str = "mascot",
        fx: str | None = None,
        grid_items: list[str] | None = None,
    ) -> str:
        """Builds a scene visual prompt adhering to the video's 3 core scene archetypes."""
        if scene_type == "ingredient_grid" or (grid_items and len(grid_items) >= 2):
            items_str = ", ".join(grid_items) if grid_items else (props or "raw ingredients")
            return (
                f"Multi-item 2x2 ingredient recipe grid breakdown, drawn in the SAME flat cel-shaded cartoon "
                f"illustration style as the mascot — bold black ink outlines, clean flat cel shading, no "
                f"photoreal or 3D-render textures, no glossy CGI materials. Clean isolated cartoon-sticker icons "
                f"displaying: {items_str}. Each item is cleanly isolated, flat-shaded, simplified — an "
                f"illustrated icon, not a rendered photograph. "
                f"NO people, NO characters, NO figures, NO hands, NO faces anywhere in the image — objects only. "
                f"Stark pure solid white background (#FFFFFF) only, zero background scenery, zero floor shadows. "
                f"No text or labels rendered directly on the image."
            )

        if scene_type == "process_action":
            return (
                f"Dynamic process demonstration action scene, drawn in the SAME flat cel-shaded cartoon "
                f"illustration style as the mascot — bold black ink outlines, clean flat cel shading, no "
                f"photoreal or 3D-render textures, no glossy CGI materials, no dramatic studio lighting. "
                f"Close-up isolated physical action, illustrated as a cartoon: {action or props or 'pouring mixture into mold'}. "
                f"Simplified flat-shaded materials and stylized illustrated motion lines, not realistic liquid "
                f"physics or photoreal rendering. "
                f"NO people, NO characters, NO figures, NO hands, NO faces anywhere in the image — the "
                f"equipment/materials act entirely on their own, as if operated by an invisible presence. "
                f"Stark pure solid white background (#FFFFFF) only, zero background scenery, zero floor shadows, clean sticker framing. "
                f"No text or labels rendered directly on the image."
            )

        # Mascot scenes (reaction, hazard, or split-canvas)
        if layout == "auto":
            layout = "split_bottom_left" if (props and not fx) else "centered"

        if layout in ("split_bottom_left", "split_bottom_right"):
            corner = "bottom-left" if layout == "split_bottom_left" else "bottom-right"
            opp_corner = "upper-right" if layout == "split_bottom_left" else "upper-left"
            prompt_parts = [
                "Split-canvas flat cel-shaded cartoon explainer composition, drawn in the SAME flat "
                "illustration style as the mascot — bold black ink outlines, flat clean shading, NOT a "
                "3D render, NOT photoreal, NOT glossy CGI — on a stark pure solid white background (#FFFFFF).",
                f"In the {corner} quadrant, the smaller full-body {self.name} mascot (occupying roughly 40% of vertical height, clearly visible and recognizable, not tiny) stands looking and pointing up with {emotion or 'an expressive engaging gesture'} as {scene_role or 'a demonstrator'}.",
            ]
            if props:
                prompt_parts.append(
                    f"In the {opp_corner} quadrant, a large floating illustrated object/diagram, drawn flat "
                    f"and cel-shaded (not a rendered 3D sticker), shows {props}."
                )
            if fx:
                prompt_parts.append(f"Visual FX: {fx}.")
            if action:
                prompt_parts.append(f"Action: {action}.")
            prompt_parts.append(
                f"Style: {self.visual_style}. Pure solid white background only, zero scenery, zero shadows, sticker framing. No text or labels."
            )
            return " ".join(prompt_parts)
        else:
            prompt_parts = [
                f"Full-body {self.name} mascot centered vertically in frame (small, occupying no more than 28% of vertical height, with generous empty white space above, below, and on both sides) on a stark pure solid white background (#FFFFFF).",
                f"Role: {scene_role or 'explainer'}. Emotion: {emotion or 'friendly enthusiastic expression'}.",
            ]
            if action:
                prompt_parts.append(f"Action: {action}.")
            if props:
                prompt_parts.append(f"Holding: {props}.")
            if fx:
                prompt_parts.append(f"Contextual Visual FX: {fx} actively billowing around or interacting with the character.")
            prompt_parts.append(
                f"Style: {self.visual_style}. Pure solid white background only, zero background scenery, zero floor shadows, sticker framing. No text or labels."
            )
            return " ".join(prompt_parts)

    def build_scene_motion_prompt(
        self,
        scene_type: str = "mascot",
        props: str | None = None,
        fx: str | None = None,
        action: str = "",
        narration: str = "",
    ) -> str:
        """Motion prompt for the continuous-AI-video path (ai_video mode,
        Hailuo/Kling) — deliberately SEPARATE from build_scene_prompt()'s
        still-image composition prompt. Reusing the still-image prompt
        verbatim as the motion source (the prior approach) produced a
        bouncing mascot plus completely frozen props: that prompt only
        describes a static composition, it never says what should keep
        moving or how.

        Encodes the pipeline-wide motion/bounce rules from user feedback
        2026-08-28: the mascot stays planted (tiny breath+blink only, no
        hop/bob/squash-stretch/repeating idle bounce); when the mascot is
        mostly still, whatever OBJECT is actually in frame keeps moving on
        its own loop (steam/drips/dust/swirl/etc., matched generically off
        this scene's fx/props/action/narration — never hardcoded to one
        topic); any pop-in is a single short scale-in on first appearance,
        never a repeating bounce."""
        object_fx = object_fx_for(fx, props, action, narration)

        if scene_type in ("ingredient_grid", "process_action"):
            # No mascot in these scene types at all (see build_scene_prompt)
            # — the props/equipment are the only thing in frame, so they
            # carry all the motion.
            parts = [
                "No character present in this shot."
                f" {action or 'The equipment/materials'} moves under its own action, as if operated by an "
                "invisible presence.",
            ]
            if scene_type == "ingredient_grid" and props:
                parts.append(
                    f"The items ({props}) appear one at a time, staggered roughly 0.6-0.9 seconds apart in "
                    "the order they'd naturally be named — never all bundled into the opening frame at once. "
                    "Each item pops in with a single short scale-in the instant it appears, then holds."
                )
            if object_fx:
                parts.append(f"Once settled, continuous looping motion: {object_fx}.")
            parts.append(
                "No bounce or wobble after each item's initial pop-in — the frame stays alive through this "
                "per-item motion, not repeated bouncing."
            )
            return " ".join(parts)

        parts = [
            "The mascot stays planted in place — only a tiny natural breath and occasional blink, no "
            "hopping, bobbing, squash-and-stretch, or repeating idle bounce of any kind.",
        ]
        if action:
            parts.append(f"Mascot action (a calm, purposeful gesture, not bouncy): {action}.")
        if object_fx:
            parts.append(
                f"Since the mascot itself stays mostly still, keep the frame alive through the props/"
                f"environment instead: {object_fx}, continuously, for the whole clip."
            )
        elif props:
            parts.append(
                f"Keep {props} subtly and continuously animated (matching its own real physical behavior) "
                "so the frame never goes fully static."
            )
        parts.append(
            "If a new object or prop enters partway through the clip, it pops in with a single short "
            "scale-in the instant it appears, then holds (aside from its own looping motion above) — never "
            "a second bounce for something already on screen, never a repeating springy idle."
        )
        return " ".join(parts)


# Generic keyword -> looping object/prop motion, used by
# build_scene_motion_prompt() to give the video model something concrete to
# animate on the PROPS themselves when the mascot stays still. Deliberately
# generic across any topic's props (not hardcoded to one topic, e.g. Roman
# concrete) — matched against whatever combination of this scene's fx/props/
# action/narration text is available.
_OBJECT_FX_KEYWORDS: list[tuple[tuple[str, ...], str]] = [
    (("volcano", "volcanic", "ash", "pyroclastic"), "fine volcanic ash drifting and gently settling"),
    (("smoke", "steam", "vapor", "vapour"), "steam/smoke gently rising and curling"),
    (("water", "drip", "wet", "damp", "liquid", "ripple", "pour"), "water rippling, dripping, or pouring steadily"),
    (("lime", "powder", "dust", "sand", "grain", "mineral"), "fine powder/dust settling and drifting"),
    (("mix", "mixing", "stir", "slurry", "blend", "paste", "mortar"), "the mixture slowly swirling and blending"),
    (("fire", "flame", "ember", "burn", "forge", "furnace", "heat"), "flames flickering and embers glowing"),
    (("bubble", "boil", "ferment"), "small bubbles steadily rising and popping"),
    (("spark", "electric", "current", "voltage"), "small sparks flickering intermittently"),
]


def object_fx_for(*texts: str | None) -> str | None:
    """Public (not module-private) since 2026-08-29 — also used by
    assembly._narrated_object_cue_start() to find the moment a scene's
    narration names something animatable, for the localized on-screen
    object-pulse feature (distinct from this function's original use here:
    describing motion for the ai_video/Hailuo continuous-video prompt)."""
    haystack = " ".join(t for t in texts if t).lower()
    for keywords, fx_desc in _OBJECT_FX_KEYWORDS:
        if any(kw in haystack for kw in keywords):
            return fx_desc
    return None


MASCOT_1 = Mascot(
    id="mascot_1",
    name="Mascot 1: Roman Legionary",
    short_desc="Cartoon Roman legionary with bronze helmet, red crest, and cape",
    hero_prompt=(
        "Full-body FLAT 2D cel-shaded cartoon illustration of a Roman legionary mascot character standing in a "
        "friendly explanatory pose facing camera, one hand gesturing forward with open palm. Bold black ink "
        "outlines, flat clean cel shading — NOT a 3D render, NOT photoreal, NOT glossy CGI. "
        "The character is small and centered vertically in frame, occupying no more than 28% of vertical height, "
        "with generous empty white space above, below, and on both sides — the character must NOT dominate or "
        "fill the frame. "
        "The character is fully clothed: wearing a classic polished bronze helmet with red brush crest, "
        "a red cape over a Roman tunic with leather armor strips and bronze buckles, and strapped sandals. "
        "Expressive cartoon eyes, animated bushy eyebrows, and engaging friendly expression. "
        "No bare skin visible except face, forearms, and calves; do not depict shirtless or undressed. "
        "Stark pure solid white background (#FFFFFF) only, zero background details, zero floor shadows, sticker framing."
    ),
    visual_style=(
        "Flat 2D cel-shaded cartoon illustration style, bold black ink outlines, clean flat shading, no gradients "
        "beyond simple cel shading — explicitly NOT a 3D render, NOT photoreal, NOT glossy CGI. "
        "Expressive Roman legionary mascot character on a pure solid white background (#FFFFFF) "
        "with zero background scenery and zero shadows. Do not render any text, words, letters, labels, or signs."
    ),
    motion_instruction=(
        "For every scene's `visual_prompt`, describe the Roman legionary mascot's dynamic physical ACTION and EMOTIONS "
        "for Hailuo-02 (e.g. 'Roman legionary character widens eyes in surprise, clutching helmet with both hands in shock', "
        "or 'Roman legionary confidently points forward with open hand while speaking, nodding cheerfully'). "
        "Keep the character centered against the clean solid white background."
    ),
    scene_role_template=(
        "Cast the Roman legionary mascot into a narrative role suited to each scene beat while preserving the bronze helmet and red cape DNA: "
        "Hook=alarmed legionary reacting to a crisis; Discovery=ancient legionary unearthing raw materials; "
        "Process=legionary builder in work tunic mixing materials with mortar tools; Challenge=legionary testing or pouring with tongs; "
        "Payoff=triumphant legionary proudly displaying the finished invention with a victory grin."
    ),
)

MASCOT_2 = Mascot(
    id="mascot_2",
    name="Mascot 2: Chibi Artisan Engineer",
    short_desc="Cute chibi anime-style inventor with leather tool apron and brass goggles",
    hero_prompt=(
        "Full-body FLAT 2D cel-shaded chibi cartoon illustration standing in a friendly explanatory gesture "
        "with one hand raised open-palmed. Thick black ink line art, flat clean cel shading — NOT a 3D render, "
        "NOT photoreal, NOT glossy CGI. "
        "The character is small and centered vertically in frame, occupying no more than 28% of vertical height, "
        "with generous empty white space above, below, and on both sides — the character must NOT dominate or "
        "fill the frame. "
        "The character has cute chibi proportions with an oversized expressive head and animated large cartoon eyes. "
        "The character is fully clothed: wearing a brown leather artisan apron with pocket tools over a navy tunic, "
        "round brass goggles resting on top of hair, and small sturdy utility boots. "
        "No bare skin visible except face and hands; do not depict shirtless or undressed. "
        "Stark pure solid white background (#FFFFFF) only, zero background scenery, zero floor shadows, sticker framing."
    ),
    visual_style=(
        "Cute flat 2D cel-shaded chibi cartoon illustration style, thick black ink line art, clean flat vector "
        "shading, vibrant saturated colors, expressive animated facial proportions — explicitly NOT a 3D render, "
        "NOT photoreal, NOT glossy CGI. "
        "Pure solid white background (#FFFFFF) with zero scenery, zero shadows. Do not render any text, words, letters, labels, or signs."
    ),
    motion_instruction=(
        "For every scene's `visual_prompt`, describe the Chibi Engineer's snappy cartoon movements and facial expressions "
        "(e.g. 'Chibi engineer scratches head in confusion with wide curious eyes, tilting head', "
        "or 'Chibi engineer enthusiastically gestures with both hands with a bright open smile while explaining'). "
        "Keep the character centered against the clean solid white background."
    ),
    scene_role_template=(
        "Cast the Chibi Engineer mascot into a narrative role suited to each scene beat while preserving the oversized cute head, goggles, and tool apron DNA: "
        "Hook=curious chibi engineer tilting head with question marks; Discovery=chibi scholar examining raw ingredients with magnifying glass; "
        "Process=chibi chemist stirring bubbling mixture with wooden paddle; Challenge=chibi builder hammering or molding with safety gloves; "
        "Payoff=beaming chibi engineer holding up the gleaming final product with sparkle stars in eyes."
    ),
)

MASCOT_3 = Mascot(
    id="mascot_3",
    name="Mascot 3: 3D Bean Scavenger",
    short_desc="Smooth coffee bean explorer with iron kettle helmet, burlap scarf, and duster coat",
    hero_prompt=(
        "Full-body FLAT 2D cel-shaded cartoon illustration combining a smooth brown bean face with a rugged "
        "medieval scavenger explorer. Bold black ink outlines, flat clean cel shading — NOT a 3D render, NOT "
        "photoreal, NOT glossy CGI, NOT cinematic lighting. "
        "The character is small and centered vertically in frame, occupying no more than 28% of vertical height, "
        "with generous empty white space above, below, and on both sides — the character must NOT dominate or "
        "fill the frame. "
        "Smooth rounded bean head with prominent expressive brow ridges, large animated green eyes, and a classic vintage handlebar mustache. "
        "The character is fully clothed: wearing an antique iron kettle helmet, a weathered brown scavenger leather coat with a frayed burlap cowl scarf, "
        "utility belt with pouches, leather gloves, and sturdy explorer boots, holding a wooden walking staff in one hand and gesturing with the other open hand. "
        "Simplified flat-shaded leather and metal textures rendered as clean illustration, not tactile/photoreal materials. "
        "Stark pure solid white background (#FFFFFF) only, zero background scenery, zero floor shadows, sticker framing."
    ),
    visual_style=(
        "Flat 2D cel-shaded cartoon illustration style, bold black ink outlines, expressive flat-shaded bean "
        "scavenger character, simplified illustrated leather and burlap textures, bright even lighting — "
        "explicitly NOT a 3D render, NOT photoreal, NOT cinematic/dramatic lighting. "
        "Pure solid white background (#FFFFFF) with zero scenery, zero shadows. Do not render any text, words, letters, labels, or signs."
    ),
    motion_instruction=(
        "For every scene's `visual_prompt`, describe the 3D Bean Scavenger's dynamic ACTION and EMOTIONS for Hailuo-02 "
        "(e.g. '3D bean character tilts helmet back and widens eyes in shock while pointing staff forward', "
        "or '3D bean character strokes his mustache thoughtfully and gestures with open hand while nodding'). "
        "Keep the character centered against the clean solid white background."
    ),
    scene_role_template=(
        "Cast the 3D Bean Scavenger mascot into a narrative role suited to each scene beat while preserving the smooth bean head, kettle helmet, and mustache DNA: "
        "Hook=shocked bean explorer pointing staff at a mysterious artifact; Discovery=bean scavenger foraging raw materials into leather pouch; "
        "Process=bean alchemist heating cauldron with bellows; Challenge=bean artisan pressing hot mold with tongs; "
        "Payoff=proud bean explorer twirling mustache while holding the finished item."
    ),
)

MASCOT_4 = Mascot(
    id="mascot_4",
    name="Mascot 4: Bearded Dwarf Scavenger Explorer (Main Mascot)",
    short_desc="Charming stylized 3D human dwarf/halfling explorer with neat brown beard, weathered leather duster coat, burlap scarf, iron kettle helmet, and walking staff",
    hero_prompt=(
        "Full-body FLAT 2D cel-shaded cartoon illustration of a stylized human dwarf / halfling scavenger explorer. "
        "Bold black ink outlines, flat clean cel shading — NOT a 3D render, NOT CGI, NOT Pixar/Dreamworks-style "
        "rendering, NOT photoreal, NOT glossy materials. A hand-drawn illustration, not a 3D-rendered asset. "
        "The character is small and centered vertically in frame, occupying no more than 28% of vertical height, "
        "with generous empty white space above, below, and on both sides — the character must NOT dominate or "
        "fill the frame. "
        "Charming expressive facial features with tousled wavy brown hair, warm expressive animated eyes, "
        "a neatly-groomed rugged full brown dwarf beard and mustache, and a friendly engaging smile. "
        "Wearing an antique metal skull-cap / iron kettle helmet, a frayed burlap cowl scarf draped around his neck, "
        "a weathered brown leather scavenger duster coat with frayed tattered hem, utility belt with pouches and brass buckles, "
        "leather work gloves, and sturdy strapped adventurer boots, holding a tall wooden walking staff in one hand and gesturing forward with open palm. "
        "Simplified flat-shaded leather, metal, and cloth textures drawn as clean illustration, not tactile/photoreal materials. "
        "Stark pure solid white background (#FFFFFF) only, zero background scenery, zero floor shadows, sticker framing."
    ),
    visual_style=(
        "Flat 2D cel-shaded cartoon illustration of a stylized human dwarf explorer mascot with a neat brown "
        "beard, bold black ink outlines, flat clean shading, wavy brown hair, antique kettle helmet, frayed "
        "burlap scarf, and weathered brown leather duster coat — explicitly NOT a 3D render, NOT CGI, NOT "
        "Pixar/Dreamworks-style, NOT photoreal. "
        "Stark pure solid white background (#FFFFFF) with zero scenery, zero shadows, clean sticker framing. "
        "Do not render any text, words, letters, labels, or signs in the base image."
    ),
    motion_instruction=(
        "For every scene's `visual_prompt`, describe the Bearded Dwarf Explorer mascot's dynamic ACTION, EMOTIONS, and GESTURES for Hailuo-02 "
        "(e.g. 'Bearded dwarf explorer points his wooden staff up and widens eyes with an amazed smile at the floating discovery', "
        "or 'Bearded dwarf explorer strokes his beard thoughtfully, gesturing with open hand with curious expression'). "
        "Keep the character isolated against the stark solid white background."
    ),
    scene_role_template=(
        "Cast the Bearded Dwarf Scavenger Explorer mascot into narrative roles across the story beats while preserving his beard, kettle helmet, burlap scarf, and leather duster coat DNA: "
        "Hook=curious bearded dwarf explorer discovering an ancient relic with raised wooden staff; "
        "Discovery=bearded dwarf scholar examining raw minerals with magnifying glass; "
        "Process=bearded dwarf craftsman boiling and mixing ingredients in rustic cauldron; "
        "Challenge=bearded dwarf builder pressing hot casting molds with leather gloves; "
        "Payoff=triumphant bearded dwarf explorer stroking beard and presenting the finished invention with a proud victory grin."
    ),
)

MASCOT_5 = Mascot(
    id="mascot_5",
    name="Mascot 5: Bushcraft Alchemist",
    short_desc="Wilderness survivor with green hooded cowl, copper distillation flask, and tool harness",
    hero_prompt=(
        "Full-body FLAT 2D cel-shaded cartoon illustration of a wilderness bushcraft alchemist and survival "
        "herbalist. Bold black ink outlines, flat clean cel shading — NOT a 3D render, NOT CGI, NOT photoreal. "
        "The character is small and centered vertically in frame, occupying no more than 28% of vertical height, "
        "with generous empty white space above, below, and on both sides — the character must NOT dominate or "
        "fill the frame. "
        "Expressive animated cartoon face with determined friendly eyes and animated brow. "
        "The character is fully clothed: wearing a moss-green weathered canvas hooded cowl over a durable wool work shirt, "
        "reinforced leather utility vest with brass buckle straps, chemical test vials strapped to chest, "
        "durable field trousers, and rugged survival climbing boots, holding an antique copper distillation flask in one hand and gesturing with the other. "
        "Simplified flat-shaded canvas and leather textures drawn as clean illustration, not tactile/photoreal materials. "
        "Stark pure solid white background (#FFFFFF) only, zero background scenery, zero floor shadows, sticker framing."
    ),
    visual_style=(
        "Flat 2D cel-shaded cartoon illustration of a bushcraft alchemist survival mascot, bold black ink "
        "outlines, simplified flat-shaded canvas and leather textures, expressive cartoon features, bright even "
        "lighting — explicitly NOT a 3D render, NOT CGI, NOT photoreal, NOT cinematic/dramatic lighting. "
        "Pure solid white background (#FFFFFF) with zero scenery, zero shadows. Do not render any text, words, letters, labels, or signs."
    ),
    motion_instruction=(
        "For every scene's `visual_prompt`, describe the Bushcraft Alchemist's dynamic physical ACTION and EMOTIONS for Hailuo-02 "
        "(e.g. 'Bushcraft alchemist raises copper flask with wide amazed eyes as steam rises', "
        "or 'Bushcraft alchemist adjusts his hood and points forward authoritatively while speaking'). "
        "Keep the character centered against the clean solid white background."
    ),
    scene_role_template=(
        "Cast the Bushcraft Alchemist mascot into a narrative role suited to each scene beat while preserving the green hooded cowl and copper flask DNA: "
        "Hook=bushcraft survivor reacting to dirty/unusable wild material; Discovery=herbalist harvesting wild ash/resins; "
        "Process=alchemist distilling liquids in copper alembic with rising vapor; Challenge=survivalist filtering through charcoal layers; "
        "Payoff=satisfied alchemist holding clean crystal-clear product with proud smile."
    ),
)

MASCOTS: dict[str, Mascot] = {
    "mascot_1": MASCOT_1,
    "mascot_2": MASCOT_2,
    "mascot_3": MASCOT_3,
    "mascot_4": MASCOT_4,
    "mascot_5": MASCOT_5,
}

DEFAULT_MASCOT_ID = "mascot_4"


def load_custom_mascots() -> dict[str, dict]:
    """Reads straight from disk every call (same pattern as
    topic_registry.load_registry()) — a custom mascot generated by one
    process must be visible to the very next call in the same process, not
    just after a restart. Each entry: {name, short_desc, hero_prompt,
    visual_style, motion_instruction, scene_role_template, keywords}."""
    if not CUSTOM_MASCOT_REGISTRY_PATH.exists():
        return {}
    return json.loads(CUSTOM_MASCOT_REGISTRY_PATH.read_text(encoding="utf-8"))


def _mascot_from_custom_entry(mascot_id: str, entry: dict) -> Mascot:
    return Mascot(
        id=mascot_id,
        name=entry["name"],
        short_desc=entry["short_desc"],
        hero_prompt=entry["hero_prompt"],
        visual_style=entry["visual_style"],
        motion_instruction=entry["motion_instruction"],
        scene_role_template=entry.get("scene_role_template", ""),
    )


def custom_mascot_slug(topic: str) -> str:
    """Deterministic id for a topic's custom mascot — the SAME topic run
    twice must find and reuse its own already-generated custom mascot
    (via this id, and independently via keyword-scoring too) instead of
    generating and paying for a new one each time."""
    slug = re.sub(r"[^a-z0-9]+", "_", topic.strip().lower()).strip("_") or "topic"
    return f"mascot_custom_{slug}"


def register_custom_mascot(mascot_id: str, design: dict[str, Any]) -> Mascot:
    """Persists a newly-designed mascot (see providers/llm.py's
    design_mascot()) so future similar topics can find and reuse it via the
    normal keyword-scoring path in select_mascot_for_story() instead of
    generating a new one every time. Atomic write (tmp + replace), same
    pattern as topic_registry.register_topic()."""
    required = {"name", "short_desc", "hero_prompt", "visual_style", "motion_instruction", "keywords"}
    missing = required - design.keys()
    if missing:
        raise ValueError(f"cannot register custom mascot, missing keys: {missing}")
    registry = load_custom_mascots()
    registry[mascot_id] = {
        "name": design["name"],
        "short_desc": design["short_desc"],
        "hero_prompt": design["hero_prompt"],
        "visual_style": design["visual_style"],
        "motion_instruction": design["motion_instruction"],
        "scene_role_template": design.get("scene_role_template", ""),
        "keywords": list(design["keywords"]),
    }
    CUSTOM_MASCOT_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = CUSTOM_MASCOT_REGISTRY_PATH.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    tmp_path.replace(CUSTOM_MASCOT_REGISTRY_PATH)
    return _mascot_from_custom_entry(mascot_id, registry[mascot_id])


def generate_custom_mascot(topic: str, brief: dict[str, Any] | None, llm_provider, cost_tracker) -> Mascot:
    """Called only when select_mascot_for_story() finds no match at all
    among the 5 registered mascots + every already-generated custom one —
    designs and persists a brand-new mascot tailored to this topic (see
    providers/llm.py's design_mascot()), so it's available for reuse on
    future similar topics without paying to generate another."""
    design = llm_provider.design_mascot(topic, brief, cost_tracker)
    mascot_id = custom_mascot_slug(topic)
    return register_custom_mascot(mascot_id, design)


def get_mascot(mascot_id: str | None) -> Mascot:
    """Returns the requested mascot (checking the 5 registered ones, then
    any generated custom mascot), or the default Mascot 4 if unspecified/unknown."""
    if not mascot_id:
        return MASCOTS[DEFAULT_MASCOT_ID]
    norm_id = mascot_id.strip().lower()
    # Support alias formats like '1', 'mascot 1', 'mascot-1'
    if norm_id in ("1", "mascot 1", "mascot-1"):
        norm_id = "mascot_1"
    elif norm_id in ("2", "mascot 2", "mascot-2"):
        norm_id = "mascot_2"
    elif norm_id in ("3", "mascot 3", "mascot-3"):
        norm_id = "mascot_3"
    elif norm_id in ("4", "mascot 4", "mascot-4"):
        norm_id = "mascot_4"
    elif norm_id in ("5", "mascot 5", "mascot-5"):
        norm_id = "mascot_5"
    if norm_id in MASCOTS:
        return MASCOTS[norm_id]
    custom = load_custom_mascots()
    if norm_id in custom:
        return _mascot_from_custom_entry(norm_id, custom[norm_id])
    return MASCOTS[DEFAULT_MASCOT_ID]


def list_mascots() -> list[Mascot]:
    """Returns all 5 mascots in numeric order."""
    return [MASCOTS[f"mascot_{i}"] for i in range(1, 6)]


# Topic/story keyword -> mascot_id, used by select_mascot_for_story() to pick
# a thematically-appropriate mascot instead of always defaulting to the same
# one. Keys are lowercase substrings checked against the topic/brief text.
MASCOT_STORY_KEYWORDS: dict[str, list[str]] = {
    "mascot_1": ["roman", "concrete", "rome", "aqueduct", "monument", "pozzolana"],
    "mascot_2": ["compass", "pump", "gear", "pottery", "wheel", "mechanic", "tool", "rope"],
    "mascot_3": ["vinegar", "food", "preservation", "ferment", "salt", "drying", "cider", "apple"],
    "mascot_4": ["soap", "charcoal", "stone", "mineral", "mining", "smelt", "furnace", "ash", "lye"],
    "mascot_5": ["water filtration", "filter", "distill", "alchemist", "chemistry", "herb", "purify"],
}


def select_mascot_for_story(
    topic: str,
    brief: dict[str, Any] | None = None,
    seed: Any = None,
) -> Mascot | None:
    """Picks a mascot thematically suited to the topic/brief: DEFAULT_MASCOT_ID
    (Mascot 4, the "Main Mascot") is the house default and wins ties among the
    5 registered mascots; the other 4 are chosen only when their specific
    keyword theme actually matches. Every already-generated custom mascot
    (see generate_custom_mascot()) is scored too, so a topic similar to one
    seen before reuses that same custom mascot instead of never finding it
    again. Scores every candidate's keyword list against the topic + brief's
    concept/angle/claims text (topic-text hits count double).

    Returns None when NOTHING matches at all — literally zero keyword hits
    across the 5 registered mascots and every custom one — signaling the
    caller (pipeline.run_pipeline) to generate a brand-new custom mascot
    for this story instead of forcing an unrelated one onto it."""
    rng = random.Random(seed) if seed is not None else random
    search_text = topic.lower()
    if brief:
        # .lower() every appended piece too — a capitalized "Roman"/"Pozzolana"
        # in the brief's own concept/claim text (real LLM prose is not
        # reliably lowercase) would otherwise silently never match any
        # lowercase keyword below (confirmed for real: this exact case broke
        # the "roman concrete" match in testing).
        search_text += f" {brief.get('concept', '')} {brief.get('angle', '')}".lower()
        for claim in brief.get("claims", []):
            if isinstance(claim, dict):
                search_text += f" {claim.get('claim', '')} {claim.get('narrative_role', '')}".lower()
            elif isinstance(claim, str):
                search_text += f" {claim}".lower()

    custom = load_custom_mascots()
    all_keywords: dict[str, list[str]] = dict(MASCOT_STORY_KEYWORDS)
    for m_id, entry in custom.items():
        all_keywords[m_id] = entry.get("keywords", [])

    scores: dict[str, int] = {m_id: 0 for m_id in all_keywords}
    topic_lower = topic.lower()
    for m_id, keywords in all_keywords.items():
        for kw in keywords:
            if kw in search_text:
                scores[m_id] += 2 if kw in topic_lower else 1

    max_score = max(scores.values()) if scores else 0
    if max_score == 0:
        return None

    candidates = [m_id for m_id, s in scores.items() if s == max_score]
    if DEFAULT_MASCOT_ID in candidates:
        chosen_id = DEFAULT_MASCOT_ID
    else:
        chosen_id = rng.choice(candidates)
    return MASCOTS[chosen_id] if chosen_id in MASCOTS else _mascot_from_custom_entry(chosen_id, custom[chosen_id])
