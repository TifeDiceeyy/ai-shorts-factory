"""Mascot registry: defines the 5 selectable character mascots for video generation.

Each mascot provides:
- id: unique identifier ('mascot_1' to 'mascot_5')
- name: user-facing display name
- short_desc: brief description of archetype and outfit
- hero_prompt: detailed generation prompt for the reference hero image (Recraft-v3 / image provider)
- visual_style: art style and palette instructions for image/video providers
- motion_instruction: guidance for the LLM when writing per-scene motion prompts (Hailuo-02)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
                f"Multi-item 2x2 ingredient recipe grid breakdown on a stark pure solid white background (#FFFFFF). "
                f"Clean isolated 3D tactile artifact stickers displaying: {items_str}. "
                f"Each item is cleanly isolated with rich realistic textures and subtle ambient contact occlusion. "
                f"Stark pure solid white background (#FFFFFF) only, zero background scenery, zero floor shadows. "
                f"No text or labels rendered directly on the image."
            )

        if scene_type == "process_action":
            return (
                f"Dynamic 3D process demonstration action scene on a stark pure solid white background (#FFFFFF). "
                f"Close-up isolated physical action: {action or props or 'pouring mixture into mold'}. "
                f"Tactile materials, realistic liquid flow/physics, and crisp lighting. "
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
                "Split-canvas 3D explainer composition on a stark pure solid white background (#FFFFFF).",
                f"In the {corner} quadrant, the smaller full-body {self.name} mascot (occupying 35-40% of vertical height) stands looking and pointing up with {emotion or 'an expressive engaging gesture'} as {scene_role or 'a demonstrator'}.",
            ]
            if props:
                prompt_parts.append(
                    f"In the {opp_corner} quadrant, a large floating 3D object sticker shows {props} with crisp tactile textures."
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
                f"Full-body {self.name} mascot centered vertically in frame (occupying 55-60% of vertical height) on a stark pure solid white background (#FFFFFF).",
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


MASCOT_1 = Mascot(
    id="mascot_1",
    name="Mascot 1: Roman Legionary",
    short_desc="Cartoon Roman legionary with bronze helmet, red crest, and cape",
    hero_prompt=(
        "Full-body cartoon Roman legionary mascot character standing in a friendly explanatory pose "
        "facing camera, one hand gesturing forward with open palm. "
        "The character is centered vertically in frame, occupying 60% of vertical height with clear space at top and bottom. "
        "The character is fully clothed: wearing a classic polished bronze helmet with red brush crest, "
        "a red cape over a Roman tunic with leather armor strips and bronze buckles, and strapped sandals. "
        "Expressive cartoon eyes, animated bushy eyebrows, and engaging friendly expression. "
        "No bare skin visible except face, forearms, and calves; do not depict shirtless or undressed. "
        "Stark pure solid white background (#FFFFFF) only, zero background details, zero floor shadows, sticker framing."
    ),
    visual_style=(
        "Vibrant 2D/3D comic cartoon illustration style, bold black ink outlines, clean cel shading, "
        "expressive Roman legionary mascot character on a pure solid white background (#FFFFFF) "
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
        "Full-body 2D/3D animated chibi cartoon mascot character standing in a friendly explanatory gesture "
        "with one hand raised open-palmed. "
        "The character is centered vertically in frame, occupying 60% of vertical height with clear space at top and bottom. "
        "The character has cute chibi proportions with an oversized expressive head and animated large cartoon eyes. "
        "The character is fully clothed: wearing a brown leather artisan apron with pocket tools over a navy tunic, "
        "round brass goggles resting on top of hair, and small sturdy utility boots. "
        "No bare skin visible except face and hands; do not depict shirtless or undressed. "
        "Stark pure solid white background (#FFFFFF) only, zero background scenery, zero floor shadows, sticker framing."
    ),
    visual_style=(
        "Cute 2D/3D chibi cartoon illustration style, thick black ink line art, clean vector cel shading, "
        "vibrant saturated colors, expressive animated facial proportions. "
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
        "Full-body 3D animated cartoon character mascot combining a smooth brown bean face with a rugged medieval scavenger explorer. "
        "The character is centered vertically in frame, occupying 60% of vertical height with clear space at top and bottom. "
        "Smooth rounded bean head with prominent expressive brow ridges, large animated green eyes, and a classic vintage handlebar mustache. "
        "The character is fully clothed: wearing an antique iron kettle helmet, a weathered brown scavenger leather coat with a frayed burlap cowl scarf, "
        "utility belt with pouches, leather gloves, and sturdy explorer boots, holding a wooden walking staff in one hand and gesturing with the other open hand. "
        "High-detail 3D animation render with tactile weathered leather and metal textures. "
        "Stark pure solid white background (#FFFFFF) only, zero background scenery, zero floor shadows, sticker framing."
    ),
    visual_style=(
        "High-detail 3D animated bean scavenger character render, expressive 3D cartoon eyes, "
        "weathered leather and burlap textures, cinematic lighting. "
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
        "Full-body 3D CGI cartoon character mascot: a stylized human dwarf / halfling scavenger explorer. "
        "The character is centered vertically in frame, occupying 60% of vertical height with clear space at top and bottom. "
        "Charming expressive facial features with tousled wavy brown hair, warm expressive animated eyes, "
        "a neatly-groomed rugged full brown dwarf beard and mustache, and a friendly engaging smile. "
        "Wearing an antique metal skull-cap / iron kettle helmet, a frayed burlap cowl scarf draped around his neck, "
        "a weathered brown leather scavenger duster coat with frayed tattered hem, utility belt with pouches and brass buckles, "
        "leather work gloves, and sturdy strapped adventurer boots, holding a tall wooden walking staff in one hand and gesturing forward with open palm. "
        "High-end 3D CGI animation render, Pixar / Dreamworks quality with rich tactile leather, metal, and cloth textures. "
        "Stark pure solid white background (#FFFFFF) only, zero background scenery, zero floor shadows, sticker framing."
    ),
    visual_style=(
        "High-end 3D CGI cartoon animation render of a stylized human dwarf explorer mascot with a neat brown beard, "
        "expressive 3D cartoon face with wavy brown hair, antique kettle helmet, frayed burlap scarf, and weathered brown leather duster coat. "
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
        "Full-body 3D animated character mascot: a wilderness bushcraft alchemist and survival herbalist. "
        "The character is centered vertically in frame, occupying 60% of vertical height with clear space at top and bottom. "
        "Expressive animated cartoon face with determined friendly eyes and animated brow. "
        "The character is fully clothed: wearing a moss-green weathered canvas hooded cowl over a durable wool work shirt, "
        "reinforced leather utility vest with brass buckle straps, chemical test vials strapped to chest, "
        "durable field trousers, and rugged survival climbing boots, holding an antique copper distillation flask in one hand and gesturing with the other. "
        "High-detail 3D CGI render with rich organic canvas and leather textures. "
        "Stark pure solid white background (#FFFFFF) only, zero background scenery, zero floor shadows, sticker framing."
    ),
    visual_style=(
        "High-detail 3D CGI animation render of a bushcraft alchemist survival mascot, rich canvas and leather textures, "
        "expressive cartoon features, vibrant lighting. "
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


def get_mascot(mascot_id: str | None) -> Mascot:
    """Returns the requested mascot, or the default Mascot 4 if unspecified/unknown."""
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
    return MASCOTS.get(norm_id, MASCOTS[DEFAULT_MASCOT_ID])


def list_mascots() -> list[Mascot]:
    """Returns all 5 mascots in numeric order."""
    return [MASCOTS[f"mascot_{i}"] for i in range(1, 6)]
