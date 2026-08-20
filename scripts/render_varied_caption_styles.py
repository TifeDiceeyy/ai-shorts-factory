import os
import sys
sys.path.insert(0, os.path.abspath("src"))
from PIL import Image
from shorts_factory.captions import draw_caption, CAPTION_STYLES

assets_dir = r"C:\Users\owner\.cursor\projects\c-Users-owner-projects-ai-shorts-factory\assets"

demo_renders = [
    {
        "input": os.path.join(assets_dir, "scene_1_hook_self_healing_concrete.png"),
        "output": os.path.join(assets_dir, "scene_1_style_electric_yellow.png"),
        "text": "ALIVE AND HEALING",
        "style": "electric_neon_yellow",
    },
    {
        "input": os.path.join(assets_dir, "scene_2_chemical_slaking_reaction.png"),
        "output": os.path.join(assets_dir, "scene_2_style_highlighter_pill.png"),
        "text": "QUICKLIME. ADD WATER",
        "style": "highlighter_yellow_pill",
    },
    {
        "input": os.path.join(assets_dir, "scene_3_triumphant_cured_concrete_payoff.png"),
        "output": os.path.join(assets_dir, "scene_3_style_cyber_cyan.png"),
        "text": "LASTS 2,000 YEARS",
        "style": "cyber_cyan_ice",
    },
    {
        "input": os.path.join(assets_dir, "scene_1_hook_self_healing_concrete.png"),
        "output": os.path.join(assets_dir, "scene_1_style_dual_tone.png"),
        "text": "HEALS ITS OWN CRACKS",
        "style": "dual_tone_fire",
    },
    {
        "input": os.path.join(assets_dir, "scene_3_triumphant_cured_concrete_payoff.png"),
        "output": os.path.join(assets_dir, "scene_3_style_dark_glass.png"),
        "text": "Outlasts Modern Skyscrapers",
        "style": "dark_glass_badge",
    },
]

for item in demo_renders:
    if os.path.exists(item["input"]):
        base = Image.open(item["input"])
        comp, box = draw_caption(base, item["text"], style=item["style"])
        comp.save(item["output"])
        print(f"Generated {item['output']} using style {item['style']} (Box: {box})")
