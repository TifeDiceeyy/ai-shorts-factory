import os
import sys
sys.path.insert(0, os.path.abspath("src"))
from PIL import Image
from shorts_factory.captions import draw_caption

assets_dir = r"C:\Users\owner\.cursor\projects\c-Users-owner-projects-ai-shorts-factory\assets"

scenes = [
    {
        "input": os.path.join(assets_dir, "scene_1_hook_self_healing_concrete.png"),
        "output": os.path.join(assets_dir, "scene_1_captioned_final.png"),
        "text": "ALIVE AND HEALING",
    },
    {
        "input": os.path.join(assets_dir, "scene_2_chemical_slaking_reaction.png"),
        "output": os.path.join(assets_dir, "scene_2_captioned_final.png"),
        "text": "QUICKLIME. ADD WATER",
    },
    {
        "input": os.path.join(assets_dir, "scene_3_triumphant_cured_concrete_payoff.png"),
        "output": os.path.join(assets_dir, "scene_3_captioned_final.png"),
        "text": "LASTS 2,000 YEARS",
    },
]

for s in scenes:
    if os.path.exists(s["input"]):
        base = Image.open(s["input"])
        comp, box = draw_caption(base, s["text"], style="comic_top")
        comp.save(s["output"])
        print(f"Rendered captioned frame: {s['output']} with box {box}")
