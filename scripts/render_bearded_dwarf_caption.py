import os
import sys
sys.path.insert(0, os.path.abspath("src"))
from PIL import Image
from shorts_factory.captions import draw_caption

assets_dir = r"C:\Users\owner\.cursor\projects\c-Users-owner-projects-ai-shorts-factory\assets"
input_path = os.path.join(assets_dir, "scene_bearded_dwarf_split_canvas.png")
output_path = os.path.join(assets_dir, "scene_bearded_dwarf_split_canvas_captioned.png")

if os.path.exists(input_path):
    base = Image.open(input_path)
    comp, box = draw_caption(base, "SELF-HEALING STONE", style="electric_neon_yellow")
    comp.save(output_path)
    print(f"Saved: {output_path} (Box: {box})")
