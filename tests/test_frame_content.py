"""Regression test for the verification gap found in review: extracting a
frame proved a file existed, not that it contained legible caption text. A
blank video could previously still pass. This proves analyze_caption_region()
actually distinguishes the two cases."""
from PIL import Image

from shorts_factory.captions import draw_caption
from shorts_factory.verify import analyze_caption_region


def test_blank_frame_has_no_visible_text(tmp_path):
    blank = Image.new("RGB", (1080, 1920), (80, 80, 80))
    frame_path = tmp_path / "blank.png"
    blank.save(frame_path)

    # A plausible caption-card box location (matches captions.py's placement),
    # but nothing was actually drawn there.
    box = {"left": 100, "top": 1400, "right": 980, "bottom": 1600}

    analysis = analyze_caption_region(frame_path, box)
    assert analysis["has_visible_text"] is False


def test_frame_with_real_caption_has_visible_text(tmp_path):
    base = Image.new("RGB", (1080, 1920), (80, 80, 80))
    composited, box = draw_caption(base, "This is a real caption with legible text.")
    frame_path = tmp_path / "with_caption.png"
    composited.save(frame_path)

    analysis = analyze_caption_region(frame_path, {
        "left": box.left, "top": box.top, "right": box.right, "bottom": box.bottom,
    })
    assert analysis["has_visible_text"] is True
