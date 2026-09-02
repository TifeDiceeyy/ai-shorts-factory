"""Rigged puppet animation for the mascot — real limb articulation, zero
marginal cost per frame.

Why this exists
---------------
Measured frame by frame off a real reference short (2026-09-01), the mascot in
that video is genuinely REDRAWN every frame: motion-compensating consecutive
frames left a 64-104% residual, so it is not a cutout being slid around, and
its silhouette height stayed constant while its interior changed, so it is not
being deformed either. It is character animation — legs stepping, arms
swinging, body bobbing.

Two approaches were ruled out by measurement before landing here:
  * a mesh warp (deforming the artwork in place) — the reference deforms
    nothing, and warping a rigid prop visibly wobbled a steel road roller;
  * rigid whole-body translation — the residual test above rules it out.

Generating the real thing frame by frame would mean ~322 distinct drawings for
18s of animated content (measured), i.e. ~$13/video at $0.04 an image against a
~$0.76 total budget. So instead the character is generated ONCE as separated
body parts and articulated procedurally here: unlimited animation, no
per-frame cost.

Parts come from a single "character sheet" image — a 3x2 grid of separated
parts on white — sliced by grid position. That reuses the same trick the
ingredient_grid reveal already relies on: this image model reliably produces a
clean, evenly-spaced grid when asked for one (confirmed against real output).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from PIL import Image

from .captions import FRAME_HEIGHT, FRAME_WIDTH

# Part order matches the 3x2 character-sheet grid, reading left-to-right,
# top-to-bottom. Kept explicit so a mis-sliced sheet is obvious rather than
# silently producing a scrambled puppet.
PART_NAMES = ("head", "torso", "arm_left", "arm_right", "leg_left", "leg_right")

# Draw order, back to front. Far-side limbs sit behind the torso so the
# character reads as having depth rather than as a flat sticker collage.
Z_ORDER = ("arm_right", "leg_right", "torso", "leg_left", "arm_left", "head")


@dataclass(frozen=True)
class Joint:
    """Where a part attaches to the body, and where it pivots.

    anchor: position of the joint in BODY space, as a fraction of the body
        box (0,0 = top-left, 1,1 = bottom-right).
    pivot: the point WITHIN the part's own image that sits on that anchor,
        again as a fraction of the part image. A shoulder pivots near the top
        of the arm; a hip near the top of the leg.
    """

    anchor: tuple[float, float]
    pivot: tuple[float, float]


# A standing cartoon character, proportioned off the reference's own mascot
# (head roughly the top third, arms from the shoulders, legs from the hips).
# Arm anchors sit at 0.18/0.82 — i.e. OUTSIDE the torso's own silhouette.
# At 0.34/0.66 (first attempt) the shoulders landed inside the torso box and
# the arms rendered completely hidden behind it, which the synthetic rig test
# caught immediately.
RIG: dict[str, Joint] = {
    "torso": Joint(anchor=(0.50, 0.55), pivot=(0.50, 0.50)),
    # Head anchored at 0.40, INSIDE the torso's top edge (the torso spans
    # ~0.38-0.72 of the body box). At 0.28 the neck sat above the torso and
    # the head visibly floated detached — caught by the synthetic rig test.
    "head": Joint(anchor=(0.50, 0.40), pivot=(0.50, 0.86)),
    "arm_left": Joint(anchor=(0.82, 0.42), pivot=(0.50, 0.08)),
    "arm_right": Joint(anchor=(0.18, 0.42), pivot=(0.50, 0.08)),
    "leg_left": Joint(anchor=(0.60, 0.68), pivot=(0.50, 0.06)),
    "leg_right": Joint(anchor=(0.40, 0.68), pivot=(0.50, 0.06)),
}

# Held on twos — a new drawing every other output frame. Measured on the
# reference: centroid values repeat in exact pairs and per-frame deltas
# alternate large/near-zero. Rendering on ones reads smoother and LESS like
# the target.
ANIMATION_STEP_FRAMES = 2


@dataclass
class Pose:
    """Per-part rotation in degrees, plus a whole-body offset in pixels."""

    rotations: dict[str, float] = field(default_factory=dict)
    offset: tuple[float, float] = (0.0, 0.0)


def _wave(t: float, period: float, phase: float = 0.0) -> float:
    return math.sin(2 * math.pi * (t / period) + phase)


def pose_idle(t: float) -> Pose:
    """Breathing and a faint weight shift. Never fully still — a frozen
    character is the exact failure this whole module exists to fix."""
    return Pose(
        rotations={
            "head": 1.5 * _wave(t, 2.6),
            "arm_left": 4 * _wave(t, 2.6, 0.4),
            "arm_right": -4 * _wave(t, 2.6, 0.4),
        },
        offset=(0.0, 1.5 * _wave(t, 2.6)),
    )


def pose_walk(t: float) -> Pose:
    """Legs alternate, arms counter-swing, body bobs at twice leg frequency."""
    period = 0.8
    swing = _wave(t, period)
    return Pose(
        rotations={
            "leg_left": 26 * swing,
            "leg_right": -26 * swing,
            "arm_left": -18 * swing,
            "arm_right": 18 * swing,
            "head": 2 * _wave(t, period / 2),
        },
        offset=(0.0, -3.0 * abs(_wave(t, period / 2))),
    )


def pose_dance(t: float) -> Pose:
    """Arms up and kicking — the celebratory beat the reference opens on."""
    period = 0.7
    s = _wave(t, period)
    return Pose(
        rotations={
            "arm_left": -95 + 22 * s,
            "arm_right": 95 - 22 * s,
            "leg_left": 20 * s,
            "leg_right": -20 * s,
            "head": 6 * s,
        },
        offset=(4.0 * s, -6.0 * abs(s)),
    )


def pose_jump(t: float) -> Pose:
    """Crouch, launch, tuck at the apex, land.

    The vertical path is taken straight off the reference measurement: rise
    ~8% of frame height over 0.40s, hang ~0.10s, fall in 0.25s.
    """
    rise, hang, fall = 0.40, 0.10, 0.25
    crouch = 0.15
    cycle = crouch + rise + hang + fall
    peak = 0.08 * FRAME_HEIGHT
    tt = t % cycle
    if tt < crouch:
        p = tt / crouch
        return Pose(
            rotations={"leg_left": -14 * p, "leg_right": 14 * p, "arm_left": 20 * p, "arm_right": -20 * p},
            offset=(0.0, 8 * p),
        )
    tt -= crouch
    if tt < rise:
        p = tt / rise
        y = -peak * (1.0 - (1.0 - p) ** 2)          # ease-out going up
    elif tt < rise + hang:
        p, y = 1.0, -peak
    else:
        p = (tt - rise - hang) / fall
        y = -peak * (1.0 - p * p)                    # ease-in coming down
    return Pose(
        rotations={
            "leg_left": -22, "leg_right": 22,
            "arm_left": -70, "arm_right": 70,
            "head": -3,
        },
        offset=(0.0, y),
    )


def pose_sad(t: float) -> Pose:
    """Head down, shoulders slumped, slow — a deliberately low-energy beat."""
    return Pose(
        rotations={
            "head": 14 + 1.5 * _wave(t, 3.4),
            "arm_left": 16 + 2 * _wave(t, 3.4),
            "arm_right": -16 - 2 * _wave(t, 3.4),
        },
        offset=(0.0, 4.0 + 1.5 * _wave(t, 3.4)),
    )


def pose_point(t: float) -> Pose:
    """One arm held out, explaining. The default 'talking to camera' beat."""
    return Pose(
        rotations={
            "arm_left": -62 + 3 * _wave(t, 2.2),
            "arm_right": 6,
            "head": 2 * _wave(t, 2.2),
        },
        offset=(0.0, 1.0 * _wave(t, 2.6)),
    )


POSES: dict[str, Callable[[float], Pose]] = {
    "idle": pose_idle,
    "walk": pose_walk,
    "dance": pose_dance,
    "jump": pose_jump,
    "sad": pose_sad,
    "point": pose_point,
}


def character_sheet_prompt(mascot) -> str:
    """Prompt for the one image the whole rig is built from.

    A 3x2 grid of SEPARATED body parts. The grid is not decoration — parts are
    recovered by slicing on cell boundaries, so even spacing and clear white
    gutters matter more than beauty. The same "ask for a clean grid" trick the
    ingredient_grid reveal already depends on, which this image model was
    confirmed to follow reliably.

    Limbs are explicitly requested straight and vertical: the rig rotates each
    part about its own joint, so a limb drawn pre-bent or at an angle animates
    from the wrong rest pose.
    """
    return (
        f"{mascot.visual_style} "
        "Character parts sheet on a pure solid white #FFFFFF background, arranged as a clean 3x2 grid "
        "with 6 evenly spaced cells and generous white gutters between every cell. Each cell contains "
        "EXACTLY ONE isolated body part of the same single character, drawn separately and NOT touching "
        "any other cell. "
        "Cell 1 (top-left): the HEAD only, including any helmet or hat, facing camera. "
        "Cell 2 (top-middle): the TORSO only - chest and hips, NO head, NO arms, NO legs. "
        "Cell 3 (top-right): ONE LEFT ARM only, straight and vertical, shoulder at the top. "
        "Cell 4 (bottom-left): ONE RIGHT ARM only, straight and vertical, shoulder at the top. "
        "Cell 5 (bottom-middle): ONE LEFT LEG only, straight and vertical, hip at the top. "
        "Cell 6 (bottom-right): ONE RIGHT LEG only, straight and vertical, hip at the top. "
        "Every part uses the same character design, colours, line weight and scale so they fit together. "
        "The two legs must be drawn as TWO SEPARATE parts, not joined by a skirt, tunic or base. "
        "No weapons, no shields, no tools, no held props of any kind - hands empty. "
        "Bold black ink outlines, flat cel shading. No text, no labels, no numbers, no grid lines, "
        "no drop shadows, no complete character - only the six separated parts."
    )


def extract_parts_from_sheet(sheet_path: Path, out_dir: Path) -> dict[str, Path]:
    """Recover the six body parts from a generated character sheet.

    Parts are found as CONNECTED COMPONENTS of non-white pixels, not by
    slicing a fixed grid. A real generated sheet (2026-09-01) ignored the
    requested 3x2 layout entirely and scattered the parts at irregular
    positions, so grid slicing would have returned garbage — but the parts
    were still cleanly separated by white space, which components handle
    regardless of where the model chose to put them.

    Classification is geometric, since the model does not label anything:
    limbs are the narrow tall components (aspect > LIMB_ASPECT), the head is
    the topmost chunky one, the torso the largest remaining. Left/right is
    assigned by horizontal position.
    """
    import numpy as np
    from collections import deque

    out_dir.mkdir(parents=True, exist_ok=True)
    rgb = np.asarray(Image.open(sheet_path).convert("RGB"))

    # Drop letterbox bars — some generations come back padded with black,
    # which is neither background nor part and derails the ink threshold.
    row_lum = rgb.mean(axis=(1, 2))
    keep = np.where(row_lum > 40)[0]
    if len(keep) == 0:
        return {}
    y_off = int(keep.min())
    rgb = rgb[y_off:int(keep.max()) + 1]
    ink = (255 - rgb.astype(int)).max(axis=2) > INK_THRESHOLD

    # Label on a downsampled grid: 16x fewer pixels to walk, and parts are
    # far larger than the step so nothing real is lost.
    step = 4
    small = ink[::step, ::step]
    h, w = small.shape
    seen = np.zeros((h, w), bool)
    boxes: list[tuple[int, int, int, int, int]] = []
    for i in range(h):
        for j in range(w):
            if not small[i, j] or seen[i, j]:
                continue
            queue = deque([(i, j)])
            seen[i, j] = True
            y0 = y1 = i
            x0 = x1 = j
            count = 0
            while queue:
                y, x = queue.popleft()
                count += 1
                y0, y1 = min(y0, y), max(y1, y)
                x0, x1 = min(x0, x), max(x1, x)
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and small[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            queue.append((ny, nx))
            bh, bw = (y1 - y0) * step, (x1 - x0) * step
            if count * step * step >= MIN_PART_PIXELS and bh > 20 and bw > 20:
                boxes.append((x0 * step, y0 * step, x1 * step, y1 * step, count))

    if len(boxes) < 4:
        return {}

    sheet = Image.open(sheet_path).convert("RGBA").crop(
        (0, y_off, rgb.shape[1], y_off + rgb.shape[0])
    )
    named = _classify_parts(boxes)
    parts: dict[str, Path] = {}
    for name, box in named.items():
        cut = _to_cutout(sheet.crop(box[:4]))
        bbox = cut.getbbox()
        if bbox is None:
            continue
        path = out_dir / f"{name}.png"
        cut.crop(bbox).save(path)
        parts[name] = path

    # Mirror a missing limb from its counterpart. A real sheet (2026-09-01)
    # came back with one clean arm, one arm holding a sword, one clean leg
    # and BOTH legs joined by a skirt — so requiring six perfect parts would
    # fail on realistic output. Mirroring is standard 2D-puppet practice and
    # yields a complete, symmetrical character from an imperfect sheet.
    for missing, source in (
        ("arm_left", "arm_right"), ("arm_right", "arm_left"),
        ("leg_left", "leg_right"), ("leg_right", "leg_left"),
    ):
        if missing not in parts and source in parts:
            img = Image.open(parts[source]).transpose(Image.FLIP_LEFT_RIGHT)
            path = out_dir / f"{missing}.png"
            img.save(path)
            parts[missing] = path

    # Legs still absent (the sheet gave no narrow leg at all): derive them
    # from an arm so the character stands rather than floating as a torso.
    if "leg_left" not in parts and "arm_left" in parts:
        for leg in ("leg_left", "leg_right"):
            img = Image.open(parts["arm_left"])
            path = out_dir / f"{leg}.png"
            img.save(path)
            parts[leg] = path
    return parts


# A component this much taller than it is wide is a limb, not a head or
# torso. Measured on a real sheet: arms/legs came out at aspect 3.8-4.1
# while head/torso sat at 1.3-1.7.
LIMB_ASPECT = 2.6
MIN_PART_PIXELS = 4000
INK_THRESHOLD = 28


def _classify_parts(boxes: list[tuple[int, int, int, int, int]]) -> dict[str, tuple[int, int, int, int, int]]:
    """Assign component boxes to rig part names using geometry alone."""
    limbs, chunky = [], []
    for b in boxes:
        x0, y0, x1, y1, _ = b
        bw, bh = max(1, x1 - x0), max(1, y1 - y0)
        (limbs if bh / bw >= LIMB_ASPECT else chunky).append(b)

    named: dict[str, tuple[int, int, int, int, int]] = {}
    chunky.sort(key=lambda b: b[1])            # topmost first
    if chunky:
        named["head"] = chunky[0]
    rest = sorted(chunky[1:], key=lambda b: b[4], reverse=True)
    if rest:
        named["torso"] = rest[0]

    # Anything long-and-thin is a limb. Upper ones are arms, lower are legs;
    # within each pair, left/right by x position.
    limbs.sort(key=lambda b: b[1])
    arms, legs = limbs[:2], limbs[2:4]
    # A sheet that merged both legs into one component leaves too few limbs —
    # fall back to reusing a leg so the rig still stands rather than failing.
    for group, names in ((arms, ("arm_right", "arm_left")), (legs, ("leg_right", "leg_left"))):
        group = sorted(group, key=lambda b: b[0])
        if len(group) == 2:
            named[names[0]], named[names[1]] = group[0], group[1]
        elif len(group) == 1:
            named[names[0]] = named[names[1]] = group[0]
    return named


def _to_cutout(img: Image.Image, threshold: int = 240) -> Image.Image:
    """Near-white -> transparent, so parts can overlap without white boxes."""
    import numpy as np

    rgba = np.asarray(img.convert("RGBA")).copy()
    rgb = rgba[:, :, :3].astype(np.int16)
    dist = (255 - rgb).max(axis=2)
    rgba[:, :, 3] = np.clip((dist - (255 - threshold)) * 16, 0, 255).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


def render_pose(
    parts: dict[str, Image.Image],
    pose: Pose,
    body_box: tuple[int, int, int, int],
    canvas_size: tuple[int, int] = (FRAME_WIDTH, FRAME_HEIGHT),
) -> Image.Image:
    """Composite one animated frame.

    body_box is where the whole character sits on the canvas (l, t, r, b);
    every joint anchor is resolved inside it, so the SAME animation works at
    any on-screen size — which is what lets a scene place the mascot large or
    small without re-authoring the motion.
    """
    canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    bl, bt, br, bb = body_box
    bw, bh = br - bl, bb - bt
    ox, oy = pose.offset

    for name in Z_ORDER:
        part = parts.get(name)
        if part is None:
            continue
        joint = RIG[name]
        angle = pose.rotations.get(name, 0.0)

        # Scale so the ASSEMBLED character fits the requested body box.
        scale = bh / _natural_height(parts)
        pw, ph = max(1, round(part.width * scale)), max(1, round(part.height * scale))
        sized = part.resize((pw, ph), Image.LANCZOS)

        # Rotate about the part's own pivot by PADDING the part until the
        # pivot IS the image centre, then rotating that.
        #
        # PIL's rotate(expand=True, center=...) is a trap here: it sizes the
        # output as if rotating about the image centre, so a limb swinging
        # far from an off-centre pivot gets silently clipped. Caught in the
        # synthetic rig test — arms were fine at walk's +-18 degrees and
        # vanished entirely at dance's +-95.
        px, py = joint.pivot[0] * pw, joint.pivot[1] * ph
        pad_x = max(px, pw - px)
        pad_y = max(py, ph - py)
        padded = Image.new("RGBA", (int(2 * pad_x), int(2 * pad_y)), (0, 0, 0, 0))
        padded.alpha_composite(sized, (int(pad_x - px), int(pad_y - py)))
        # expand=True is REQUIRED here. The padded canvas is tall and narrow
        # (a 58x230 arm becomes 58x423), so rotating without expanding clips
        # the limb to the canvas width and it renders as a thin spike — seen
        # directly in the rig test. With the pivot now at the centre, expand
        # sizes the output correctly, which it could not do off-centre.
        rotated = padded.rotate(angle, resample=Image.BICUBIC, expand=True)

        ax = bl + joint.anchor[0] * bw + ox
        ay = bt + joint.anchor[1] * bh + oy
        canvas.alpha_composite(rotated, (round(ax - rotated.width / 2), round(ay - rotated.height / 2)))
    return canvas


def _natural_height(parts: dict[str, "Image.Image"]) -> float:
    """How tall the character stands at the parts' own pixel scale.

    Measured from the parts themselves rather than assumed. A fixed constant
    was tried first and was wrong in a real render: extracted parts come back
    at whatever size the generated sheet happened to use (head 324px + torso
    352px + leg 332px is a ~1000px character), so scaling against a hardcoded
    900 made the mascot overflow its box and swamp the frame.

    Head and torso overlap at the neck and the hip joint sits inside the
    torso, so the stack is discounted rather than summed naively.
    """
    head = parts["head"].height if "head" in parts else 0
    torso = parts["torso"].height if "torso" in parts else 0
    leg = max(
        (parts[n].height for n in ("leg_left", "leg_right") if n in parts),
        default=0,
    )
    return max(1.0, head * 0.72 + torso * 0.85 + leg * 0.80)


def animation_frames(
    parts: dict[str, Image.Image],
    pose_name: str,
    duration: float,
    body_box: tuple[int, int, int, int],
    fps: int = 30,
    step: int = ANIMATION_STEP_FRAMES,
) -> list[Image.Image]:
    """Every output frame for one animated beat, held on twos.

    The pose is evaluated only every `step` frames and the drawing repeated in
    between — this is the measured cadence of the reference, not a performance
    shortcut (though it is also ~2x cheaper to render).
    """
    fn = POSES.get(pose_name, pose_idle)
    total = max(1, int(round(duration * fps)))
    frames: list[Image.Image] = []
    current: Image.Image | None = None
    for i in range(total):
        if i % step == 0 or current is None:
            current = render_pose(parts, fn(i / fps), body_box)
        frames.append(current)
    return frames
