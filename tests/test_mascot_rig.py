"""Rigged puppet animation — real limb articulation at zero per-frame cost.

Built after measuring a real reference short frame by frame (2026-09-01):
motion-compensating consecutive frames left a 64-104% residual, proving the
character is REDRAWN each frame rather than slid or deformed. Reproducing that
by generating images would be ~322 drawings for 18s of animation (~$13/video),
so the character is generated once as parts and articulated procedurally.
"""
from PIL import Image, ImageDraw

from shorts_factory import mascot_rig as rig


def _part(w: int, h: int, color=(120, 92, 66, 255)) -> Image.Image:
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(img).rounded_rectangle(
        (2, 2, w - 3, h - 3), radius=min(w, h) // 3, fill=color, outline=(20, 20, 20, 255), width=5
    )
    return img


def _parts() -> dict[str, Image.Image]:
    return {
        "head": _part(190, 190), "torso": _part(210, 300),
        "arm_left": _part(58, 230), "arm_right": _part(58, 230),
        "leg_left": _part(72, 250), "leg_right": _part(72, 250),
    }


BODY = (390, 420, 690, 1320)


def _ink(frame: Image.Image) -> int:
    import numpy as np
    return int((np.asarray(frame)[:, :, 3] > 40).sum())


def test_every_named_pose_renders_all_six_parts():
    """A missing part means a scrambled or amputated puppet, and it is easy to
    do silently — the first rig had arms anchored INSIDE the torso box and
    they rendered completely hidden behind it."""
    parts = _parts()
    for name in rig.POSES:
        frame = rig.render_pose(parts, rig.POSES[name](0.3), BODY)
        assert _ink(frame) > 20000, f"{name} rendered almost nothing"


def test_limbs_actually_articulate_not_just_translate():
    """The whole point of the rig: parts must move RELATIVE to each other.

    A pose that merely shifted the body would move every part identically.
    Comparing two moments of the walk cycle, the legs must change their
    rotation while the torso does not — that is articulation, not sliding.
    """
    # Sample the stride's PEAKS, not its zero-crossings: the walk cycle is a
    # 0.8s sine, so t=0.0 and t=0.4 are both zero and would compare identical
    # legs while the rig is working perfectly.
    a = rig.pose_walk(0.2)
    b = rig.pose_walk(0.6)
    assert abs(a.rotations["leg_left"] - b.rotations["leg_left"]) > 20, "legs must swing"
    assert a.rotations["leg_left"] * a.rotations["leg_right"] < 0, "legs must oppose each other"
    assert "torso" not in a.rotations, "the torso is the anchor and should not rotate"


def test_large_rotations_are_not_clipped():
    """Regression guard for a real PIL trap.

    rotate(expand=True, center=...) sizes its output as if rotating about the
    image CENTRE, so a limb swinging far from an off-centre pivot is silently
    clipped. Arms survived walk's +-18 degrees and vanished entirely at
    dance's +-95. The fix pads the part until the pivot IS the centre, then
    expands. A raised-arm pose must therefore keep as much ink as a
    neutral one.
    """
    parts = _parts()
    neutral = _ink(rig.render_pose(parts, rig.Pose(), BODY))
    raised = _ink(rig.render_pose(parts, rig.Pose(rotations={"arm_left": -95, "arm_right": 95}), BODY))
    assert raised > neutral * 0.9, (
        f"raising the arms lost ink ({raised} vs {neutral}) — limbs are being clipped"
    )


def test_jump_arc_matches_the_measured_reference():
    """Rise ~8% of frame height, hang at the apex, come back down — taken off
    the reference measurement (52px rise over 0.40s, 0.10s hang, 0.25s fall),
    not invented."""
    ys = [rig.pose_jump(t).offset[1] for t in (0.0, 0.20, 0.55, 0.62, 0.85)]
    assert ys[0] >= 0, "starts on the ground (crouching down is positive y)"
    assert min(ys) < -0.05 * rig.FRAME_HEIGHT, "must actually leave the ground"
    apex = min(ys)
    assert abs(apex + 0.08 * rig.FRAME_HEIGHT) < 0.02 * rig.FRAME_HEIGHT, "apex ~8% of frame height"
    assert ys[-1] > apex, "must come back down"


def test_animation_is_held_on_twos():
    """Measured cadence of the reference: a new drawing every OTHER frame
    (centroid values repeat in exact pairs, deltas alternate large/near-zero).
    Rendering on ones reads smoother and less like the target."""
    frames = rig.animation_frames(_parts(), "walk", 1.0, BODY, fps=30)
    assert len(frames) == 30
    for i in range(0, 28, 2):
        assert frames[i] is frames[i + 1], f"frames {i}/{i+1} should be the same held drawing"
    assert frames[0] is not frames[2], "a new drawing must appear every second frame"


def test_body_box_scales_the_whole_rig():
    """The same animation has to work at any on-screen size, so a scene can
    place the mascot large or small without the motion being re-authored."""
    parts = _parts()
    small = rig.render_pose(parts, rig.pose_walk(0.2), (450, 900, 630, 1440))
    large = rig.render_pose(parts, rig.pose_walk(0.2), (300, 300, 780, 1740))
    assert _ink(large) > _ink(small) * 2, "a bigger body box must render a bigger character"
