"""Post-render verification: proves the skeleton acceptance criteria with
real ffprobe/ffmpeg command output, not eyeballing. Writes
verification-report.json with a pass/fail per criterion and returns overall
pass/fail so the pipeline can exit non-zero on failure.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from PIL import Image, ImageStat

EXPECTED_WIDTH = 1080
EXPECTED_HEIGHT = 1920
DURATION_TOLERANCE_S = 0.5
LOUDNESS_TARGET = -14.0
LOUDNESS_TOLERANCE_LU = 1.0

# Thresholds for "does this region actually contain rendered caption text",
# not just "does a frame file exist". Caption cards are a dark translucent
# box with white text (captions.py: BG_COLOR ~ (0,0,0,165), TEXT_COLOR ~
# white) — real text creates high local pixel variance (dark card + bright
# glyphs) and a meaningful count of near-white pixels. A blank/uniform frame
# (e.g. a rendering regression that drops the caption entirely) would show
# neither: flat color has near-zero variance and no white pixels.
MIN_STDDEV_FOR_TEXT = 15.0
MIN_NEAR_WHITE_FRACTION = 0.01
NEAR_WHITE_THRESHOLD = 200


def ffprobe_json(mp4_path: Path) -> dict[str, Any]:
    cmd = [
        "ffprobe", "-v", "error",
        "-print_format", "json",
        "-show_format", "-show_streams",
        str(mp4_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    return json.loads(result.stdout), " ".join(cmd), result.stdout


def extract_frame(mp4_path: Path, at_seconds: float, out_png: Path) -> tuple[bool, str]:
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{at_seconds:.2f}", "-i", str(mp4_path),
        "-frames:v", "1", str(out_png),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0 and out_png.exists(), " ".join(cmd)


def measure_final_loudness(mp4_path: Path) -> dict:
    filt = f"loudnorm=I={LOUDNESS_TARGET}:print_format=json"
    cmd = ["ffmpeg", "-y", "-loglevel", "info", "-i", str(mp4_path), "-af", filt, "-f", "null", "-"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    stderr = result.stderr
    start = stderr.rfind("{")
    end = stderr.rfind("}")
    if start == -1 or end == -1:
        raise RuntimeError(f"could not parse loudnorm stats:\n{stderr}")
    return json.loads(stderr[start:end + 1]), " ".join(cmd)


def analyze_caption_region(frame_path: Path, box: dict[str, int]) -> dict[str, Any]:
    """Inspect actual pixels inside a scene's known caption-card bounding box
    (from captions.meta.json) for evidence of real rendered text, rather than
    trusting that a caption was drawn just because metadata says so."""
    img = Image.open(frame_path).convert("L")  # grayscale: text/card contrast is luminance, not color
    crop = img.crop((box["left"], box["top"], box["right"], box["bottom"]))

    stat = ImageStat.Stat(crop)
    stddev = stat.stddev[0]

    pixels = list(crop.getdata())
    near_white_count = sum(1 for p in pixels if p >= NEAR_WHITE_THRESHOLD)
    near_white_fraction = near_white_count / len(pixels) if pixels else 0.0

    has_text = stddev >= MIN_STDDEV_FOR_TEXT and near_white_fraction >= MIN_NEAR_WHITE_FRACTION
    return {
        "stddev": round(stddev, 2),
        "stddev_threshold": MIN_STDDEV_FOR_TEXT,
        "near_white_fraction": round(near_white_fraction, 4),
        "near_white_fraction_threshold": MIN_NEAR_WHITE_FRACTION,
        "has_visible_text": has_text,
    }


def sha256_of(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_verification(
    mp4_path: Path,
    scripted_total_seconds: float,
    captions_meta_path: Path,
    cost_report_path: Path,
    budget_cap_usd: float,
    artifacts_dir: Path,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: Any):
        checks.append({"criterion": name, "passed": bool(passed), "detail": detail})

    probe, probe_cmd, probe_raw = ffprobe_json(mp4_path)
    fmt = probe.get("format", {})
    streams = probe.get("streams", [])
    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]

    add("ffprobe_command_ran", True, {"command": probe_cmd})

    width = video_streams[0]["width"] if video_streams else None
    height = video_streams[0]["height"] if video_streams else None
    add(
        "resolution_1080x1920",
        width == EXPECTED_WIDTH and height == EXPECTED_HEIGHT,
        {"width": width, "height": height},
    )

    duration = float(fmt.get("duration", -1))
    duration_ok = abs(duration - scripted_total_seconds) <= DURATION_TOLERANCE_S
    add(
        "duration_matches_script_within_0.5s",
        duration_ok,
        {"actual_duration": duration, "scripted_total": scripted_total_seconds, "tolerance": DURATION_TOLERANCE_S},
    )

    add(
        "exactly_one_video_one_audio_stream",
        len(video_streams) == 1 and len(audio_streams) == 1,
        {"video_stream_count": len(video_streams), "audio_stream_count": len(audio_streams)},
    )

    # Representative rasterized frame — grab one from the middle of the render.
    mid_t = max(0.1, duration / 2) if duration > 0 else 1.0
    frame_out = artifacts_dir / "verification_frame.png"
    frame_ok, frame_cmd = extract_frame(mp4_path, mid_t, frame_out)
    add(
        "representative_frame_extracted",
        frame_ok,
        {"command": frame_cmd, "path": str(frame_out) if frame_ok else None, "at_seconds": mid_t},
    )

    # Caption timing + safe positioning, from the recorded caption metadata.
    if captions_meta_path.exists():
        meta = json.loads(captions_meta_path.read_text(encoding="utf-8"))
        scenes_meta = meta.get("scenes", [])

        # Pixel-level check that the extracted frame ACTUALLY shows caption
        # text, not just that a frame file exists (a blank/regressed render
        # would previously still pass "representative_frame_extracted").
        active_scene = next((s for s in scenes_meta if s["start"] <= mid_t < s["end"]), None)
        if frame_ok and active_scene is not None:
            text_analysis = analyze_caption_region(frame_out, active_scene["box"])
            add(
                "frame_contains_visible_caption_text",
                text_analysis["has_visible_text"],
                {**text_analysis, "scene_checked": active_scene["caption"], "box": active_scene["box"]},
            )
        else:
            add(
                "frame_contains_visible_caption_text",
                False,
                {"error": "no extracted frame or no matching scene at mid_t", "mid_t": mid_t},
            )

        all_inside = all(s["inside_safe_area"] for s in scenes_meta)
        timing_monotonic = all(
            scenes_meta[i]["end"] == scenes_meta[i + 1]["start"] for i in range(len(scenes_meta) - 1)
        ) if len(scenes_meta) > 1 else True
        timing_covers_duration = bool(scenes_meta) and abs(scenes_meta[-1]["end"] - scripted_total_seconds) < 1e-3
        add(
            "captions_inside_safe_margins",
            all_inside,
            {"scene_count": len(scenes_meta), "boxes": [s["box"] for s in scenes_meta]},
        )
        add(
            "caption_timing_valid",
            timing_monotonic and timing_covers_duration,
            {"timing_monotonic": timing_monotonic, "covers_scripted_duration": timing_covers_duration},
        )
    else:
        add("frame_contains_visible_caption_text", False, {"error": "captions.meta.json not found"})
        add("captions_inside_safe_margins", False, {"error": "captions.meta.json not found"})
        add("caption_timing_valid", False, {"error": "captions.meta.json not found"})

    loud_stats, loud_cmd = measure_final_loudness(mp4_path)
    input_i = float(loud_stats.get("input_i", "999"))
    loud_ok = abs(input_i - LOUDNESS_TARGET) <= LOUDNESS_TOLERANCE_LU
    add(
        "integrated_loudness_within_1LU_of_-14",
        loud_ok,
        {"command": loud_cmd, "measured_input_i": input_i, "target": LOUDNESS_TARGET, "tolerance_lu": LOUDNESS_TOLERANCE_LU},
    )

    if cost_report_path.exists():
        cost_report = json.loads(cost_report_path.read_text(encoding="utf-8"))
        under_cap = cost_report.get("total_spent_usd", float("inf")) <= budget_cap_usd
        add(
            "total_cost_under_budget_cap",
            under_cap,
            {"total_spent_usd": cost_report.get("total_spent_usd"), "budget_cap_usd": budget_cap_usd},
        )
    else:
        add("total_cost_under_budget_cap", False, {"error": "cost-report.json not found"})

    add("output_sha256", True, {"sha256": sha256_of(mp4_path)})

    overall_pass = all(c["passed"] for c in checks)
    return {
        "overall_pass": overall_pass,
        "checks": checks,
    }
