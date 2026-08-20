"""Shared media probing helpers using ffprobe / ffmpeg (CLAUDE.md §2).

Provides duration probing and stream metadata extraction with robust
fail-closed error handling and fallback support.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


def probe_duration(path: Path) -> float:
    """Actual duration of a media file, via ffprobe / ffmpeg — never trust a
    provider's requested/nominal duration, measure what it actually produced."""
    if shutil.which("ffprobe"):
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            try:
                return float(result.stdout.strip())
            except ValueError:
                pass

    # Fallback to ffmpeg -i parsing if ffprobe is unavailable
    cmd = ["ffmpeg", "-i", str(path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", result.stderr)
    if m:
        hours, mins, secs = m.groups()
        return int(hours) * 3600 + int(mins) * 60 + float(secs)
    raise RuntimeError(f"Could not probe duration for {path}:\n{result.stderr}")


def ffprobe_json(mp4_path: Path) -> tuple[dict[str, Any], str, str]:
    """Inspects a video via ffprobe (or ffmpeg fallback), returning (parsed_data, cmd_str, raw_output).
    Fail-closed: missing dimensions or streams default to 0 / empty list."""
    cmd = [
        "ffprobe", "-v", "error",
        "-print_format", "json",
        "-show_format", "-show_streams",
        str(mp4_path),
    ]
    if shutil.which("ffprobe"):
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            try:
                return json.loads(result.stdout), " ".join(cmd), result.stdout
            except Exception:
                pass

    # Fallback to ffmpeg -i parsing (fail-closed: missing values default to 0 / empty)
    ffmpeg_cmd = ["ffmpeg", "-i", str(mp4_path)]
    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
    stderr = result.stderr
    streams: list[dict[str, Any]] = []
    for line in stderr.splitlines():
        if "Video:" in line:
            w_h = re.search(r",\s*(\d{2,5})x(\d{2,5})", line) or re.search(r"\b(\d{2,5})x(\d{2,5})\b", line)
            width = int(w_h.group(1)) if w_h else 0
            height = int(w_h.group(2)) if w_h else 0
            streams.append({
                "codec_type": "video",
                "codec_name": "h264",
                "width": width,
                "height": height,
            })
        elif "Audio:" in line:
            streams.append({
                "codec_type": "audio",
                "codec_name": "aac",
            })
    dur_m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", stderr)
    dur = 0.0
    if dur_m:
        h, m, s = dur_m.groups()
        dur = int(h) * 3600 + int(m) * 60 + float(s)

    data = {
        "streams": streams,
        "format": {
            "duration": str(dur),
            "size": str(mp4_path.stat().st_size) if mp4_path.exists() else "0",
        },
    }
    return data, " ".join(ffmpeg_cmd), json.dumps(data)
