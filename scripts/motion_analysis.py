"""Detect scene cuts and caption motion in a reference video."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageChops, ImageStat


def diff_stats(a: Image.Image, b: Image.Image) -> float:
    d = ImageChops.difference(a, b)
    return sum(ImageStat.Stat(d).mean) / 3


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    args = parser.parse_args()
    video = args.video.resolve()
    if not video.exists():
        print(f"Missing: {video}", file=sys.stderr)
        return 1

    ff = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg_dir = Path(ff).parent
    shim = ffmpeg_dir / "ffmpeg.exe"
    if not shim.exists():
        shutil.copy2(ff, shim)
    os.environ["PATH"] = str(ffmpeg_dir) + os.pathsep + os.environ.get("PATH", "")

    tmp = Path("artifacts/_roman_ref_tmp")
    tmp.mkdir(parents=True, exist_ok=True)

    def grab(t: float) -> Image.Image:
        out = tmp / f"f{t:.2f}.png"
        subprocess.run(
            [str(shim), "-y", "-loglevel", "error", "-ss", f"{t:.3f}", "-i", str(video), "-frames:v", "1", str(out)],
            check=True,
        )
        return Image.open(out).convert("RGB")

    cuts: list[tuple[float, float]] = []
    prev: Image.Image | None = None
    for i in range(0, 180):
        t = i * 0.5
        if t > 89:
            break
        frame = grab(t)
        if prev is not None:
            delta = diff_stats(prev, frame)
            if delta > 12:
                cuts.append((t, round(delta, 1)))
        prev = frame

    print("High-diff moments (>12 mean RGB delta):")
    for t, delta in cuts:
        print(f"  t={t:5.1f}s  diff={delta}")

    print("\nCaption zone motion t=8-10 (0.1s steps, top 22%):")
    base: Image.Image | None = None
    for step in range(0, 21):
        t = 8.0 + step * 0.1
        frame = grab(t)
        top = frame.crop((0, 0, frame.width, int(frame.height * 0.22)))
        if base is None:
            base = top
            continue
        print(f"  t={t:.1f} diff={diff_stats(base, top):.2f}")
        base = top

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
