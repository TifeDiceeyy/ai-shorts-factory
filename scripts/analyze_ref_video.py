"""Extract metadata and frames from a reference Short."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--out", type=Path, default=Path("artifacts/_ref_analysis"))
    parser.add_argument(
        "--times",
        nargs="*",
        type=float,
        default=[0, 1, 2, 3, 5, 8, 10, 12, 15, 18, 20, 22, 25, 28, 30, 32, 35, 38, 40, 42, 45, 48, 50],
    )
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

    probe = subprocess.run([str(shim), "-i", str(video)], capture_output=True, text=True)
    print("=== METADATA ===")
    print(probe.stderr[-3500:])

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    for t in args.times:
        dest = out / f"frame_{int(t):02d}.jpg"
        subprocess.run(
            [str(shim), "-y", "-loglevel", "error", "-ss", str(t), "-i", str(video), "-frames:v", "1", str(dest)],
            check=False,
        )
    frames = sorted(out.glob("frame_*.jpg"))
    print(f"Extracted {len(frames)} frames to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
