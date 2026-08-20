import json
import os
import subprocess
import imageio_ffmpeg

def analyze():
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    downloads_dir = r"C:\Users\owner\Downloads"
    target_file = None
    for f in os.listdir(downloads_dir):
        if "Roman concrete" in f and f.endswith(".mp4"):
            target_file = os.path.join(downloads_dir, f)
            break
            
    if not target_file or not os.path.exists(target_file):
        print("Target video file not found!")
        return

    print(f"Found reference video: {target_file}")
    out_dir = os.path.abspath(r"assets/reference_roman_concrete")
    os.makedirs(out_dir, exist_ok=True)

    # 1. Probe video info using ffmpeg -i
    res = subprocess.run([ffmpeg_exe, "-i", target_file], capture_output=True, text=True)
    print("--- Video Metadata ---")
    print(res.stderr[:1000])

    # 2. Extract 1 frame per second
    extract_cmd = [
        ffmpeg_exe, "-y", "-i", target_file,
        "-vf", "fps=1",
        os.path.join(out_dir, "frame_%03d.png")
    ]
    subprocess.run(extract_cmd, capture_output=True, check=True)
    frames = [f for f in os.listdir(out_dir) if f.endswith(".png")]
    print(f"Extracted {len(frames)} frames to {out_dir}")

if __name__ == "__main__":
    analyze()
